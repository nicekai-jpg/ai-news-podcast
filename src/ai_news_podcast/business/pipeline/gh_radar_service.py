"""项目雷达:每天发现一个正在起势、可上手的 AI 开源项目。

双轨制中的「项目轨」,与新闻管线数据完全隔离:
播客栏目「项目雷达」只主推评分第一名,备选(评分第 2、3 名)只进日报章节。
数字全部来自 GitHub API 实测并由代码写入,安装命令逐字来自 README 摘录,
LLM 无权自报热度或改写命令。

v4 选型设计 = 「两道硬闸 + 一个评分场」(详见 docs 与计划):
发现层 D —— 四源合并,优先复用现成榜单信号(每日 ≤2 search + ≤11 GET):
  D1 GitHub Trending 镜像 RSS(RSSHub,备源 mshibanami)抽 slug 后 GET 验活;
  D2 搜索新星榜 `created:>30d stars:>=500`(现有,结构兜底);
  D3 AI 子方向轮换榜 `topic:<轮换> created:>90d stars:>=1000`(确定性轮换);
  D4 当日新闻流挂载的 owner/repo(_fetch_github_api 已抓 AI 热门仓,零新增请求)。
考期层 T —— 读最近 3 天快照得 days_in_pool,进池首日速度 ×0.85 折;
  velocity 一律按天归一(delta/快照间隔天数),无基线退化为 stars/仓库年龄斜率。
评分场 S —— 0.35·V(0.6 池内秩百分位 + 0.4 绝对归一) + 0.25·AI 契约度(词法)
  + 0.25·可上手 + 0.10·新闻提及;同语言罚 −0.05,连续渐变、永不硬淘汰。
约束层 C —— 硬闸仅两个:可上手硬过滤(_hard_filter)与排重窗口(30 天,
  兜底冷却回选 ≥7 天按增速);扩池补搜沿 v1。AI 方向闸默认 soft
  (仅评分权重),strict 模式 A=0 出局、AI 全无时缓释放行。
观测环 O —— meta 记录昨日主推今日增速、四源占比与方向降级,零 search 增量。
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from ai_news_podcast.business.pipeline.gh_client_service import GhClient
from ai_news_podcast.data.utils_dao import read_json, write_json

logger = logging.getLogger(__name__)

_INSTALL_COMMANDS = (
    "pip install",
    "npm install",
    "cargo install",
    "brew install",
    "docker run",
)

_INSTALL_HINTS = (
    *_INSTALL_COMMANDS,
    "quickstart",
    "getting started",
    "安装",
    "快速开始",
)

_EXCERPT_HEADINGS = (
    "quickstart",
    "getting started",
    "installation",
    "install",
    "usage",
    "安装",
    "快速开始",
    "上手",
)
# 去掉空白后再匹配:"Quick start"/"Getting started" 这类带空格的标题也能命中
_EXCERPT_HEADINGS_COMPACT = tuple(re.sub(r"\s+", "", h) for h in _EXCERPT_HEADINGS)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

_GITHUB_SLUG_RE = re.compile(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")


def _readme_intro(readme: str, max_chars: int) -> str:
    """从 README 开头抽取一段「项目是做什么的」简介。

    剥掉 HTML 标签与 badge 图片,取首个章节标题之前的连续描述性文字。
    纯文本抽取,不含 LLM,保证与原文一致。
    """
    if not readme:
        return ""
    text = _MD_IMAGE_RE.sub(" ", _HTML_TAG_RE.sub(" ", readme))
    out: list[str] = []
    size = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if out:
                break
            continue
        if set(line) <= set("-=*_>· "):  # 分隔线/装饰
            continue
        out.append(line)
        size += len(line) + 1
        if size >= max_chars:
            break
    return " ".join(out).strip()[:max_chars]


@dataclass
class RadarProject:
    """单个候选项目的雷达视图。"""

    repo: str
    url: str
    description: str
    language: str
    topics: list[str]
    stars: int
    delta_stars: int | None
    is_new: bool
    created_at: str
    pushed_at: str
    license: str
    has_install_docs: bool
    mentions: int
    readme_excerpt: str = ""
    readme_intro: str = ""
    score: float = 0.0
    score_parts: dict[str, float] = field(default_factory=dict)
    affinity: float = 0.0


def _tokenize_repo_name(full_name: str) -> list[str]:
    name = full_name.split("/", 1)[-1].lower()
    return [p for p in re.split(r"[-_.]", name) if len(p) >= 4]


def count_news_mentions(full_name: str, news_titles: list[str]) -> int:
    """统计当日新闻标题对项目的提及次数(短 token 忽略以免误伤)。"""
    tokens = _tokenize_repo_name(full_name)
    if not tokens:
        return 0
    return sum(1 for title in news_titles if any(tok in title.lower() for tok in tokens))


def _is_excluded(full_name: str, topics: list[str], patterns: list[str]) -> bool:
    hay = (full_name + " " + " ".join(topics)).lower()
    return any(p.lower() in hay for p in patterns)


def _has_install_docs(readme: str) -> bool:
    if not readme:
        return False
    low = readme.lower()
    return any(hint in low for hint in _INSTALL_HINTS)


def _hands_on_excerpt(readme: str, max_chars: int) -> str:
    """从 README 截取上手小节(安装/快速开始)原文,供文案逐字引用。

    优先找标题行(Quickstart/安装 等);找不到则退化为第一条安装命令所在行起。
    """
    if not readme:
        return ""
    lines = readme.splitlines()
    start = -1
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("#") and any(
            h in re.sub(r"\s+", "", stripped) for h in _EXCERPT_HEADINGS_COMPACT
        ):
            start = i + 1
            break
    if start < 0:
        for i, line in enumerate(lines):
            low = line.lower()
            if any(h in low for h in _INSTALL_COMMANDS):
                start = max(0, i)
                break
    if start < 0:
        # 退化为第一个围栏代码块(常是安装/运行命令)
        m = re.search(r"```[^\n]*\n(.*?)```", readme, re.DOTALL)
        return m.group(1).strip()[:max_chars] if m else ""
    picked: list[str] = []
    size = 0
    for line in lines[start:]:
        if line.strip().startswith("#") and picked:
            break
        picked.append(line)
        size += len(line) + 1
        if size >= max_chars:
            break
    return "\n".join(picked).strip()[:max_chars]


def _hard_filter(item: dict[str, Any], gcfg: dict[str, Any], now: datetime) -> bool:
    """可上手硬过滤:合集/无描述/无许可证/已停更的直接出局。"""
    full_name = str(item.get("full_name", ""))
    topics = [str(t) for t in (item.get("topics") or [])]
    if _is_excluded(full_name, topics, list(gcfg.get("excluded_name_patterns", []))):
        return False
    if not str(item.get("description") or "").strip():
        return False
    license_info = item.get("license") or {}
    spdx = str(license_info.get("spdx_id") or "")
    if not spdx or spdx == "NOASSERTION":
        return False
    pushed_at = str(item.get("pushed_at") or "")
    if not pushed_at:
        return False
    pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
    return (now - pushed).days <= int(gcfg.get("recent_push_days", 21))


def _to_project(
    item: dict[str, Any], prev_stars: dict[str, int], news_titles: list[str]
) -> RadarProject:
    full_name = str(item["full_name"])
    stars = int(item.get("stargazers_count", 0))
    prev = prev_stars.get(full_name)
    license_info = item.get("license") or {}
    return RadarProject(
        repo=full_name,
        url=str(item.get("html_url") or f"https://github.com/{full_name}"),
        description=str(item.get("description") or "").strip(),
        language=str(item.get("language") or ""),
        topics=[str(t) for t in (item.get("topics") or [])],
        stars=stars,
        delta_stars=stars - prev if prev is not None else None,
        is_new=prev is None,
        created_at=str(item.get("created_at") or ""),
        pushed_at=str(item.get("pushed_at") or ""),
        license=str(license_info.get("spdx_id") or ""),
        has_install_docs=False,
        mentions=count_news_mentions(full_name, news_titles),
    )


# ── AI 方向契约度(词法) ────────────────────────────────────────────────


@lru_cache(maxsize=8)
def _compile_lexicon(
    key: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    pairs: list[tuple[str, re.Pattern[str]]] = []
    for cat, kw in key:
        # 词边界匹配以免 "rag" 误中 "storage";连字符变体(空格/下划线)同类别算命中
        word = re.escape(kw.lower()).replace(r"\-", "[-_ ]")
        pairs.append((cat, re.compile(rf"\b{word}s?\b")))
    return tuple(pairs)


def ai_affinity(item: dict[str, Any], lexicon: dict[str, list[str]]) -> float:
    """AI 方向契约度:min(1, 命中类别数 / 2);匹配范围 = 仓库名∪topics∪描述。

    词法确定性实现,零依赖;LLM 不参与方向判定。空词库(未配置)恒 0。
    """
    if not lexicon:
        return 0.0
    key = tuple((cat, kw) for cat in sorted(lexicon) for kw in lexicon[cat])
    topics = " ".join(str(t) for t in (item.get("topics") or []))
    hay = f"{item.get('full_name', '')} {topics} {item.get('description') or ''}".lower()
    cats = {cat for cat, pat in _compile_lexicon(key) if pat.search(hay)}
    return min(1.0, len(cats) / 2.0)


# ── 排重历史与快照 ────────────────────────────────────────────────────


def _picks_from_radar(data: Any) -> set[str]:
    """从单份历史 radar JSON 提取已上榜的仓库(含主推与备选)。"""
    picked: set[str] = set()
    if not isinstance(data, dict):
        return picked
    for p in data.get("projects") or []:
        if isinstance(p, dict) and p.get("repo"):
            picked.add(str(p["repo"]))
    meta = data.get("meta") or {}
    for key in ("pick_repo", "runner_up_repos"):
        v = meta.get(key)
        if isinstance(v, str) and v:
            picked.add(v)
        elif isinstance(v, list):
            picked.update(str(x) for x in v if x)
    return picked


def _load_recent_pick_dates(output_dir: Path, window_days: int, today: str) -> dict[str, str]:
    """收集最近 window_days 天内每个已推荐仓库的最近一次上榜日期(防霸榜排重)。"""
    if window_days <= 0 or not output_dir.exists():
        return {}
    base = datetime.fromisoformat(today)
    cutoff = (base - timedelta(days=window_days)).strftime("%Y-%m-%d")
    picked: dict[str, str] = {}
    for f in sorted(output_dir.glob("radar_*.json")):
        m = re.search(r"radar_(\d{4}-\d{2}-\d{2})\.json$", f.name)
        if not m or not (cutoff <= m.group(1) < today):
            continue
        try:
            data = read_json(f)
        except Exception:  # 单份历史损坏/字段类型异常(如 projects 为数字)不影响其余
            continue
        for repo in _picks_from_radar(data):
            if repo not in picked or m.group(1) > picked[repo]:
                picked[repo] = m.group(1)
    return picked


def _load_yesterday_pick(output_dir: Path, today: str) -> dict[str, str]:
    """最近一份(早于今天)历史雷达的主推 {repo, language};用于观测环与多样性。"""
    if not output_dir.exists():
        return {}
    best: dict[str, str] = {}
    best_date = ""
    for f in sorted(output_dir.glob("radar_*.json")):
        m = re.search(r"radar_(\d{4}-\d{2}-\d{2})\.json$", f.name)
        if not m or m.group(1) >= today or m.group(1) <= best_date:
            continue
        try:
            data = read_json(f)
        except Exception:
            continue
        pick = str(data.get("meta", {}).get("pick_repo") or "")
        if not pick:
            continue
        language = ""
        for p in data.get("projects") or []:
            if isinstance(p, dict) and p.get("repo") == pick:
                language = str(p.get("language") or "")
                break
        best = {"repo": pick, "language": language, "date": m.group(1)}
        best_date = m.group(1)
    return best


def _load_recent_snapshots(
    snapshot_dir: Path, today: str, lookback: int
) -> list[tuple[str, dict[str, int]]]:
    """收集最近 lookback 天内的星数快照 [(日期, stars)],日期升序。"""
    if not snapshot_dir.exists() or lookback <= 0:
        return []
    cutoff = (datetime.fromisoformat(today) - timedelta(days=lookback)).strftime("%Y-%m-%d")
    out: list[tuple[str, dict[str, int]]] = []
    for f in sorted(snapshot_dir.glob("snap_*.json")):
        m = re.search(r"snap_(\d{4}-\d{2}-\d{2})\.json$", f.name)
        if not m or not (cutoff <= m.group(1) < today):
            continue
        try:
            data = read_json(f)
            stars = data.get("stars", {})
            if isinstance(stars, dict):
                out.append((m.group(1), {str(k): int(v) for k, v in stars.items()}))
        except Exception:  # 单份快照损坏不影响其余
            continue
    return out


def _load_previous_snapshot(snapshot_dir: Path, today: str) -> tuple[str, dict[str, int]]:
    """取今天之前最近一份快照 → (日期, 星数表);无则 ("", {})。

    当天重跑不会拿到自己的快照。
    """
    best: dict[str, int] = {}
    best_date = ""
    if not snapshot_dir.exists():
        return "", best
    for f in sorted(snapshot_dir.glob("snap_*.json")):
        try:
            data = read_json(f)
            if not isinstance(data, dict):
                continue
            d = str(data.get("date", ""))
            stars = data.get("stars", {})
            # 先解析再赋值:解析失败(如 stars 值为 "1k")时不能污染 best_date
            parsed = {str(k): int(v) for k, v in stars.items()} if isinstance(stars, dict) else None
            if d < today and d > best_date and parsed is not None:
                best_date, best = d, parsed
        except Exception:  # 单份快照损坏/字段类型异常不影响其余
            continue
    return best_date, best


def _days_in_pool(snaps: list[tuple[str, dict[str, int]]], repo: str, today: str) -> int:
    """考期:仓库首次出现在快照以来的天数(0 = 今天新进池)。"""
    hits = [d for d, stars in snaps if repo in stars]
    if not hits:
        return 0
    return (datetime.fromisoformat(today) - datetime.fromisoformat(min(hits))).days


def _delta_of(item: dict[str, Any], prev_stars: dict[str, int]) -> int:
    """候选自上一份快照的星数增速;无基线返回 -1(排序时垫底)。"""
    prev = prev_stars.get(str(item.get("full_name", "")))
    stars = int(item.get("stargazers_count", 0))
    return stars - prev if prev is not None else -1


def _vel_per_day(
    item: dict[str, Any], prev_stars: dict[str, int], snap_gap_days: int, now: datetime
) -> float:
    """每日速度:优先快照差分/间隔天数;无基线退化为 stars/仓库年龄斜率。"""
    stars = int(item.get("stargazers_count", 0))
    prev = prev_stars.get(str(item.get("full_name", "")))
    if prev is not None and snap_gap_days >= 1:
        return (stars - prev) / snap_gap_days
    created = str(item.get("created_at") or "")
    try:
        age = max(1, (now - datetime.fromisoformat(created.replace("Z", "+00:00"))).days)
    except ValueError:
        age = 90
    return stars / age


# ── 候选发现(v4 四源合并) ─────────────────────────────────────────────


def _merge_hard_filtered(
    seen: dict[str, dict[str, Any]],
    sources: dict[str, str],
    items: list[dict[str, Any]],
    gcfg: dict[str, Any],
    now: datetime,
    src: str,
) -> None:
    """把一批结果经硬过滤后并入候选表(full_name 去重,先到先得,记录来源)。"""
    for it in items:
        fn = str(it.get("full_name", ""))
        if fn and fn not in seen and _hard_filter(it, gcfg, now):
            seen[fn] = it
            sources[fn] = src


def _parse_trending_slugs(rss_text: str, cap: int) -> list[str]:
    """从 Trending 镜像 RSS 文本抽 owner/repo slug(保持榜单顺序,去重)。"""
    out: list[str] = []
    seen_slug: set[str] = set()
    for m in _GITHUB_SLUG_RE.finditer(rss_text):
        slug = m.group(1).rstrip("/")
        low = slug.lower()
        if low.startswith("github/") or low.endswith(".git") or slug in seen_slug:
            continue
        seen_slug.add(slug)
        out.append(slug)
        if len(out) >= cap:
            break
    return out


async def _hydrate_slugs(
    client: GhClient,
    slugs: list[str],
    seen: dict[str, dict[str, Any]],
    sources: dict[str, str],
    gcfg: dict[str, Any],
    now: datetime,
    src: str,
) -> int:
    """把(slug)榜单逐个 GET /repos 验活并入候选池;返回成功入池数量。"""
    added = 0
    for slug in slugs:
        if slug in seen:
            continue
        try:
            data = await client.fetch_repo(slug)
        except Exception as e:  # 单仓库失败不致命
            logger.warning("repo fetch failed for %s: %s", slug, e)
            continue
        if not data:
            continue
        before = len(seen)
        _merge_hard_filtered(seen, sources, [data], gcfg, now, src)
        if len(seen) > before:
            added += 1
    return added


async def _hydrate_trending(
    client: GhClient,
    gcfg: dict[str, Any],
    now: datetime,
    seen: dict[str, dict[str, Any]],
    sources: dict[str, str],
) -> int:
    """D1:拉取 GitHub Trending 镜像 RSS,抽 slug 后实测入池。

    主源(RSSHub)失败自动换备源(mshibanami 静态馈);全部失败静默跳过。
    """
    urls = [
        u
        for u in (
            str(gcfg.get("trending_rss_url", "") or ""),
            str(gcfg.get("trending_rss_url_fallback", "") or ""),
        )
        if u
    ]
    cap = int(gcfg.get("trending_top", 6))
    for url in urls:
        try:
            text = await client.fetch_text(url)
        except Exception as e:
            logger.warning("trending RSS fetch failed for %s: %s", url, e)
            continue
        slugs = _parse_trending_slugs(text, cap)
        added = await _hydrate_slugs(client, slugs, seen, sources, gcfg, now, "trending")
        if added:
            return added
    return 0


async def _collect_candidates(
    client: GhClient,
    gcfg: dict[str, Any],
    now: datetime,
    *,
    date_str: str,
    cutoff: str,
    top_n: int,
    min_stars: int,
    news_repos: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """发现层:四源合并(D1 trending 镜像 / D2 搜索新星 / D3 主题轮换 / D4 新闻挂载)。

    返回 (候选表, 来源表);任一源失败静默跳过,不阻断其余源(可失败原则)。
    """
    seen: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    strict_query = f"created:>{cutoff} stars:>={min_stars}"
    _merge_hard_filtered(
        seen, sources, await client.search_repos(strict_query, per_page=top_n), gcfg, now, "new"
    )
    await _hydrate_trending(client, gcfg, now, seen, sources)
    if news_repos:
        cap = int(gcfg.get("news_repo_top", 4))
        hydrated = await _hydrate_slugs(
            client, list(dict.fromkeys(news_repos))[:cap], seen, sources, gcfg, now, "news"
        )
        logger.info("radar D4 news-repos hydrated: %d", hydrated)
    ai_topics = [str(t).strip() for t in gcfg.get("ai_topics", []) if str(t).strip()]
    if ai_topics:
        idx = (datetime.fromisoformat(date_str) - datetime(2000, 1, 1)).days % len(ai_topics)
        tcut = (
            datetime.fromisoformat(date_str)
            - timedelta(days=int(gcfg.get("topic_rotation_days", 90)))
        ).strftime("%Y-%m-%d")
        topic_query = (
            f"topic:{ai_topics[idx]} created:>{tcut} "
            f"stars:>={int(gcfg.get('topic_min_stars', 1000))}"
        )
        _merge_hard_filtered(
            seen,
            sources,
            await client.search_repos(topic_query, per_page=top_n),
            gcfg,
            now,
            "topic",
        )
    return seen, sources


async def _widen_search(
    client: GhClient,
    seen: dict[str, dict[str, Any]],
    sources: dict[str, str],
    gcfg: dict[str, Any],
    now: datetime,
    *,
    cutoff: str,
    min_stars: int,
    recent_pick_dates: dict[str, str],
) -> bool:
    """扩池补搜:新品不足时放宽星数门槛再搜一轮,把起势项目纳入候选。

    仅在需要时多打一次 search API;不满足触发条件或门槛不可放宽时返回 False。
    """
    ratio = float(gcfg.get("widen_min_stars_ratio", 0.4))
    pick_count = int(gcfg.get("pick_count", 1))
    fresh_count = sum(1 for fn in seen if fn not in recent_pick_dates)
    floor = int(min_stars * ratio)
    if ratio <= 0 or fresh_count >= pick_count or not (0 < floor < min_stars):
        return False
    widen_query = f"created:>{cutoff} stars:>={floor}"
    wider = await client.search_repos(widen_query, per_page=int(gcfg.get("top_n", 30)))
    _merge_hard_filtered(seen, sources, wider, gcfg, now, "new")
    return True


def _select_probe_list(
    candidates: list[dict[str, Any]],
    prev_stars: dict[str, int],
    recent_pick_dates: dict[str, str],
    *,
    date_str: str,
    pick_count: int,
    probe_limit: int,
    repeat_gap_days: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """分层挑选 README 探测列表:新品优先,不足时回选冷却满 gap 天的旧榜项目。

    返回 (fresh, probe_list):fresh 是全部未上榜过的新品,repeats 已按增速排入 probe_list。
    """
    fresh = [it for it in candidates if str(it.get("full_name", "")) not in recent_pick_dates]
    repeats: list[dict[str, Any]] = []
    if len(fresh) < pick_count:
        repeats = _repeat_eligible(candidates, recent_pick_dates, date_str, repeat_gap_days)
        repeats.sort(
            key=lambda it: (_delta_of(it, prev_stars), int(it.get("stargazers_count", 0))),
            reverse=True,
        )
    probe_list = fresh[:probe_limit]
    probe_list += repeats[: max(0, probe_limit - len(probe_list))]
    return fresh, probe_list


def _repeat_eligible(
    candidates: list[dict[str, Any]],
    recent_pick_dates: dict[str, str],
    date_str: str,
    repeat_gap_days: int,
) -> list[dict[str, Any]]:
    """回选兜底候选:排重窗口内已上榜、但距最近一次上榜至少 gap 天的项目。"""
    base = datetime.fromisoformat(date_str)
    out: list[dict[str, Any]] = []
    for it in candidates:
        last = recent_pick_dates.get(str(it.get("full_name", "")))
        if last and (base - datetime.fromisoformat(last)).days >= repeat_gap_days:
            out.append(it)
    return out


def _apply_direction_gate(
    probe_list: list[dict[str, Any]],
    affinity_map: dict[str, float],
    gcfg: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    """strict 方向闸:契约度 0 的候选出局;全场无 AI 候选时缓释放行(方向降级)。

    soft 模式方向仅作评分权重,不做二元淘汰,返回 (False, 原 probe_list)。
    """
    if str(gcfg.get("direction_gate", "soft")) != "strict":
        return False, probe_list
    ai_only = [it for it in probe_list if affinity_map.get(str(it.get("full_name", "")), 0.0) > 0]
    if not ai_only:
        return True, probe_list
    return False, ai_only


def _hands_on_of(p: RadarProject, now: datetime) -> float:
    """可上手分:0.4 README 含安装命令 + 0.3 有许可证 + 0.3/0.15 近期还在维护。"""
    hands_on = 0.4 * (1.0 if p.has_install_docs else 0.0)
    if p.license:
        hands_on += 0.3
    pushed = datetime.fromisoformat(p.pushed_at.replace("Z", "+00:00"))
    days_since_push = (now - pushed).days
    if days_since_push <= 7:
        hands_on += 0.3
    elif days_since_push <= 21:
        hands_on += 0.15
    return hands_on


def score_project(
    p: RadarProject,
    *,
    now: datetime,
    preferred_topics: list[str] = (),
    pool_ctx: dict[str, Any] | None = None,
) -> None:
    """确定性评分。

    旧口径(不传 pool_ctx,兼容历史调用/测试):
        0.5·增速(绝对/1500) + 0.3·可上手 + 0.2·交叉热度 + 0.1·AI 主题加分
    v4 评分场(传 pool_ctx):
        0.35·V + 0.25·affinity + 0.25·hands_on + 0.10·cross
        V = vel_rel·池内秩百分位 + vel_abs·绝对速度(cap=velocity_cap,星/天)
        附加渐变项:进池首日 ×fresh_penalty;与昨日主推同语言 −same_language_penalty
    """
    if pool_ctx is None:
        if p.delta_stars is not None:
            velocity = min(p.delta_stars, 1500) / 1500
        else:
            velocity = min(p.stars, 10000) / 10000 * 0.5

        hands_on = _hands_on_of(p, now)
        cross = min(p.mentions, 3) / 3
        topic_set = {t.lower() for t in p.topics}
        ai_bonus = 0.1 if any(t in topic_set for t in preferred_topics) else 0.0

        p.score = round(
            max(0.0, min(1.0, 0.5 * velocity + 0.3 * hands_on + 0.2 * cross + ai_bonus)), 4
        )
        p.score_parts = {
            "velocity": round(velocity, 4),
            "hands_on": round(hands_on, 4),
            "cross_heat": round(cross, 4),
            "ai_bonus": ai_bonus,
        }
        return

    vel_map = dict(pool_ctx.get("vel_by_repo") or {})
    mine = float(vel_map.get(p.repo, 0.0))
    cap = float(pool_ctx.get("velocity_cap", 1500)) or 1.0
    if len(vel_map) > 1:
        vel_rel = sum(1 for v in vel_map.values() if v < mine) / (len(vel_map) - 1)
    else:
        vel_rel = 0.5
    vel_abs = min(mine, cap) / cap
    velocity = (
        float(pool_ctx.get("vel_rel", 0.6)) * vel_rel
        + float(pool_ctx.get("vel_abs", 0.4)) * vel_abs
    )

    hands_on = _hands_on_of(p, now)
    cross = min(p.mentions, 3) / 3

    score = 0.35 * velocity + 0.25 * p.affinity + 0.25 * hands_on + 0.10 * cross
    days_in_pool = int(pool_ctx.get("days_in_pool", {}).get(p.repo, 0))
    if days_in_pool <= 1:
        score *= float(pool_ctx.get("fresh_penalty", 0.85))
    last_lang = str(pool_ctx.get("last_pick_language") or "").lower()
    if last_lang and p.language.lower() == last_lang:
        score -= float(pool_ctx.get("same_language_penalty", 0.05))

    p.score = round(max(0.0, min(1.0, score)), 4)
    p.score_parts = {
        "velocity": round(velocity, 4),
        "vel_rel": round(vel_rel, 4),
        "vel_abs": round(vel_abs, 4),
        "hands_on": round(hands_on, 4),
        "cross_heat": round(cross, 4),
        "affinity": round(p.affinity, 4),
        "days_in_pool": days_in_pool,
    }


async def _probe_readme(
    client: GhClient,
    p: RadarProject,
    *,
    excerpt_chars: int,
    intro_chars: int,
    intro_fallback: Callable[[str, str, str], str | None] | None,
) -> None:
    """探测单个项目的 README:抽取上手摘录 + 项目简介。

    简介优先用 LLM 输出中文(README 多为英文);LLM 不可用时退回 README 原文摘要。
    """
    try:
        readme = await client.fetch_readme_text(p.repo)
    except Exception as e:  # 单仓库 README 拉取失败不致命
        logger.warning("README probe failed for %s: %s", p.repo, e)
        readme = ""
    p.has_install_docs = _has_install_docs(readme)
    p.readme_excerpt = _hands_on_excerpt(readme, excerpt_chars)
    intro = _readme_intro(readme, intro_chars)
    if intro_fallback:
        try:
            generated = intro_fallback(p.repo, p.description, readme)
            if generated and str(generated).strip():
                intro = str(generated).strip()[:intro_chars]
        except Exception as e:
            logger.warning("radar intro LLM failed for %s: %s", p.repo, e)
    p.readme_intro = intro


async def _probe_and_score(
    client: GhClient,
    probe_list: list[dict[str, Any]],
    *,
    pool_ctx: dict[str, Any],
    preferred: list[str],
    gcfg: dict[str, Any],
    news_titles: list[str],
    now: datetime,
    intro_fallback: Callable[[str, str, str], str | None] | None,
    affinity_map: dict[str, float],
) -> list[RadarProject]:
    """探测 README 并在评分场中统一排序(主推与备选在此产生)。"""
    projects: list[RadarProject] = []
    excerpt_chars = int(gcfg.get("readme_excerpt_chars", 1200))
    intro_chars = int(gcfg.get("readme_intro_chars", 800))
    for item in probe_list:
        p = _to_project(item, pool_ctx["prev_stars"], news_titles)
        p.affinity = affinity_map.get(p.repo, 0.0)
        await _probe_readme(
            client,
            p,
            excerpt_chars=excerpt_chars,
            intro_chars=intro_chars,
            intro_fallback=intro_fallback,
        )
        score_project(p, now=now, preferred_topics=preferred, pool_ctx=pool_ctx)
        projects.append(p)
    projects.sort(key=lambda x: x.score, reverse=True)
    return projects


async def _observe_yesterday(
    client: GhClient, yesterday: dict[str, str], prev_stars: dict[str, int]
) -> tuple[str, int | None]:
    """观测环:昨日主推仓库今日增速(今日星数 − 最近快照星数),失败静默。"""
    repo = str(yesterday.get("repo") or "")
    if not repo:
        return "", None
    prev = prev_stars.get(repo)
    try:
        data = await client.fetch_repo(repo)
    except Exception as e:
        logger.warning("yesterday pick fetch failed for %s: %s", repo, e)
        return repo, None
    if not data or prev is None:
        return repo, None
    return repo, int(data.get("stargazers_count", 0)) - prev


async def build_radar(
    gcfg: dict[str, Any],
    date_str: str,
    data_dir: Path,
    news_titles: list[str],
    *,
    client: GhClient | None = None,
    now: datetime | None = None,
    intro_fallback: Callable[[str, str, str], str | None] | None = None,
    news_repos: list[str] | None = None,
) -> dict[str, Any]:
    """构建当日项目雷达。硬失败直接抛异常,由调用方决定降级。"""
    now = now or datetime.now(tz=UTC)
    if not gcfg.get("enabled", True):
        return {"date": date_str, "projects": [], "meta": {"degraded": True, "reason": "disabled"}}

    top_n = int(gcfg.get("top_n", 30))
    min_stars = int(gcfg.get("min_stars", 500))
    window_days = int(gcfg.get("created_window_days", 30))
    # cutoff 与排重窗口同源:都用剧集日期(Asia/Shanghai),而非 UTC 的 now
    cutoff = (datetime.fromisoformat(date_str) - timedelta(days=window_days)).strftime("%Y-%m-%d")

    owns_client = client is None
    if client is None:
        client = GhClient(httpx.AsyncClient(timeout=30.0))

    try:
        # 发现层:四源合并(D1/D2/D3/D4,任一源失败静默跳过)
        seen, sources = await _collect_candidates(
            client,
            gcfg,
            now,
            date_str=date_str,
            cutoff=cutoff,
            top_n=top_n,
            min_stars=min_stars,
            news_repos=news_repos or [],
        )

        snapshot_dir = data_dir / str(gcfg.get("snapshot_dir", "gh_snapshots"))
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        output_dir = data_dir / str(gcfg.get("output_dir", "gh_radar"))
        recent_pick_dates = _load_recent_pick_dates(
            output_dir, int(gcfg.get("repeat_window_days", 30)), date_str
        )

        # 扩池补搜(新品不足时放宽星数门槛,仅多打一次 search)
        widened = await _widen_search(
            client,
            seen,
            sources,
            gcfg,
            now,
            cutoff=cutoff,
            min_stars=min_stars,
            recent_pick_dates=recent_pick_dates,
        )

        candidates = sorted(
            seen.values(), key=lambda it: int(it.get("stargazers_count", 0)), reverse=True
        )
        prev_date, prev_stars = _load_previous_snapshot(snapshot_dir, date_str)
        snap_gap_days = (
            (datetime.fromisoformat(date_str) - datetime.fromisoformat(prev_date)).days
            if prev_date
            else 0
        )
        # 快照在排重和 README 截断之前写:被排重的项目、以及今天新进榜的热门项目,
        # 次日都能算出增速
        write_json(
            snapshot_dir / f"snap_{date_str}.json",
            {
                "date": date_str,
                "stars": {
                    str(it.get("full_name")): int(it.get("stargazers_count", 0))
                    for it in candidates
                },
            },
        )

        pick_count = int(gcfg.get("pick_count", 1))
        fresh, probe_list = _select_probe_list(
            candidates,
            prev_stars,
            recent_pick_dates,
            date_str=date_str,
            pick_count=pick_count,
            probe_limit=int(gcfg.get("readme_probe_limit", 12)),
            repeat_gap_days=int(gcfg.get("repeat_gap_days", 7)),
        )

        lexicon = dict(gcfg.get("ai_lexicon") or {})
        affinity_map = {str(it.get("full_name", "")): ai_affinity(it, lexicon) for it in probe_list}
        direction_fallback, probe_list = _apply_direction_gate(probe_list, affinity_map, gcfg)
        snaps = _load_recent_snapshots(snapshot_dir, date_str, 3)
        yesterday = _load_yesterday_pick(output_dir, date_str)
        pool_ctx: dict[str, Any] = {
            "prev_stars": prev_stars,
            "velocity_cap": float(gcfg.get("velocity_cap", 1500)),
            "vel_rel": float(gcfg.get("vel_rel", 0.6)),
            "vel_abs": float(gcfg.get("vel_abs", 0.4)),
            "fresh_penalty": float(gcfg.get("fresh_penalty", 0.85)),
            "same_language_penalty": float(gcfg.get("same_language_penalty", 0.05)),
            "snap_gap_days": snap_gap_days,
            "vel_by_repo": {
                str(it.get("full_name", "")): _vel_per_day(it, prev_stars, snap_gap_days, now)
                for it in probe_list
            },
            "days_in_pool": {fn: _days_in_pool(snaps, fn, date_str) for fn in affinity_map},
            "last_pick_language": str(yesterday.get("language") or ""),
        }

        preferred = [str(t).lower() for t in gcfg.get("preferred_topics", [])]
        projects = await _probe_and_score(
            client,
            probe_list,
            pool_ctx=pool_ctx,
            preferred=preferred,
            gcfg=gcfg,
            news_titles=news_titles,
            now=now,
            intro_fallback=intro_fallback,
            affinity_map=affinity_map,
        )

        keep = pick_count + int(gcfg.get("runner_up_count", 2))
        projects = projects[:keep]
        pick_repo = projects[0].repo if projects else ""
        # fresh 定义上不在排重表;上榜项目若命中 recent_pick_dates 即为回选兜底
        repeat_used = any(p.repo in recent_pick_dates for p in projects)

        yesterday_pick, yesterday_delta = await _observe_yesterday(client, yesterday, prev_stars)
        source_counts = Counter(
            sources.get(str(p_dict.get("full_name", "")), "new") for p_dict in candidates
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        radar = {
            "date": date_str,
            "generated_at": now.isoformat(),
            "projects": [asdict(p) for p in projects],
            "meta": {
                "pick_repo": pick_repo,
                "runner_up_repos": [p.repo for p in projects[1:]],
                "degraded": repeat_used or direction_fallback,
                "reason": (
                    "recent-repeat"
                    if repeat_used
                    else ("off-direction" if direction_fallback else "")
                ),
                "candidates": len(candidates),
                "excluded_recent": len(candidates) - len(fresh),
                "fresh_candidates": len(fresh),
                "widen_pass": widened,
                "pool_by_source": dict(source_counts),
                "direction_fallback": direction_fallback,
                "yesterday_pick": yesterday_pick,
                "yesterday_pick_delta": yesterday_delta,
            },
        }
        write_json(output_dir / f"radar_{date_str}.json", radar)
        return radar
    finally:
        if owns_client:
            await client.aclose()
