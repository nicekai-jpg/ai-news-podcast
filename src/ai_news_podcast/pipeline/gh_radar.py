"""项目雷达:每天发现一个正在起势、可上手的开源项目。

双轨制中的「项目轨」,与新闻管线数据完全隔离:
播客栏目「项目雷达」只主推评分第一名,备选(评分第 2、3 名)只进日报章节。
数字全部来自 GitHub API 实测并由代码写入,安装命令逐字来自 README 摘录,
LLM 无权自报热度或改写命令。
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from ai_news_podcast.pipeline.gh_client import GhClient
from ai_news_podcast.utils import read_json, write_json

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
    score: float = 0.0
    score_parts: dict[str, float] = field(default_factory=dict)


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
        if stripped.startswith("#") and any(h in stripped for h in _EXCERPT_HEADINGS):
            start = i + 1
            break
    if start < 0:
        for i, line in enumerate(lines):
            low = line.lower()
            if any(h in low for h in _INSTALL_COMMANDS):
                start = max(0, i)
                break
    if start < 0:
        return ""
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


def score_project(p: RadarProject, *, now: datetime, preferred_topics: list[str]) -> None:
    """确定性评分:0.5 增速 + 0.3 可上手 + 0.2 交叉热度 + 0.1 AI 主题加分。

    前置条件:pushed_at 必须是可解析的非空 ISO 时间戳(由 _hard_filter 保证)。
    """
    if p.delta_stars is not None:
        velocity = min(p.delta_stars, 1500) / 1500
    else:
        velocity = min(p.stars, 10000) / 10000 * 0.5

    hands_on = 0.4 * (1.0 if p.has_install_docs else 0.0)
    if p.license:
        hands_on += 0.3
    pushed = datetime.fromisoformat(p.pushed_at.replace("Z", "+00:00"))
    days_since_push = (now - pushed).days
    if days_since_push <= 7:
        hands_on += 0.3
    elif days_since_push <= 21:
        hands_on += 0.15

    cross = min(p.mentions, 3) / 3
    topic_set = {t.lower() for t in p.topics}
    ai_bonus = 0.1 if any(t in topic_set for t in preferred_topics) else 0.0

    p.score = round(max(0.0, min(1.0, 0.5 * velocity + 0.3 * hands_on + 0.2 * cross + ai_bonus)), 4)
    p.score_parts = {
        "velocity": round(velocity, 4),
        "hands_on": round(hands_on, 4),
        "cross_heat": round(cross, 4),
        "ai_bonus": ai_bonus,
    }


def _load_previous_snapshot(snapshot_dir: Path, today: str) -> dict[str, int]:
    """取今天之前最近一份快照的星数表(当天重跑不会拿到自己的快照)。"""
    best: dict[str, int] = {}
    best_date = ""
    if not snapshot_dir.exists():
        return best
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
    return best


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


def _load_recent_picks(output_dir: Path, window_days: int, today: str) -> set[str]:
    """收集最近 window_days 天已推荐过的仓库,防止热门项目天天霸榜。"""
    if window_days <= 0 or not output_dir.exists():
        return set()
    base = datetime.fromisoformat(today)
    cutoff = (base - timedelta(days=window_days)).strftime("%Y-%m-%d")
    picked: set[str] = set()
    for f in sorted(output_dir.glob("radar_*.json")):
        m = re.search(r"radar_(\d{4}-\d{2}-\d{2})\.json$", f.name)
        if not m or not (cutoff <= m.group(1) < today):
            continue
        try:
            data = read_json(f)
            picked |= _picks_from_radar(data)
        except Exception:  # 单份历史损坏/字段类型异常(如 projects 为数字)不影响其余
            continue
    return picked


async def build_radar(
    gcfg: dict[str, Any],
    date_str: str,
    data_dir: Path,
    news_titles: list[str],
    *,
    client: GhClient | None = None,
    now: datetime | None = None,
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
    query = f"created:>{cutoff} stars:>={min_stars}"

    owns_client = client is None
    if client is None:
        client = GhClient(httpx.AsyncClient(timeout=30.0))

    try:
        items = await client.search_repos(query, per_page=top_n)
        candidates = [it for it in items if _hard_filter(it, gcfg, now)]
        candidates.sort(key=lambda it: int(it.get("stargazers_count", 0)), reverse=True)

        snapshot_dir = data_dir / str(gcfg.get("snapshot_dir", "gh_snapshots"))
        snapshot_dir.mkdir(parents=True, exist_ok=True)
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
        # README 只探测榜单前几名,截断必须在快照之后
        candidates = candidates[: int(gcfg.get("readme_probe_limit", 12))]

        prev_stars = _load_previous_snapshot(snapshot_dir, date_str)
        output_dir = data_dir / str(gcfg.get("output_dir", "gh_radar"))
        recent_picks = _load_recent_picks(
            output_dir, int(gcfg.get("repeat_window_days", 30)), date_str
        )
        excluded_recent = sum(
            1 for it in candidates if str(it.get("full_name", "")) in recent_picks
        )
        preferred = [str(t).lower() for t in gcfg.get("preferred_topics", [])]
        excerpt_chars = int(gcfg.get("readme_excerpt_chars", 1200))

        projects: list[RadarProject] = []
        for item in candidates:
            if str(item.get("full_name", "")) in recent_picks:
                continue
            p = _to_project(item, prev_stars, news_titles)
            try:
                readme = await client.fetch_readme_text(p.repo)
                p.has_install_docs = _has_install_docs(readme)
                p.readme_excerpt = _hands_on_excerpt(readme, excerpt_chars)
            except Exception as e:  # 单仓库 README 拉取失败不致命
                logger.warning("README probe failed for %s: %s", p.repo, e)
            score_project(p, now=now, preferred_topics=preferred)
            projects.append(p)

        projects.sort(key=lambda x: x.score, reverse=True)
        keep = int(gcfg.get("pick_count", 1)) + int(gcfg.get("runner_up_count", 2))
        projects = projects[:keep]
        pick_repo = projects[0].repo if projects else ""

        output_dir.mkdir(parents=True, exist_ok=True)
        radar = {
            "date": date_str,
            "generated_at": now.isoformat(),
            "projects": [asdict(p) for p in projects],
            "meta": {
                "pick_repo": pick_repo,
                "runner_up_repos": [p.repo for p in projects[1:]],
                "degraded": False,
                "reason": "",
                "candidates": len(candidates),
                "excluded_recent": excluded_recent,
            },
        }
        write_json(output_dir / f"radar_{date_str}.json", radar)
        return radar
    finally:
        if owns_client:
            await client.aclose()
