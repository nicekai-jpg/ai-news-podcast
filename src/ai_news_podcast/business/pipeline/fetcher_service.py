"""Stage 1 — 新闻抓取模块

职责：RSS 抓取 → 全文提取 → URL 规范化 → 输出 RawItem 列表。
特性：readability-lxml 与 BeautifulSoup4 高效正文提取、tenacity 指数退避重试、
      同域名 1 req/sec 限速、全局并发 ≤ 4。
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RawItem schema (PLAN §1.5)
# ---------------------------------------------------------------------------


@dataclass
class RawItem:
    id: str  # sha256(normalized_link)
    title: str
    link: str
    normalized_link: str
    source_name: str
    source_category: str  # official / research / news / product / analysis / tools / events / other
    published_at: str  # RFC 3339
    summary: str
    full_text_snippet: str  # 1200‑2000 chars
    category: str  # model / product / research / open_source / policy / tool / other
    language: str  # zh / en

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RawItem:
        return cls(
            id=d.get("id", ""),
            title=d.get("title", ""),
            link=d.get("link", ""),
            normalized_link=d.get("normalized_link", ""),
            source_name=d.get("source_name", ""),
            source_category=d.get("source_category", ""),
            published_at=d.get("published_at", ""),
            summary=d.get("summary", ""),
            full_text_snippet=d.get("full_text_snippet", ""),
            category=d.get("category", ""),
            language=d.get("language", ""),
        )


# ---------------------------------------------------------------------------
# URL 规范化 (PLAN §1.3)
# ---------------------------------------------------------------------------

_UTM_PARAMS = re.compile(r"^utm_", re.IGNORECASE)


def normalize_url(url: str) -> str:
    """去 utm_*、统一 https、去尾斜杠。"""
    url = (url or "").strip()
    if not url:
        return ""
    try:
        parsed = urlparse(url)
    except (ValueError, UnicodeError):
        return url
    # 统一 https
    scheme = "https" if parsed.scheme in ("http", "https") else parsed.scheme
    # 去 utm_* 参数
    qs = parse_qs(parsed.query, keep_blank_values=True)
    cleaned = {k: v for k, v in qs.items() if not _UTM_PARAMS.match(k)}
    query = urlencode(cleaned, doseq=True) if cleaned else ""
    # 去尾斜杠
    path = parsed.path.rstrip("/") or ""
    return urlunparse((scheme, parsed.netloc.lower(), path, parsed.params, query, ""))


def _item_id(normalized_link: str) -> str:
    return hashlib.sha256(normalized_link.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 全文提取 (readability-lxml)
# ---------------------------------------------------------------------------


def _extract_fulltext(html: str, url: str, *, min_chars: int = 1200, max_chars: int = 2000) -> str:
    """使用 readability-lxml 提取网页正文内容。"""
    text = ""
    try:
        readability = importlib.import_module("readability")
        bs4 = importlib.import_module("bs4")
        doc = readability.Document(html)
        soup = bs4.BeautifulSoup(doc.summary(), "html.parser")
        text = soup.get_text(" ", strip=True)
    except (OSError, ValueError, UnicodeDecodeError):
        logger.debug("readability-lxml extraction failed for %s", url)

    # truncate
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


# ---------------------------------------------------------------------------
# HTML 去标签 (简要摘要用)
# ---------------------------------------------------------------------------


def _strip_html(s: str) -> str:
    if not s:
        return ""
    if "<" not in s and "&" not in s:
        return re.sub(r"\s+", " ", s).strip()
    bs4 = importlib.import_module("bs4")
    soup = bs4.BeautifulSoup(s, "html.parser")
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 日期解析
# ---------------------------------------------------------------------------


def _parse_dt(entry: Any) -> datetime:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = getattr(entry, key, None)
        if isinstance(parsed, tuple) and len(parsed) >= 6:
            return datetime(
                parsed[0],
                parsed[1],
                parsed[2],
                parsed[3],
                parsed[4],
                parsed[5],
                tzinfo=UTC,
            )
    return datetime.now(tz=UTC)


# ---------------------------------------------------------------------------
# 简单语言检测
# ---------------------------------------------------------------------------

_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_JUNK_SUMMARY_PATTERNS = [
    r"illustration of",
    r"collage of",
    r"photo of",
    r"image of",
    r"an illustration",
    r"a collage",
    r"a photo",
    r"featured image",
]


def _is_junk_summary(text: str) -> bool:
    """检测摘要是否为无意义的图片描述或占位符。"""
    if not text:
        return True
    low = text.lower()
    # 如果长度很短且包含图片描述词
    if len(text) < 200:
        for p in _JUNK_SUMMARY_PATTERNS:
            if re.search(p, low):
                return True
    return False


def _detect_lang(text: str) -> str:
    zh_ratio = len(_ZH_RE.findall(text)) / max(len(text), 1)
    return "zh" if zh_ratio > 0.1 else "en"


# ---------------------------------------------------------------------------
# 简单类别推断
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "model": [
        "模型",
        "model",
        "参数",
        "parameter",
        "benchmark",
        "SOTA",
        "训练",
        "training",
    ],
    "product": ["发布", "推出", "上线", "launch", "release", "产品", "API"],
    "research": ["论文", "paper", "arxiv", "研究", "research", "算法", "method"],
    "open_source": ["开源", "GitHub", "open source", "repo", "仓库"],
    "policy": [
        "政策",
        "监管",
        "安全",
        "法案",
        "regulation",
        "policy",
        "safety",
        "合规",
    ],
    "tool": ["工具", "tool", "插件", "plugin", "框架", "framework", "SDK", "库"],
}


def _infer_category(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    best, best_count = "other", 0
    for cat, kws in _CATEGORY_KEYWORDS.items():
        count = sum(1 for kw in kws if kw.lower() in text)
        if count > best_count:
            best, best_count = cat, count
    return best


# ---------------------------------------------------------------------------
# 域名限速器
# ---------------------------------------------------------------------------


class _DomainThrottle:
    """同域名 1 req/sec。"""

    def __init__(self, interval: float = 1.0):
        self._interval = interval
        self._last: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def wait(self, domain: str) -> None:
        async with self._lock:
            now = time.monotonic()
            last = self._last.get(domain, 0.0)
            wait_time = max(0.0, self._interval - (now - last))
            self._last[domain] = now + wait_time
        if wait_time > 0:
            await asyncio.sleep(wait_time)


# ---------------------------------------------------------------------------
# HTTP 抓取 (tenacity 指数退避 2s→6s)
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
async def _http_get(
    client: httpx.AsyncClient,
    url: str,
    throttle: _DomainThrottle,
) -> httpx.Response:
    domain = urlparse(url).netloc
    await throttle.wait(domain)
    resp = await client.get(url)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# 单 feed 抓取 + 全文提取
# ---------------------------------------------------------------------------


def _entry_content_html(entry: Any) -> str:
    """取 RSS 的 ``content:encoded`` 全文 HTML。

    部分源(如 wechat2rss 生成的公众号 feed)只有 ``content:encoded`` 有全文,
    ``summary`` 只有几十字; 优先用全文可以省掉一次正文抓取, 也避免微信页反爬导致
    素材只剩一个短摘要。
    """
    for block in getattr(entry, "content", None) or []:
        value = block.get("value") if isinstance(block, dict) else getattr(block, "value", "")
        if value:
            return str(value)
    return ""


async def _process_single_entry(
    entry: Any,
    *,
    name: str,
    src_category: str,
    client: httpx.AsyncClient,
    throttle: _DomainThrottle,
    max_pages: int,
    pages_counter: list[int],
) -> RawItem | None:
    """处理单个 RSS entry，返回 RawItem 或 None（如果无效）。"""
    title = _strip_html(str(getattr(entry, "title", "") or "")).strip()
    link = str(getattr(entry, "link", "") or "").strip()
    if not title or not link:
        return None

    norm_link = normalize_url(link)
    item_id = _item_id(norm_link)
    published_at = _parse_dt(entry)

    content_html = _entry_content_html(entry)
    content_text = _strip_html(content_html)
    summary_raw = (
        content_text
        or str(getattr(entry, "summary", "") or "")
        or str(getattr(entry, "description", "") or "")
    )
    summary = _strip_html(summary_raw)

    # 如果是垃圾摘要，清空它以便后续逻辑强制尝试抓取正文
    if _is_junk_summary(summary):
        logger.debug("Detected junk summary for %s, will try to fetch full text", link)
        summary = ""

    if len(summary) > 500:
        summary = summary[:497].rstrip() + "..."

    # content:encoded 已含正文时直接使用, 不再额外抓页面(公众号链接常被反爬)。
    full_text = content_text[:2000].strip()
    if not full_text and (pages_counter[0] < max_pages or not summary):
        # 如果摘要没了（是垃圾），我们稍微放宽一点限制，或者至少优先抓取
        try:
            page_resp = await _http_get(client, link, throttle)
            if not summary:  # 只有没摘要时才增加计数，避免浪费配额
                pages_counter[0] += 1
            full_text = _extract_fulltext(page_resp.text, link, min_chars=1200, max_chars=2000)
        except (httpx.HTTPError, OSError, ValueError):
            logger.debug("Full-text fetch failed: %s", link)

    # 用 summary 兜底
    if not full_text:
        full_text = summary[:2000] if summary else ""

    # 如果连正文也没抓到，且摘要是空的，这条新闻就没意义了
    if not full_text and not summary:
        return None

    lang = _detect_lang(f"{title} {summary}")
    category = _infer_category(title, summary)

    return RawItem(
        id=item_id,
        title=title,
        link=link,
        normalized_link=norm_link,
        source_name=name,
        source_category=src_category,
        published_at=published_at.isoformat(),
        summary=summary,
        full_text_snippet=full_text,
        category=category,
        language=lang,
    )


async def _fetch_one_feed(
    src: dict[str, Any],
    *,
    client: httpx.AsyncClient,
    throttle: _DomainThrottle,
    max_items: int,
    max_pages: int,
    pages_counter: list[int],  # mutable counter [current]
) -> list[RawItem]:
    """抓取单个 RSS 源并返回 RawItem 列表。"""
    feedparser = importlib.import_module("feedparser")
    name = str(src.get("name") or "").strip()
    url = str(src.get("url") or "").strip()
    src_category = str(src.get("category") or "other").strip()
    if not name or not url:
        return []

    # 获取 feed
    try:
        resp = await _http_get(client, url, throttle)
        feed = feedparser.parse(resp.content)
    except (httpx.HTTPError, OSError, ValueError):
        logger.warning("Failed to fetch feed: %s (%s)", name, url)
        return []

    entries = list(getattr(feed, "entries", []) or [])
    items: list[RawItem] = []

    for i, entry in enumerate(entries):
        if i >= max_items:
            break
        item = await _process_single_entry(
            entry,
            name=name,
            src_category=src_category,
            client=client,
            throttle=throttle,
            max_pages=max_pages,
            pages_counter=pages_counter,
        )
        if item is not None:
            items.append(item)

    return items


async def _fetch_github_api(
    src: dict[str, Any],
    *,
    client: httpx.AsyncClient,
    throttle: _DomainThrottle,
    max_items: int,
) -> list[RawItem]:
    """通过 GitHub Search API 抓取热门 AI 仓库。"""
    name = str(src.get("name") or "").strip()
    src_category = str(src.get("category") or "other").strip()
    if not name:
        return []

    # GitHub 搜索 API：最近更新的 AI 相关仓库
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    api_url = f"https://api.github.com/search/repositories?q=topic:artificial-intelligence+created:>{today}&sort=updated&order=desc&per_page=30"

    try:
        resp = await _http_get(client, api_url, throttle)
        data = resp.json()
    except (httpx.HTTPError, OSError, ValueError):
        logger.warning("Failed to fetch GitHub API: %s", name)
        return []

    items: list[RawItem] = []
    repos = data.get("items", [])

    for repo in repos[:max_items]:
        title = _strip_html(str(repo.get("full_name", "")))
        link = repo.get("html_url", "")
        if not title or not link:
            continue

        norm_link = normalize_url(link)
        item_id = _item_id(norm_link)

        # GitHub 返回 ISO 8601 时间
        created_at = repo.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            published_at = dt.isoformat()
        except ValueError:
            published_at = datetime.now(tz=UTC).isoformat()

        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)
        language = repo.get("language", "")
        description = repo.get("description", "") or ""

        summary = f"GitHub 热门仓库（⭐{stars} stars, 🍴{forks} forks）{(language and f'[{language}] ') or ''}{description[:200]}"
        full_text = (
            f"{description}\n\nStars: {stars}\nForks: {forks}\nLanguage: {language}\nURL: {link}"[
                :2000
            ]
        )

        lang = _detect_lang(f"{title} {summary}")
        category = _infer_category(title, summary)

        items.append(
            RawItem(
                id=item_id,
                title=title,
                link=link,
                normalized_link=norm_link,
                source_name=name,
                source_category=src_category,
                published_at=published_at,
                summary=summary,
                full_text_snippet=full_text,
                category=category,
                language=lang,
            )
        )

    logger.info("Fetched %d items from GitHub API: %s", len(items), name)
    return items


async def _fetch_hn_api(
    src: dict[str, Any],
    *,
    client: httpx.AsyncClient,
    throttle: _DomainThrottle,
    max_items: int,
) -> list[RawItem]:
    """通过 Hacker News Algolia API 抓取帖子。"""
    name = str(src.get("name") or "").strip()
    src_category = str(src.get("category") or "other").strip()
    if not name:
        return []

    # Algolia 搜索 API，获取 front page 内容
    api_url = "https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage=30"

    try:
        resp = await _http_get(client, api_url, throttle)
        data = resp.json()
    except (httpx.HTTPError, OSError, ValueError):
        logger.warning("Failed to fetch HN API: %s", name)
        return []

    items: list[RawItem] = []
    hits = data.get("hits", [])

    for hit in hits[:max_items]:
        title = _strip_html(str(hit.get("title", "")))
        story_url = hit.get("url", "")
        object_id = hit.get("objectID", "")
        # 如果帖子没有外部链接，用 HN 讨论页
        link = story_url or f"https://news.ycombinator.com/item?id={object_id}"
        if not title or not link:
            continue

        norm_link = normalize_url(link)
        item_id = _item_id(norm_link)

        # Algolia 返回的是毫秒级 unix 时间戳
        ts = hit.get("created_at_i", 0)
        published_at = datetime.fromtimestamp(ts, tz=UTC).isoformat()

        points = hit.get("points", 0)
        num_comments = hit.get("num_comments", 0)
        author = hit.get("author", "")
        summary = f"HN 热门讨论（{points} points, {num_comments} 评论）by {author}"

        # HN 帖子需要尝试抓取外部链接的全文
        full_text = ""
        if story_url:
            try:
                page_resp = await _http_get(client, story_url, throttle)
                full_text = _extract_fulltext(
                    page_resp.text, story_url, min_chars=600, max_chars=2000
                )
            except (httpx.HTTPError, OSError, ValueError):
                logger.debug("Full-text fetch failed for HN story: %s", story_url)

        if not full_text:
            full_text = summary

        lang = _detect_lang(f"{title} {summary}")
        category = _infer_category(title, summary)

        items.append(
            RawItem(
                id=item_id,
                title=title,
                link=link,
                normalized_link=norm_link,
                source_name=name,
                source_category=src_category,
                published_at=published_at,
                summary=summary,
                full_text_snippet=full_text,
                category=category,
                language=lang,
            )
        )

    logger.info("Fetched %d items from HN API: %s", len(items), name)
    return items


# ---------------------------------------------------------------------------
# DeepSeek 官方新闻（docs 站点没有 RSS: 用 sitemap 发现 news 页, 抓中文正文）
# ---------------------------------------------------------------------------

_DEEPSEEK_NEWS_PATH_RE = re.compile(r"/news/news(\d{2})(\d{2})(\d{2})/?$")
_DEEPSEEK_NEWS_RECENT_DAYS = 14
_DEEPSEEK_CHANGELOG_RECENT_DAYS = 14
_DEEPSEEK_NEWS_DEFAULT_SITEMAP = "https://api-docs.deepseek.com/sitemap.xml"


def _deepseek_published_at(year2: str, month: str, day: str) -> datetime | None:
    """news 页只给日期(如 2026/10/17), 统一按北京时间 20:00(UTC 12:00)近似发布时间。

    官方新闻以傍晚发布为主(如 V4.1-Flash-1017 当日 20:21 见报); 用 12:00 UTC 比用
    00:00 UTC 更接近真实发布时间, 避免时间窗/时效扣分把当天发布误判成"旧闻"。
    """
    try:
        return datetime(2000 + int(year2), int(month), int(day), 12, 0, tzinfo=UTC)
    except ValueError:
        return None


def _deepseek_zh_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, f"/zh-cn{parsed.path}", "", "", ""))


def _deepseek_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    if not match:
        return ""
    title = _strip_html(match.group(1)).strip()
    return re.sub(r"\s*\|\s*DeepSeek API Docs\s*$", "", title, flags=re.I).strip()


def _deepseek_sitemap_entries(sitemap_text: str) -> list[tuple[datetime, str]]:
    """从 sitemap 中解析出 (发布日期, 英文新闻页 URL), 按日期倒序。"""
    entries: list[tuple[datetime, str]] = []
    for loc in re.findall(r"<loc>\s*([^<]+?)\s*</loc>", sitemap_text):
        match = _DEEPSEEK_NEWS_PATH_RE.search(loc)
        if not match:
            continue
        published = _deepseek_published_at(*match.groups())
        if published is not None:
            entries.append((published, loc))
    entries.sort(key=lambda entry: entry[0], reverse=True)
    return entries


async def _get_page_text(
    client: httpx.AsyncClient,
    throttle: _DomainThrottle,
    url: str,
) -> str | None:
    """抓取页面文本, 失败返回 None(由调用方决定是否回退)。"""
    try:
        resp = await _http_get(client, url, throttle)
    except (httpx.HTTPError, OSError, ValueError):
        return None
    return resp.text


async def _fetch_deepseek_page(
    client: httpx.AsyncClient,
    throttle: _DomainThrottle,
    en_url: str,
) -> tuple[str, str] | None:
    """优先抓 zh-cn 版新闻页(中文正文), 失败回退英文页。返回 (最终链接, HTML)。"""
    zh_url = _deepseek_zh_url(en_url)
    zh_text = await _get_page_text(client, throttle, zh_url)
    if zh_text is not None:
        return zh_url, zh_text
    logger.debug("DeepSeek zh news fetch failed, fallback to en: %s", zh_url)

    en_text = await _get_page_text(client, throttle, en_url)
    if en_text is not None:
        return en_url, en_text
    logger.warning("DeepSeek news page fetch failed: %s", en_url)
    return None


def _build_deepseek_item(
    *,
    name: str,
    src_category: str,
    published: datetime,
    link: str,
    page_text: str,
) -> RawItem | None:
    """把官方新闻页 HTML 转成 RawItem; 标题缺失时返回 None。"""
    title = _deepseek_title(page_text)
    if not title:
        return None
    body = _extract_fulltext(page_text, link, min_chars=400, max_chars=2000)
    if not body:
        body = _strip_html(page_text)
    summary = body[:300].strip()
    full_text = body[:2000].strip() or summary
    norm_link = normalize_url(link)
    return RawItem(
        id=_item_id(norm_link),
        title=title,
        link=link,
        normalized_link=norm_link,
        source_name=name,
        source_category=src_category,
        published_at=published.isoformat(),
        summary=summary,
        full_text_snippet=full_text,
        category=_infer_category(title, summary),
        language="zh",
    )


def _parse_deepseek_changelog(html: str) -> list[tuple[datetime, str, str]]:
    """解析更新日志页: 每个 ``时间: YYYY-MM-DD`` 小节下的 h3 条目。

    返回 (发布日期(UTC 12:00 近似), 标题, 正文文本)。更新日志是**稳定地址**,
    不随版本发布换 URL, 可覆盖那些没有单独 news 页的 API/版本更新。
    """
    entries: list[tuple[datetime, str, str]] = []
    for section in re.finditer(r"<h2[^>]*>(.*?)</h2>(.*?)(?=<h2[^>]*>|$)", html, re.S):
        heading = _strip_html(section.group(1))
        date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", heading)
        if not date_match:
            continue
        try:
            published = datetime(
                int(date_match.group(1)),
                int(date_match.group(2)),
                int(date_match.group(3)),
                12,
                0,
                tzinfo=UTC,
            )
        except ValueError:
            continue
        for entry in re.finditer(
            r"<h3[^>]*>(.*?)</h3>(.*?)(?=<h3[^>]*>|<h2[^>]*>|$)",
            section.group(2),
            re.S,
        ):
            # Docusaurus 的标题锚点会带零宽空格(\u200b), 不清掉会污染标题。
            title = _strip_html(entry.group(1)).replace("\u200b", "").strip()
            body = _strip_html(entry.group(2)).replace("\u200b", "").strip()
            if title and body:
                entries.append((published, title, body))
    return entries


def _deepseek_changelog_items(
    *,
    name: str,
    src_category: str,
    changelog_url: str,
    html: str,
    news_dates: set[str],
) -> list[RawItem]:
    """更新日志 → RawItem; 同日期已有 news 页时跳过(新闻页信息更全, 避免重复播报)。"""
    cutoff = datetime.now(tz=UTC) - timedelta(days=_DEEPSEEK_CHANGELOG_RECENT_DAYS)
    items: list[RawItem] = []
    seen: set[str] = set()
    for published, title, body in _parse_deepseek_changelog(html):
        if published < cutoff or published.date().isoformat() in news_dates:
            continue
        # 所有更新日志条目共用同一个页面: 用 query 参数区分, 否则跨期 URL 去重
        # 会把「之前播过任意一条更新日志」误当成整个页面都播过, 从而漏掉新条目。
        slug = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
        link = f"{changelog_url}?date={published:%Y-%m-%d}&entry={slug}"
        norm_link = normalize_url(link)
        if norm_link in seen:
            continue
        seen.add(norm_link)
        summary = body[:300].strip()
        full_text = body[:2000].strip() or summary
        items.append(
            RawItem(
                id=_item_id(norm_link),
                title=title,
                link=link,
                normalized_link=norm_link,
                source_name=name,
                source_category=src_category,
                published_at=published.isoformat(),
                summary=summary,
                full_text_snippet=full_text,
                category=_infer_category(title, summary),
                language="zh",
            )
        )
    return items


async def _fetch_deepseek_news(
    src: dict[str, Any],
    *,
    client: httpx.AsyncClient,
    throttle: _DomainThrottle,
    max_items: int,
) -> list[RawItem]:
    """抓取 DeepSeek 官方新闻(一手发布信息)。

    DeepSeek 官网/docs 没有提供 RSS:
    - ``/news/newsYYMMDD`` 新闻页只出现在 sitemap 里, 每次发布都是**新地址**,
      因此每次运行都重新解析 sitemap, 不硬编码任何一期的 URL;
    - ``/zh-cn/updates`` 更新日志是**稳定地址**, 用来兜底覆盖没有单独 news 页的更新。
    """
    name = str(src.get("name") or "").strip()
    src_category = str(src.get("category") or "other").strip()
    if not name:
        return []
    sitemap_url = str(src.get("url") or "").strip() or _DEEPSEEK_NEWS_DEFAULT_SITEMAP

    try:
        sitemap_resp = await _http_get(client, sitemap_url, throttle)
    except (httpx.HTTPError, OSError, ValueError):
        logger.warning("Failed to fetch DeepSeek news sitemap: %s", name)
        return []

    entries = _deepseek_sitemap_entries(sitemap_resp.text)
    if not entries:
        logger.info("No DeepSeek news entries found in sitemap: %s", name)
        return []

    cutoff = datetime.now(tz=UTC) - timedelta(days=_DEEPSEEK_NEWS_RECENT_DAYS)
    selected = [entry for entry in entries if entry[0] >= cutoff][:max_items]
    if not selected:
        # 长期没有新版本时仍抓最新一条: 让跨期去重判断是否已播, 避免漏掉唯一一次官方发布。
        selected = entries[:1]

    items: list[RawItem] = []
    for published, en_url in selected:
        page = await _fetch_deepseek_page(client, throttle, en_url)
        if page is None:
            continue
        link, page_text = page
        item = _build_deepseek_item(
            name=name,
            src_category=src_category,
            published=published,
            link=link,
            page_text=page_text,
        )
        if item is not None:
            items.append(item)

    changelog_url = str(src.get("changelog_url") or "").strip()
    if changelog_url:
        changelog_html = await _get_page_text(client, throttle, changelog_url)
        if changelog_html:
            news_dates = {entry[0].date().isoformat() for entry in entries}
            items.extend(
                _deepseek_changelog_items(
                    name=name,
                    src_category=src_category,
                    changelog_url=changelog_url,
                    html=changelog_html,
                    news_dates=news_dates,
                )
            )

    logger.info("Fetched %d items from DeepSeek official news: %s", len(items), name)
    return items


# ---------------------------------------------------------------------------
# 抓取器注册表
# ---------------------------------------------------------------------------

_FETCHER_REGISTRY: dict[str, Any] = {
    "rss": _fetch_one_feed,
    "github_api": _fetch_github_api,
    "hn_api": _fetch_hn_api,
    "deepseek_news": _fetch_deepseek_news,
}


# ---------------------------------------------------------------------------
# 主入口 — 并发抓取所有启用源（全局并发 ≤ 4）
# ---------------------------------------------------------------------------


async def fetch_all(
    sources: list[dict[str, Any]],
    *,
    timeout_seconds: int = 20,
    connect_timeout: int = 5,
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    max_items_per_feed: int = 30,
    max_pages: int = 80,
) -> list[RawItem]:
    """抓取所有已启用源，返回 RawItem 列表。

    根据 source 中的 ``fetcher_type`` 字段自动路由到对应的抓取器：
    - ``rss``（默认）：标准 RSS 抓取
    - ``github_api``：GitHub Search API
    - ``hn_api``：Hacker News Algolia API
    - ``deepseek_news``：DeepSeek 官方新闻页(sitemap 发现, 无 RSS)
    """
    enabled = [s for s in sources if s.get("enabled", False)]
    if not enabled:
        logger.warning("No enabled sources")
        return []

    throttle = _DomainThrottle(interval=1.0)
    pages_counter: list[int] = [0]
    semaphore = asyncio.Semaphore(4)  # 全局并发 ≤ 4

    async def _guarded(src: dict[str, Any]) -> list[RawItem]:
        async with semaphore:
            fetcher_type = str(src.get("fetcher_type") or "rss").strip()
            fetcher = _FETCHER_REGISTRY.get(fetcher_type)
            if fetcher is None:
                logger.warning(
                    "Unknown fetcher_type '%s' for source '%s', falling back to rss",
                    fetcher_type,
                    src.get("name"),
                )
                fetcher = _fetch_one_feed

            if fetcher_type == "rss":
                return await fetcher(
                    src,
                    client=client,
                    throttle=throttle,
                    max_items=max_items_per_feed,
                    max_pages=max_pages,
                    pages_counter=pages_counter,
                )
            # API 抓取器不需要 pages_counter
            return await fetcher(
                src,
                client=client,
                throttle=throttle,
                max_items=max_items_per_feed,
            )

    timeout = httpx.Timeout(timeout_seconds, connect=connect_timeout)
    async with httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": user_agent},
        follow_redirects=True,
    ) as client:
        tasks = [asyncio.create_task(_guarded(src)) for src in enabled]
        try:
            # Enforce an absolute maximum of 10 minutes (600 seconds) for fetching all pages
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=600.0
            )
        except TimeoutError:
            logger.exception("Fetch all exactly hit 10 minute timeout limit. Aborting remainder.")
            # Gather any tasks that may have finished but returned before the timeout
            # We filter for tasks that are actually done to avoid hanging or raising errors
            results = [t.result() for t in tasks if t.done() and not t.cancelled()]

    all_items: list[RawItem] = []
    for result in results:
        if isinstance(result, list):
            all_items.extend(result)
        elif isinstance(result, Exception):
            logger.warning("Feed fetch error: %s", result)

    logger.info(
        "Fetched %d raw items from %d feeds (pages: %d/%d)",
        len(all_items),
        len(enabled),
        pages_counter[0],
        max_pages,
    )
    return all_items
