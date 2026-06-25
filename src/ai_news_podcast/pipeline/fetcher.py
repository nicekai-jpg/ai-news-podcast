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
from datetime import datetime, timezone
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
    normalized = urlunparse((scheme, parsed.netloc.lower(), path, parsed.params, query, ""))
    return normalized


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
                tzinfo=timezone.utc,
            )
    return datetime.now(tz=timezone.utc)


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
        title = _strip_html(str(getattr(entry, "title", "") or "")).strip()
        link = str(getattr(entry, "link", "") or "").strip()
        if not title or not link:
            continue

        norm_link = normalize_url(link)
        item_id = _item_id(norm_link)
        published_at = _parse_dt(entry)

        summary_raw = str(getattr(entry, "summary", "") or "") or str(
            getattr(entry, "description", "") or ""
        )
        summary = _strip_html(summary_raw)

        # 如果是垃圾摘要，清空它以便后续逻辑强制尝试抓取正文
        if _is_junk_summary(summary):
            logger.debug("Detected junk summary for %s, will try to fetch full text", link)
            summary = ""

        if len(summary) > 500:
            summary = summary[:497].rstrip() + "..."

        # 全文提取 (受 max_pages 限制)
        full_text = ""
        # 如果摘要没了（是垃圾），我们稍微放宽一点限制，或者至少优先抓取
        if pages_counter[0] < max_pages or not summary:
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
            continue

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
                published_at=published_at.isoformat(),
                summary=summary,
                full_text_snippet=full_text,
                category=category,
                language=lang,
            )
        )

    return items


# ---------------------------------------------------------------------------
# JSON API 源抓取（GitHub / Hacker News）
# ---------------------------------------------------------------------------


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
    api_url = "https://api.github.com/search/repositories?q=topic:artificial-intelligence+created:>2026-06-01&sort=updated&order=desc&per_page=30"

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
            published_at = datetime.now(tz=timezone.utc).isoformat()

        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)
        language = repo.get("language", "")
        description = repo.get("description", "") or ""

        summary = f"GitHub 热门仓库（⭐{stars} stars, 🍴{forks} forks）{language and f'[{language}] ' or ''}{description[:200]}"
        full_text = f"{description}\n\nStars: {stars}\nForks: {forks}\nLanguage: {language}\nURL: {link}"[:2000]

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
        published_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        points = hit.get("points", 0)
        num_comments = hit.get("num_comments", 0)
        author = hit.get("author", "")
        summary = f"HN 热门讨论（{points} points, {num_comments} 评论）by {author}"

        # HN 帖子需要尝试抓取外部链接的全文
        full_text = ""
        if story_url:
            try:
                page_resp = await _http_get(client, story_url, throttle)
                full_text = _extract_fulltext(page_resp.text, story_url, min_chars=600, max_chars=2000)
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
# 抓取器注册表
# ---------------------------------------------------------------------------

_FETCHER_REGISTRY: dict[str, Any] = {
    "rss": _fetch_one_feed,
    "github_api": _fetch_github_api,
    "hn_api": _fetch_hn_api,
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
                logger.warning("Unknown fetcher_type '%s' for source '%s', falling back to rss", fetcher_type, src.get("name"))
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
            else:
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
        except asyncio.TimeoutError:
            logger.error("Fetch all exactly hit 10 minute timeout limit. Aborting remainder.")
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
