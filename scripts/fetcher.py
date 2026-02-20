"""Stage 1 — 新闻抓取模块

职责：RSS 抓取 → 全文提取 → URL 规范化 → 输出 RawItem 列表。
特性：trafilatura + readability-lxml 双引擎全文提取、tenacity 指数退避重试、
      同域名 1 req/sec 限速、全局并发 ≤ 4。
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import logging
import re
import time
from dataclasses import asdict, dataclass, field
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
    source_category: (
        str  # official / research / news / product / analysis / tools / events / other
    )
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
    except Exception:
        return url
    # 统一 https
    scheme = "https" if parsed.scheme in ("http", "https") else parsed.scheme
    # 去 utm_* 参数
    qs = parse_qs(parsed.query, keep_blank_values=True)
    cleaned = {k: v for k, v in qs.items() if not _UTM_PARAMS.match(k)}
    query = urlencode(cleaned, doseq=True) if cleaned else ""
    # 去尾斜杠
    path = parsed.path.rstrip("/") or ""
    normalized = urlunparse(
        (scheme, parsed.netloc.lower(), path, parsed.params, query, "")
    )
    return normalized


def _item_id(normalized_link: str) -> str:
    return hashlib.sha256(normalized_link.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 全文提取 (PLAN §1.3 — trafilatura + readability‑lxml fallback)
# ---------------------------------------------------------------------------


def _extract_fulltext(
    html: str, url: str, *, min_chars: int = 1200, max_chars: int = 2000
) -> str:
    """优先 trafilatura，不足 min_chars 时 fallback readability‑lxml。"""
    text = ""
    # 1) trafilatura
    try:
        trafilatura = importlib.import_module("trafilatura")
        text = trafilatura.extract(html, url=url, include_comments=False) or ""
    except Exception:
        logger.debug("trafilatura failed for %s", url)

    # 2) readability-lxml fallback
    if len(text) < min_chars:
        try:
            readability = importlib.import_module("readability")
            bs4 = importlib.import_module("bs4")
            doc = readability.Document(html)
            soup = bs4.BeautifulSoup(doc.summary(), "html.parser")
            alt = soup.get_text(" ", strip=True)
            if len(alt) > len(text):
                text = alt
        except Exception:
            logger.debug("readability-lxml fallback failed for %s", url)

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
        if parsed is not None:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# 简单语言检测
# ---------------------------------------------------------------------------

_ZH_RE = re.compile(r"[\u4e00-\u9fff]")


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
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last[domain] = time.monotonic()


# ---------------------------------------------------------------------------
# HTTP 抓取 (tenacity 指数退避 2s→6s)
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=6),
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
    except Exception:
        logger.warning("Failed to fetch feed: %s (%s)", name, url)
        return []

    entries = list(getattr(feed, "entries", []) or [])[:max_items]
    items: list[RawItem] = []

    for entry in entries:
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
        if len(summary) > 500:
            summary = summary[:497].rstrip() + "..."

        # 全文提取 (受 max_pages 限制)
        full_text = ""
        if pages_counter[0] < max_pages:
            try:
                page_resp = await _http_get(client, link, throttle)
                pages_counter[0] += 1
                full_text = _extract_fulltext(
                    page_resp.text, link, min_chars=1200, max_chars=2000
                )
            except Exception:
                logger.debug("Full-text fetch failed: %s", link)

        # 用 summary 兜底
        if not full_text:
            full_text = summary[:2000] if summary else ""

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
# 主入口 — 并发抓取所有启用源（全局并发 ≤ 4）
# ---------------------------------------------------------------------------


async def fetch_all(
    sources: list[dict[str, Any]],
    *,
    timeout_seconds: int = 20,
    connect_timeout: int = 5,
    user_agent: str = "ai-news-podcast/0.1",
    max_items_per_feed: int = 30,
    max_pages: int = 80,
) -> list[RawItem]:
    """抓取所有已启用 RSS 源，返回 RawItem 列表。"""
    enabled = [s for s in sources if s.get("enabled", False)]
    if not enabled:
        logger.warning("No enabled sources")
        return []

    throttle = _DomainThrottle(interval=1.0)
    pages_counter: list[int] = [0]
    semaphore = asyncio.Semaphore(4)  # 全局并发 ≤ 4

    async def _guarded(src: dict[str, Any]) -> list[RawItem]:
        async with semaphore:
            return await _fetch_one_feed(
                src,
                client=client,
                throttle=throttle,
                max_items=max_items_per_feed,
                max_pages=max_pages,
                pages_counter=pages_counter,
            )

    timeout = httpx.Timeout(timeout_seconds, connect=connect_timeout)
    async with httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": user_agent},
        follow_redirects=True,
    ) as client:
        tasks = [_guarded(src) for src in enabled]
        results = await asyncio.gather(*tasks, return_exceptions=True)

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
