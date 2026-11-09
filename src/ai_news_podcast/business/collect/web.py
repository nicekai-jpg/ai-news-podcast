"""C4 网页抓取通道：7 个 SSR 源，配置驱动的 HTML 解析。

两种 parser（由 ``config/sources_v3.yaml`` 的 ``parser`` 字段选择）：

- ``generic``：按配置里的 CSS 选择器解析列表页（``item/title/link/date/summary``），
  缺选择器时退化为「锚点 + 邻近日期」启发式；适合智谱 / MiniMax / 字节 Seed /
  商汤 / 工信部 / 网信办这类 SSR 列表页；
- ``deepseek``：DeepSeek 官网（docs 站）没有 RSS，用 ``sitemap.xml`` 动态发现
  ``/news/newsYYMMDD`` 页（每期都是新地址，代码里不硬编码），再用稳定地址
  ``/zh-cn/updates`` 更新日志兜底没有单独 news 页的更新。

单源内多列表页之间失败隔离：某个 URL 抓取/解析失败只跳过该 URL；
列表页解析为空也视为失败（多半是选择器过期或被反爬）。
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from ai_news_podcast.business.contracts import Item

from .base import (
    CollectError,
    ConfigError,
    FetchContext,
    SourceConfig,
    build_item,
    detect_lang,
    extract_article_text,
    find_date_in_url,
    find_datetime,
    strip_html,
    try_fetch_text,
)

logger = logging.getLogger(__name__)

DEEPSEEK_SITEMAP_PATH_RE = re.compile(r"/news/news(\d{2})(\d{2})(\d{2})/?$")
_SKIP_HREF_PREFIXES = ("#", "javascript:", "mailto:", "tel:")
_SKIP_TITLES = {"更多", "查看更多", "阅读全文", "read more", "more", "首页", "上一页", "下一页"}
_MIN_TITLE_CHARS = 6


@dataclass
class WebEntry:
    """列表页里一条新闻的原始结构（正文在回源后补）。"""

    title: str
    link: str
    published_at: datetime | None = None
    summary: str = ""


# ---------------------------------------------------------------------------
# 通用列表页解析
# ---------------------------------------------------------------------------


def parse_html_list(
    html: str,
    *,
    page_url: str,
    selectors: dict[str, str] | None = None,
    now: datetime | None = None,
    limit: int = 50,
) -> list[WebEntry]:
    """解析列表页 HTML 为 ``WebEntry`` 列表（去重、截断）。"""
    soup = BeautifulSoup(html or "", "html.parser")
    selectors = selectors or {}
    heuristic = not selectors.get("item")
    nodes = list(soup.select(selectors["item"])) if not heuristic else list(soup.select("a[href]"))
    entries: list[WebEntry] = []
    seen: set[str] = set()
    for node in nodes:
        entry = _entry_from_node(
            node,
            page_url=page_url,
            selectors=selectors,
            now=now,
            require_date=heuristic,
        )
        if entry is None or entry.link in seen:
            continue
        seen.add(entry.link)
        entries.append(entry)
        if len(entries) >= limit:
            break
    return entries


def _entry_from_node(
    node: Any,
    *,
    page_url: str,
    selectors: dict[str, str],
    now: datetime | None,
    require_date: bool,
) -> WebEntry | None:
    anchor = _find_anchor(node, selectors)
    if anchor is None:
        return None
    href = str(anchor.get("href") or anchor.get("data-href") or "").strip()
    if not href or href.startswith(_SKIP_HREF_PREFIXES):
        return None
    link = urljoin(page_url, href)
    if not link.startswith(("http://", "https://")) or link == page_url:
        return None

    title = _strip_title_date(_node_title(node, anchor, selectors))
    if len(title) < _MIN_TITLE_CHARS or title.strip().lower() in _SKIP_TITLES:
        return None

    date_text = _select_text(node, selectors.get("date"))
    published = find_datetime(date_text, now=now)
    if published is None:
        published = find_datetime(_node_context_text(node), now=now)
    if published is None:
        published = find_date_in_url(link)
    if require_date and published is None:
        return None

    return WebEntry(
        title=title,
        link=link,
        published_at=published,
        summary=_select_text(node, selectors.get("summary")),
    )


def _find_anchor(node: Any, selectors: dict[str, str]) -> Any | None:
    selector = selectors.get("link")
    if selector:
        found = node.select_one(selector)
        if found is not None:
            if getattr(found, "name", "") == "a":
                return found
            return found.find("a", href=True) or found
    if getattr(node, "name", "") == "a" and node.get("href"):
        return node
    return node.find("a", href=True)


def _node_title(node: Any, anchor: Any, selectors: dict[str, str]) -> str:
    title_selector = selectors.get("title")
    if title_selector:
        found = node.select_one(title_selector)
        if found is not None:
            return _clean(found.get_text(" ", strip=True))
    title_attr = selectors.get("title_attr")
    if title_attr and anchor.get(title_attr):
        return _clean(str(anchor.get(title_attr)))
    if getattr(node, "name", "") == "a":
        return _clean(node.get_text(" ", strip=True))
    return _clean(anchor.get_text(" ", strip=True))


def _select_text(node: Any, selector: str | None) -> str:
    if not selector:
        return ""
    found = node.select_one(selector)
    if found is None:
        return ""
    return _clean(found.get_text(" ", strip=True))


def _node_context_text(node: Any) -> str:
    """向上最多取 3 层祖先的文本（日期常和 ``<a>`` 同在一个列表项里）。

    一旦到达「包含多个链接的容器」（如 ``<ul>``）就停止：否则会把相邻新闻的日期
    误当成当前锚点的日期，启发式解析会大量误收导航链接。
    """
    chunks: list[str] = []
    current = getattr(node, "parent", None)
    for _ in range(3):
        if current is None or getattr(current, "name", "") in {"body", "html", "[document]"}:
            break
        if len(current.find_all("a", href=True)) > 1:
            break
        text = _clean(current.get_text(" ", strip=True))
        if text:
            chunks.append(text[:200])
        if find_datetime(text) is not None:
            break
        current = getattr(current, "parent", None)
    return " ".join(chunks)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


_TITLE_DATE_SUFFIX_RE = re.compile(r"\s*20\d{2}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}\s*日?\s*$")


def _strip_title_date(title: str) -> str:
    """去掉锚点文本末尾附带的日期（部分列表页把标题和日期放在同一个 ``<a>`` 里）。"""
    return _TITLE_DATE_SUFFIX_RE.sub("", title).strip()


# ---------------------------------------------------------------------------
# 通用列表页 → Item
# ---------------------------------------------------------------------------


def _fetch_generic(source: SourceConfig, ctx: FetchContext) -> list[Item]:
    urls = source.page_urls()
    if not urls:
        raise ConfigError(f"源 {source.name} (c4_web) 缺少 url/urls")

    entries: list[WebEntry] = []
    failures: list[str] = []
    for page_url in urls:
        html = try_fetch_text(ctx, page_url)
        if html is None:
            failures.append(f"{page_url} (抓取失败)")
            continue
        try:
            parsed = parse_html_list(
                html,
                page_url=page_url,
                selectors=source.selectors,
                now=ctx.now,
                limit=max(source.limit, 20),
            )
        except (TypeError, ValueError, LookupError) as exc:
            failures.append(f"{page_url} (解析失败: {exc})")
            continue
        if not parsed:
            failures.append(f"{page_url} (未解析到条目)")
        entries.extend(parsed)

    if not entries:
        raise CollectError(f"源 {source.name} 列表页全部失败: {'; '.join(failures[:3])}")

    items: list[Item] = []
    for entry in entries:
        item = _build_web_item(entry, source, ctx)
        if item is not None:
            items.append(item)
    return items


def _build_web_item(entry: WebEntry, source: SourceConfig, ctx: FetchContext) -> Item | None:
    body = ""
    status = "none"
    if ctx.fetch_fulltext:
        page = try_fetch_text(ctx, entry.link)
        if page is None:
            status = "failed"
        else:
            body = extract_article_text(page, entry.link)
            status = "ok" if body else "failed"
    extra: dict[str, Any] = {}
    if entry.published_at is None:
        extra["no_published_at"] = True
    return build_item(
        channel=source.channel,
        source=source.name,
        source_label=source.label,
        category=source.category,
        title=entry.title,
        link=entry.link,
        published_at=entry.published_at.isoformat() if entry.published_at else "",
        summary=entry.summary or body,
        full_text=body,
        full_text_status=status,
        lang=detect_lang(f"{entry.title} {entry.summary}"),
        extra=extra,
        fetched_at=ctx.fetched_at,
    )


# ---------------------------------------------------------------------------
# DeepSeek 官网（sitemap 动态发现 + 更新日志兜底）
# ---------------------------------------------------------------------------


def _news_date(year2: str, month: str, day: str) -> datetime | None:
    """news 页只给日期，统一按 UTC 12:00（北京时间 20:00）近似发布时间。"""
    try:
        return datetime(2000 + int(year2), int(month), int(day), 12, 0, tzinfo=UTC)
    except ValueError:
        return None


def parse_sitemap_news(sitemap_text: str) -> list[tuple[datetime, str]]:
    """sitemap → ``(发布日期, 英文新闻页 URL)``，按日期倒序。"""
    entries: list[tuple[datetime, str]] = []
    for loc in re.findall(r"<loc>\s*([^<]+?)\s*</loc>", sitemap_text or ""):
        match = DEEPSEEK_SITEMAP_PATH_RE.search(loc)
        if match is None:
            continue
        published = _news_date(*match.groups())
        if published is not None:
            entries.append((published, loc))
    entries.sort(key=lambda entry: entry[0], reverse=True)
    return entries


def to_zh_url(url: str) -> str:
    """英文 news 页 URL → 中文（``/zh-cn`` 前缀）页 URL。"""
    parts = urlsplit(url)
    if parts.path.startswith("/zh-cn"):
        return url
    return urlunsplit((parts.scheme, parts.netloc, f"/zh-cn{parts.path}", parts.query, ""))


def parse_page_title(html: str) -> str:
    """从新闻页 ``<title>`` 提取标题（去掉站点后缀）。"""
    match = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.S | re.I)
    if match is None:
        return ""
    title = strip_html(match.group(1))
    return re.sub(r"\s*[|｜-]\s*DeepSeek.*$", "", title, flags=re.I).strip()


def parse_changelog(html: str) -> list[tuple[datetime, str, str]]:
    """解析更新日志页：``<h2>时间: YYYY-MM-DD</h2>`` 下的 ``<h3>`` 条目。

    返回 ``(发布日期, 标题, 正文文本)``；Docusaurus 标题锚点里的零宽空格会清掉。
    """
    entries: list[tuple[datetime, str, str]] = []
    for section in re.finditer(r"<h2[^>]*>(.*?)</h2>(.*?)(?=<h2[^>]*>|$)", html or "", re.S):
        heading = strip_html(section.group(1))
        date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", heading)
        if date_match is None:
            continue
        published = _news_date(date_match.group(1)[2:], date_match.group(2), date_match.group(3))
        if published is None:
            continue
        for entry in re.finditer(
            r"<h3[^>]*>(.*?)</h3>(.*?)(?=<h3[^>]*>|<h2[^>]*>|$)",
            section.group(2),
            re.S,
        ):
            title = strip_html(entry.group(1)).replace("\u200b", "").strip()
            body = strip_html(entry.group(2)).replace("\u200b", "").strip()
            if title and body:
                entries.append((published, title, body))
    return entries


def _fetch_deepseek_page(ctx: FetchContext, en_url: str) -> tuple[str, str] | None:
    """优先中文页，失败回退英文页；返回 ``(最终链接, HTML)``。"""
    for candidate in (to_zh_url(en_url), en_url):
        page = try_fetch_text(ctx, candidate)
        if page is not None:
            return candidate, page
    logger.warning("DeepSeek 新闻页抓取失败: %s", en_url)
    return None


def _deepseek_changelog_items(
    *,
    source: SourceConfig,
    ctx: FetchContext,
    changelog_url: str,
    html: str,
    news_dates: set[str],
    cutoff: datetime,
) -> list[Item]:
    """更新日志 → item；同日期已有 news 页时跳过（news 页信息更全）。"""
    items: list[Item] = []
    for published, title, body in parse_changelog(html):
        if published < cutoff or published.date().isoformat() in news_dates:
            continue
        # 所有更新日志条目共用同一个页面：用 query 参数区分，避免跨期 URL 去重
        # 把「之前播过任意一条更新日志」误当成整个页面都播过。
        slug = hashlib.sha256(title.encode("utf-8")).hexdigest()[:8]
        link = f"{changelog_url}?date={published:%Y-%m-%d}&entry={slug}"
        items.append(
            build_item(
                channel=source.channel,
                source=source.name,
                source_label=source.label,
                category=source.category,
                title=title,
                link=link,
                published_at=published.isoformat(),
                summary=body,
                full_text=body,
                full_text_status="ok",
                lang="zh",
                extra={"entry": "changelog"},
                fetched_at=ctx.fetched_at,
            )
        )
    return items


def _fetch_deepseek(source: SourceConfig, ctx: FetchContext) -> list[Item]:
    """DeepSeek 官方新闻：sitemap 动态发现 news 页 + changelog 兜底。"""
    sitemap_url = source.url or (source.page_urls()[0] if source.page_urls() else "")
    if not sitemap_url:
        raise ConfigError(f"源 {source.name} (deepseek) 缺少 sitemap url")
    sitemap_text = try_fetch_text(ctx, sitemap_url)
    if sitemap_text is None:
        raise CollectError(f"源 {source.name} sitemap 抓取失败: {sitemap_url}")

    news_entries = parse_sitemap_news(sitemap_text)
    cutoff = ctx.now - timedelta(hours=ctx.window_hours)
    selected = [entry for entry in news_entries if entry[0] >= cutoff][: max(source.limit, 1)]

    items: list[Item] = []
    for published, en_url in selected:
        page = _fetch_deepseek_page(ctx, en_url)
        if page is None:
            continue
        link, html = page
        title = parse_page_title(html)
        if not title:
            continue
        body = extract_article_text(html, link)
        if not body:
            body = strip_html(html)
        items.append(
            build_item(
                channel=source.channel,
                source=source.name,
                source_label=source.label,
                category=source.category,
                title=title,
                link=link,
                published_at=published.isoformat(),
                summary=body,
                full_text=body,
                full_text_status="ok" if body else "failed",
                lang="zh",
                extra={"entry": "news"},
                fetched_at=ctx.fetched_at,
            )
        )

    if source.changelog_url:
        changelog_html = try_fetch_text(ctx, source.changelog_url)
        if changelog_html:
            news_dates = {entry[0].date().isoformat() for entry in news_entries}
            items.extend(
                _deepseek_changelog_items(
                    source=source,
                    ctx=ctx,
                    changelog_url=source.changelog_url,
                    html=changelog_html,
                    news_dates=news_dates,
                    cutoff=cutoff,
                )
            )
    if not items:
        logger.info("DeepSeek 官网窗口内无新条目: %s", source.name)
    return items


# ---------------------------------------------------------------------------
# 通道入口
# ---------------------------------------------------------------------------

_PARSERS: dict[str, Callable[[SourceConfig, FetchContext], list[Item]]] = {
    "generic": _fetch_generic,
    "deepseek": _fetch_deepseek,
}


def fetch_web(source: SourceConfig, ctx: FetchContext) -> list[Item]:
    """按配置的 ``parser`` 抓取一个 C4 网页源。"""
    parser = (source.parser or "generic").strip().lower()
    handler = _PARSERS.get(parser)
    if handler is None:
        raise ConfigError(f"源 {source.name} 的 c4_web parser 未知: {parser!r}")
    return handler(source, ctx)


__all__ = [
    "WebEntry",
    "fetch_web",
    "parse_changelog",
    "parse_html_list",
    "parse_page_title",
    "parse_sitemap_news",
    "to_zh_url",
]
