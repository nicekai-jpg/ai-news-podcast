"""通用 RSS 2.0 / Atom 1.0 解析（feedparser）。

C1（固定 RSS）、C2（Atom）、C3（RSSHub）、C5（wechat2rss）共用这一层：

- feedparser 统一展开 RSS/Atom，``content:encoded`` 与 Atom ``content`` 都落在
  ``entry.content``，wechat2rss 这类「只有全文、摘要只有几十字」的 feed 因此可用；
- 正文优先级：feed 全文 → 长摘要（≥800 字）→ 抓文章页用 readability 抽正文；
  全部失败时 ``full_text_status=failed``、正文留空、摘要保留（下游按摘要降级）；
- 只做单源内的时效过滤 / URL 去重 / 条数上限（``base.finalize_items``），
  跨源去重与重要性判断交给 Stage 2。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import feedparser

from ai_news_podcast.business.contracts import Item

from .base import (
    FULL_TEXT_MIN_CHARS,
    CollectError,
    ConfigError,
    FetchContext,
    ParseError,
    SourceConfig,
    build_item,
    detect_lang,
    extract_article_text,
    fetch_bytes,
    parse_datetime,
    strip_html,
    try_fetch_text,
)

logger = logging.getLogger(__name__)


@dataclass
class FeedEntry:
    """feedparser 条目归一化后的中间结构。"""

    title: str = ""
    link: str = ""
    published_at: datetime | None = None
    summary_html: str = ""
    content_html: str = ""
    author: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def parse_feed(content: bytes | str) -> list[FeedEntry]:
    """解析 RSS/Atom 字节流为 ``FeedEntry`` 列表。

    feed 本身可解析但没有条目（如空 releases feed）时返回空列表；
    bozo（格式损坏）且无条目时抛 ``ParseError``，交由 runner 记录为单源失败。
    """
    parsed = feedparser.parse(content)
    entries = list(getattr(parsed, "entries", []) or [])
    if not entries and getattr(parsed, "bozo", False):
        raise ParseError(f"feed 解析失败: {getattr(parsed, 'bozo_exception', '')}")
    return [_to_feed_entry(entry) for entry in entries]


def _to_feed_entry(entry: Any) -> FeedEntry:
    return FeedEntry(
        title=strip_html(str(entry.get("title") or "")),
        link=_entry_link(entry),
        published_at=_entry_datetime(entry),
        summary_html=_entry_summary_html(entry),
        content_html=_entry_content_html(entry),
        author=strip_html(str(entry.get("author") or "")),
    )


def _entry_link(entry: Any) -> str:
    links = list(entry.get("links") or [])
    for candidate in links:
        href = str(candidate.get("href") or "").strip()
        if href and str(candidate.get("rel") or "alternate") == "alternate":
            return href
    for candidate in links:
        href = str(candidate.get("href") or "").strip()
        if href:
            return href
    link = str(entry.get("link") or "").strip()
    if link:
        return link
    identifier = str(entry.get("id") or "").strip()
    return identifier if identifier.startswith(("http://", "https://")) else ""


def _entry_datetime(entry: Any) -> datetime | None:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        value = entry.get(key)
        if not value:
            continue
        try:
            return datetime(*value[:6], tzinfo=UTC)
        except (TypeError, ValueError):
            continue
    for key in ("published", "updated", "created", "dc_date"):
        parsed = parse_datetime(str(entry.get(key) or ""))
        if parsed is not None:
            return parsed
    return None


def _entry_summary_html(entry: Any) -> str:
    detail = entry.get("summary_detail") or {}
    if isinstance(detail, dict):
        value = detail.get("value")
        if value:
            return str(value)
    return str(entry.get("summary") or entry.get("description") or "")


def _entry_content_html(entry: Any) -> str:
    """取 feed 自带全文：RSS ``content:encoded`` 与 Atom ``content``。"""
    for block in entry.get("content") or []:
        value = block.get("value") if isinstance(block, dict) else getattr(block, "value", "")
        if value:
            return str(value)
    return str(entry.get("content_encoded") or "")


def build_item_from_entry(
    entry: FeedEntry,
    source: SourceConfig,
    ctx: FetchContext,
    *,
    allow_page_fetch: bool = True,
) -> Item | None:
    """单个 feed 条目 → ``contracts.Item``（含正文获取优先级）。"""
    if not entry.title or not entry.link:
        return None

    content_text = strip_html(entry.content_html)
    summary_text = strip_html(entry.summary_html)
    summary = summary_text or content_text
    body = ""
    status = "none"

    if content_text:
        body, status = content_text, "ok"
    elif len(summary_text) >= FULL_TEXT_MIN_CHARS:
        body, status = summary_text, "ok"
    elif allow_page_fetch and ctx.fetch_fulltext:
        body, status = _fetch_page_body(ctx, entry.link)
    elif summary_text:
        # 不回源抓取（如公众号 feed）：正文缺失但保留摘要，标记 failed 供下游降级。
        status = "failed"

    extra = dict(entry.extra)
    if entry.author:
        extra["author"] = entry.author
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
        summary=summary,
        full_text=body,
        full_text_status=status,
        lang=detect_lang(f"{entry.title} {summary}"),
        extra=extra,
        fetched_at=ctx.fetched_at,
    )


def _fetch_page_body(ctx: FetchContext, link: str) -> tuple[str, str]:
    """回源文章页抽正文，返回 ``(正文, full_text_status)``。"""
    page = try_fetch_text(ctx, link)
    if page is None:
        return "", "failed"
    text = extract_article_text(page, link)
    if not text:
        return "", "failed"
    return text, "ok"


def fetch_feed(
    source: SourceConfig,
    ctx: FetchContext,
    *,
    urls: list[str] | None = None,
    allow_page_fetch: bool = True,
) -> list[Item]:
    """抓取一个或多个 feed URL 并解析为 item 列表。

    多 URL（如 C4 的多栏目）之间失败隔离：某个 URL 失败只跳过该 URL；
    全部失败才抛异常（由 runner 标记单源失败 / 尝试 failover）。
    """
    targets = list(urls) if urls else source.page_urls()
    if not targets:
        raise ConfigError(f"源 {source.name} 缺少 url/urls")

    items: list[Item] = []
    errors: list[str] = []
    for url in targets:
        try:
            content = fetch_bytes(ctx, url)
        except CollectError as exc:
            errors.append(str(exc))
            logger.warning("feed 抓取失败 %s (%s): %s", source.name, url, exc)
            continue
        try:
            entries = parse_feed(content)
        except ParseError as exc:
            errors.append(str(exc))
            logger.warning("feed 解析失败 %s (%s): %s", source.name, url, exc)
            continue
        for entry in entries:
            item = build_item_from_entry(
                entry,
                source,
                ctx,
                allow_page_fetch=allow_page_fetch,
            )
            if item is not None:
                items.append(item)

    if not items and errors:
        raise CollectError(f"源 {source.name} 全部 feed 抓取失败: {'; '.join(errors[:3])}")
    return items
