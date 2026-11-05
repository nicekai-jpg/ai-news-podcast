"""Stage 1.5 — 抓取结果过滤（时间窗 / 话题关键词 / 单源配额）。

此前 ``config.yaml`` 里的 ``selection.prefer_recent_hours`` / ``fallback_recent_hours`` /
``per_feed_cap`` / ``include_keywords`` / ``exclude_keywords`` 全部没有接线，导致：

- 8 个月前的 arXiv 论文也能被当成「今日头条」（published_at 完全没被使用）；
- 单个 feed（如 OpenAI Blog 23 条）可以全部进入 main；
- 非 AI 话题（DeepMind 的森林制图）也能进素材池。

这个模块只做纯函数过滤，方便单测；抓取逻辑仍留在 fetcher_service。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from ai_news_podcast.business.pipeline.fetcher_service import RawItem

log = logging.getLogger(__name__)

# 时间窗内条目太少时降到 fallback 窗口；再少就不再放宽（宁可少也不要旧闻）
_MIN_ITEMS_FOR_PREFERRED_WINDOW = 20


def _parse_published(value: str) -> datetime | None:
    """解析 RFC3339 时间；解析失败返回 None（调用方按「新条目」处理，避免误删）。"""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _matches_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword and keyword.lower() in lowered for keyword in keywords)


def _item_text(item: RawItem) -> str:
    return f"{item.title} {item.summary}"


def _recency_filter(
    items: list[RawItem],
    *,
    now: datetime,
    prefer_hours: int,
    fallback_hours: int,
) -> tuple[list[RawItem], str]:
    """按时间窗过滤；窗口内条目过少时退回更大窗口。返回 (items, 使用的窗口名)。"""

    def within(hours: int) -> list[RawItem]:
        cutoff = now - timedelta(hours=hours)
        kept: list[RawItem] = []
        for item in items:
            published = _parse_published(item.published_at)
            if published is None or published >= cutoff:
                kept.append(item)
        return kept

    preferred = within(prefer_hours)
    if len(preferred) >= _MIN_ITEMS_FOR_PREFERRED_WINDOW:
        return preferred, f"prefer_{prefer_hours}h"
    fallback = within(fallback_hours)
    if len(fallback) > len(preferred):
        log.info(
            "Recency filter: prefer %sh kept %d items (<%d), falling back to %sh → %d items",
            prefer_hours,
            len(preferred),
            _MIN_ITEMS_FOR_PREFERRED_WINDOW,
            fallback_hours,
            len(fallback),
        )
        return fallback, f"fallback_{fallback_hours}h"
    return preferred, f"prefer_{prefer_hours}h"


def _keyword_filter(
    items: list[RawItem],
    *,
    include_keywords: list[str],
    exclude_keywords: list[str],
) -> list[RawItem]:
    kept: list[RawItem] = []
    for item in items:
        text = _item_text(item)
        if exclude_keywords and _matches_any(text, exclude_keywords):
            continue
        if include_keywords and not _matches_any(text, include_keywords):
            continue
        kept.append(item)
    return kept


def _per_feed_cap(items: list[RawItem], cap: int) -> list[RawItem]:
    """每个源只保留最新的 cap 条，避免单一 feed 霸榜（如 OpenAI Blog 23 条全进 main）。"""
    if cap <= 0:
        return items
    grouped: dict[str, list[RawItem]] = {}
    for item in items:
        grouped.setdefault(item.source_name, []).append(item)
    kept: list[RawItem] = []
    for group in grouped.values():
        selected = group
        if len(group) > cap:
            selected = sorted(
                group,
                key=lambda it: (
                    _parse_published(it.published_at) or datetime.min.replace(tzinfo=UTC)
                ),
                reverse=True,
            )[:cap]
        kept.extend(selected)
    # 保持原顺序（抓取顺序对后续去重/聚类影响更小）
    allowed = {id(item) for item in kept}
    return [item for item in items if id(item) in allowed]


def filter_ingested_items(
    items: list[RawItem],
    *,
    selection_cfg: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> tuple[list[RawItem], dict[str, Any]]:
    """抓取后过滤：时间窗 → 关键词 → 单源配额。返回 (保留条目, 统计信息)。"""
    cfg = selection_cfg or {}
    prefer_hours = int(cfg.get("prefer_recent_hours") or 36)
    fallback_hours = int(cfg.get("fallback_recent_hours") or 96)
    per_feed_cap = int(cfg.get("per_feed_cap") or 0)
    include_keywords = [str(k) for k in (cfg.get("include_keywords") or [])]
    exclude_keywords = [str(k) for k in (cfg.get("exclude_keywords") or [])]

    reference = now or datetime.now(tz=UTC)
    stats: dict[str, Any] = {
        "input": len(items),
        "prefer_hours": prefer_hours,
        "fallback_hours": fallback_hours,
        "include_keywords": len(include_keywords),
        "exclude_keywords": len(exclude_keywords),
        "per_feed_cap": per_feed_cap,
    }

    kept, window = _recency_filter(
        items,
        now=reference,
        prefer_hours=prefer_hours,
        fallback_hours=fallback_hours,
    )
    stats["after_recency"] = len(kept)
    stats["window"] = window

    kept = _keyword_filter(
        kept,
        include_keywords=include_keywords,
        exclude_keywords=exclude_keywords,
    )
    stats["after_keywords"] = len(kept)

    kept = _per_feed_cap(kept, per_feed_cap)
    stats["after_feed_cap"] = len(kept)

    log.info(
        "Ingest filter: %d → %d (window=%s, keywords=%d, feed_cap=%d)",
        stats["input"],
        stats["after_feed_cap"],
        window,
        len(include_keywords),
        per_feed_cap,
    )
    return kept, stats
