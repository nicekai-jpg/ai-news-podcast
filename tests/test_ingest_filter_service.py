"""Stage 1.5 过滤单测：时间窗 / 关键词 / 单源配额。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_news_podcast.business.pipeline.fetcher_service import RawItem
from ai_news_podcast.business.pipeline.ingest_filter_service import filter_ingested_items

NOW = datetime(2026, 11, 5, 6, 0, tzinfo=UTC)


def _item(title: str, *, hours_ago: float = 1, source: str = "Feed") -> RawItem:
    return RawItem(
        id=title,
        title=title,
        link=f"https://example.com/{title}",
        normalized_link=f"https://example.com/{title}",
        source_name=source,
        source_category="news",
        published_at=(NOW - timedelta(hours=hours_ago)).isoformat(),
        summary="summary",
        full_text_snippet="snippet",
        category="other",
        language="zh",
    )


def test_recency_filter_drops_stale_items() -> None:
    items = [_item("fresh", hours_ago=2) for _ in range(25)] + [_item("stale", hours_ago=24 * 30)]
    kept, stats = filter_ingested_items(items, selection_cfg={}, now=NOW)
    assert all(item.title != "stale" for item in kept)
    assert stats["window"] == "prefer_36h"


def test_recency_falls_back_to_wider_window_when_too_few() -> None:
    items = [_item(f"old{i}", hours_ago=60) for i in range(5)]
    kept, stats = filter_ingested_items(items, selection_cfg={}, now=NOW)
    assert len(kept) == 5
    assert stats["window"] == "fallback_96h"


def test_per_feed_cap_keeps_newest() -> None:
    items = [
        _item("a-new", hours_ago=1, source="One Feed"),
        _item("a-old", hours_ago=10, source="One Feed"),
        _item("b", hours_ago=1, source="Other"),
    ]
    kept, _ = filter_ingested_items(items, selection_cfg={"per_feed_cap": 1}, now=NOW)
    titles = {item.title for item in kept}
    assert titles == {"a-new", "b"}


def test_include_and_exclude_keywords() -> None:
    items = [
        _item("AI 模型发布"),
        _item("森林制图"),
        _item("AI 工具但广告"),
    ]
    kept, _ = filter_ingested_items(
        items,
        selection_cfg={
            "include_keywords": ["AI"],
            "exclude_keywords": ["广告"],
        },
        now=NOW,
    )
    assert [item.title for item in kept] == ["AI 模型发布"]
