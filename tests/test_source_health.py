"""Tests for collect/health.py — 采集源健康度聚合。"""

from __future__ import annotations

from ai_news_podcast.business.collect.health import aggregate_source_health, render_markdown
from ai_news_podcast.business.contracts import Item


def _item(source: str, status: str = "ok") -> Item:
    return Item(
        item_id=f"{source}-{status}",
        channel="c1_rss",
        source=source,
        source_label=source,
        category="news",
        title="t",
        link="https://example.com",
        full_text_status=status,
    )


def _manifest(stats: list[dict]) -> dict:
    return {"source_stats": stats}


def _stat(name: str, status: str, count: int, duration: int = 100) -> dict:
    return {
        "name": name,
        "label": name.upper(),
        "channel": "c1_rss",
        "status": status,
        "count": count,
        "error": "",
        "failover": False,
        "duration_ms": duration,
    }


class TestAggregateSourceHealth:
    def test_failed_source_is_bad(self) -> None:
        snapshots = [
            (_manifest([_stat("a", "failed", 0)]), []),
            (_manifest([_stat("a", "ok", 3)]), [_item("a"), _item("a", "failed")]),
        ]
        health = aggregate_source_health(snapshots)[0]
        assert health.name == "a"
        assert health.failures == 1
        assert health.health == "bad"
        assert health.fulltext_ok == 1
        assert health.fulltext_total == 2

    def test_healthy_source(self) -> None:
        snapshots = [
            (_manifest([_stat("b", "ok", 5)]), [_item("b"), _item("b")]),
            (_manifest([_stat("b", "ok", 5)]), [_item("b")]),
        ]
        health = aggregate_source_health(snapshots)[0]
        assert health.health == "ok"
        assert health.avg_duration_ms == 100
        assert health.fulltext_ok_rate == 1.0

    def test_empty_source_is_bad(self) -> None:
        snapshots = [
            (_manifest([_stat("c", "empty", 0)]), []),
            (_manifest([_stat("c", "empty", 0)]), []),
        ]
        health = aggregate_source_health(snapshots)[0]
        assert health.health == "bad"
        assert health.empty_days == 2

    def test_sorted_bad_first(self) -> None:
        snapshots = [
            (_manifest([_stat("ok1", "ok", 2)]), [_item("ok1")]),
            (_manifest([_stat("bad1", "failed", 0)]), []),
        ]
        healths = aggregate_source_health(snapshots)
        assert healths[0].name == "bad1"

    def test_render_markdown(self) -> None:
        healths = aggregate_source_health([(_manifest([_stat("x", "failed", 0)]), [])])
        markdown = render_markdown(healths)
        assert "采集源健康度" in markdown
        assert "x" in markdown
