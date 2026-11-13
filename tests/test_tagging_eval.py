"""Tests for eval/tagging_eval.py."""

from __future__ import annotations

from ai_news_podcast.business.contracts import Event, Labels
from ai_news_podcast.business.eval.tagging_eval import GoldRow, evaluate_tagging, render_markdown


def _event(event_id: str, relevance: str, topics: list[str]) -> Event:
    return Event(
        event_id=event_id,
        representative_title="t",
        labels=Labels(ai_relevance=relevance, topics=topics),
    )


def test_single_and_multi_metrics() -> None:
    gold = [
        GoldRow("e1", {"ai_relevance": "core", "topics": ["技术"]}),
        GoldRow("e2", {"ai_relevance": "none", "topics": []}),
    ]
    events = {"e1": _event("e1", "core", ["技术"]), "e2": _event("e2", "core", [])}
    report = evaluate_tagging(gold, events, fields=("ai_relevance", "topics"))
    assert report["matched"] == 2
    assert report["fields"]["ai_relevance"]["accuracy"] == 0.5
    assert report["fields"]["topics"]["micro_f1"] == 1.0


def test_unmatched_reported() -> None:
    report = evaluate_tagging([GoldRow("missing", {"ai_relevance": "core"})], {})
    assert report["unmatched"] == 1
    assert "未匹配" in render_markdown(report)
