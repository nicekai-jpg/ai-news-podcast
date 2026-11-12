"""Tests for eval/label_coverage.py."""

from __future__ import annotations

from ai_news_podcast.business.contracts import Event, Labels
from ai_news_podcast.business.eval.label_coverage import label_coverage, render_markdown


def _event(event_id: str, *, topics=None, subjects=None) -> Event:
    return Event(
        event_id=event_id,
        representative_title="t",
        labels=Labels(topics=topics or [], subjects=subjects or []),
    )


def test_coverage_counts() -> None:
    events = [
        _event("a", topics=["技术"], subjects=[{"id": "openai", "name": "OpenAI"}]),
        _event("b"),
    ]
    report = label_coverage(events, fields=("topics", "subjects"), warn_threshold=0.3)
    assert report["fields"]["topics"]["empty"] == 1
    assert report["fields"]["topics"]["empty_rate"] == 0.5
    assert report["warnings"]


def test_low_expectation_field_not_warned() -> None:
    report = label_coverage([_event("a")], fields=("sensitivity",), warn_threshold=0.3)
    assert report["warnings"] == []


def test_render_markdown() -> None:
    report = label_coverage([_event("a", topics=["技术"])], fields=("topics",))
    assert "标签覆盖率" in render_markdown(report)
