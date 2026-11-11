"""Tests for business/eval/recommend_eval.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_news_podcast.business.contracts import Event, Labels
from ai_news_podcast.business.eval.recommend_eval import (
    EvalLabel,
    evaluate,
    load_labels,
    match_labels,
    render_markdown,
)

REFERENCE = datetime(2026, 11, 11, 8, tzinfo=UTC)


def _event(event_id: str, title: str, *, age_hours: float = 1.0) -> Event:
    seen = (REFERENCE - timedelta(hours=age_hours)).isoformat()
    return Event(
        event_id=event_id,
        representative_title=title,
        summary="摘要",
        item_ids=[f"item-{event_id}"],
        source_count=3,
        independent_sources=2,
        first_seen_at=seen,
        last_update_at=seen,
        labels=Labels(
            topics=["技术"], tech_stack=["llm"], lifecycle="burst", evidence_tier="medium"
        ),
        evidence_score=3.0,
    )


EVENTS = [
    _event("ai", "OpenAI 发布新模型，支持 Agent 工具"),
    _event("car", "玛莎拉蒂将与尊界联合造车，明年量产"),
]


def test_match_labels_by_normalized_title() -> None:
    matched, unmatched = match_labels(
        EVENTS,
        [EvalLabel(title="OpenAI 发布新模型，支持 Agent 工具", ai_relevance="core")],
    )
    assert list(matched) == ["ai"]
    assert unmatched == []


def test_match_labels_reports_unmatched() -> None:
    _, unmatched = match_labels(EVENTS, [EvalLabel(title="不存在的标题", ai_relevance="none")])
    assert unmatched == ["不存在的标题"]


def test_evaluate_stage_metrics() -> None:
    labels = [
        EvalLabel(title=EVENTS[0].representative_title, ai_relevance="core"),
        EvalLabel(title=EVENTS[1].representative_title, ai_relevance="none"),
    ]
    report = evaluate(EVENTS, labels, reference_time=REFERENCE)
    metrics = report["metrics"]
    assert metrics["recall"]["positive_recall"] == 1.0
    assert metrics["recall"]["negative_leak_rate"] == 0.0
    assert metrics["shortlist"]["precision"] == 1.0
    assert metrics["shortlist"]["negative_rate"] == 0.0
    assert report["metrics"]["label_coverage"] == {
        "labeled": 2,
        "matched": 2,
        "unmatched": 0,
        "reviewed": 2,
    }


def test_invalid_relevance_rejected() -> None:
    with pytest.raises(ValueError):
        EvalLabel(title="t", ai_relevance="bad")


def test_load_labels_jsonl(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"title":"A","ai_relevance":"core"}',
                '{"title":"B","ai_relevance":"none","reviewed":false}',
                "",
            ]
        ),
        encoding="utf-8",
    )
    labels = load_labels(path)
    assert [label.ai_relevance for label in labels] == ["core", "none"]
    assert labels[1].reviewed is False


def test_render_markdown_contains_metrics() -> None:
    report = evaluate(
        EVENTS,
        [EvalLabel(title=EVENTS[0].representative_title, ai_relevance="core")],
        reference_time=REFERENCE,
    )
    markdown = render_markdown(report)
    assert "推荐流程评测报告" in markdown
    assert "top8" in markdown
    assert "精确率" in markdown
