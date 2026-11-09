"""Tests for recommend/rerank.py — MMR 多样性、去重、时效衰减、top N。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_news_podcast.business.contracts import Event, Labels
from ai_news_podcast.business.recommend.config import RerankSettings
from ai_news_podcast.business.recommend.fine import FineScore
from ai_news_podcast.business.recommend.rerank import rerank

REFERENCE = datetime(2026, 11, 9, 8, tzinfo=UTC)


def _make_event(
    event_id: str,
    *,
    subject: str = "openai",
    tech: str = "agent",
    topic: str = "技术",
    age_hours: float = 2.0,
    title: str | None = None,
) -> Event:
    seen = (REFERENCE - timedelta(hours=age_hours)).isoformat()
    return Event(
        event_id=event_id,
        representative_title=title or f"事件 {event_id}",
        summary="摘要",
        item_ids=[f"item-{event_id}"],
        source_count=3,
        independent_sources=2,
        first_seen_at=seen,
        last_update_at=seen,
        labels=Labels(
            subjects=[{"id": subject, "name": subject, "weight": 0.9}],
            tech_stack=[tech],
            topics=[topic],
        ),
    )


def _make_fine(event_id: str, score: float, **event_kw) -> FineScore:
    event = _make_event(event_id, **event_kw)
    return FineScore(
        event=event,
        components={},
        rule_score=score,
        final_score=score,
        reasons=[f"规则分 {score:.2f}"],
    )


class TestTopNAndOrder:
    def test_returns_top_n_by_relevance(self) -> None:
        scored = [
            _make_fine("a", 4.0),
            _make_fine("b", 3.0, subject="nvidia", tech="chip"),
            _make_fine("c", 2.0, subject="meta", tech="vision"),
        ]
        result = rerank(scored, RerankSettings(shortlist_size=2), reference_time=REFERENCE)
        assert [item.event.event_id for item in result] == ["a", "b"]
        assert result[0].mmr_utility >= result[1].mmr_utility

    def test_empty_input(self) -> None:
        assert rerank([], RerankSettings(), reference_time=REFERENCE) == []

    def test_lambda_one_keeps_pure_relevance_order(self) -> None:
        scored = [
            _make_fine("a", 2.0),
            _make_fine("b", 4.0, subject="nvidia", tech="chip"),
            _make_fine("c", 3.0, subject="meta", tech="vision"),
        ]
        settings = RerankSettings(shortlist_size=3, mmr_lambda=1.0)
        result = rerank(scored, settings, reference_time=REFERENCE)
        assert [item.event.event_id for item in result] == ["b", "c", "a"]


class TestDiversity:
    def test_same_subject_capped_at_two(self) -> None:
        scored = [
            _make_fine("a", 5.0, tech="agent"),
            _make_fine("b", 4.5, tech="chip"),
            _make_fine("c", 4.0, tech="robot"),
            _make_fine("d", 3.5, tech="voice"),
            _make_fine("e", 3.0, subject="nvidia", tech="vision"),
        ]
        result = rerank(scored, RerankSettings(), reference_time=REFERENCE)
        subjects = [item.event.labels.subjects[0]["id"] for item in result]
        assert subjects.count("openai") == 2
        assert "e" in {item.event.event_id for item in result}

    def test_all_same_subject_returns_cap_count(self) -> None:
        scored = [
            _make_fine(f"e{index}", 5.0 - index * 0.1, tech=f"tech{index}") for index in range(5)
        ]
        result = rerank(scored, RerankSettings(), reference_time=REFERENCE)
        assert len(result) == 2

    def test_same_tech_stack_capped_at_two(self) -> None:
        scored = [
            _make_fine("a", 5.0, subject="openai", tech="agent"),
            _make_fine("b", 4.0, subject="nvidia", tech="agent"),
            _make_fine("c", 3.0, subject="meta", tech="agent"),
            _make_fine("d", 2.0, subject="google", tech="vision"),
        ]
        result = rerank(scored, RerankSettings(), reference_time=REFERENCE)
        techs = [item.event.labels.tech_stack[0] for item in result]
        assert techs.count("agent") == 2
        assert "d" in {item.event.event_id for item in result}

    def test_generic_tech_tags_do_not_trigger_cap(self) -> None:
        scored = [
            _make_fine(f"e{index}", 5.0 - index * 0.1, subject=f"s{index}", tech="llm")
            for index in range(4)
        ]
        result = rerank(scored, RerankSettings(), reference_time=REFERENCE)
        assert len(result) == 4


class TestDedupeAndDecay:
    def test_near_duplicate_titles_keep_higher_score(self) -> None:
        scored = [
            _make_fine("a", 5.0, title="OpenAI 发布 GPT-5 模型"),
            _make_fine("b", 4.0, title="OpenAI 发布 GPT-5 模型！"),
            _make_fine("c", 3.0, subject="deepseek", tech="chip", title="DeepSeek 开源新框架"),
        ]
        result = rerank(scored, RerankSettings(), reference_time=REFERENCE)
        assert [item.event.event_id for item in result] == ["a", "c"]

    def test_covered_ids_and_covered_before_are_excluded(self) -> None:
        covered = _make_fine("covered", 5.0)
        covered_before_event = _make_event("before")
        covered_before_event.covered_before = True
        before = FineScore(
            event=covered_before_event,
            components={},
            rule_score=4.0,
            final_score=4.0,
        )
        normal = _make_fine("normal", 3.0, subject="nvidia")
        result = rerank(
            [covered, before, normal],
            RerankSettings(),
            covered_event_ids={"covered"},
            reference_time=REFERENCE,
        )
        assert [item.event.event_id for item in result] == ["normal"]

    def test_stale_events_are_decayed(self) -> None:
        fresh = _make_fine("fresh", 4.0, age_hours=2)
        old = _make_fine("old", 4.0, subject="nvidia", tech="chip", age_hours=100)
        result = rerank([old, fresh], RerankSettings(), reference_time=REFERENCE)
        assert [item.event.event_id for item in result] == ["fresh", "old"]
        assert result[0].freshness == pytest.approx(1.0)
        assert result[1].freshness == pytest.approx(0.85)
        assert result[1].final_score == pytest.approx(4.0 * 0.85)
        assert any("时效衰减" in reason for reason in result[1].reasons)

    def test_shortlist_size_override(self) -> None:
        scored = [
            _make_fine(f"e{index}", 5.0 - index * 0.1, subject=f"s{index}", tech=f"t{index}")
            for index in range(5)
        ]
        result = rerank(
            scored,
            RerankSettings(),
            reference_time=REFERENCE,
            shortlist_size=3,
        )
        assert len(result) == 3
