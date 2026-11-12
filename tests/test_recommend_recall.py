"""Tests for recommend/recall.py — 多路召回。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_news_podcast.business.contracts import Event, Labels
from ai_news_podcast.business.recommend.config import RecallSettings
from ai_news_podcast.business.recommend.recall import RECALL_CHANNELS, recall_events

REFERENCE = datetime(2026, 11, 9, 8, tzinfo=UTC)


def _make_event(
    event_id: str = "e1",
    *,
    age_hours: float | None = 1.0,
    lifecycle: str = "burst",
    evidence_tier: str = "medium",
    source_count: int = 4,
    independent_sources: int = 3,
    evidence_score: float = 4.0,
    sequence: int = 1,
    is_update: bool = False,
    reappeared: bool = False,
    continuity_id: str = "",
    subjects: list[dict] | None = None,
    topics: list[str] | None = None,
    tech_stack: list[str] | None = None,
    novelty_numbers: list[dict] | None = None,
    title: str | None = None,
    ai_relevance: str = "unknown",
    source_tier: int = 3,
) -> Event:
    seen = "" if age_hours is None else (REFERENCE - timedelta(hours=age_hours)).isoformat()
    labels = Labels(
        subjects=(
            [{"id": "openai", "name": "OpenAI", "weight": 0.9, "confidence": 1.0}]
            if subjects is None
            else subjects
        ),
        topics=["技术"] if topics is None else topics,
        tech_stack=["llm"] if tech_stack is None else tech_stack,
        lifecycle=lifecycle,
        evidence_tier=evidence_tier,
        novelty_numbers=[] if novelty_numbers is None else novelty_numbers,
        ai_relevance=ai_relevance,
    )
    return Event(
        event_id=event_id,
        representative_title=title or f"AI 模型事件 {event_id}",
        item_ids=[f"item-{event_id}"],
        source_count=source_count,
        independent_sources=independent_sources,
        first_seen_at=seen,
        last_update_at=seen,
        continuity_id=continuity_id,
        sequence=sequence,
        is_update=is_update,
        labels=labels,
        evidence_score=evidence_score,
        best_source_tier=source_tier,
        momentum_growth=0.0,
        reappeared=reappeared,
    )


class TestRecallChannels:
    def test_hits_all_six_channels(self) -> None:
        event = _make_event(continuity_id="c1", sequence=2, is_update=True)
        candidates = recall_events([event], RecallSettings(), reference_time=REFERENCE)
        assert len(candidates) == 1
        assert candidates[0].channels == RECALL_CHANNELS

    def test_unknown_age_is_treated_as_fresh(self) -> None:
        event = _make_event(
            age_hours=None,
            subjects=[],
            topics=[],
            tech_stack=[],
            source_count=0,
            independent_sources=0,
            evidence_score=0.0,
        )
        candidates = recall_events([event], RecallSettings(), reference_time=REFERENCE)
        assert candidates[0].channels == ("freshness",)

    def test_unknown_age_can_be_disabled(self) -> None:
        event = _make_event(
            age_hours=None,
            subjects=[],
            topics=[],
            tech_stack=[],
            source_count=0,
            independent_sources=0,
            evidence_score=0.0,
        )
        settings = RecallSettings(unknown_age_as_fresh=False)
        assert recall_events([event], settings, reference_time=REFERENCE) == []

    def test_event_without_any_channel_is_dropped(self) -> None:
        event = _make_event(
            age_hours=400,
            lifecycle="fading",
            subjects=[],
            topics=[],
            tech_stack=[],
            source_count=0,
            independent_sources=0,
            evidence_score=0.0,
        )
        assert recall_events([event], RecallSettings(), reference_time=REFERENCE) == []

    def test_stale_long_tail_requires_new_progress(self) -> None:
        stale = _make_event(
            age_hours=400,
            lifecycle="long_tail",
            subjects=[],
            topics=[],
            tech_stack=[],
            source_count=0,
            independent_sources=0,
            evidence_score=0.0,
        )
        progress = _make_event(
            age_hours=400,
            lifecycle="long_tail",
            subjects=[],
            topics=[],
            tech_stack=[],
            source_count=0,
            independent_sources=0,
            evidence_score=0.0,
            novelty_numbers=[{"type": "version", "value": "v9"}],
        )
        candidates = recall_events([stale, progress], RecallSettings(), reference_time=REFERENCE)
        assert [candidate.event.event_id for candidate in candidates] == ["e1"]
        assert candidates[0].channels == ("freshness", "continuity")

    def test_subject_recall_uses_weight_and_confidence(self) -> None:
        weak = _make_event(
            age_hours=400,
            lifecycle="fading",
            subjects=[{"id": "tiny", "name": "Tiny", "weight": 0.4, "confidence": 1.0}],
            topics=[],
            tech_stack=[],
            source_count=0,
            independent_sources=0,
            evidence_score=0.0,
        )
        strong = _make_event(
            age_hours=400,
            lifecycle="fading",
            subjects=[{"id": "openai", "name": "OpenAI", "weight": 0.9, "confidence": 0.8}],
            topics=[],
            tech_stack=[],
            source_count=0,
            independent_sources=0,
            evidence_score=0.0,
        )
        candidates = recall_events([weak, strong], RecallSettings(), reference_time=REFERENCE)
        assert [candidate.event.event_id for candidate in candidates] == ["e1"]
        assert candidates[0].channels == ("subject",)

    def test_same_event_id_merges_channels(self) -> None:
        topic_only = _make_event(
            event_id="dup",
            age_hours=400,
            lifecycle="fading",
            subjects=[],
            tech_stack=[],
            source_count=0,
            independent_sources=0,
            evidence_score=0.0,
            topics=["技术"],
        )
        tech_only = _make_event(
            event_id="dup",
            age_hours=400,
            lifecycle="fading",
            subjects=[],
            topics=[],
            source_count=0,
            independent_sources=0,
            evidence_score=0.0,
            tech_stack=["agent"],
        )
        candidates = recall_events(
            [topic_only, tech_only], RecallSettings(), reference_time=REFERENCE
        )
        assert len(candidates) == 1
        assert set(candidates[0].channels) == {"topic", "tech_stack"}


class TestRecallEvidence:
    def test_independent_sources_trigger_evidence(self) -> None:
        event = _make_event(
            age_hours=400,
            lifecycle="fading",
            independent_sources=2,
            evidence_score=0.0,
            source_count=0,
            subjects=[],
            topics=[],
            tech_stack=[],
        )
        candidates = recall_events([event], RecallSettings(), reference_time=REFERENCE)
        assert "evidence" in candidates[0].channels

    def test_single_source_not_recalled_by_evidence(self) -> None:
        event = _make_event(
            age_hours=400,
            lifecycle="fading",
            independent_sources=1,
            evidence_score=0.0,
            source_count=1,
            subjects=[],
            topics=[],
            tech_stack=[],
        )
        assert recall_events([event], RecallSettings(), reference_time=REFERENCE) == []

    def test_channel_order_is_canonical(self) -> None:
        event = _make_event(sequence=2)
        candidates = recall_events([event], RecallSettings(), reference_time=REFERENCE)
        channels = candidates[0].channels
        assert channels == tuple(channel for channel in RECALL_CHANNELS if channel in channels)


class TestAiPrecondition:
    def test_rejects_non_ai_event_even_if_fresh_and_subject(self) -> None:
        car = _make_event(
            "car",
            title="玛莎拉蒂将与尊界联合造车，明年正式量产",
            subjects=[{"id": "huawei", "name": "华为", "weight": 0.9, "confidence": 1.0}],
        )
        assert recall_events([car], RecallSettings(), reference_time=REFERENCE) == []

    def test_keeps_ai_event(self) -> None:
        ai = _make_event("ai", title="OpenAI 发布新模型，支持 Agent 工具")
        candidates = recall_events([ai], RecallSettings(), reference_time=REFERENCE)
        assert [candidate.event.event_id for candidate in candidates] == ["ai"]

    def test_switch_can_disable_precondition(self) -> None:
        car = _make_event("car2", title="某车企发布新车")
        candidates = recall_events(
            [car], RecallSettings(require_ai_relevance=False), reference_time=REFERENCE
        )
        assert [candidate.event.event_id for candidate in candidates] == ["car2"]

    def test_label_grade_needs_head_signal_for_news(self) -> None:
        car = _make_event("car3", title="某车企发布新车", ai_relevance="adjacent")
        assert recall_events([car], RecallSettings(), reference_time=REFERENCE) == []

    def test_label_grade_trusted_for_official(self) -> None:
        car = _make_event("car4", title="某车企发布新车", ai_relevance="adjacent", source_tier=1)
        assert [
            c.event.event_id
            for c in recall_events([car], RecallSettings(), reference_time=REFERENCE)
        ] == ["car4"]

    def test_label_none_rejects_ai_text(self) -> None:
        ai = _make_event("ai2", title="OpenAI 发布新模型", ai_relevance="none")
        assert recall_events([ai], RecallSettings(), reference_time=REFERENCE) == []
