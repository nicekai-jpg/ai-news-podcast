"""Tests for recommend/coarse.py — 规则粗排与淘汰原因。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_news_podcast.business.contracts import Event, Labels
from ai_news_podcast.business.recommend.coarse import (
    REJECT_COARSE_CUT,
    REJECT_COVERED,
    REJECT_COVERED_BEFORE,
    REJECT_FACT_CONFLICT,
    REJECT_JUNK,
    REJECT_NOT_AI,
    REJECT_SINGLE_SOURCE_RUMOR,
    REJECT_SUBJECT_CAP,
    coarse_rank,
)
from ai_news_podcast.business.recommend.config import CoarseSettings
from ai_news_podcast.business.recommend.recall import RecallCandidate

REFERENCE = datetime(2026, 11, 9, 8, tzinfo=UTC)


def _make_event(
    event_id: str = "e1",
    *,
    age_hours: float = 2.0,
    title: str | None = None,
    summary: str = "事件摘要",
    subjects: list[dict] | None = None,
    lifecycle: str = "burst",
    evidence_tier: str = "medium",
    source_count: int = 4,
    independent_sources: int = 3,
    confidence: dict | None = None,
    covered_before: bool = False,
    ai_relevance: str = "unknown",
) -> Event:
    seen = (REFERENCE - timedelta(hours=age_hours)).isoformat()
    labels = Labels(
        subjects=(
            [{"id": "openai", "name": "OpenAI", "weight": 0.9, "confidence": 1.0}]
            if subjects is None
            else subjects
        ),
        primary_action="发布",
        actions=["发布", "开源"],
        topics=["技术"],
        lifecycle=lifecycle,
        evidence_tier=evidence_tier,
        confidence={} if confidence is None else confidence,
        ai_relevance=ai_relevance,
    )
    return Event(
        event_id=event_id,
        representative_title=title or f"OpenAI 发布新模型 {event_id}",
        summary=summary,
        item_ids=[f"item-{event_id}"],
        source_count=source_count,
        independent_sources=independent_sources,
        first_seen_at=seen,
        last_update_at=seen,
        labels=labels,
        evidence_score=3.0,
        covered_before=covered_before,
    )


def _candidate(event: Event) -> RecallCandidate:
    return RecallCandidate(event=event, channels=("topic",), age_hours=2.0)


def _reasons(outcome) -> list[str]:
    return [item.reason for item in outcome.eliminated]


class TestHardGates:
    def test_covered_event_is_eliminated(self) -> None:
        outcome = coarse_rank(
            [_candidate(_make_event("e1"))], CoarseSettings(), covered_event_ids={"e1"}
        )
        assert outcome.candidates == []
        assert _reasons(outcome) == [REJECT_COVERED]
        assert outcome.eliminated[0].detail

    def test_covered_before_flag_is_eliminated(self) -> None:
        event = _make_event("e1", covered_before=True)
        outcome = coarse_rank([_candidate(event)], CoarseSettings(), reference_time=REFERENCE)
        assert _reasons(outcome) == [REJECT_COVERED_BEFORE]

    def test_junk_recruit_post_is_eliminated(self) -> None:
        event = _make_event("e1", title="急招大模型算法工程师（内推）")
        outcome = coarse_rank([_candidate(event)], CoarseSettings(), reference_time=REFERENCE)
        assert _reasons(outcome) == [REJECT_JUNK]
        assert "内推" in outcome.eliminated[0].detail

    def test_junk_english_hiring_is_eliminated(self) -> None:
        event = _make_event("e1", title="We are hiring AI engineers")
        outcome = coarse_rank([_candidate(event)], CoarseSettings(), reference_time=REFERENCE)
        assert _reasons(outcome) == [REJECT_JUNK]

    def test_single_source_rumor_is_eliminated(self) -> None:
        event = _make_event("e1", independent_sources=1, evidence_tier="rumor")
        outcome = coarse_rank([_candidate(event)], CoarseSettings(), reference_time=REFERENCE)
        assert _reasons(outcome) == [REJECT_SINGLE_SOURCE_RUMOR]

    def test_single_source_non_rumor_is_kept_but_demoted(self) -> None:
        single = _make_event("e1", independent_sources=1, source_count=1)
        multi = _make_event("e2", subjects=[{"id": "nvidia", "name": "NVIDIA", "weight": 0.9}])
        outcome = coarse_rank(
            [_candidate(single), _candidate(multi)],
            CoarseSettings(),
            reference_time=REFERENCE,
        )
        assert {candidate.event.event_id for candidate in outcome.candidates} == {"e1", "e2"}
        single_candidate = next(c for c in outcome.candidates if c.event.event_id == "e1")
        multi_candidate = next(c for c in outcome.candidates if c.event.event_id == "e2")
        assert single_candidate.coarse_score < multi_candidate.coarse_score
        assert "单一来源降权" in single_candidate.reasons

    def test_fact_conflict_is_eliminated(self) -> None:
        conflict = _make_event("e1", confidence={"fact_consistency": 0.2})
        confirmed = _make_event(
            "e2",
            subjects=[{"id": "nvidia", "name": "NVIDIA"}],
            confidence={"fact_consistency": 0.9},
        )
        outcome = coarse_rank(
            [_candidate(conflict), _candidate(confirmed)],
            CoarseSettings(),
            reference_time=REFERENCE,
        )
        assert [item.event_id for item in outcome.eliminated] == ["e1"]
        assert _reasons(outcome)[0] == REJECT_FACT_CONFLICT

    def test_no_rejection_when_hard_gates_pass(self) -> None:
        outcome = coarse_rank(
            [_candidate(_make_event())], CoarseSettings(), reference_time=REFERENCE
        )
        assert len(outcome.candidates) == 1
        assert outcome.candidates[0].reasons
        assert outcome.considered == 1


class TestCapsAndOrdering:
    def test_subject_cap_keeps_top_two(self) -> None:
        events = [
            _make_event("e1", age_hours=2, independent_sources=4),
            _make_event("e2", age_hours=30, independent_sources=2),
            _make_event("e3", age_hours=2, independent_sources=3),
        ]
        outcome = coarse_rank(
            [_candidate(event) for event in events],
            CoarseSettings(),
            reference_time=REFERENCE,
        )
        kept = {candidate.event.event_id for candidate in outcome.candidates}
        assert kept == {"e1", "e3"}
        assert _reasons(outcome) == [REJECT_SUBJECT_CAP]
        assert "openai" in outcome.eliminated[0].detail

    def test_max_candidates_cut(self) -> None:
        events = [
            _make_event(f"e{index}", subjects=[{"id": f"s{index}", "name": f"S{index}"}])
            for index in range(4)
        ]
        outcome = coarse_rank(
            [_candidate(event) for event in events],
            CoarseSettings(max_candidates=2),
            reference_time=REFERENCE,
        )
        assert len(outcome.candidates) == 2
        assert _reasons(outcome) == [REJECT_COARSE_CUT, REJECT_COARSE_CUT]

    def test_coarse_score_orders_strong_before_weak(self) -> None:
        strong = _make_event(
            "strong", age_hours=2, independent_sources=5, source_count=8, evidence_tier="strong"
        )
        weak = _make_event(
            "weak",
            age_hours=400,
            independent_sources=1,
            source_count=1,
            subjects=[],
            lifecycle="fading",
            evidence_tier="weak",
        )
        outcome = coarse_rank(
            [_candidate(weak), _candidate(strong)],
            CoarseSettings(),
            reference_time=REFERENCE,
        )
        assert [candidate.event.event_id for candidate in outcome.candidates] == [
            "strong",
            "weak",
        ]

    def test_lifecycle_and_evidence_tier_affect_coarse_score(self) -> None:
        burst = _make_event("a", lifecycle="burst", evidence_tier="strong")
        fading = _make_event(
            "b",
            lifecycle="fading",
            evidence_tier="rumor",
            subjects=[{"id": "nvidia", "name": "NVIDIA"}],
        )
        outcome = coarse_rank(
            [_candidate(burst), _candidate(fading)],
            CoarseSettings(),
            reference_time=REFERENCE,
        )
        scores = {
            candidate.event.event_id: candidate.coarse_score for candidate in outcome.candidates
        }
        assert scores["a"] > scores["b"]


class TestAiRelevance:
    def test_rejects_non_ai_event(self) -> None:
        car = _make_event(
            "car",
            title="玛莎拉蒂将与尊界联合造车，明年正式量产",
            summary="双方将联合开发新车，首发车型或为双门 GT。",
            subjects=[{"id": "huawei", "name": "华为", "weight": 0.9, "confidence": 1.0}],
        )
        outcome = coarse_rank(
            [RecallCandidate(event=car, channels=("subject",))],
            CoarseSettings(),
            reference_time=REFERENCE,
        )
        assert outcome.candidates == []
        assert [item.reason for item in outcome.eliminated] == [REJECT_NOT_AI]

    def test_keeps_ai_event(self) -> None:
        ai = _make_event("ai", title="OpenAI 发布新模型，支持 Agent 工具")
        outcome = coarse_rank(
            [RecallCandidate(event=ai, channels=("subject",))],
            CoarseSettings(),
            reference_time=REFERENCE,
        )
        assert [item.event.event_id for item in outcome.candidates] == ["ai"]

    def test_switch_can_disable_gate(self) -> None:
        car = _make_event(
            "car2",
            title="某车企发布新车",
            summary="新车明年初上市。",
            subjects=[{"id": "x", "name": "某车企", "weight": 0.9, "confidence": 1.0}],
        )
        outcome = coarse_rank(
            [RecallCandidate(event=car, channels=("subject",))],
            CoarseSettings(require_ai_relevance=False),
            reference_time=REFERENCE,
        )
        assert [item.event.event_id for item in outcome.candidates] == ["car2"]

    def test_label_grade_can_keep_adjacent(self) -> None:
        car = _make_event("car3", title="某车企发布新车", ai_relevance="adjacent")
        outcome = coarse_rank(
            [RecallCandidate(event=car, channels=("subject",))],
            CoarseSettings(),
            reference_time=REFERENCE,
        )
        assert [item.event.event_id for item in outcome.candidates] == ["car3"]

    def test_label_none_rejects_ai_title(self) -> None:
        ai = _make_event("ai3", title="OpenAI 发布新模型", ai_relevance="none")
        outcome = coarse_rank(
            [RecallCandidate(event=ai, channels=("subject",))],
            CoarseSettings(),
            reference_time=REFERENCE,
        )
        assert [item.reason for item in outcome.eliminated] == [REJECT_NOT_AI]
