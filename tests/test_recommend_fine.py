"""Tests for recommend/fine.py — 精排特征、LLM rubric 融合与防循环约束。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_news_podcast.business.contracts import Event, Labels
from ai_news_podcast.business.recommend.base import DIMENSIONS
from ai_news_podcast.business.recommend.config import FineSettings
from ai_news_podcast.business.recommend.fine import apply_judgements, score_events
from ai_news_podcast.business.recommend.prompts import build_rubric_prompt

REFERENCE = datetime(2026, 11, 9, 8, tzinfo=UTC)
ALL_DIMS_FIVE = {
    "impact": 5,
    "novelty": 4,
    "momentum": 4,
    "audience_fit": 4,
    "narratability": 4,
    "evidence_risk": 4,
}


def _make_event(
    event_id: str = "e1",
    *,
    age_hours: float = 2.0,
    title: str = "OpenAI 发布新一代推理模型",
    **label_overrides,
) -> Event:
    label_values: dict = {
        "subjects": [{"id": "openai", "name": "OpenAI", "weight": 0.9, "confidence": 1.0}],
        "object_type": ["model"],
        "actions": ["发布", "开源"],
        "primary_action": "发布",
        "tech_stack": ["agent"],
        "regions": ["全球"],
        "lifecycle": "burst",
        "topics": ["技术"],
        "primary_topic": "技术",
        "evidence_tier": "medium",
        "novelty_numbers": [{"type": "version", "value": "v3.0"}],
        "narrative_angle": ["首次"],
        "audience_persona": ["开发者"],
        "sensitivity": [],
        "quote_availability": True,
        "confidence": {"fact_consistency": 0.9},
    }
    label_values.update(label_overrides)
    seen = (REFERENCE - timedelta(hours=age_hours)).isoformat()
    event_stats = {
        key: label_values.pop(key)
        for key in (
            "source_count",
            "independent_sources",
            "evidence_score",
            "momentum_growth",
            "sequence",
            "is_update",
            "reappeared",
        )
        if key in label_values
    }
    return Event(
        event_id=event_id,
        representative_title=title,
        summary="摘要",
        item_ids=[f"item-{event_id}"],
        first_seen_at=seen,
        last_update_at=seen,
        source_count=event_stats.get("source_count", 4),
        independent_sources=event_stats.get("independent_sources", 3),
        evidence_score=event_stats.get("evidence_score", 3.0),
        momentum_growth=event_stats.get("momentum_growth", 2.0),
        sequence=event_stats.get("sequence", 1),
        is_update=event_stats.get("is_update", False),
        reappeared=event_stats.get("reappeared", False),
        labels=Labels(**label_values),
    )


def _score(event: Event, settings: FineSettings | None = None):
    return score_events([event], settings or FineSettings())[0]


class TestRuleFeatures:
    def test_rule_score_bounds_and_components(self) -> None:
        scored = _score(_make_event())
        assert 0.0 < scored.rule_score <= 5.0
        assert set(DIMENSIONS).issubset(scored.components)
        assert "risk_penalty" in scored.components
        assert len(scored.reasons) == 7
        assert scored.reasons[0].startswith("影响力")

    def test_lifecycle_and_evidence_tier_do_not_change_fine_score(self) -> None:
        base = _score(_make_event("a", lifecycle="burst", evidence_tier="strong"))
        rogue = _score(_make_event("b", lifecycle="fading", evidence_tier="rumor"))
        assert base.components == rogue.components
        assert base.rule_score == pytest.approx(rogue.rule_score)

    def test_evidence_score_is_not_used_by_fine(self) -> None:
        low = _score(_make_event("a", evidence_score=0.0))
        high = _score(_make_event("b", evidence_score=5.0))
        assert low.components == high.components

    def test_subject_confidence_lowers_impact(self) -> None:
        confident = _score(
            _make_event(
                "a", subjects=[{"id": "openai", "name": "OpenAI", "weight": 0.9, "confidence": 1.0}]
            )
        )
        doubtful = _score(
            _make_event(
                "b", subjects=[{"id": "openai", "name": "OpenAI", "weight": 0.9, "confidence": 0.1}]
            )
        )
        assert doubtful.components["impact"] < confident.components["impact"]

    def test_low_label_confidence_increases_risk_penalty(self) -> None:
        clean = _score(_make_event("a", confidence={"fact_consistency": 0.9}))
        noisy = _score(_make_event("b", confidence={"fact_consistency": 0.2, "topic": 0.3}))
        assert noisy.components["risk_penalty"] > clean.components["risk_penalty"]
        assert noisy.components["risk"] < clean.components["risk"]

    def test_novelty_numbers_raise_novelty(self) -> None:
        none = _score(_make_event("a", novelty_numbers=[]))
        many = _score(
            _make_event(
                "b",
                novelty_numbers=[
                    {"type": "version", "value": "v4"},
                    {"type": "amount", "value": "10 亿美元"},
                    {"type": "score", "value": "92.1"},
                ],
            )
        )
        assert none.components["novelty"] == 0.0
        assert many.components["novelty"] > none.components["novelty"]

    def test_more_sources_and_growth_raise_momentum(self) -> None:
        quiet = _score(_make_event("a", source_count=1, independent_sources=0, momentum_growth=0.0))
        hot = _score(
            _make_event(
                "b",
                source_count=8,
                independent_sources=5,
                momentum_growth=4.0,
                sequence=3,
                reappeared=True,
            )
        )
        assert hot.components["momentum"] > quiet.components["momentum"]

    def test_clickbait_title_lowers_risk_safety(self) -> None:
        plain = _score(_make_event("a", title="OpenAI 发布新模型"))
        clickbait = _score(_make_event("b", title="震惊！OpenAI 史上最强模型炸裂发布"))
        assert clickbait.components["risk"] < plain.components["risk"]


class TestLlmJudge:
    def test_judgement_is_blended_with_rule_score(self) -> None:
        scored = [_score(_make_event())]
        judgements = {"e1": {**ALL_DIMS_FIVE, "reason": "值得讲"}}
        blended = apply_judgements(scored, judgements, FineSettings())
        assert blended[0].llm_score == pytest.approx(4.2)
        expected = 0.6 * scored[0].rule_score + 0.4 * 4.2
        assert blended[0].final_score == pytest.approx(expected)
        assert blended[0].llm_reason == "值得讲"
        assert any("LLM 复核分" in reason for reason in blended[0].reasons)

    def test_missing_judgement_falls_back_to_rule_score(self) -> None:
        scored = [_score(_make_event())]
        blended = apply_judgements(scored, {"other": ALL_DIMS_FIVE}, FineSettings())
        assert blended[0].llm_score is None
        assert blended[0].final_score == pytest.approx(scored[0].rule_score)

    def test_out_of_range_and_string_scores_are_clamped(self) -> None:
        scored = [_score(_make_event())]
        judgements = {
            "e1": {
                "impact": 9,
                "novelty": -3,
                "momentum": "4",
                "audience_fit": 4,
                "narratability": 4,
                "evidence_risk": 4,
            }
        }
        blended = apply_judgements(scored, judgements, FineSettings())
        assert blended[0].llm_score == pytest.approx(3.48)

    def test_empty_judgements_leave_scores_untouched(self) -> None:
        scored = [_score(_make_event())]
        assert apply_judgements(scored, {}, FineSettings()) == scored

    def test_judge_weights_map_evidence_and_risk_to_one_dimension(self) -> None:
        from ai_news_podcast.business.recommend.config import judge_dimension_weights

        weights = judge_dimension_weights(FineSettings())
        assert weights["evidence_risk"] == pytest.approx(0.20)
        assert sum(weights.values()) == pytest.approx(1.0)


class TestRubricPrompt:
    def test_prompt_contains_six_dimensions_and_levels(self) -> None:
        prompt = build_rubric_prompt([_make_event("judge-me")])
        for dim in ALL_DIMS_FIVE:
            assert dim in prompt
        assert "0 分：" in prompt
        assert "5 分：" in prompt
        assert "judge-me" in prompt
        assert "JSON" in prompt

    def test_prompt_does_not_leak_lifecycle_or_evidence_tier(self) -> None:
        prompt = build_rubric_prompt([_make_event()])
        assert "lifecycle" not in prompt
        assert "evidence_tier" not in prompt
