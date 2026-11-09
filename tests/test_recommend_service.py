"""Tests for recommend/service.py — 端到端推荐、配置加载、LLM 注入与边界。"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from ai_news_podcast.business.contracts import Event, Labels
from ai_news_podcast.business.recommend.config import RecommendConfig
from ai_news_podcast.business.recommend.service import (
    load_events,
    recommend,
    recommend_with_audit,
)

REFERENCE = datetime(2026, 11, 9, 8, tzinfo=UTC)
TECHS = (
    "agent",
    "chip",
    "robot",
    "multimodal",
    "voice",
    "vision",
    "rag",
    "mcp",
    "reasoning",
    "training",
    "finetune",
    "diffusion",
)
FIVE = {
    "impact": 5,
    "novelty": 5,
    "momentum": 5,
    "audience_fit": 5,
    "narratability": 5,
    "evidence_risk": 5,
    "reason": "主编认为值得讲",
}


def _make_event(
    index: int,
    *,
    subject: str | None = None,
    age_hours: float = 2.0,
    title: str | None = None,
    lifecycle: str = "burst",
    evidence_tier: str = "medium",
    source_count: int = 4,
    independent_sources: int = 3,
    confidence: dict | None = None,
) -> Event:
    seen = (REFERENCE - timedelta(hours=age_hours)).isoformat()
    subject_id = subject or f"company{index}"
    return Event(
        event_id=f"e{index}",
        representative_title=title or f"{subject_id} 发布新模型 v{index}",
        summary="事件摘要",
        item_ids=[f"item-{index}-a", f"item-{index}-b"],
        source_count=source_count,
        independent_sources=independent_sources,
        first_seen_at=seen,
        last_update_at=seen,
        sequence=1,
        labels=Labels(
            subjects=[
                {"id": subject_id, "name": subject_id.title(), "weight": 0.9, "confidence": 1.0}
            ],
            object_type=["model"],
            actions=["发布", "开源"],
            primary_action="发布",
            tech_stack=[TECHS[index % len(TECHS)]],
            regions=["全球"],
            lifecycle=lifecycle,
            topics=["技术"],
            primary_topic="技术",
            evidence_tier=evidence_tier,
            novelty_numbers=[{"type": "version", "value": f"v{index}"}],
            narrative_angle=["首次"],
            audience_persona=["开发者"],
            sensitivity=[],
            quote_availability=True,
            confidence=confidence if confidence is not None else {"fact_consistency": 0.9},
        ),
        evidence_score=4.0,
        momentum_growth=2.0,
    )


def _valid_events(count: int = 10) -> list[Event]:
    return [_make_event(index) for index in range(count)]


class TestConfigLoading:
    def test_default_config_values(self) -> None:
        config = RecommendConfig.load()
        assert config.recall.new_hours == 48
        assert config.coarse.max_candidates == 20
        assert config.coarse.subject_cap == 2
        assert config.fine.llm_top_k == 20
        assert config.fine.blend_rule == pytest.approx(0.6)
        assert config.fine.blend_llm == pytest.approx(0.4)
        assert config.rerank.shortlist_size == 8
        assert config.rerank.mmr_lambda == pytest.approx(0.72)
        assert sum(config.fine.weights.values()) == pytest.approx(1.0)

    def test_partial_config_uses_defaults(self, tmp_path) -> None:
        path = tmp_path / "partial.yaml"
        path.write_text("rerank:\n  shortlist_size: 3\n", encoding="utf-8")
        config = RecommendConfig.load(path)
        assert config.rerank.shortlist_size == 3
        assert config.rerank.mmr_lambda == pytest.approx(0.72)
        assert config.fine.llm_top_k == 20

    def test_missing_config_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            RecommendConfig.load(tmp_path / "missing.yaml")


class TestRecommendEndToEnd:
    def test_returns_shortlist_with_reasons_and_components(self) -> None:
        result = recommend(_valid_events(10), reference_time=REFERENCE)
        assert len(result) == 8
        assert [entry.rank for entry in result] == list(range(1, 9))
        assert len({entry.event_id for entry in result}) == 8
        for entry in result:
            assert 0.0 <= entry.score <= 5.0
            assert entry.rule_score > 0.0
            assert entry.llm_score == 0.0
            assert entry.reasons
            assert {"impact", "novelty", "momentum", "risk"} <= set(entry.components)
            assert entry.evidence_item_ids
            assert entry.rejected_reason == ""

    def test_covered_and_junk_events_are_not_recommended(self) -> None:
        events = _valid_events(10)
        events.append(_make_event(10, subject="covered", title="已被播报的事件"))
        events.append(_make_event(11, subject="junk", title="大模型算法工程师招聘（内推）"))
        result = recommend_with_audit(
            events,
            reference_time=REFERENCE,
            covered_event_ids={"e10"},
        )
        ids = {entry.event_id for entry in result.shortlist}
        assert "e10" not in ids
        assert "e11" not in ids
        reasons = {item.reason for item in result.coarse.eliminated}
        assert "covered" in reasons
        assert "junk" in reasons

    def test_no_candidates_returns_empty_list(self) -> None:
        stale = _make_event(
            0,
            age_hours=400,
            lifecycle="fading",
            evidence_tier="weak",
            source_count=0,
            independent_sources=0,
        )
        stale.labels.subjects = []
        stale.labels.topics = []
        stale.labels.primary_topic = ""
        stale.labels.tech_stack = []
        stale.labels.novelty_numbers = []
        stale.evidence_score = 0.0
        assert recommend([stale], reference_time=REFERENCE) == []

    def test_all_same_subject_is_capped(self) -> None:
        events = [_make_event(index, subject="openai") for index in range(6)]
        result = recommend(events, reference_time=REFERENCE)
        assert 1 <= len(result) <= 2

    def test_deterministic_output(self) -> None:
        events = _valid_events(10)
        first = [entry.event_id for entry in recommend(events, reference_time=REFERENCE)]
        second = [entry.event_id for entry in recommend(events, reference_time=REFERENCE)]
        assert first == second

    def test_accepts_serialized_event_dicts(self) -> None:
        events = [event.to_dict() for event in _valid_events(3)]
        result = recommend(events, reference_time=REFERENCE)
        assert len(result) == 3


class TestLlmJudgeIntegration:
    def test_judge_called_for_top_k_and_blended(self, tmp_path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "fine:\n  llm_top_k: 3\nrerank:\n  shortlist_size: 5\n",
            encoding="utf-8",
        )
        calls: list[list[str]] = []

        def fake_judge(events: list[Event]) -> dict[str, dict]:
            calls.append([event.event_id for event in events])
            return {event.event_id: dict(FIVE) for event in events}

        result = recommend_with_audit(
            _valid_events(10),
            config_path=path,
            llm_judge=fake_judge,
            reference_time=REFERENCE,
        )
        assert len(calls) == 1
        assert len(calls[0]) == 3
        judged_ids = set(calls[0])
        fine_by_id = {item.event.event_id: item for item in result.fine_scores}
        assert result.shortlist
        for entry in result.shortlist:
            if entry.event_id in judged_ids:
                assert entry.llm_score == pytest.approx(5.0)
                rule = fine_by_id[entry.event_id].rule_score
                assert entry.score == pytest.approx(0.6 * rule + 0.4 * 5.0, abs=1e-3)
            else:
                assert entry.llm_score == 0.0

    def test_judge_failure_falls_back_to_rule_scores(self) -> None:
        def broken_judge(events: list[Event]) -> dict[str, dict]:
            raise RuntimeError("LLM 不可用")

        result = recommend(_valid_events(10), llm_judge=broken_judge, reference_time=REFERENCE)
        assert len(result) == 8
        assert all(entry.llm_score == 0.0 for entry in result)

    def test_judge_returning_empty_dict_falls_back(self) -> None:
        result = recommend(_valid_events(5), llm_judge=lambda events: {}, reference_time=REFERENCE)
        assert len(result) == 5
        assert all(entry.llm_score == 0.0 for entry in result)


class TestLoadEvents:
    def test_load_events_from_list_json(self, tmp_path) -> None:
        events = _valid_events(2)
        path = tmp_path / "events.json"
        path.write_text(
            json.dumps([event.to_dict() for event in events], ensure_ascii=False),
            encoding="utf-8",
        )
        loaded = load_events(path)
        assert [event.event_id for event in loaded] == ["e0", "e1"]
        assert loaded[0].labels.subjects[0]["id"] == "company0"

    def test_load_events_from_wrapped_json(self, tmp_path) -> None:
        path = tmp_path / "events.json"
        path.write_text(
            json.dumps({"events": [_make_event(0).to_dict()]}, ensure_ascii=False),
            encoding="utf-8",
        )
        loaded = load_events(path)
        assert len(loaded) == 1
        assert loaded[0].event_id == "e0"

    def test_load_events_missing_file(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_events(tmp_path / "missing.json")
