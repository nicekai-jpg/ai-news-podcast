"""Tests for ocean/relevance.py — AI 相关性分级。"""

from __future__ import annotations

from ai_news_podcast.business.ocean.relevance import classify_text, stronger_grade


class TestClassifyText:
    def test_core_model_news(self) -> None:
        grade, evidence, confidence = classify_text("OpenAI 发布新模型，支持 Agent")
        assert grade == "core"
        assert evidence
        assert confidence >= 0.7

    def test_adjacent_chip_news(self) -> None:
        grade, evidence, _ = classify_text("英伟达 GPU 供应紧张")
        assert grade == "adjacent"
        assert "gpu" in evidence or "英伟达" in evidence

    def test_none_for_car_news(self) -> None:
        grade, evidence, confidence = classify_text("玛莎拉蒂与尊界联合造车，明年量产")
        assert grade == "none"
        assert evidence == []
        assert confidence >= 0.5

    def test_unknown_for_too_short(self) -> None:
        grade, _, _ = classify_text("新车")
        assert grade == "unknown"

    def test_body_only_evidence_is_ignored_for_news(self) -> None:
        grade, _, _ = classify_text(
            "某车企发布新车",
            summary="",
            full_text="该车搭载智能驾驶系统，支持 AI 语音助手",
        )
        assert grade == "none"

    def test_body_evidence_allowed_for_research(self) -> None:
        grade, _, confidence = classify_text(
            "A Novel Method for X",
            summary="",
            full_text="该方法使用大模型与 Agent 工具",
            category="research",
        )
        assert grade == "core"
        assert confidence < 0.7


def test_stronger_grade_order() -> None:
    assert stronger_grade("unknown", "none") == "none"
    assert stronger_grade("none", "adjacent") == "adjacent"
    assert stronger_grade("adjacent", "core") == "core"
    assert stronger_grade("core", "adjacent") == "core"
