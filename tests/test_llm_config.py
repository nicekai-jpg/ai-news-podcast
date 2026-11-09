"""Tests for LLMConfig per-stage model resolution."""

from __future__ import annotations

from ai_news_podcast.data.config.models_dao import LLMConfig


class TestStageModels:
    def test_default_model(self) -> None:
        cfg = LLMConfig()
        assert cfg.resolve_model("stage5") == "MiniMax-M3"

    def test_stage_override(self) -> None:
        cfg = LLMConfig(
            model="MiniMax-M3",
            stage_models={"stage2": "MiniMax-M3.1-Flash", "stage3": "MiniMax-M3.1-Flash"},
        )
        assert cfg.resolve_model("stage2") == "MiniMax-M3.1-Flash"
        assert cfg.resolve_model("stage5") == "MiniMax-M3"

    def test_unknown_stage_falls_back(self) -> None:
        cfg = LLMConfig(stage_models={"stage4": "X"})
        assert cfg.resolve_model("stage9") == cfg.model
