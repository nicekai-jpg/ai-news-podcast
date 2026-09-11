"""Tests for podcastwriter prompt building functions."""

from __future__ import annotations

from datetime import datetime

from ai_news_podcast.pipeline.material import build_material_text as _build_material_text
from ai_news_podcast.prompts import (
    build_editor_prompt,
    build_writer_prompt,
)


class TestBuildEditorPrompt:
    def test_contains_date_and_material(self) -> None:
        material = "【素材1】\n标题：Test\n"
        prompt = build_editor_prompt(material, datetime(2024, 3, 15))
        assert "2024年3月15日" in prompt
        assert "Test" in prompt
        assert "## 金句" in prompt
        assert "## 头条 1" in prompt
        assert "## 快讯" in prompt


class TestBuildWriterPrompt:
    def test_contains_banned_words(self) -> None:
        style_cfg = {"banned_words": ["炸裂", "王炸"]}
        prompt = build_writer_prompt(
            '{"thesis": "x"}',
            datetime(2024, 3, 15),
            "Test Podcast",
            style_cfg,
        )
        assert "炸裂" in prompt
        assert "王炸" in prompt
        assert "Test Podcast" in prompt
        assert "[Host A]" in prompt
        assert "[Host B]" in prompt

    def test_uses_default_banned_words(self) -> None:
        prompt = build_writer_prompt(
            '{"thesis": "x"}',
            datetime(2024, 3, 15),
            "Podcast",
            {},
        )
        assert "废话不多说" in prompt  # from DEFAULT_BANNED_WORDS


class TestBuildMaterialTextBackgroundAndSources:
    def test_includes_background_and_sources(self) -> None:
        brief = {
            "stories": [
                {
                    "role": "main",
                    "total_score": 14,
                    "representative_title": "AI News",
                    "context": {
                        "factual_summary": ["Summary."],
                        "historical_background": "Previously...",
                        "sources_ranked": [
                            {"name": "Source A"},
                            {"name": "Source B"},
                        ],
                    },
                }
            ]
        }
        text = _build_material_text(brief, max_stories=5)
        assert "AI News" in text
        assert "Previously..." in text
        assert "Source A" in text
        assert "Source B" in text

    def test_role_labels(self) -> None:
        brief = {
            "stories": [
                {
                    "role": "main",
                    "total_score": 14,
                    "representative_title": "A",
                    "context": {},
                },
                {
                    "role": "supporting",
                    "total_score": 10,
                    "representative_title": "B",
                    "context": {},
                },
                {
                    "role": "quick",
                    "total_score": 6,
                    "representative_title": "C",
                    "context": {},
                },
            ]
        }
        text = _build_material_text(brief)
        assert "重要" in text
        assert "次要" in text
        assert "简讯" in text


class TestRadarPromptSections:
    def test_editor_prompt_without_radar_omits_section(self) -> None:
        prompt = build_editor_prompt("素材", datetime(2026, 9, 9))
        assert "项目雷达素材" not in prompt

    def test_editor_prompt_with_radar(self) -> None:
        prompt = build_editor_prompt("素材", datetime(2026, 9, 9), radar_material="雷达素材")
        assert "项目雷达素材" in prompt
        assert "雷达素材" in prompt
        assert "[主推]" in prompt

    def test_writer_prompt_radar_rules_toggle(self) -> None:
        base = build_writer_prompt("大纲", datetime(2026, 9, 9), "AI 每日先锋", {})
        assert "项目雷达栏目规范" not in base
        with_radar = build_writer_prompt(
            "大纲", datetime(2026, 9, 9), "AI 每日先锋", {}, has_radar=True
        )
        assert "项目雷达栏目规范" in with_radar
        assert "300-500 字" in with_radar
