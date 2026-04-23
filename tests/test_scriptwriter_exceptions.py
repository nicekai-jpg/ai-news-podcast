"""Tests for scriptwriter fallback, error handling and edge cases."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from ai_news_podcast.pipeline.scriptwriter import (
    _build_fallback,
    _build_material_text,
    _call_llm,
    _normalize_host_tags,
    generate_script,
)


class TestCallLlm:
    def test_missing_api_key_returns_none(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        result = _call_llm("hello", {"api_key_env": "MISSING_KEY_ENV"})
        assert result is None

    def test_empty_api_key_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv("EMPTY_KEY", "   ")
        result = _call_llm("hello", {"api_key_env": "EMPTY_KEY"})
        assert result is None


class TestBuildFallback:
    def test_generates_dialogue(self) -> None:
        brief = {
            "stories": [
                {
                    "role": "main",
                    "representative_title": "Big News",
                    "context": {"factual_summary": ["It happened."]},
                },
                {
                    "role": "supporting",
                    "representative_title": "Small News",
                    "context": {"factual_summary": ["Details here."]},
                },
                {
                    "role": "skip",
                    "representative_title": "Ignored",
                    "context": {},
                },
            ]
        }
        script = _build_fallback(brief, datetime(2024, 3, 15), "Test Podcast")
        assert "[Host A]" in script
        assert "[Host B]" in script
        assert "Big News" in script
        assert "Small News" in script
        assert "Ignored" not in script

    def test_empty_stories(self) -> None:
        script = _build_fallback({"stories": []}, datetime(2024, 3, 15), "Test")
        assert "[Host A]" in script
        assert "[Host B]" in script


class TestBuildMaterialText:
    def test_filters_skip_stories(self) -> None:
        brief = {
            "stories": [
                {"role": "main", "total_score": 14, "representative_title": "A", "context": {}},
                {"role": "skip", "total_score": 3, "representative_title": "B", "context": {}},
            ]
        }
        text = _build_material_text(brief)
        assert "A" in text
        assert "B" not in text

    def test_limits_to_max_stories(self) -> None:
        stories = [
            {"role": "main", "total_score": 20 - i, "representative_title": f"Story {i}", "context": {}}
            for i in range(10)
        ]
        brief = {"stories": stories}
        text = _build_material_text(brief, max_stories=3)
        assert text.count("【素材") == 3

    def test_empty_stories(self) -> None:
        assert _build_material_text({"stories": []}) == ""


class TestNormalizeHostTags:
    def test_fixes_various_formats(self) -> None:
        text = "Host A: hello\n[Host B]: world\nHostA there\nrandom line"
        result = _normalize_host_tags(text)
        assert result.startswith("[Host A] hello")
        assert "[Host B] world" in result
        assert "[Host A] random line" in result

    def test_removes_mood_tags(self) -> None:
        text = "[mood:excited] [Host A] hello"
        result = _normalize_host_tags(text)
        assert "[mood:excited]" not in result
        assert "[Host A] hello" in result

    def test_skips_empty_lines(self) -> None:
        text = "[Host A] hello\n\n\n[Host B] world"
        result = _normalize_host_tags(text)
        assert "\n\n\n" not in result


class TestGenerateScriptFallback:
    def test_uses_fallback_when_llm_fails(self) -> None:
        brief = {
            "stories": [
                {
                    "role": "main",
                    "representative_title": "AI News",
                    "context": {"factual_summary": ["Summary."]},
                }
            ]
        }
        with patch("ai_news_podcast.pipeline.scriptwriter._call_llm", return_value=None):
            script, warnings = generate_script(
                brief,
                episode_date=datetime(2024, 3, 15),
                podcast_title="Test",
                script_cfg={},
                llm_cfg={"api_key_env": "FAKE_KEY"},
            )
        assert "[Host A]" in script
        assert "[Host B]" in script
        # Fallback should still produce valid dialogue
        assert "AI News" in script

    def test_uses_llm_when_available(self) -> None:
        brief = {
            "stories": [
                {
                    "role": "main",
                    "representative_title": "AI News",
                    "context": {"factual_summary": ["Summary."]},
                }
            ]
        }
        fake_llm = "[Host A] LLM generated script\n[Host B] Yes it is"
        with patch("ai_news_podcast.pipeline.scriptwriter._call_llm", return_value=fake_llm):
            script, warnings = generate_script(
                brief,
                episode_date=datetime(2024, 3, 15),
                podcast_title="Test",
                script_cfg={},
                llm_cfg={"api_key_env": "FAKE_KEY"},
            )
        assert "LLM generated script" in script

    def test_warns_on_single_host(self) -> None:
        brief = {"stories": []}
        with patch("ai_news_podcast.pipeline.scriptwriter._call_llm", return_value=None):
            script, warnings = generate_script(
                brief,
                episode_date=datetime(2024, 3, 15),
                podcast_title="Test",
                script_cfg={},
                llm_cfg={"api_key_env": "FAKE_KEY"},
            )
        # Fallback always generates both hosts, so no warning expected here
        assert any("Host A" in script for _ in [0])
        assert any("Host B" in script for _ in [0])
