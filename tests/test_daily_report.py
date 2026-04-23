"""Tests for ai_news_podcast.cli.daily_report."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_news_podcast.cli.daily_report import _call_llm_ollama_direct, build_report_prompt


class TestBuildReportPrompt:
    def test_includes_date(self) -> None:
        brief = {
            "stories": [
                {
                    "role": "main",
                    "total_score": 14,
                    "representative_title": "GPT-5",
                    "context": {"factual_summary": ["It is big."]},
                }
            ]
        }
        prompt = build_report_prompt(brief, "2024年03月15日")
        assert "2024年03月15日" in prompt
        assert "科技新闻日报" in prompt

    def test_filters_skip_stories(self) -> None:
        brief = {
            "stories": [
                {
                    "role": "main",
                    "total_score": 14,
                    "representative_title": "Keep",
                    "context": {},
                },
                {
                    "role": "skip",
                    "total_score": 3,
                    "representative_title": "Drop",
                    "context": {},
                },
            ]
        }
        prompt = build_report_prompt(brief, "2024-03-15")
        assert "Keep" in prompt
        assert "Drop" not in prompt

    def test_limits_to_15_stories(self) -> None:
        stories = [
            {
                "role": "main",
                "total_score": 20 - i,
                "representative_title": f"Story {i}",
                "context": {"factual_summary": ["Summary."]},
            }
            for i in range(20)
        ]
        brief = {"stories": stories}
        prompt = build_report_prompt(brief, "2024-03-15")
        # Count 【素材 markers
        assert prompt.count("【素材") == 15

    def test_empty_stories(self) -> None:
        prompt = build_report_prompt({"stories": []}, "2024-03-15")
        assert "【素材" not in prompt


class TestCallLlmOllamaDirect:
    def test_success(self) -> None:
        fake_lines = [
            b'{"message": {"content": "Hello "}}',
            b'{"message": {"content": "world"}}',
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = fake_lines
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = _call_llm_ollama_direct("prompt", "qwen")
        assert result == "Hello world"

    def test_failure_returns_none(self) -> None:
        with patch("requests.post", side_effect=Exception("network down")):
            result = _call_llm_ollama_direct("prompt", "qwen")
        assert result is None

    def test_skips_malformed_json_lines(self) -> None:
        fake_lines = [
            b'bad json',
            b'{"message": {"content": "OK"}}',
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = fake_lines
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = _call_llm_ollama_direct("prompt", "qwen")
        assert result == "OK"
