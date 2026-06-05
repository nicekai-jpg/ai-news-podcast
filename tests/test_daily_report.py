"""Tests for ai_news_podcast.cli.daily_report."""

from __future__ import annotations

from ai_news_podcast.cli.daily_report import build_report_prompt


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

    def test_limits_to_5_stories(self) -> None:
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
        assert prompt.count("【素材") == 5

    def test_empty_stories(self) -> None:
        prompt = build_report_prompt({"stories": []}, "2024-03-15")
        assert "【素材" not in prompt



