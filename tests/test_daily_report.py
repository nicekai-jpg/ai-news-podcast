"""Tests for ai_news_podcast.cli.podcast_report."""

from __future__ import annotations

from typing import Any

from ai_news_podcast.cli.podcast_report import (
    build_radar_report_section,
    build_report_prompt,
)


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


class TestRadarReportSection:
    def test_empty_radar_returns_empty(self) -> None:
        assert build_radar_report_section(None, "2026年9月9日") == ""
        assert build_radar_report_section({"projects": []}, "2026年9月9日") == ""

    def test_renders_pick_and_runner_ups(self) -> None:
        radar: dict[str, Any] = {
            "projects": [
                {
                    "repo": "owner/hot",
                    "url": "https://github.com/owner/hot",
                    "stars": 1500,
                    "delta_stars": 1000,
                    "language": "Python",
                    "license": "MIT",
                    "description": "Fast LLM harness",
                },
                {
                    "repo": "owner/next",
                    "url": "https://github.com/owner/next",
                    "stars": 800,
                    "delta_stars": None,
                    "language": "Rust",
                    "license": "Apache-2.0",
                    "description": "Quick agent runtime",
                },
            ],
            "meta": {"pick_repo": "owner/hot"},
        }
        section = build_radar_report_section(radar, "2026年9月9日")
        assert "## 📡 项目雷达 | 2026年9月9日" in section
        assert "🥇 主推 [owner/hot]" in section
        assert "https://github.com/owner/hot" in section
        assert "⭐ 1500" in section and "+1000/天" in section
        assert "备选 [owner/next]" in section
        assert "首日" in section
