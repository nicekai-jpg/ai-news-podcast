"""Tests for scriptwriter generate_show_notes (Markdown)."""

from __future__ import annotations

from datetime import datetime

from ai_news_podcast.pipeline.scriptwriter import generate_show_notes


class TestGenerateShowNotes:
    def test_full_structure(self) -> None:
        brief = {
            "thesis": "AI is evolving fast",
            "stories": [
                {
                    "role": "main",
                    "representative_title": "GPT-5",
                    "context": {
                        "factual_summary": ["It understands everything."],
                    },
                    "total_score": 14,
                    "items": [
                        {"source_name": "OpenAI", "link": "https://openai.com"}
                    ],
                },
                {
                    "role": "supporting",
                    "representative_title": "Claude 4",
                    "context": {
                        "factual_summary": ["Better reasoning."],
                    },
                    "total_score": 10,
                    "items": [],
                },
                {
                    "role": "quick",
                    "representative_title": "Mini Update",
                    "context": {"factual_summary": []},
                    "total_score": 6,
                    "items": [
                        {"source_name": "TechCrunch", "link": "https://tc.com"}
                    ],
                },
                {
                    "role": "skip",
                    "representative_title": "Ignored",
                    "context": {},
                    "total_score": 2,
                    "items": [],
                },
            ],
        }
        md = generate_show_notes(
            brief,
            episode_title="AI News | 2024-03-15",
            episode_date=datetime(2024, 3, 15),
        )
        assert "# AI News | 2024-03-15" in md
        assert "2024年3月15日" in md
        assert "> AI is evolving fast" in md
        assert "🔴 主要报道" in md
        assert "🟡 支撑消息" in md
        assert "🟢 快讯" in md
        assert "### GPT-5" in md
        assert "It understands everything." in md
        assert "[OpenAI](https://openai.com)" in md
        assert "### Claude 4" in md
        assert "### Mini Update" in md
        assert "[TechCrunch](https://tc.com)" in md
        assert "综合评分: 14/15" in md
        assert "Ignored" not in md
        assert "本期由 AI 自动生成" in md

    def test_no_thesis(self) -> None:
        brief = {
            "thesis": "",
            "stories": [
                {
                    "role": "main",
                    "representative_title": "Only Story",
                    "context": {"factual_summary": ["Summary."]},
                    "total_score": 12,
                    "items": [],
                }
            ],
        }
        md = generate_show_notes(brief, episode_title="T", episode_date=datetime(2024, 1, 1))
        assert "> " not in md  # No thesis block

    def test_empty_stories(self) -> None:
        md = generate_show_notes(
            {"stories": []},
            episode_title="Empty",
            episode_date=datetime(2024, 1, 1),
        )
        assert "# Empty" in md
        assert "🔴 主要报道" not in md

    def test_items_limit_to_5(self) -> None:
        items = [{"source_name": f"S{i}", "link": f"https://s{i}.com"} for i in range(10)]
        brief = {
            "stories": [
                {
                    "role": "main",
                    "representative_title": "Many Sources",
                    "context": {"factual_summary": []},
                    "total_score": 12,
                    "items": items,
                }
            ],
        }
        md = generate_show_notes(brief, episode_title="T", episode_date=datetime(2024, 1, 1))
        # Only first 5 items should appear
        assert md.count("[S") == 5
