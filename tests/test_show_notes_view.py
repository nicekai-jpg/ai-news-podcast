"""Tests for show_notes_view.generate_show_notes_html（生产唯一入口）。"""

from __future__ import annotations

from datetime import datetime

from ai_news_podcast.presentation.site_builder.show_notes_view import generate_show_notes_html


class TestGenerateShowNotesHtml:
    def test_full_structure(self) -> None:
        brief = {
            "stories": [
                {
                    "role": "main",
                    "representative_title": "GPT-5",
                    "role_emoji": "🔴",
                    "items": [{"source_name": "OpenAI", "link": "https://openai.com"}],
                },
                {
                    "role": "skip",
                    "representative_title": "Ignored",
                    "role_emoji": "⚪",
                    "items": [{"source_name": "X", "link": "https://x.example"}],
                },
            ]
        }
        html = generate_show_notes_html(
            brief,
            episode_title="AI News | 2024-03-15",
            episode_date=datetime(2024, 3, 15),
        )
        assert html.startswith("<!doctype html>")
        assert "<title>AI News | 2024-03-15</title>" in html
        assert "2024年03月15日" in html
        assert '<a href="https://openai.com">GPT-5</a>' in html
        assert "🔴" in html
        assert "(OpenAI)" in html
        assert "Ignored" not in html

    def test_empty_stories(self) -> None:
        html = generate_show_notes_html(
            {},
            episode_title="Empty",
            episode_date=datetime(2024, 1, 1),
        )
        assert "<h1>Empty</h1>" in html
        assert "2024年01月01日" in html
        assert "<ol>" in html
        assert "<li>" not in html

    def test_escapes_title_and_source(self) -> None:
        brief = {
            "stories": [
                {
                    "role": "main",
                    "representative_title": "A & B <C>",
                    "items": [{"source_name": "S <1>", "link": "https://example.com"}],
                }
            ]
        }
        html = generate_show_notes_html(
            brief,
            episode_title="T & <U>",
            episode_date=datetime(2024, 1, 1),
        )
        assert "<title>T &amp; &lt;U&gt;</title>" in html
        assert "A &amp; B &lt;C&gt;" in html
        assert "S &lt;1&gt;" in html
