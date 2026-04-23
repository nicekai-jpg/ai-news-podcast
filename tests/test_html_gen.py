"""Tests for ai_news_podcast.site_builder.html_gen."""

from __future__ import annotations

from pathlib import Path

from ai_news_podcast.pipeline.scriptwriter import generate_show_notes_html
from ai_news_podcast.site_builder.html_gen import build_index_html


class TestBuildIndexHtml:
    def test_creates_index_html(self, tmp_path: Path) -> None:
        episodes = [
            {
                "guid": "https://example.com/episodes/2024-03-15",
                "title": "Ep 1",
                "description": "<p>First episode</p>",
                "pubDate": "Mon, 01 Jan 2024 00:00:00 +0000",
                "enclosure_url": "https://example.com/episodes/2024-03-15.mp3",
                "enclosure_length": 1048576,
            }
        ]
        build_index_html(tmp_path, "Test Podcast", episodes, "https://example.com")
        index_path = tmp_path / "index.html"
        assert index_path.exists()
        html = index_path.read_text(encoding="utf-8")
        assert "Test Podcast" in html
        assert "Ep 1" in html
        assert "https://example.com/episodes/2024-03-15.mp3" in html
        assert "1.0 MB" in html
        assert "<audio controls" in html

    def test_empty_episodes(self, tmp_path: Path) -> None:
        build_index_html(tmp_path, "Empty Podcast", [], "https://example.com")
        html = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "暂无节目，请稍后再来。" in html

    def test_html_escaping(self, tmp_path: Path) -> None:
        episodes = [
            {
                "guid": "https://example.com/episodes/1",
                "title": "Title",
                "description": '<script>alert("xss")</script>',
                "pubDate": "Mon, 01 Jan 2024 00:00:00 +0000",
                "enclosure_length": 0,
            }
        ]
        build_index_html(tmp_path, "Podcast", episodes, "https://example.com")
        html = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_limits_to_30_episodes(self, tmp_path: Path) -> None:
        episodes = [
            {
                "guid": f"https://example.com/episodes/{i:02d}",
                "title": f"Episode {i}",
                "description": "",
                "pubDate": "Mon, 01 Jan 2024 00:00:00 +0000",
                "enclosure_length": 0,
            }
            for i in range(35)
        ]
        build_index_html(tmp_path, "Podcast", episodes, "https://example.com")
        html = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert html.count('class="ep-card"') == 30


class TestGenerateShowNotesHtml:
    def test_generates_html_structure(self) -> None:
        from datetime import datetime

        brief = {
            "stories": [
                {
                    "role": "main",
                    "role_emoji": "🔴",
                    "representative_title": "GPT-5 发布",
                    "items": [
                        {"link": "https://openai.com", "source_name": "OpenAI"}
                    ],
                },
                {
                    "role": "skip",
                    "representative_title": "Ignored",
                    "items": [],
                },
            ]
        }
        html = generate_show_notes_html(
            brief,
            episode_title="AI 新闻快报 | 2024-03-15",
            episode_date=datetime(2024, 3, 15),
        )
        assert "<!doctype html>" in html
        assert "AI 新闻快报 | 2024-03-15" in html
        assert "2024年3月15日" in html
        assert "GPT-5 发布" in html
        assert "https://openai.com" in html
        assert "OpenAI" in html
        assert "Ignored" not in html

    def test_escapes_special_chars(self) -> None:
        from datetime import datetime

        brief = {
            "stories": [
                {
                    "role": "main",
                    "role_emoji": "🔴",
                    "representative_title": "A & B <release>",
                    "items": [
                        {"link": "https://a.com", "source_name": "A&B Corp"}
                    ],
                }
            ]
        }
        html = generate_show_notes_html(
            brief,
            episode_title="Test",
            episode_date=datetime(2024, 3, 15),
        )
        assert "A &amp; B &lt;release&gt;" in html
        assert "A&amp;B Corp" in html

    def test_empty_stories(self) -> None:
        from datetime import datetime

        html = generate_show_notes_html(
            {"stories": []},
            episode_title="Empty",
            episode_date=datetime(2024, 3, 15),
        )
        assert "<ol>\n\n</ol>" in html or "<ol>" in html
