"""html_gen 站点生成测试。"""

import tempfile
from pathlib import Path

from ai_news_podcast.site_builder.html_gen import format_friendly_date, build_index_html


class TestFormatFriendlyDate:
    def test_iso_format(self):
        result = format_friendly_date("2026-06-04")
        assert "2026" in result
        assert "06-04" in result

    def test_rfc_format(self):
        result = format_friendly_date("Thu, 04 Jun 2026 01:05:50 +0000")
        assert "06-04" in result

    def test_empty_string(self):
        assert format_friendly_date("") == ""

    def test_invalid_returns_original(self):
        assert format_friendly_date("not-a-date") == "not-a-date"

    def test_none_like_empty(self):
        assert format_friendly_date("") == ""


class TestBuildIndexHtml:
    def test_creates_index_file(self, tmp_path):
        site_dir = tmp_path / "site"
        episodes = [
            {
                "id": "2026-06-04",
                "title": "测试",
                "pubDate": "Thu, 04 Jun 2026 01:05:50 +0000",
            }
        ]
        build_index_html(site_dir, "测试播客", episodes, "http://localhost:8000/site")
        assert (site_dir / "index.html").exists()

    def test_index_contains_title(self, tmp_path):
        site_dir = tmp_path / "site"
        episodes = [
            {
                "id": "2026-06-04",
                "title": "测试",
                "pubDate": "Thu, 04 Jun 2026 01:05:50 +0000",
            }
        ]
        build_index_html(site_dir, "测试播客", episodes, "http://localhost:8000/site")
        content = (site_dir / "index.html").read_text(encoding="utf-8")
        assert "测试播客" in content

    def test_index_contains_episodes_map(self, tmp_path):
        site_dir = tmp_path / "site"
        episodes = [
            {
                "id": "2026-06-04",
                "title": "AI 新闻快报 | 2026-06-04",
                "enclosure_url": "https://example.com/episodes/2026-06-04.mp3",
            }
        ]
        build_index_html(site_dir, "Test Podcast", episodes, "https://example.com")
        html = (site_dir / "index.html").read_text(encoding="utf-8")
        assert '"2026-06-04"' in html
        assert "2026-06-04.mp3" in html

    def test_index_no_episodes(self, tmp_path):
        site_dir = tmp_path / "site"
        build_index_html(site_dir, "Test Podcast", [], "https://example.com")
        html = (site_dir / "index.html").read_text(encoding="utf-8")
        assert "<title>Test Podcast</title>" in html
        assert "DATES = []" in html

    def test_index_contains_key_elements(self, tmp_path):
        site_dir = tmp_path / "site"
        episodes = [
            {
                "id": "2026-06-04",
                "title": "AI 新闻快报 | 2026-06-04",
                "enclosure_url": "https://example.com/episodes/2026-06-04.mp3",
            }
        ]
        build_index_html(site_dir, "Test Podcast", episodes, "https://example.com")
        html = (site_dir / "index.html").read_text(encoding="utf-8")
        assert "layout-grid" in html or "podcast-workspace" in html
        assert "player-bar" in html or "console-player" in html
        assert "./feed.xml" in html
        assert "parseTranscript" in html

    def test_index_base_url(self, tmp_path):
        site_dir = tmp_path / "site"
        episodes = [
            {
                "id": "2026-06-04",
                "title": "AI 新闻快报 | 2026-06-04",
            }
        ]
        build_index_html(site_dir, "Test Podcast", episodes, "https://mypod.example.com")
        html = (site_dir / "index.html").read_text(encoding="utf-8")
        assert "https://mypod.example.com" in html

    def test_index_transcript_parser(self, tmp_path):
        site_dir = tmp_path / "site"
        episodes = [
            {
                "id": "2026-06-04",
                "title": "AI 新闻快报 | 2026-06-04",
                "enclosure_url": "https://example.com/episodes/2026-06-04.mp3",
            }
        ]
        build_index_html(site_dir, "Test Podcast", episodes, "https://example.com")
        html = (site_dir / "index.html").read_text(encoding="utf-8")
        assert "parseTranscript" in html
        assert "dialogue-line" in html or "transcript-row" in html
        assert "speaker-label" in html or "speaker-meta" in html
        assert "host-a" in html
        assert "host-b" in html
