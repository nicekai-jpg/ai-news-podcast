import json
from pathlib import Path

from ai_news_podcast.site_builder.html_gen import build_index_html


def _make_episodes(n=3):
    episodes = []
    for i in range(n):
        date_str = f"2026-06-0{3 - i}"
        episodes.append(
            {
                "id": date_str,
                "title": f"AI 新闻快报 | {date_str}",
                "enclosure_url": f"https://example.com/episodes/{date_str}.mp3",
            }
        )
    return episodes


def test_build_index_html_creates_file(tmp_path):
    site_dir = tmp_path / "site"
    build_index_html(site_dir, "Test Podcast", _make_episodes(), "https://example.com")
    assert (site_dir / "index.html").exists()


def test_build_index_html_contains_key_elements(tmp_path):
    site_dir = tmp_path / "site"
    episodes = _make_episodes()
    build_index_html(site_dir, "Test Podcast", episodes, "https://example.com")
    html = (site_dir / "index.html").read_text(encoding="utf-8")

    # Title
    assert "<title>Test Podcast</title>" in html

    # Layout structure
    assert "layout-grid" in html
    assert "date-list" in html
    assert "content-split" in html
    assert "report-markdown" in html
    assert "transcript-content" in html

    # No old sidebar/tabs
    assert "intro-sidebar" not in html
    assert "tab-container" not in html
    assert "tab-nav" not in html

    # Player bar
    assert "player-bar" in html

    # RSS link
    assert "./feed.xml" in html

    # Dates in JS
    assert '"2026-06-03"' in html
    assert '"2026-06-02"' in html
    assert '"2026-06-01"' in html


def test_build_index_html_episodes_map(tmp_path):
    site_dir = tmp_path / "site"
    episodes = _make_episodes()
    build_index_html(site_dir, "Test Podcast", episodes, "https://example.com")
    html = (site_dir / "index.html").read_text(encoding="utf-8")

    # Episodes map should contain mp3 URLs
    assert "2026-06-03.mp3" in html
    assert "2026-06-02.mp3" in html


def test_build_index_html_no_episodes(tmp_path):
    site_dir = tmp_path / "site"
    build_index_html(site_dir, "Test Podcast", [], "https://example.com")
    html = (site_dir / "index.html").read_text(encoding="utf-8")

    assert "<title>Test Podcast</title>" in html
    assert "layout-grid" in html
    # dates array should be empty
    assert "const dates = [];" in html


def test_build_index_html_base_url(tmp_path):
    site_dir = tmp_path / "site"
    build_index_html(site_dir, "Test Podcast", _make_episodes(), "https://mypod.example.com")
    html = (site_dir / "index.html").read_text(encoding="utf-8")

    assert "https://mypod.example.com" in html


def test_build_index_html_transcript_parser(tmp_path):
    site_dir = tmp_path / "site"
    build_index_html(site_dir, "Test Podcast", _make_episodes(), "https://example.com")
    html = (site_dir / "index.html").read_text(encoding="utf-8")

    # Transcript parser functions should exist
    assert "parseTranscript" in html
    assert "dialogue-line" in html
    assert "speaker-label" in html
    assert "host-a" in html
    assert "host-b" in html
