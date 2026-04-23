"""Tests for ai_news_podcast.site_builder modules."""

from __future__ import annotations

from ai_news_podcast.site_builder.rss_gen import build_feed_xml


class TestBuildFeedXml:
    def test_contains_required_elements(self) -> None:
        xml = build_feed_xml(
            base_url="https://example.com",
            podcast_title="Test Podcast",
            podcast_description="A test podcast.",
            podcast_language="zh-cn",
            podcast_author="Tester",
            podcast_category="Technology",
            podcast_explicit=False,
            episodes=[
                {
                    "title": "Ep 1",
                    "description": "<p>First episode</p>",
                    "pubDate": "Mon, 01 Jan 2024 00:00:00 +0000",
                    "guid": "https://example.com/episodes/2024-01-01",
                    "link": "https://example.com/episodes/2024-01-01.html",
                    "enclosure_url": "https://example.com/episodes/2024-01-01.mp3",
                    "enclosure_length": 12345,
                }
            ],
        )
        assert xml.startswith("<?xml version='1.0' encoding='utf-8'?>")
        assert "<rss" in xml
        assert 'version="2.0"' in xml
        assert "<title>Test Podcast</title>" in xml
        assert "<link>https://example.com/</link>" in xml
        assert "<language>zh-cn</language>" in xml
        assert "<enclosure" in xml
        assert 'url="https://example.com/episodes/2024-01-01.mp3"' in xml
        assert 'type="audio/mpeg"' in xml
        assert "itunes:author" in xml
        assert "itunes:explicit>no<" in xml

    def test_empty_episodes(self) -> None:
        xml = build_feed_xml(
            base_url="https://example.com",
            podcast_title="Empty",
            podcast_description="No episodes yet.",
            podcast_language="zh-cn",
            podcast_author="Tester",
            podcast_category="Technology",
            podcast_explicit=False,
            episodes=[],
        )
        assert "<title>Empty</title>" in xml
        assert "<item>" not in xml
