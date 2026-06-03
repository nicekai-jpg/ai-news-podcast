"""Tests for runner.py pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from ai_news_podcast.pipeline.runner import get_recent_broadcasted_urls
from ai_news_podcast.utils import write_json


def test_get_recent_broadcasted_urls_missing(tmp_path: Path) -> None:
    path = tmp_path / "episodes.json"
    urls = get_recent_broadcasted_urls(path)
    assert isinstance(urls, set)
    assert len(urls) == 0


def test_get_recent_broadcasted_urls_valid(tmp_path: Path) -> None:
    path = tmp_path / "episodes.json"
    episodes = [
        {
            "id": "2026-06-03",
            "description": '<p>Demo description</p><ol><li>🔴 <a href="https://example.com/moment-1">Moment 1</a></li><li>🔴 <a href="http://example.com/moment-2/">Moment 2</a></li></ol>',
            "enclosure_url": "https://example.com/demo.mp3",
        },
        {
            "id": "2026-06-02",
            "description": '<p>Demo description</p><ol><li>🔴 <a href="https://example.com/moment-3?utm_source=test">Moment 3</a></li></ol>',
            "enclosure_url": "https://example.com/demo2.mp3",
        }
    ]
    write_json(path, episodes)

    urls = get_recent_broadcasted_urls(path, limit=2)
    assert isinstance(urls, set)
    # The links should be normalized:
    # 1. https://example.com/moment-1
    # 2. http://example.com/moment-2/ -> https://example.com/moment-2
    # 3. https://example.com/moment-3?utm_source=test -> https://example.com/moment-3
    assert "https://example.com/moment-1" in urls
    assert "https://example.com/moment-2" in urls
    assert "https://example.com/moment-3" in urls
    # Ensure mp3 is not included
    assert "https://example.com/demo.mp3" not in urls
