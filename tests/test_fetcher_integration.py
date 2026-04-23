"""Integration tests for fetcher with mocked HTTP and feedparser."""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ai_news_podcast.pipeline.fetcher import fetch_all


def _make_entry(
    *,
    title: str = "Test Article",
    link: str = "https://example.com/article",
    summary: str = "Summary text.",
    published_parsed: tuple | None = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        title=title,
        link=link,
        summary=summary,
        description="",
        published_parsed=published_parsed or (2024, 1, 1, 0, 0, 0, 0, 1, 0),
        updated_parsed=None,
        created_parsed=None,
    )


def _make_feed(entries: list[types.SimpleNamespace]) -> types.SimpleNamespace:
    return types.SimpleNamespace(entries=entries)


@pytest.mark.asyncio
async def test_fetch_all_success(
    mock_httpx_client,
    mock_feedparser,
    mock_trafilatura,
    mock_readability,
    mock_bs4,
) -> None:
    """fetch_all should return RawItems when feeds parse successfully."""
    rss_xml = b"<rss><channel></channel></rss>"

    # HTTP responses: first for feed, second for full-text page
    responses = [
        MagicMock(content=rss_xml, text="<html>page</html>", status_code=200),
        MagicMock(content=rss_xml, text="<html>page</html>", status_code=200),
    ]
    mock_httpx_client.get = AsyncMock(side_effect=responses)

    feed_entries = [
        _make_entry(title="Article One", link="https://a.com/1"),
        _make_entry(title="Article Two", link="https://a.com/2"),
    ]
    mock_feedparser.parse.side_effect = [
        _make_feed(feed_entries),
        _make_feed(feed_entries),
    ]

    mock_trafilatura.extract.return_value = "This is the full text content of the article." * 30

    with (
        patch("httpx.AsyncClient", return_value=mock_httpx_client),
        patch.dict(sys.modules, {"feedparser": mock_feedparser}),
        patch.dict(sys.modules, {"trafilatura": mock_trafilatura}),
        patch.dict(sys.modules, {"readability": mock_readability}),
        patch.dict(sys.modules, {"bs4": mock_bs4}),
    ):
        sources = [
            {
                "name": "Source A",
                "url": "https://a.com/feed.xml",
                "category": "news",
                "enabled": True,
            },
            {
                "name": "Source B",
                "url": "https://b.com/feed.xml",
                "category": "official",
                "enabled": True,
            },
        ]
        items = await fetch_all(
            sources,
            timeout_seconds=5,
            connect_timeout=2,
            user_agent="test",
            max_items_per_feed=5,
            max_pages=10,
        )

    assert len(items) == 4  # 2 feeds × 2 entries each
    titles = {item.title for item in items}
    assert titles == {"Article One", "Article Two"}


@pytest.mark.asyncio
async def test_fetch_all_no_enabled_sources() -> None:
    """fetch_all should return empty list when no sources are enabled."""
    sources = [
        {"name": "A", "url": "https://a.com", "enabled": False},
    ]
    items = await fetch_all(sources)
    assert items == []


@pytest.mark.asyncio
async def test_fetch_all_http_error_continues(
    mock_httpx_client,
    mock_feedparser,
) -> None:
    """If one feed fails with HTTP error, fetch_all should continue with others."""
    rss_xml = b"<rss><channel></channel></rss>"
    ok_response = MagicMock(content=rss_xml, text="<html>page</html>", status_code=200)

    responses = [
        httpx.HTTPError("boom"),  # first feed fails
        ok_response,              # second feed ok
        ok_response,              # full text for second feed
    ]
    mock_httpx_client.get = AsyncMock(side_effect=responses)

    mock_feedparser.parse.return_value = _make_feed(
        [_make_entry(title="Survivor", link="https://b.com/1")]
    )

    with (
        patch("httpx.AsyncClient", return_value=mock_httpx_client),
        patch.dict(sys.modules, {"feedparser": mock_feedparser}),
    ):
        sources = [
            {"name": "Bad", "url": "https://bad.com/feed", "enabled": True},
            {"name": "Good", "url": "https://good.com/feed", "enabled": True},
        ]
        items = await fetch_all(sources, max_pages=10)

    assert len(items) == 1
    assert items[0].title == "Survivor"
