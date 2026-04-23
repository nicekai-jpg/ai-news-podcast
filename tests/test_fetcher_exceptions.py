"""Exception and edge-case tests for fetcher."""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ai_news_podcast.pipeline.fetcher import (
    _extract_fulltext,
    _http_get,
    fetch_all,
)


class TestExtractFulltext:
    def test_trafilatura_success(self) -> None:
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract.return_value = "x" * 1500

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura}):
            text = _extract_fulltext("<html>body</html>", "https://example.com")
        assert len(text) == 1500

    def test_fallback_to_readability(self) -> None:
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract.return_value = "short"  # below min_chars

        doc = MagicMock()
        doc.summary.return_value = "<div>long fallback text</div>"
        mock_readability = MagicMock()
        mock_readability.Document = MagicMock(return_value=doc)

        soup = MagicMock()
        soup.get_text = MagicMock(return_value="y" * 1500)
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup = MagicMock(return_value=soup)

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura, "readability": mock_readability, "bs4": mock_bs4}):
            text = _extract_fulltext("<html>body</html>", "https://example.com")
        assert len(text) == 1500

    def test_both_engines_fail_returns_empty(self) -> None:
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract.return_value = None
        mock_readability = MagicMock()
        mock_readability.Document = MagicMock(side_effect=Exception("boom"))

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura, "readability": mock_readability}):
            text = _extract_fulltext("<html>body</html>", "https://example.com")
        assert text == ""

    def test_truncates_when_too_long(self) -> None:
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract.return_value = "z" * 3000

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura}):
            text = _extract_fulltext("<html>body</html>", "https://example.com", max_chars=100)
        assert text.endswith("...")
        assert len(text) <= 100


class TestHttpGetRetry:
    @pytest.mark.asyncio
    async def test_retries_on_httpx_error(self) -> None:
        """_http_get should retry up to 3 times on httpx errors."""
        from ai_news_podcast.pipeline.fetcher import _DomainThrottle

        client = AsyncMock()
        client.get = AsyncMock(side_effect=[
            httpx.HTTPError("fail 1"),
            httpx.HTTPError("fail 2"),
            MagicMock(status_code=200),
        ])
        throttle = _DomainThrottle(interval=0.01)

        resp = await _http_get(client, "https://example.com", throttle)
        assert resp.status_code == 200
        assert client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_exhausted_retries(self) -> None:
        from ai_news_podcast.pipeline.fetcher import _DomainThrottle

        client = AsyncMock()
        client.get = AsyncMock(side_effect=httpx.HTTPError("always fails"))
        throttle = _DomainThrottle(interval=0.01)

        with pytest.raises(httpx.HTTPError):
            await _http_get(client, "https://example.com", throttle)
        assert client.get.call_count == 3


class TestFetchAllEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_entries_feed(self, mock_httpx_client, mock_feedparser) -> None:
        """Feed with no entries should return empty list."""
        rss_xml = b"<rss><channel></channel></rss>"
        mock_httpx_client.get = AsyncMock(return_value=MagicMock(content=rss_xml, text="", status_code=200))
        mock_feedparser.parse.return_value = types.SimpleNamespace(entries=[])

        with (
            patch("httpx.AsyncClient", return_value=mock_httpx_client),
            patch.dict(sys.modules, {"feedparser": mock_feedparser}),
        ):
            sources = [
                {"name": "Empty Feed", "url": "https://empty.com/feed", "enabled": True},
            ]
            items = await fetch_all(sources, max_pages=10)
        assert items == []

    @pytest.mark.asyncio
    async def test_feed_missing_title_or_link_skipped(self, mock_httpx_client, mock_feedparser) -> None:
        """Entries without title or link should be skipped."""
        rss_xml = b"<rss><channel></channel></rss>"
        mock_httpx_client.get = AsyncMock(return_value=MagicMock(content=rss_xml, text="", status_code=200))
        mock_feedparser.parse.return_value = types.SimpleNamespace(entries=[
            types.SimpleNamespace(title="", link="https://a.com", summary="", published_parsed=None),
            types.SimpleNamespace(title="Has Title", link="", summary="", published_parsed=None),
            types.SimpleNamespace(title="Good", link="https://a.com/good", summary="s", published_parsed=None),
        ])

        with (
            patch("httpx.AsyncClient", return_value=mock_httpx_client),
            patch.dict(sys.modules, {"feedparser": mock_feedparser}),
        ):
            sources = [
                {"name": "Mixed", "url": "https://mixed.com/feed", "enabled": True},
            ]
            items = await fetch_all(sources, max_pages=10)
        assert len(items) == 1
        assert items[0].title == "Good"
