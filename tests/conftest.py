"""Shared pytest fixtures and helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_news_podcast.pipeline.fetcher import RawItem

# ---------------------------------------------------------------------------
# RawItem factory
# ---------------------------------------------------------------------------


def make_raw_item(
    *,
    title: str = "Test Title",
    link: str = "https://example.com/article",
    normalized_link: str | None = None,
    source_name: str = "Test Source",
    source_category: str = "news",
    published_at: str | None = None,
    summary: str = "Test summary.",
    full_text_snippet: str = "Test full text snippet.",
    category: str = "product",
    language: str = "zh",
) -> RawItem:
    """Factory for RawItem test instances."""
    norm = normalized_link or link
    item_id = f"hashof-{norm}"
    return RawItem(
        id=item_id,
        title=title,
        link=link,
        normalized_link=norm,
        source_name=source_name,
        source_category=source_category,
        published_at=published_at or datetime.now(tz=timezone.utc).isoformat(),
        summary=summary,
        full_text_snippet=full_text_snippet,
        category=category,
        language=language,
    )


@pytest.fixture
def raw_item_factory():
    """Return the make_raw_item factory function."""
    return make_raw_item


# ---------------------------------------------------------------------------
# HTTP / parser mocks (used by fetcher integration & exception tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_feedparser():
    """Return a mock feedparser module that yields controlled feeds."""
    module = MagicMock()
    module.parse = MagicMock()
    return module


@pytest.fixture
def mock_httpx_client():
    """Return a mock httpx.AsyncClient factory."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.fixture
def mock_trafilatura():
    module = MagicMock()
    module.extract = MagicMock(return_value=None)
    return module


@pytest.fixture
def mock_readability():
    doc = MagicMock()
    doc.summary.return_value = "<html><body>fallback text</body></html>"
    module = MagicMock()
    module.Document = MagicMock(return_value=doc)
    return module


@pytest.fixture
def mock_bs4():
    soup = MagicMock()
    soup.get_text = MagicMock(return_value="fallback text")
    module = MagicMock()
    module.BeautifulSoup = MagicMock(return_value=soup)
    return module
