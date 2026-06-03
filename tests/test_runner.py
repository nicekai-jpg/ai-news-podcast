"""Tests for runner.py pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
import pytest
from ai_news_podcast.pipeline.runner import get_recent_broadcasted_urls, get_recent_broadcasted_texts
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


def test_get_recent_broadcasted_texts(tmp_path: Path) -> None:
    path = tmp_path / "episodes.json"
    episodes = [
        {
            "id": "2026-06-03",
            "description": '<p>Demo description</p><ol><li>🔴 <a href="https://example.com/moment-1">Google I/O 2026 recap</a></li><li>🔴 <a href="http://example.com/moment-2/">Anthropic Claude 3.5 Sonnet</a></li></ol>',
            "enclosure_url": "https://example.com/demo.mp3",
        }
    ]
    write_json(path, episodes)

    records = get_recent_broadcasted_texts(path, limit=1)
    assert isinstance(records, list)
    assert len(records) == 2
    texts = [r.text for r in records]
    assert "Google I/O 2026 recap" in texts
    assert "Anthropic Claude 3.5 Sonnet" in texts


@pytest.mark.asyncio
async def test_run_pipeline_semantic_dedup(tmp_path: Path, raw_item_factory) -> None:
    from unittest.mock import patch, AsyncMock
    from ai_news_podcast.pipeline.runner import run_pipeline

    episodes_path = tmp_path / "episodes.json"
    episodes = [
        {
            "id": "2026-06-02",
            "description": '<p>Demo</p><ol><li>🔴 <a href="https://example.com/item1">Catch up on 12 major I/O 2026 moments</a></li></ol>',
        }
    ]
    write_json(episodes_path, episodes)

    cfg = {
        "processing": {
            "dedup": {
                "semantic_sim_threshold": 0.20,
                "embedding_sim_threshold": 0.20,
            }
        }
    }

    item_similar = raw_item_factory(
        title="Google I/O 2026 developer collection",
        link="https://example.com/similar-item",
        summary="A compilation of developer resources from I/O 2026.",
    )
    item_different = raw_item_factory(
        title="MiniMax M3 model released",
        link="https://example.com/diff-item",
        summary="MiniMax released their new M3 model today with excellent multi-modal support.",
    )

    with patch("ai_news_podcast.pipeline.runner.fetch_all", new_callable=AsyncMock) as mock_fetch, \
         patch("ai_news_podcast.pipeline.runner.process") as mock_process:

        mock_fetch.return_value = [item_similar, item_different]
        mock_process.return_value = {"stories": []}

        brief = await run_pipeline(
            cfg=cfg,
            sources=[],
            date_str="2026-06-03",
            data_dir=tmp_path,
            force_refresh=True,
        )

        called_args = mock_process.call_args[0][0]
        assert len(called_args) == 1
        assert called_args[0].title == "MiniMax M3 model released"

        assert "metadata" in brief
        assert "dedup_details" in brief["metadata"]
        dedup_details = brief["metadata"]["dedup_details"]
        assert len(dedup_details) == 1
        assert dedup_details[0]["title"] == "Google I/O 2026 developer collection"
        assert dedup_details[0]["reason"] == "cross_episode_semantic"
        assert dedup_details[0]["matched_story_title"] == "Catch up on 12 major I/O 2026 moments"


@pytest.mark.asyncio
async def test_run_pipeline_semantic_dedup_tfidf_fallback(tmp_path: Path, raw_item_factory) -> None:
    from unittest.mock import patch, AsyncMock
    from ai_news_podcast.pipeline.runner import run_pipeline

    episodes_path = tmp_path / "episodes.json"
    episodes = [
        {
            "id": "2026-06-02",
            "description": '<p>Demo</p><ol><li>🔴 <a href="https://example.com/item1">Catch up on 12 major I/O 2026 moments</a></li></ol>',
        }
    ]
    write_json(episodes_path, episodes)

    cfg = {
        "processing": {
            "dedup": {
                "semantic_sim_threshold": 0.20,
            }
        }
    }

    item_similar = raw_item_factory(
        title="Google I/O 2026 developer collection",
        link="https://example.com/similar-item",
        summary="A compilation of developer resources from I/O 2026.",
    )
    item_different = raw_item_factory(
        title="MiniMax M3 model released",
        link="https://example.com/diff-item",
        summary="MiniMax released their new M3 model today with excellent multi-modal support.",
    )

    with patch("ai_news_podcast.pipeline.runner.fetch_all", new_callable=AsyncMock) as mock_fetch, \
         patch("ai_news_podcast.pipeline.runner.process") as mock_process, \
         patch.dict("sys.modules", {"sentence_transformers": None}):

        mock_fetch.return_value = [item_similar, item_different]
        mock_process.return_value = {"stories": []}

        brief = await run_pipeline(
            cfg=cfg,
            sources=[],
            date_str="2026-06-03",
            data_dir=tmp_path,
            force_refresh=True,
        )

        called_args = mock_process.call_args[0][0]
        assert len(called_args) == 1
        assert called_args[0].title == "MiniMax M3 model released"

        assert "metadata" in brief
        assert "dedup_details" in brief["metadata"]
        dedup_details = brief["metadata"]["dedup_details"]
        assert len(dedup_details) == 1
        assert dedup_details[0]["title"] == "Google I/O 2026 developer collection"
        assert dedup_details[0]["reason"] == "cross_episode_semantic"
        assert dedup_details[0]["matched_story_title"] == "Catch up on 12 major I/O 2026 moments"


@pytest.mark.asyncio
async def test_run_pipeline_skips_current_episode(tmp_path: Path, raw_item_factory) -> None:
    from unittest.mock import patch, AsyncMock
    from ai_news_podcast.pipeline.runner import run_pipeline

    episodes_path = tmp_path / "episodes.json"
    # Create episodes list containing the current day (2026-06-03)
    episodes = [
        {
            "id": "2026-06-03",
            "description": '<p>Demo</p><ol><li>🔴 <a href="https://example.com/item1">Catch up on 12 major I/O 2026 moments</a></li></ol>',
        }
    ]
    write_json(episodes_path, episodes)

    cfg = {
        "processing": {
            "dedup": {
                "semantic_sim_threshold": 0.20,
                "embedding_sim_threshold": 0.20,
            }
        }
    }

    # This item has the exact same link and a highly similar title, which would normally trigger deduplication
    item_similar = raw_item_factory(
        title="Google I/O 2026 developer collection",
        link="https://example.com/item1",
        summary="A compilation of developer resources from I/O 2026.",
    )

    with patch("ai_news_podcast.pipeline.runner.fetch_all", new_callable=AsyncMock) as mock_fetch, \
         patch("ai_news_podcast.pipeline.runner.process") as mock_process:

        mock_fetch.return_value = [item_similar]
        mock_process.return_value = {"stories": []}

        brief = await run_pipeline(
            cfg=cfg,
            sources=[],
            date_str="2026-06-03",
            data_dir=tmp_path,
            force_refresh=True,
        )

        called_args = mock_process.call_args[0][0]
        # Because we skip current_episode_id ("2026-06-03") during deduplication, the item is NOT deduplicated!
        assert len(called_args) == 1
        assert called_args[0].title == "Google I/O 2026 developer collection"
        assert len(brief.get("metadata", {}).get("dedup_details", [])) == 0

