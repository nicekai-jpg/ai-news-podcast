"""Tests for ai_news_podcast.pipeline.gh_radar 与相关配置。"""

from __future__ import annotations

import httpx
import pytest

from ai_news_podcast.config.models import AppConfig
from ai_news_podcast.pipeline.gh_client import GhClient


class TestGhRadarConfig:
    def test_defaults(self) -> None:
        cfg = AppConfig.from_dict({})
        assert cfg.gh_radar.enabled is True
        assert cfg.gh_radar.min_stars == 500
        assert cfg.gh_radar.pick_count == 1
        assert cfg.gh_radar.runner_up_count == 2
        assert cfg.gh_radar.repeat_window_days == 30
        assert cfg.gh_radar.snapshot_dir == "gh_snapshots"

    def test_overrides(self) -> None:
        cfg = AppConfig.from_dict(
            {
                "gh_radar": {
                    "min_stars": 100,
                    "preferred_topics": ["rag"],
                    "pick_count": 2,
                }
            }
        )
        assert cfg.gh_radar.min_stars == 100
        assert cfg.gh_radar.preferred_topics == ["rag"]
        assert cfg.gh_radar.pick_count == 2


def _gh(handler, *, token: str | None = "t0k", sleep_seconds: float = 0.0) -> GhClient:
    transport = httpx.MockTransport(handler)
    return GhClient(
        httpx.AsyncClient(transport=transport), token=token, sleep_seconds=sleep_seconds
    )


class TestGhClient:
    @pytest.mark.asyncio
    async def test_search_repos_parses_items_and_auth(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("Authorization", "")
            return httpx.Response(200, json={"items": [{"full_name": "owner/repo"}]})

        gh = _gh(handler)
        items = await gh.search_repos("created:>2026-08-01 stars:>=500")
        assert items[0]["full_name"] == "owner/repo"
        assert seen["auth"] == "Bearer t0k"
        assert "search/repositories" in seen["url"]
        await gh.aclose()

    @pytest.mark.asyncio
    async def test_fetch_readme_404_returns_empty(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/readme"):
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json={"items": []})

        gh = _gh(handler)
        assert await gh.fetch_readme_text("owner/repo") == ""
        await gh.aclose()
