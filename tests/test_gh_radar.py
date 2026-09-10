"""Tests for ai_news_podcast.pipeline.gh_radar 与相关配置。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from ai_news_podcast.config.models import AppConfig
from ai_news_podcast.pipeline.gh_client import GhClient
from ai_news_podcast.pipeline.gh_radar import (
    RadarProject,
    _hands_on_excerpt,
    build_radar,
    count_news_mentions,
    score_project,
)
from ai_news_podcast.utils import write_json


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
            seen["sort"] = request.url.params["sort"]
            seen["per_page"] = request.url.params["per_page"]
            return httpx.Response(200, json={"items": [{"full_name": "owner/repo"}]})

        gh = _gh(handler)
        items = await gh.search_repos("created:>2026-08-01 stars:>=500")
        assert items[0]["full_name"] == "owner/repo"
        assert seen["auth"] == "Bearer t0k"
        assert "search/repositories" in seen["url"]
        assert seen["sort"] == "stars"
        assert seen["per_page"] == "30"
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

    @pytest.mark.asyncio
    async def test_fetch_readme_success_raw_accept_and_truncation(self) -> None:
        body = "x" * 7000
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["accept"] = request.headers["Accept"]
            return httpx.Response(200, text=body)

        gh = _gh(handler)
        text = await gh.fetch_readme_text("owner/repo")
        assert seen["accept"] == "application/vnd.github.raw+json"
        assert text == body[:6000]
        await gh.aclose()


class FakeGhClient:
    """与 GhClient 同接口的测试替身。"""

    def __init__(self, items: list[dict], readmes: dict[str, str] | None = None) -> None:
        self.items = items
        self.readmes = readmes or {}

    async def search_repos(self, query: str, *, sort: str = "stars", per_page: int = 30):
        return self.items

    async def fetch_readme_text(self, repo: str) -> str:
        return self.readmes.get(repo, "")

    async def aclose(self) -> None:
        return None


_NOW = datetime(2026, 9, 9, tzinfo=UTC)


def _repo_item(
    full_name: str,
    stars: int,
    *,
    pushed: str = "2026-09-08T00:00:00Z",
    license_id: str = "MIT",
    topics: list[str] | None = None,
    desc: str = "A real tool",
) -> dict:
    return {
        "full_name": full_name,
        "html_url": f"https://github.com/{full_name}",
        "description": desc,
        "language": "Python",
        "topics": topics or [],
        "stargazers_count": stars,
        "created_at": "2026-08-15T00:00:00Z",
        "pushed_at": pushed,
        "license": {"spdx_id": license_id},
    }


GCFG = {
    "enabled": True,
    "min_stars": 500,
    "created_window_days": 30,
    "recent_push_days": 21,
    "top_n": 30,
    "pick_count": 1,
    "runner_up_count": 2,
    "readme_probe_limit": 12,
    "repeat_window_days": 30,
    "readme_excerpt_chars": 1200,
    "excluded_name_patterns": ["awesome", "list"],
    "preferred_topics": ["llm"],
    "snapshot_dir": "gh_snapshots",
    "output_dir": "gh_radar",
}


class TestCountNewsMentions:
    def test_counts_title_hits(self) -> None:
        titles = ["BigLang released", "biglang v2 out now", "unrelated news"]
        assert count_news_mentions("acme/biglang", titles) == 2

    def test_short_tokens_ignored(self) -> None:
        assert count_news_mentions("acme/gpt", ["gpt is everywhere"]) == 0


class TestHandsOnExcerpt:
    def test_quickstart_heading_extracted(self) -> None:
        readme = "# Hot\nA tool.\n\n## Quickstart\npip install hot\nhot run\n\n## License\nMIT"
        text = _hands_on_excerpt(readme, 1200)
        assert "pip install hot" in text
        assert "MIT" not in text

    def test_no_heading_falls_back_to_install_line(self) -> None:
        readme = "Some intro.\nJust run `pip install hot` to start."
        assert "pip install hot" in _hands_on_excerpt(readme, 1200)

    def test_empty_readme(self) -> None:
        assert _hands_on_excerpt("", 1200) == ""


class TestScoreProject:
    def test_full_score_computation(self) -> None:
        p = _mk(
            delta_stars=3000,
            has_install_docs=True,
            license="MIT",
            pushed_at="2026-09-08T00:00:00Z",
            mentions=2,
            topics=["llm"],
        )
        score_project(p, now=_NOW, preferred_topics=["llm"])
        # velocity=1.0; hands_on=0.4+0.3+0.3=1.0; cross=2/3; bonus=0.1
        assert p.score_parts["velocity"] == 1.0
        assert p.score_parts["hands_on"] == 1.0
        assert p.score > 0.9


def _mk(**kw) -> RadarProject:
    defaults = {
        "repo": "owner/repo",
        "url": "https://github.com/owner/repo",
        "description": "d",
        "language": "Python",
        "topics": [],
        "stars": 1000,
        "delta_stars": None,
        "is_new": True,
        "created_at": "2026-08-15T00:00:00Z",
        "pushed_at": "2026-09-08T00:00:00Z",
        "license": "MIT",
        "has_install_docs": False,
        "mentions": 0,
        "readme_excerpt": "",
        "score": 0.0,
        "score_parts": {},
    }
    defaults.update(kw)
    return RadarProject(**defaults)


class TestBuildRadar:
    @pytest.mark.asyncio
    async def test_end_to_end_with_snapshot_delta(self, tmp_path: Path) -> None:
        snap_dir = tmp_path / "gh_snapshots"
        snap_dir.mkdir()
        write_json(
            snap_dir / "snap_2026-09-08.json",
            {"date": "2026-09-08", "stars": {"owner/hot": 500}},
        )
        items = [
            _repo_item("owner/hot", 1500, topics=["llm"]),
            _repo_item("owner/awesome-ai", 9000, desc="A list of AI stuff"),
            _repo_item("owner/stale", 600, pushed="2026-05-01T00:00:00Z"),
            _repo_item("owner/nolicense", 600, license_id=""),
        ]
        gh = FakeGhClient(items, {"owner/hot": "pip install hot\n# quickstart"})
        radar = await build_radar(GCFG, "2026-09-09", tmp_path, [], client=gh, now=_NOW)

        repos = [p["repo"] for p in radar["projects"]]
        assert "owner/hot" in repos
        assert "owner/awesome-ai" not in repos  # 合集被排除
        assert "owner/stale" not in repos  # 42 天无 commit
        assert "owner/nolicense" not in repos  # 无许可证
        hot = next(p for p in radar["projects"] if p["repo"] == "owner/hot")
        assert hot["delta_stars"] == 1000  # 1500 - 500
        assert hot["has_install_docs"] is True
        assert radar["meta"]["pick_repo"] == radar["projects"][0]["repo"]
        assert (tmp_path / "gh_radar" / "radar_2026-09-09.json").exists()
        snap = snap_dir / "snap_2026-09-09.json"
        assert snap.exists()  # 当日快照已写

    @pytest.mark.asyncio
    async def test_readme_excerpt_in_pick(self, tmp_path: Path) -> None:
        readme = "# Hot\n\n## Quickstart\npip install hot\n\n## License\nMIT"
        gh = FakeGhClient([_repo_item("owner/hot", 800)], {"owner/hot": readme})
        radar = await build_radar(GCFG, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
        assert "pip install hot" in radar["projects"][0]["readme_excerpt"]

    @pytest.mark.asyncio
    async def test_recent_picks_excluded(self, tmp_path: Path) -> None:
        out = tmp_path / "gh_radar"
        out.mkdir()
        write_json(
            out / "radar_2026-09-08.json",
            {
                "date": "2026-09-08",
                "projects": [{"repo": "owner/hot"}, {"repo": "owner/second"}],
                "meta": {"pick_repo": "owner/hot", "runner_up_repos": ["owner/second"]},
            },
        )
        items = [
            _repo_item("owner/hot", 1500),
            _repo_item("owner/second", 900),
            _repo_item("owner/fresh2", 800),
        ]
        gh = FakeGhClient(items)
        radar = await build_radar(GCFG, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
        repos = [p["repo"] for p in radar["projects"]]
        assert "owner/hot" not in repos
        assert "owner/second" not in repos
        assert "owner/fresh2" in repos
        assert radar["meta"]["excluded_recent"] == 2

    @pytest.mark.asyncio
    async def test_first_day_no_delta(self, tmp_path: Path) -> None:
        gh = FakeGhClient([_repo_item("owner/fresh", 800)])
        radar = await build_radar(GCFG, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
        assert radar["projects"][0]["delta_stars"] is None
        assert radar["projects"][0]["is_new"] is True

    @pytest.mark.asyncio
    async def test_disabled_returns_degraded(self, tmp_path: Path) -> None:
        gh = FakeGhClient([])
        radar = await build_radar(
            {"enabled": False}, "2026-09-09", tmp_path, [], client=gh, now=_NOW
        )
        assert radar["meta"]["degraded"] is True
        assert radar["meta"]["reason"] == "disabled"

    @pytest.mark.asyncio
    async def test_search_failure_propagates(self, tmp_path: Path) -> None:
        class Boom(FakeGhClient):
            async def search_repos(self, query, *, sort="stars", per_page=30):
                raise RuntimeError("api down")

        gh = Boom([])
        with pytest.raises(RuntimeError):
            await build_radar(GCFG, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
