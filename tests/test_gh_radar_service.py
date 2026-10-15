"""Tests for ai_news_podcast.business.pipeline.gh_radar_service 与相关配置。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import httpx
import pytest

from ai_news_podcast.business.pipeline.gh_client_service import _README_MAX_CHARS, GhClient
from ai_news_podcast.business.pipeline.gh_radar_service import (
    RadarProject,
    _hands_on_excerpt,
    _parse_trending_slugs,
    _readme_intro,
    ai_affinity,
    build_radar,
    count_news_mentions,
    score_project,
)
from ai_news_podcast.data.config.models_dao import AppConfig
from ai_news_podcast.data.utils_dao import read_json, write_json


class TestGhRadarConfig:
    def test_defaults(self) -> None:
        cfg = AppConfig.from_dict({})
        assert cfg.gh_radar.enabled is True
        assert cfg.gh_radar.min_stars == 500
        assert cfg.gh_radar.pick_count == 1
        assert cfg.gh_radar.runner_up_count == 2
        assert cfg.gh_radar.repeat_window_days == 30
        assert cfg.gh_radar.widen_min_stars_ratio == 0.4
        assert cfg.gh_radar.repeat_gap_days == 7
        assert cfg.gh_radar.direction_gate == "soft"
        assert cfg.gh_radar.ai_topics[0] == "agents"
        assert "https://rsshub.app/github/trending/daily" in cfg.gh_radar.trending_rss_url
        assert set(cfg.gh_radar.ai_lexicon) >= {"agents", "gateway", "rag", "mcp", "app"}
        assert cfg.gh_radar.vel_rel + cfg.gh_radar.vel_abs == 1.0
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
        body = "x" * (_README_MAX_CHARS + 1000)
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["accept"] = request.headers["Accept"]
            return httpx.Response(200, text=body)

        gh = _gh(handler)
        text = await gh.fetch_readme_text("owner/repo")
        assert seen["accept"] == "application/vnd.github.raw+json"
        assert text == body[:_README_MAX_CHARS]
        await gh.aclose()


class FakeGhClient:
    """与 GhClient 同接口的测试替身。"""

    def __init__(
        self,
        items: list[dict],
        readmes: dict[str, str] | None = None,
        repos: dict[str, dict] | None = None,
        rss: str | dict[str, str] = "",
    ) -> None:
        self.items = items
        self.readmes = readmes or {}
        self.repos = repos or {}  # slug → GET /repos 返回
        self.rss = rss
        self.queries: list[str] = []
        self.readme_calls: list[str] = []
        self.repo_calls: list[str] = []
        self.rss_calls: list[str] = []

    async def search_repos(self, query: str, *, sort: str = "stars", per_page: int = 30):
        self.queries.append(query)
        return self.items

    async def fetch_readme_text(self, repo: str) -> str:
        self.readme_calls.append(repo)
        return self.readmes.get(repo, "")

    async def fetch_repo(self, repo: str) -> dict | None:
        self.repo_calls.append(repo)
        return self.repos.get(repo)

    async def fetch_text(self, url: str) -> str:
        self.rss_calls.append(url)
        if isinstance(self.rss, dict):
            return self.rss.get(url, "")
        return self.rss

    async def aclose(self) -> None:
        return None


class QueryGh(FakeGhClient):
    """按查询条件返回不同结果的测试替身(模拟扩池补搜)。"""

    def __init__(self, respond, readmes: dict[str, str] | None = None) -> None:
        self._respond = respond
        self.readmes = readmes or {}
        self.queries: list[str] = []
        self.readme_calls: list[str] = []

    async def search_repos(self, query: str, *, sort: str = "stars", per_page: int = 30):
        self.queries.append(query)
        return self._respond(query)


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
    "widen_min_stars_ratio": 0.4,
    "repeat_gap_days": 7,
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
        # 搜索查询:剧集日期 2026-09-09 往前 30 天
        assert len(gh.queries) == 1
        assert "created:>2026-08-10" in gh.queries[0]
        assert "stars:>=500" in gh.queries[0]

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
    async def test_snapshot_covers_all_before_probe_truncation(self, tmp_path: Path) -> None:
        gcfg = dict(GCFG, readme_probe_limit=1)
        items = [_repo_item("owner/hot", 1500), _repo_item("owner/warm", 1200)]
        gh = FakeGhClient(items, {"owner/hot": "pip install hot"})
        radar = await build_radar(gcfg, "2026-09-09", tmp_path, [], client=gh, now=_NOW)

        snap = read_json(tmp_path / "gh_snapshots" / "snap_2026-09-09.json")
        assert set(snap["stars"]) == {"owner/hot", "owner/warm"}  # 截断前快照,新进榜也有基线
        assert gh.readme_calls == ["owner/hot"]  # 只探测截断后的第一名
        assert [p["repo"] for p in radar["projects"]] == ["owner/hot"]

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

    @pytest.mark.asyncio
    async def test_widen_pass_when_fresh_exhausted(self, tmp_path: Path) -> None:
        out = tmp_path / "gh_radar"
        out.mkdir()
        write_json(
            out / "radar_2026-09-08.json",
            {
                "date": "2026-09-08",
                "projects": [{"repo": "owner/hot"}],
                "meta": {"pick_repo": "owner/hot", "runner_up_repos": []},
            },
        )
        strict = [_repo_item("owner/hot", 1500)]
        widened = [_repo_item("owner/lowc", 300)]

        def respond(query: str) -> list[dict]:
            return strict if "stars:>=500" in query else widened

        gh = QueryGh(respond, {"owner/lowc": "pip install lowc"})
        radar = await build_radar(GCFG, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
        # 新品为 0 → 触发扩池补搜,门槛放宽到 500×0.4=200
        assert len(gh.queries) == 2
        assert "stars:>=500" in gh.queries[0]
        assert "stars:>=200" in gh.queries[1]

        repos = [p["repo"] for p in radar["projects"]]
        assert repos == ["owner/lowc"]  # 扩池捞到的起势项目成为主推
        assert "owner/hot" not in repos  # 近期推过的仍旧不重复上榜
        meta = radar["meta"]
        assert meta["widen_pass"] is True
        assert meta["degraded"] is False
        assert meta["fresh_candidates"] == 1
        assert meta["excluded_recent"] == 1
        snap = read_json(tmp_path / "gh_snapshots" / "snap_2026-09-09.json")
        assert set(snap["stars"]) == {"owner/hot", "owner/lowc"}  # 扩池结果也进快照

    @pytest.mark.asyncio
    async def test_repeat_fallback_after_widen_fails(self, tmp_path: Path) -> None:
        out = tmp_path / "gh_radar"
        out.mkdir()
        # owner/old 上次上榜 8 天前(≥7 天可回选);owner/newbie 2 天前(不可回选)
        write_json(
            out / "radar_2026-09-01.json",
            {
                "date": "2026-09-01",
                "projects": [{"repo": "owner/old"}],
                "meta": {"pick_repo": "owner/old", "runner_up_repos": []},
            },
        )
        write_json(
            out / "radar_2026-09-07.json",
            {
                "date": "2026-09-07",
                "projects": [{"repo": "owner/newbie"}],
                "meta": {"pick_repo": "owner/newbie", "runner_up_repos": []},
            },
        )
        pool = [_repo_item("owner/old", 1500), _repo_item("owner/newbie", 1200)]
        gh = QueryGh(lambda query: pool)
        radar = await build_radar(GCFG, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
        repos = [p["repo"] for p in radar["projects"]]
        assert repos == ["owner/old"]  # 只回选冷却满 7 天的
        meta = radar["meta"]
        assert meta["degraded"] is True
        assert meta["reason"] == "recent-repeat"
        assert meta["pick_repo"] == "owner/old"

    @pytest.mark.asyncio
    async def test_repeat_gap_blocks_recent_pick(self, tmp_path: Path) -> None:
        out = tmp_path / "gh_radar"
        out.mkdir()
        write_json(
            out / "radar_2026-09-07.json",
            {
                "date": "2026-09-07",
                "projects": [{"repo": "owner/hot"}],
                "meta": {"pick_repo": "owner/hot", "runner_up_repos": []},
            },
        )
        pool = [_repo_item("owner/hot", 1500)]
        gh = QueryGh(lambda query: pool)
        radar = await build_radar(GCFG, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
        # 唯一候选 2 天前刚上榜,冷却期未满 → 宁缺毋滥,当日无主推
        assert radar["projects"] == []
        assert radar["meta"]["pick_repo"] == ""
        assert radar["meta"]["widen_pass"] is True


class TestReadmeIntro:
    def test_html_and_badges_stripped(self) -> None:
        readme = (
            '<h1 align="center">LightCraft</h1>\n'
            '<p align="center"><b>Photo library and raw development in pure Rust.</b><br>'
            "Native on macOS and Windows.</p>\n"
            '<img src="https://img.shields.io/badge/pure-Rust-x">\n'
            "## Quick start\ncargo run\n"
        )
        intro = _readme_intro(readme, 400)
        assert "pure Rust" in intro
        assert "Native on macOS" in intro
        assert "shields.io" not in intro
        assert "<" not in intro
        assert "Quick start" not in intro  # 只取首个章节标题之前

    def test_empty_readme(self) -> None:
        assert _readme_intro("", 400) == ""

    def test_excerpt_matches_heading_with_space(self) -> None:
        readme = "# Hot\nintro\n\n## Quick start\ncargo run --release\n\n## License\nMIT"
        excerpt = _hands_on_excerpt(readme, 1200)
        assert "cargo run" in excerpt
        assert "MIT" not in excerpt


class TestAiAffinity:
    LEXICON: ClassVar[dict] = {"agents": ["agent"], "gateway": ["llm", "gateway"], "mcp": ["mcp"]}

    def test_two_categories_cap_one(self) -> None:
        item = {
            "full_name": "acme/agent-gateway",
            "topics": ["llm"],
            "description": "An agent orchestration and inference gateway",
        }
        assert ai_affinity(item, self.LEXICON) == 1.0

    def test_word_boundary_no_substring(self) -> None:
        item = {
            "full_name": "acme/storage-lib",
            "topics": [],
            "description": "A fast storage engine",
        }
        assert ai_affinity(item, {"rag": ["rag"]}) == 0.0

    def test_one_category_half(self) -> None:
        item = {"full_name": "acme/xyz", "topics": [], "description": "UI for agent apps"}
        assert ai_affinity(item, {"agents": ["agent"], "app": ["chatbot"]}) == 0.5

    def test_hyphen_variant_matches(self) -> None:
        item = {"full_name": "acme/xyz", "topics": [], "description": "A multi agent system"}
        assert ai_affinity(item, {"agents": ["multi-agent"]}) == 0.5

    def test_empty_lexicon_zero(self) -> None:
        item = {"full_name": "a/llm-agent", "topics": ["agent"], "description": "LLM agent"}
        assert ai_affinity(item, {}) == 0.0


class TestParseTrendingSlugs:
    def test_extract_in_order_with_cap(self) -> None:
        rss = (
            '<item><link href="https://github.com/owner/tool1"/></item>'
            "<item>github.com/mona/tool2/tree/main</item>"
            "github.com/owner/tool3/issues/1"
        )
        assert _parse_trending_slugs(rss, 2) == ["owner/tool1", "mona/tool2"]

    def test_dedupe_and_github_nav_filtered(self) -> None:
        rss = "github.com/github/docs github.com/a/b github.com/a/b"
        assert _parse_trending_slugs(rss, 5) == ["a/b"]


class TestScoreProjectV2:
    @staticmethod
    def _ctx(**kw) -> dict:
        base: dict = {
            "prev_stars": {},
            "vel_by_repo": {"a/x": 1000.0, "b/y": 100.0},
            "velocity_cap": 1500,
            "vel_rel": 0.6,
            "vel_abs": 0.4,
            "fresh_penalty": 0.85,
            "same_language_penalty": 0.05,
            "snap_gap_days": 1,
            "days_in_pool": {},
            "last_pick_language": "",
        }
        base.update(kw)
        return base

    def test_relative_velocity_and_affinity(self) -> None:
        big = _mk(repo="a/x", delta_stars=1000, affinity=1.0, language="Python")
        small = _mk(repo="b/y", delta_stars=100, affinity=0.0, language="Python")
        ctx = self._ctx()
        score_project(big, now=_NOW, pool_ctx=ctx)
        score_project(small, now=_NOW, pool_ctx=ctx)
        assert big.score > small.score
        assert big.score_parts["vel_rel"] == 1.0
        assert small.score_parts["vel_rel"] == 0.0
        assert big.score_parts["affinity"] == 1.0
        assert small.score_parts["affinity"] == 0.0

    def test_fresh_penalty_first_day(self) -> None:
        new = _mk(repo="a/x", delta_stars=1000, language="Go")
        kept = _mk(repo="b/y", delta_stars=1000, language="Go")
        ctx = self._ctx(
            vel_by_repo={"a/x": 1000.0, "b/y": 1000.0}, days_in_pool={"a/x": 0, "b/y": 5}
        )
        score_project(new, now=_NOW, pool_ctx=ctx)
        score_project(kept, now=_NOW, pool_ctx=ctx)
        assert new.score < kept.score
        assert new.score_parts["days_in_pool"] == 0

    def test_same_language_penalty(self) -> None:
        same = _mk(repo="a/x", delta_stars=1000, language="Rust")
        other = _mk(repo="b/y", delta_stars=1000, language="Python")
        ctx = self._ctx(
            vel_by_repo={"a/x": 1000.0, "b/y": 1000.0},
            days_in_pool={"a/x": 5, "b/y": 5},
            last_pick_language="Rust",
        )
        score_project(same, now=_NOW, pool_ctx=ctx)
        score_project(other, now=_NOW, pool_ctx=ctx)
        assert same.score < other.score


class TestBuildRadarV4:
    _TCFG: ClassVar[dict] = dict(
        GCFG,
        trending_rss_url="https://rss.example/primary",
        trending_rss_url_fallback="https://rss.example/fallback",
    )

    @pytest.mark.asyncio
    async def test_trending_rss_hydrates(self, tmp_path: Path) -> None:
        rss = {
            "https://rss.example/primary": (
                "<item>github.com/owner/tr1</item><item>github.com/owner/tr2</item>"
            ),
            "https://rss.example/fallback": "<item>github.com/owner/third</item>",
        }
        repos = {
            "owner/tr1": _repo_item("owner/tr1", 2000, topics=["llm", "agent"]),
            "owner/tr2": _repo_item("owner/tr2", 900, topics=["chatbot"]),
        }
        gh = FakeGhClient([], repos=repos, rss=rss)
        radar = await build_radar(self._TCFG, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
        picks = [p["repo"] for p in radar["projects"]]
        assert "owner/tr1" in picks
        assert radar["meta"]["pool_by_source"] == {"trending": 2}
        # 主源一次成功(有两个 slug 入池)即不再打备源;slug 逐个 GET 验活
        assert gh.rss_calls == ["https://rss.example/primary"]
        assert gh.repo_calls == ["owner/tr1", "owner/tr2"]

    @pytest.mark.asyncio
    async def test_trending_primary_empty_falls_back(self, tmp_path: Path) -> None:
        rss = {
            "https://rss.example/primary": "<rss>no links here</rss>",
            "https://rss.example/fallback": "<item>github.com/owner/tr2</item>",
        }
        repos = {"owner/tr2": _repo_item("owner/tr2", 900, topics=["chatbot"])}
        gh = FakeGhClient([], repos=repos, rss=rss)
        radar = await build_radar(self._TCFG, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
        assert [p["repo"] for p in radar["projects"]] == ["owner/tr2"]
        assert gh.rss_calls == [
            "https://rss.example/primary",
            "https://rss.example/fallback",
        ]
        assert radar["meta"]["pool_by_source"] == {"trending": 1}

    @pytest.mark.asyncio
    async def test_topic_rotation_daily(self, tmp_path: Path) -> None:
        tcfg = dict(
            GCFG, ai_topics=["a-topic", "b-topic"], topic_min_stars=1000, topic_rotation_days=90
        )
        items = [_repo_item("owner/agentx", 3000, topics=["agents"])]
        gh = FakeGhClient(items)
        await build_radar(tcfg, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
        await build_radar(tcfg, "2026-09-10", tmp_path, [], client=gh, now=_NOW)
        topic_queries = [q for q in gh.queries if "topic:" in q]
        assert len(topic_queries) == 2
        assert topic_queries[0] != topic_queries[1]
        # 各自的剧集日期 −topic_rotation_days(90 天)作为创建窗口
        assert "created:>2026-06-11" in topic_queries[0]  # 2026-09-09 − 90d
        assert "created:>2026-06-12" in topic_queries[1]  # 2026-09-10 − 90d
        for q in topic_queries:
            assert "stars:>=1000" in q
        topics_used = {q.split()[0].split(":")[1] for q in topic_queries}
        assert topics_used == {"a-topic", "b-topic"}

    @pytest.mark.asyncio
    async def test_news_repos_deduped_and_capped(self, tmp_path: Path) -> None:
        repos = {
            "owner/plain": _repo_item("owner/plain", 800),
            "owner/aitool": _repo_item("owner/aitool", 1200, topics=["agent"]),
            "owner/spam": _repo_item("owner/spam", 900, topics=["llm"]),
        }
        gh = FakeGhClient([], repos=repos)
        gcfg = dict(GCFG, news_repo_top=2)
        radar = await build_radar(
            gcfg,
            "2026-09-09",
            tmp_path,
            [],
            client=gh,
            now=_NOW,
            news_repos=["owner/plain", "owner/aitool", "owner/spam", "owner/spam"],
        )
        assert gh.repo_calls == ["owner/plain", "owner/aitool"]  # 去重 + cap
        assert radar["meta"]["pool_by_source"].get("news") == 2

    @pytest.mark.asyncio
    async def test_strict_gate_filters_non_ai(self, tmp_path: Path) -> None:
        gcfg = dict(GCFG, direction_gate="strict", ai_lexicon={"gateway": ["llm"]})
        items = [
            _repo_item("owner/plain", 1500, topics=["react"]),
            _repo_item("owner/aitool", 800, topics=["llm"]),
        ]
        gh = FakeGhClient(items)
        radar = await build_radar(gcfg, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
        assert [p["repo"] for p in radar["projects"]] == ["owner/aitool"]
        meta = radar["meta"]
        assert meta["direction_fallback"] is False
        assert meta["degraded"] is False

    @pytest.mark.asyncio
    async def test_strict_gate_relief_when_no_ai(self, tmp_path: Path) -> None:
        gcfg = dict(GCFG, direction_gate="strict", ai_lexicon={"gateway": ["llm"]})
        gh = FakeGhClient([_repo_item("owner/plain", 1500, topics=["react"])])
        radar = await build_radar(gcfg, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
        assert [p["repo"] for p in radar["projects"]] == ["owner/plain"]
        meta = radar["meta"]
        assert meta["direction_fallback"] is True
        assert meta["degraded"] is True
        assert meta["reason"] == "off-direction"

    @pytest.mark.asyncio
    async def test_yesterday_pick_observed(self, tmp_path: Path) -> None:
        out = tmp_path / "gh_radar"
        out.mkdir()
        write_json(
            out / "radar_2026-09-08.json",
            {
                "date": "2026-09-08",
                "projects": [{"repo": "owner/prev", "language": "Swift"}],
                "meta": {"pick_repo": "owner/prev", "runner_up_repos": []},
            },
        )
        gh = FakeGhClient(
            [_repo_item("owner/hot", 1500)],
            repos={"owner/prev": _repo_item("owner/prev", 2600, topics=["swift-ui"])},
        )
        radar = await build_radar(GCFG, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
        meta = radar["meta"]
        assert meta["yesterday_pick"] == "owner/prev"
        # 无历史快照基线 → 增速不可得,静默 None
        assert meta["yesterday_pick_delta"] is None
