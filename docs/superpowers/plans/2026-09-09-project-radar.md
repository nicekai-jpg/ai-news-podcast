# 项目雷达 (Project Radar) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在每日流水线中新增独立「项目轨」——每天扫描 GitHub 发现正在起势、可上手的开源项目,产出播客固定栏目「项目雷达」(3 快讯 + 1 深评)与日报雷达章节;与新闻管线数据完全隔离,失败不致命,数字零幻觉。

**Architecture:** 双轨制。新闻轨(fetch→process→打分)不变;新增 `pipeline/gh_client.py`(GitHub REST 轻客户端)与 `pipeline/gh_radar.py`(候选→硬过滤→快照差分→评分→选优)。runner 在 stage1 内挂载雷达(try/except 包裹,失败仅发 StageFailed 事件),结果以 `brief["radar"]` 随 brief 持久化,writer/report 从 brief 自然读到。播客栏目文案由 LLM 生成但所有数字由代码注入素材;日报雷达章节完全由代码生成。

**Tech Stack:** Python 3.11 / httpx(已有依赖)/ GitHub REST API(免费额度,`GITHUB_TOKEN` 可选)/ pytest + pytest-asyncio。

**关键约束(设计已定,不可偏离):**
- 雷达是"可失败环节":任何异常不得阻断正片。
- LLM 禁止自报数字:stars/增速等由代码写进素材文本。
- 同步纪律:快照 `data/gh_snapshots/snap_{date}.json` 必须随 stage1 提交到 main(次日 CI 才能算增速)。
- 命名:内部 `gh_radar`,中文显示名「项目雷达」。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `src/ai_news_podcast/config/models.py` | 修改 | 新增 `GhRadarConfig` + `AppConfig.gh_radar` |
| `config/config.yaml` | 修改 | 新增 `gh_radar:` 配置块 |
| `src/ai_news_podcast/pipeline/gh_client.py` | 新建 | GitHub REST 轻客户端(search + readme) |
| `src/ai_news_podcast/pipeline/gh_radar.py` | 新建 | 雷达编排:过滤/评分/快照/产物 |
| `src/ai_news_podcast/pipeline/runner.py` | 修改 | stage1 挂载雷达,注入 `brief["radar"]` |
| `src/ai_news_podcast/pipeline/material.py` | 修改 | 新增 `build_radar_text()` |
| `src/ai_news_podcast/prompts.py` | 修改 | editor/writer 模板增加雷达段 |
| `src/ai_news_podcast/pipeline/podcastwriter.py` | 修改 | 接线:素材+prompt 传参 |
| `src/ai_news_podcast/cli/podcast_report.py` | 修改 | 代码生成雷达章节 |
| `.github/workflows/daily.yml` | 修改 | stage1 传 `GITHUB_TOKEN` + 提交雷达/快照目录 |
| `AGENTS.md` | 修改 | 数据流与纪律说明 |
| `tests/test_gh_radar.py` | 新建 | 配置/客户端/雷达核心测试 |
| `tests/test_runner.py`、`tests/test_material.py`、`tests/test_podcastwriter_prompt.py`、`tests/test_podcastwriter.py`、`tests/test_daily_report.py` | 修改 | 各集成点测试 |

层级合规:全部新代码位于 `pipeline` 层(或 cli 层),只依赖同层与 `utils`,符合 `.importlinter` 契约。

---

### Task 1: 配置层 — `GhRadarConfig` 与 config.yaml

**Files:**
- Modify: `src/ai_news_podcast/config/models.py`
- Modify: `config/config.yaml`
- Test: `tests/test_gh_radar.py`(新建)

- [ ] **Step 1: 写失败测试**

新建 `tests/test_gh_radar.py`:

```python
"""Tests for ai_news_podcast.pipeline.gh_radar 与相关配置。"""

from __future__ import annotations

from ai_news_podcast.config.models import AppConfig


class TestGhRadarConfig:
    def test_defaults(self) -> None:
        cfg = AppConfig.from_dict({})
        assert cfg.gh_radar.enabled is True
        assert cfg.gh_radar.min_stars == 500
        assert cfg.gh_radar.quick_count == 3
        assert cfg.gh_radar.deep_dive_count == 1
        assert cfg.gh_radar.snapshot_dir == "gh_snapshots"

    def test_overrides(self) -> None:
        cfg = AppConfig.from_dict(
            {"gh_radar": {"min_stars": 100, "preferred_topics": ["rag"]}}
        )
        assert cfg.gh_radar.min_stars == 100
        assert cfg.gh_radar.preferred_topics == ["rag"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_gh_radar.py -v`
Expected: FAIL —— `AttributeError: 'AppConfig' object has no attribute 'gh_radar'`(dataclass 无该字段)

- [ ] **Step 3: 实现 `GhRadarConfig`**

`src/ai_news_podcast/config/models.py` 中,在 `class BuildConfig` 之后、`class EntitiesConfig` 之前插入:

```python
@dataclass(frozen=True)
class GhRadarConfig:
    """GitHub 项目雷达(独立项目轨)配置。"""

    enabled: bool = True
    min_stars: int = 500
    created_window_days: int = 30
    recent_push_days: int = 21
    top_n: int = 30
    quick_count: int = 3
    deep_dive_count: int = 1
    readme_probe_limit: int = 12
    excluded_name_patterns: list[str] = field(
        default_factory=lambda: [
            "awesome",
            "tutorial",
            "roadmap",
            "interview",
            "books",
            "cheatsheet",
            "list",
        ]
    )
    preferred_topics: list[str] = field(
        default_factory=lambda: [
            "llm",
            "ai",
            "agents",
            "agent",
            "rag",
            "inference",
            "mcp",
            "transformers",
        ]
    )
    snapshot_dir: str = "gh_snapshots"
    output_dir: str = "gh_radar"
```

`AppConfig` 增加字段(放在 `entities` 之前):

```python
    gh_radar: GhRadarConfig = field(default_factory=GhRadarConfig)
```

`from_dict` 中增加一行(放在 `entities=_build_entities(data),` 之前):

```python
            gh_radar=_build_gh_radar(data),
```

文件末尾追加 builder:

```python
def _build_gh_radar(data: dict[str, Any]) -> GhRadarConfig:
    g = data.get("gh_radar", {})
    return GhRadarConfig(
        **{k: v for k, v in g.items() if k in GhRadarConfig.__dataclass_fields__}
    )
```

- [ ] **Step 4: config.yaml 增加配置块**

在 `config/config.yaml` 的 `build:` 块之后、`entities:` 之前插入:

```yaml
gh_radar:
  enabled: true
  min_stars: 500                  # 新星榜最低星数门槛
  created_window_days: 30         # 新星榜创建时间窗口(天)
  recent_push_days: 21            # 最近一次 commit 必须在此窗口内(项目还活着)
  top_n: 30                       # 搜索拉取的候选数量
  quick_count: 3                  # 每期快讯项目数
  deep_dive_count: 1              # 每期深评项目数(取最高分)
  readme_probe_limit: 12          # 探测 README 的候选上限(控 API 用量)
  excluded_name_patterns:         # 命中即排除(资料合集类不可上手)
    - "awesome"
    - "tutorial"
    - "roadmap"
    - "interview"
    - "books"
    - "cheatsheet"
    - "list"
  preferred_topics:               # AI 工具链优先:命中主题加分
    - "llm"
    - "ai"
    - "agents"
    - "agent"
    - "rag"
    - "inference"
    - "mcp"
    - "transformers"
  snapshot_dir: "gh_snapshots"    # 相对 data/ 目录
  output_dir: "gh_radar"          # 相对 data/ 目录
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_gh_radar.py -v`
Expected: PASS(2 passed)

- [ ] **Step 6: Commit**

```bash
uv run ruff format src/ tests/ && uv run ruff check src/ tests/ && git add src/ai_news_podcast/config/models.py config/config.yaml tests/test_gh_radar.py && git commit -m "feat(radar): add gh_radar config block and typed settings"
```

---

### Task 2: `pipeline/gh_client.py` — GitHub REST 轻客户端

**Files:**
- Create: `src/ai_news_podcast/pipeline/gh_client.py`
- Test: `tests/test_gh_radar.py`(追加)

- [ ] **Step 1: 写失败测试(追加到 tests/test_gh_radar.py)**

```python
import httpx
import pytest

from ai_news_podcast.pipeline.gh_client import GhClient


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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_gh_radar.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'ai_news_podcast.pipeline.gh_client'`

- [ ] **Step 3: 实现客户端**

新建 `src/ai_news_podcast/pipeline/gh_client.py`:

```python
"""GitHub REST API 轻量客户端(项目雷达专用)。

只封装雷达需要的两个端点:仓库搜索与 README 读取。
未显式传 token 时读取 GITHUB_TOKEN 环境变量;都缺失则匿名访问
(限流更紧,但每日一次的扫描量足够)。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"


class GhClient:
    """异步 GitHub REST 客户端。测试时传入 MockTransport 构造的 AsyncClient。"""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        token: str | None = None,
        sleep_seconds: float = 1.0,
    ) -> None:
        self._client = client
        self._sleep_seconds = sleep_seconds
        headers = {"Accept": "application/vnd.github+json"}
        resolved = token if token is not None else os.environ.get("GITHUB_TOKEN") or ""
        if resolved:
            headers["Authorization"] = f"Bearer {resolved}"
        self._headers = headers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search_repos(
        self, query: str, *, sort: str = "stars", per_page: int = 30
    ) -> list[dict[str, Any]]:
        params = {"q": query, "sort": sort, "order": "desc", "per_page": per_page}
        resp = await self._client.get(
            f"{_GITHUB_API}/search/repositories", params=params, headers=self._headers
        )
        resp.raise_for_status()
        await asyncio.sleep(self._sleep_seconds)  # search API 限流:匿名 10 次/分钟
        return list(resp.json().get("items", []))

    async def fetch_readme_text(self, repo: str) -> str:
        """返回 README 原文(截断),404 视为无 README 返回空串。"""
        headers = dict(self._headers)
        headers["Accept"] = "application/vnd.github.raw+json"
        resp = await self._client.get(
            f"{_GITHUB_API}/repos/{repo}/readme", headers=headers
        )
        if resp.status_code == 404:
            return ""
        resp.raise_for_status()
        return resp.text[:6000]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_gh_radar.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/ tests/ && uv run ruff check src/ tests/ && git add src/ai_news_podcast/pipeline/gh_client.py tests/test_gh_radar.py && git commit -m "feat(radar): add minimal GitHub REST client for radar"
```

---

### Task 3: `pipeline/gh_radar.py` — 雷达核心(过滤/评分/快照/编排)

**Files:**
- Create: `src/ai_news_podcast/pipeline/gh_radar.py`
- Test: `tests/test_gh_radar.py`(追加)

- [ ] **Step 1: 写失败测试(追加到 tests/test_gh_radar.py)**

```python
from datetime import UTC, datetime
from pathlib import Path

from ai_news_podcast.pipeline.gh_radar import (
    build_radar,
    count_news_mentions,
    score_project,
)
from ai_news_podcast.utils import write_json


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


def _repo_item(full_name: str, stars: int, *, pushed: str = "2026-09-08T00:00:00Z",
               license_id: str = "MIT", topics: list[str] | None = None,
               desc: str = "A real tool") -> dict:
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


GCFG = {"enabled": True, "min_stars": 500, "created_window_days": 30,
        "recent_push_days": 21, "top_n": 30, "quick_count": 3, "deep_dive_count": 1,
        "readme_probe_limit": 12, "excluded_name_patterns": ["awesome", "list"],
        "preferred_topics": ["llm"], "snapshot_dir": "gh_snapshots",
        "output_dir": "gh_radar"}


class TestCountNewsMentions:
    def test_counts_title_hits(self) -> None:
        titles = ["BigLang released", "biglang v2 out now", "unrelated news"]
        assert count_news_mentions("acme/biglang", titles) == 2

    def test_short_tokens_ignored(self) -> None:
        assert count_news_mentions("acme/gpt", ["gpt is everywhere"]) == 0


class TestScoreProject:
    def test_full_score_computation(self) -> None:
        p = _mk(delta=3000, install=True, license_id="MIT", pushed="2026-09-08T00:00:00Z",
                mentions=2, topics=["llm"])
        score_project(p, now=_NOW, preferred_topics=["llm"])
        # velocity=1.0; hands_on=0.4+0.3+0.3=1.0; cross=2/3; bonus=0.1
        assert p.score_parts["velocity"] == 1.0
        assert p.score_parts["hands_on"] == 1.0
        assert p.score > 0.9


def _mk(**kw) -> "RadarProject":
    from ai_news_podcast.pipeline.gh_radar import RadarProject

    defaults = dict(
        repo="owner/repo", url="https://github.com/owner/repo", description="d",
        language="Python", topics=[], stars=1000, delta_stars=None, is_new=True,
        created_at="2026-08-15T00:00:00Z", pushed_at="2026-09-08T00:00:00Z",
        license="MIT", has_install_docs=False, mentions=0, score=0.0, score_parts={},
    )
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
        assert "owner/awesome-ai" not in repos      # 合集被排除
        assert "owner/stale" not in repos           # 42 天无 commit
        assert "owner/nolicense" not in repos       # 无许可证
        hot = next(p for p in radar["projects"] if p["repo"] == "owner/hot")
        assert hot["delta_stars"] == 1000           # 1500 - 500
        assert hot["has_install_docs"] is True
        assert radar["meta"]["deep_dive_repo"] == radar["projects"][0]["repo"]
        assert (tmp_path / "gh_radar" / "radar_2026-09-09.json").exists()
        snap = (snap_dir / "snap_2026-09-09.json")
        assert snap.exists()                        # 当日快照已写

    @pytest.mark.asyncio
    async def test_first_day_no_delta(self, tmp_path: Path) -> None:
        gh = FakeGhClient([_repo_item("owner/fresh", 800)])
        radar = await build_radar(GCFG, "2026-09-09", tmp_path, [], client=gh, now=_NOW)
        assert radar["projects"][0]["delta_stars"] is None
        assert radar["projects"][0]["is_new"] is True

    @pytest.mark.asyncio
    async def test_disabled_returns_degraded(self, tmp_path: Path) -> None:
        gh = FakeGhClient([])
        radar = await build_radar({"enabled": False}, "2026-09-09", tmp_path, [],
                                  client=gh, now=_NOW)
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_gh_radar.py -v`
Expected: FAIL —— `ModuleNotFoundError: ... gh_radar`

- [ ] **Step 3: 实现 gh_radar.py**

新建 `src/ai_news_podcast/pipeline/gh_radar.py`:

```python
"""项目雷达:发现正在起势、可上手的开源项目。

双轨制中的「项目轨」,与新闻管线数据完全隔离:
产物只进入播客固定栏目「项目雷达」与日报章节两个出口。
数字全部来自 GitHub API 实测并由代码写入,LLM 无权自报热度。
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_news_podcast.pipeline.gh_client import GhClient
from ai_news_podcast.utils import read_json, write_json

logger = logging.getLogger(__name__)

_INSTALL_HINTS = (
    "pip install",
    "npm install",
    "cargo install",
    "brew install",
    "docker run",
    "quickstart",
    "getting started",
    "安装",
    "快速开始",
)


@dataclass
class RadarProject:
    """单个候选项目的雷达视图。"""

    repo: str
    url: str
    description: str
    language: str
    topics: list[str]
    stars: int
    delta_stars: int | None
    is_new: bool
    created_at: str
    pushed_at: str
    license: str
    has_install_docs: bool
    mentions: int
    score: float = 0.0
    score_parts: dict[str, float] = field(default_factory=dict)


def _tokenize_repo_name(full_name: str) -> list[str]:
    name = full_name.split("/", 1)[-1].lower()
    return [p for p in re.split(r"[-_.]", name) if len(p) >= 4]


def count_news_mentions(full_name: str, news_titles: list[str]) -> int:
    """统计当日新闻标题对项目的提及次数(短 token 忽略以免误伤)。"""
    tokens = _tokenize_repo_name(full_name)
    if not tokens:
        return 0
    return sum(1 for title in news_titles if any(tok in title.lower() for tok in tokens))


def _is_excluded(full_name: str, topics: list[str], patterns: list[str]) -> bool:
    hay = (full_name + " " + " ".join(topics)).lower()
    return any(p.lower() in hay for p in patterns)


def _has_install_docs(readme: str) -> bool:
    if not readme:
        return False
    low = readme.lower()
    return any(hint in low for hint in _INSTALL_HINTS)


def _hard_filter(item: dict[str, Any], gcfg: dict[str, Any], now: datetime) -> bool:
    """可上手硬过滤:合集/无描述/无许可证/已停更的直接出局。"""
    full_name = str(item.get("full_name", ""))
    topics = [str(t) for t in (item.get("topics") or [])]
    if _is_excluded(full_name, topics, list(gcfg.get("excluded_name_patterns", []))):
        return False
    if not str(item.get("description") or "").strip():
        return False
    license_info = item.get("license") or {}
    spdx = str(license_info.get("spdx_id") or "")
    if not spdx or spdx == "NOASSERTION":
        return False
    pushed_at = str(item.get("pushed_at") or "")
    if not pushed_at:
        return False
    pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
    if (now - pushed).days > int(gcfg.get("recent_push_days", 21)):
        return False
    return True


def _to_project(
    item: dict[str, Any], prev_stars: dict[str, int], news_titles: list[str]
) -> RadarProject:
    full_name = str(item["full_name"])
    stars = int(item.get("stargazers_count", 0))
    prev = prev_stars.get(full_name)
    license_info = item.get("license") or {}
    return RadarProject(
        repo=full_name,
        url=str(item.get("html_url") or f"https://github.com/{full_name}"),
        description=str(item.get("description") or "").strip(),
        language=str(item.get("language") or ""),
        topics=[str(t) for t in (item.get("topics") or [])],
        stars=stars,
        delta_stars=stars - prev if prev is not None else None,
        is_new=prev is None,
        created_at=str(item.get("created_at") or ""),
        pushed_at=str(item.get("pushed_at") or ""),
        license=str(license_info.get("spdx_id") or ""),
        has_install_docs=False,
        mentions=count_news_mentions(full_name, news_titles),
    )


def score_project(
    p: RadarProject, *, now: datetime, preferred_topics: list[str]
) -> None:
    """确定性评分:0.5 增速 + 0.3 可上手 + 0.2 交叉热度 + 0.1 AI 主题加分。"""
    if p.delta_stars is not None:
        velocity = min(p.delta_stars, 1500) / 1500
    else:
        velocity = min(p.stars, 10000) / 10000 * 0.5

    hands_on = 0.4 * (1.0 if p.has_install_docs else 0.0)
    if p.license:
        hands_on += 0.3
    pushed = datetime.fromisoformat(p.pushed_at.replace("Z", "+00:00"))
    days_since_push = (now - pushed).days
    if days_since_push <= 7:
        hands_on += 0.3
    elif days_since_push <= 21:
        hands_on += 0.15

    cross = min(p.mentions, 3) / 3
    topic_set = {t.lower() for t in p.topics}
    ai_bonus = 0.1 if any(t in topic_set for t in preferred_topics) else 0.0

    p.score = round(min(1.0, 0.5 * velocity + 0.3 * hands_on + 0.2 * cross + ai_bonus), 4)
    p.score_parts = {
        "velocity": round(velocity, 4),
        "hands_on": round(hands_on, 4),
        "cross_heat": round(cross, 4),
        "ai_bonus": ai_bonus,
    }


def _load_previous_snapshot(snapshot_dir: Path, today: str) -> dict[str, int]:
    """取今天之前最近一份快照的星数表(当天重跑不会拿到自己的快照)。"""
    best: dict[str, int] = {}
    best_date = ""
    if not snapshot_dir.exists():
        return best
    for f in sorted(snapshot_dir.glob("snap_*.json")):
        try:
            data = read_json(f)
        except Exception:  # noqa: BLE001 — 单份快照损坏不影响其余
            continue
        if not isinstance(data, dict):
            continue
        d = str(data.get("date", ""))
        if d < today and d > best_date:
            best_date = d
            stars = data.get("stars", {})
            best = {str(k): int(v) for k, v in stars.items()} if isinstance(stars, dict) else {}
    return best


async def build_radar(
    gcfg: dict[str, Any],
    date_str: str,
    data_dir: Path,
    news_titles: list[str],
    *,
    client: GhClient | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """构建当日项目雷达。硬失败直接抛异常,由调用方决定降级。"""
    now = now or datetime.now(tz=UTC)
    if not gcfg.get("enabled", True):
        return {"date": date_str, "projects": [], "meta": {"degraded": True, "reason": "disabled"}}

    top_n = int(gcfg.get("top_n", 30))
    min_stars = int(gcfg.get("min_stars", 500))
    window_days = int(gcfg.get("created_window_days", 30))
    cutoff = (now - timedelta(days=window_days)).strftime("%Y-%m-%d")
    query = f"created:>{cutoff} stars:>={min_stars}"

    owns_client = client is None
    if client is None:
        import httpx

        client = GhClient(httpx.AsyncClient(timeout=30.0))

    try:
        items = await client.search_repos(query, per_page=top_n)
        candidates = [it for it in items if _hard_filter(it, gcfg, now)]
        candidates.sort(key=lambda it: int(it.get("stargazers_count", 0)), reverse=True)
        candidates = candidates[: int(gcfg.get("readme_probe_limit", 12))]

        snapshot_dir = data_dir / str(gcfg.get("snapshot_dir", "gh_snapshots"))
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            snapshot_dir / f"snap_{date_str}.json",
            {
                "date": date_str,
                "stars": {
                    str(it.get("full_name")): int(it.get("stargazers_count", 0))
                    for it in candidates
                },
            },
        )

        prev_stars = _load_previous_snapshot(snapshot_dir, date_str)
        preferred = [str(t).lower() for t in gcfg.get("preferred_topics", [])]

        projects: list[RadarProject] = []
        for item in candidates:
            p = _to_project(item, prev_stars, news_titles)
            try:
                p.has_install_docs = _has_install_docs(await client.fetch_readme_text(p.repo))
            except Exception as e:  # noqa: BLE001 — 单仓库 README 拉取失败不致命
                logger.warning("README probe failed for %s: %s", p.repo, e)
            score_project(p, now=now, preferred_topics=preferred)
            projects.append(p)

        projects.sort(key=lambda x: x.score, reverse=True)
        quick_count = int(gcfg.get("quick_count", 3))
        deep_count = int(gcfg.get("deep_dive_count", 1))
        projects = projects[: quick_count + deep_count]
        deep_dive_repo = projects[0].repo if projects else ""

        output_dir = data_dir / str(gcfg.get("output_dir", "gh_radar"))
        output_dir.mkdir(parents=True, exist_ok=True)
        radar = {
            "date": date_str,
            "generated_at": now.isoformat(),
            "projects": [asdict(p) for p in projects],
            "meta": {
                "quick_repos": [p.repo for p in projects[deep_count:]],
                "deep_dive_repo": deep_dive_repo,
                "degraded": False,
                "reason": "",
                "candidates": len(candidates),
            },
        }
        write_json(output_dir / f"radar_{date_str}.json", radar)
        return radar
    finally:
        if owns_client:
            await client.aclose()
```

注意 `meta.quick_repos` 用 `projects[deep_count:]`:深评取第一名,快讯取其后 N 个。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_gh_radar.py -v`
Expected: PASS(全部通过)

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/ tests/ && uv run ruff check src/ tests/ && git add src/ai_news_podcast/pipeline/gh_radar.py tests/test_gh_radar.py && git commit -m "feat(radar): implement project radar scoring, snapshot and builder"
```

---

### Task 4: runner 挂载 — `brief["radar"]` 与可失败纪律

**Files:**
- Modify: `src/ai_news_podcast/pipeline/runner.py`
- Test: `tests/test_runner.py`(追加)

- [ ] **Step 1: 写失败测试(追加到 tests/test_runner.py)**

```python
@pytest.mark.asyncio
async def test_run_pipeline_attaches_radar(tmp_path: Path, raw_item_factory) -> None:
    from unittest.mock import AsyncMock, patch

    from ai_news_podcast.pipeline.runner import run_pipeline

    radar = {
        "date": "2026-06-03",
        "projects": [{"repo": "owner/hot", "stars": 1500, "delta_stars": 1000}],
        "meta": {"deep_dive_repo": "owner/hot", "quick_repos": [], "degraded": False},
    }
    with (
        patch("ai_news_podcast.pipeline.runner.fetch_all", new_callable=AsyncMock) as mock_fetch,
        patch("ai_news_podcast.pipeline.runner.process") as mock_process,
        patch(
            "ai_news_podcast.pipeline.runner.build_radar", new_callable=AsyncMock
        ) as mock_radar,
    ):
        mock_fetch.return_value = [raw_item_factory()]
        mock_process.return_value = {"stories": []}
        mock_radar.return_value = radar

        brief = await run_pipeline(
            cfg={}, sources=[], date_str="2026-06-03", data_dir=tmp_path, force_refresh=True
        )

        assert brief["radar"]["meta"]["deep_dive_repo"] == "owner/hot"
        saved = (tmp_path / "briefs" / "brief_2026-06-03.json").read_text(encoding="utf-8")
        assert '"deep_dive_repo"' in saved


@pytest.mark.asyncio
async def test_run_pipeline_survives_radar_failure(tmp_path: Path, raw_item_factory) -> None:
    from unittest.mock import AsyncMock, patch

    from ai_news_podcast.pipeline.runner import run_pipeline

    with (
        patch("ai_news_podcast.pipeline.runner.fetch_all", new_callable=AsyncMock) as mock_fetch,
        patch("ai_news_podcast.pipeline.runner.process") as mock_process,
        patch(
            "ai_news_podcast.pipeline.runner.build_radar",
            new_callable=AsyncMock,
            side_effect=RuntimeError("github down"),
        ),
    ):
        mock_fetch.return_value = [raw_item_factory()]
        mock_process.return_value = {"stories": []}

        brief = await run_pipeline(
            cfg={}, sources=[], date_str="2026-06-03", data_dir=tmp_path, force_refresh=True
        )

        assert "radar" not in brief  # 雷达失败,正片照常
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_runner.py -k radar -v`
Expected: FAIL —— `AttributeError: <module ...> does not have the attribute 'build_radar'`

- [ ] **Step 3: 实现 runner 挂载**

`src/ai_news_podcast/pipeline/runner.py`:

imports 区增加:

```python
from ai_news_podcast.pipeline.gh_radar import build_radar
```

在 `brief.setdefault("metadata", {})["dedup_details"] = dedup_details` 之后、`data_dir.mkdir(...)` 之前插入:

```python
    # ── Stage 2b: 项目雷达(独立项目轨,可失败不致命) ──────────────────────
    gh_radar_cfg = cfg_dict.get("gh_radar", {})
    if gh_radar_cfg.get("enabled", True):
        event_bus.emit(
            StageStarted(stage="gh_radar", episode_id=date_str, timestamp=datetime.now(tz=UTC))
        )
        try:
            news_titles = [item.title for item in raw_items]
            brief["radar"] = await build_radar(gh_radar_cfg, date_str, data_dir, news_titles)
            event_bus.emit(
                StageCompleted(
                    stage="gh_radar",
                    episode_id=date_str,
                    duration_ms=0,
                    result={"projects": len(brief["radar"].get("projects", []))},
                )
            )
        except Exception as e:  # noqa: BLE001 — 雷达必须可失败不致命
            log.warning("项目雷达失败,当期正片不含雷达栏目: %s", e)
            event_bus.emit(
                StageFailed(
                    stage="gh_radar",
                    episode_id=date_str,
                    error=str(e),
                    timestamp=datetime.now(tz=UTC),
                )
            )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_runner.py -v`
Expected: PASS(原有用例 + 2 个新用例;原有用例 cfg 不含 gh_radar → `enabled` 默认 True → build_radar 会被调用?**注意**:原用例没有 mock build_radar,会真发网络请求!)

**修正(必须做)**:原有两个 `test_run_pipeline_semantic_dedup*` 用例的 cfg dict 无 `gh_radar` 键,会触发真实调用。给原用例的 cfg 增加 `"gh_radar": {"enabled": False}`,并在 degraded 测试中同样使用 `cfg={"gh_radar": {"enabled": False}, ...}` 时无需 mock——上面 Step 1 的两个新用例 cfg 用 `{}`(enabled 默认开)所以必须 mock,保持原样;原用例加 `"gh_radar": {"enabled": False}` 即可关闭。

Run: `uv run pytest tests/test_runner.py -v`
Expected: PASS(全部)

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/ tests/ && uv run ruff check src/ tests/ && git add src/ai_news_podcast/pipeline/runner.py tests/test_runner.py && git commit -m "feat(radar): attach radar to stage1 brief with fail-safe integration"
```

---

### Task 5: `material.build_radar_text` — 雷达素材文本

**Files:**
- Modify: `src/ai_news_podcast/pipeline/material.py`
- Test: `tests/test_material.py`(追加)

- [ ] **Step 1: 写失败测试(追加到 tests/test_material.py)**

```python
from ai_news_podcast.pipeline.material import build_radar_text


class TestBuildRadarText:
    def test_empty_radar_returns_empty(self) -> None:
        assert build_radar_text(None) == ""
        assert build_radar_text({"projects": []}) == ""

    def test_formats_projects_with_numbers(self) -> None:
        radar = {
            "projects": [
                {"repo": "owner/hot", "url": "https://github.com/owner/hot", "stars": 1500,
                 "delta_stars": 1000, "language": "Python", "license": "MIT",
                 "description": "Fast LLM harness"},
                {"repo": "owner/next", "url": "https://github.com/owner/next", "stars": 800,
                 "delta_stars": None, "language": "Rust", "license": "Apache-2.0",
                 "description": "Quick agent runtime"},
            ],
            "meta": {"deep_dive_repo": "owner/hot"},
        }
        text = build_radar_text(radar)
        assert "[深评]" in text and "[快讯]" in text
        assert "owner/hot" in text and "⭐1500" in text and "较昨日 +1000" in text
        assert "首日无对比数据" in text
        assert "禁止编造" in text
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_material.py -k radar -v`
Expected: FAIL —— `ImportError: cannot import name 'build_radar_text'`

- [ ] **Step 3: 实现**

`src/ai_news_podcast/pipeline/material.py` 末尾追加:

```python
def build_radar_text(radar: dict[str, Any] | None) -> str:
    """把项目雷达结果格式化为固定栏目素材文本(空雷达返回空串)。

    数字与链接全部来自代码注入的结构化数据,LLM 不得增删。
    """
    if not radar or not radar.get("projects"):
        return ""
    meta = radar.get("meta", {})
    deep_repo = meta.get("deep_dive_repo")
    lines = ["以下是「项目雷达」栏目的结构化素材(与新闻无关,单独成栏)。"]
    for p in radar.get("projects", []):
        role = "深评" if p.get("repo") == deep_repo else "快讯"
        delta = p.get("delta_stars")
        delta_str = f"较昨日 +{delta}" if isinstance(delta, int) else "首日无对比数据"
        lines.append(
            f"- [{role}] {p.get('repo')}(⭐{p.get('stars')},{delta_str},"
            f"语言:{p.get('language') or '未知'},许可证:{p.get('license') or '无'}):"
            f"{str(p.get('description') or '').strip()}"
        )
        lines.append(f"  链接:{p.get('url')}")
    lines.append(
        "使用规则:所有仓库名与数字必须原样引用,禁止编造或修改;"
        "禁止把项目与新闻混在同一栏目。"
    )
    return "\n".join(lines)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_material.py -v`
Expected: PASS(原有用例 + 新用例)

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/ tests/ && uv run ruff check src/ tests/ && git add src/ai_news_podcast/pipeline/material.py tests/test_material.py && git commit -m "feat(radar): format radar material text for prompts"
```

---

### Task 6: prompts.py — editor/writer 模板雷达段

**Files:**
- Modify: `src/ai_news_podcast/prompts.py`
- Test: `tests/test_podcastwriter_prompt.py`(追加)

- [ ] **Step 1: 写失败测试(追加到 tests/test_podcastwriter_prompt.py)**

```python
class TestRadarPromptSections:
    def test_editor_prompt_without_radar_omits_section(self) -> None:
        prompt = build_editor_prompt("素材", datetime(2026, 9, 9))
        assert "项目雷达素材" not in prompt

    def test_editor_prompt_with_radar(self) -> None:
        prompt = build_editor_prompt("素材", datetime(2026, 9, 9), radar_material="雷达素材")
        assert "项目雷达素材" in prompt
        assert "雷达素材" in prompt
        assert "3 个快讯项目" in prompt and "1 个深评项目" in prompt

    def test_writer_prompt_radar_rules_toggle(self) -> None:
        base = build_writer_prompt("大纲", datetime(2026, 9, 9), "AI 每日先锋", {})
        assert "项目雷达栏目规范" not in base
        with_radar = build_writer_prompt(
            "大纲", datetime(2026, 9, 9), "AI 每日先锋", {}, has_radar=True
        )
        assert "项目雷达栏目规范" in with_radar
        assert "400-600 字" in with_radar
```

(注意 import:复用该文件已有的 `build_editor_prompt`/`build_writer_prompt`/`datetime` 导入。)

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_podcastwriter_prompt.py -k radar -v`
Expected: FAIL —— `TypeError: build_editor_prompt() got an unexpected keyword argument 'radar_material'`

- [ ] **Step 3: 实现模板扩展**

`src/ai_news_podcast/prompts.py`:

`build_editor_prompt` 之后追加段模板并修改 builder:

```python
EDITOR_RADAR_SECTION = """

## 项目雷达素材(固定栏目,与新闻无关)
{radar_material}

除上述新闻大纲外,请在输出末尾追加一节「## 项目雷达」:选出 {quick_count} 个快讯项目和
{deep_dive_count} 个深评项目(深评固定选给出的最高分项目)。快讯每条一句话(50 字以内,
必须点出「能拿来干什么」);深评回答三件事:第一步怎么跑起来、生态缺口在哪、fork 后
最小改动能做出什么差异化。所有仓库名与数字必须原样引用,禁止编造。"""


def build_editor_prompt(
    material: str,
    episode_date: datetime,
    radar_material: str = "",
    quick_count: int = 3,
    deep_dive_count: int = 1,
) -> str:
    """第一阶段:主编 Agent,负责精简素材和定调(可选附项目雷达)。"""
    date_str = _cn_date(episode_date)
    prompt = EDITOR_USER_TEMPLATE.format(date_str=date_str, material=material)
    if radar_material.strip():
        prompt += EDITOR_RADAR_SECTION.format(
            radar_material=radar_material,
            quick_count=quick_count,
            deep_dive_count=deep_dive_count,
        )
    return prompt
```

`build_writer_prompt` 修改为:

```python
WRITER_RADAR_SECTION = """

## 项目雷达栏目规范(大纲中含「## 项目雷达」时必须遵守)
1. 用固定转场自然开启栏目,例如苏晴说「新闻说完了,接下来进入今天的项目雷达时间」。
   不要每天一字不差。
2. 大纲里的仓库名、星数、增速数字必须原样引用,禁止编造、取整或夸大。
3. 快讯每个 2-3 句:是什么 + 为什么热 + 「能拿来干什么」。
4. 深评务必回答三件事:第一步怎么跑起来(如 pip install xx)、生态缺口在哪、
   fork 后最小改动能做出什么差异化。
5. 雷达部分总字数控制在 400-600 字,整体字数上限可放宽至 3500 字。"""


def build_writer_prompt(
    editor_plan_json: str,
    episode_date: datetime,
    podcast_title: str,
    style_cfg: dict[str, Any],
    *,
    has_radar: bool = False,
) -> str:
    """第二阶段:撰稿 Agent,将主编定下的大纲转化为双人对谈剧本。"""
    banned = style_cfg.get("banned_words", DEFAULT_BANNED_WORDS)
    banned_str = "、".join(banned)
    date_str = _cn_date(episode_date)

    prompt = WRITER_USER_TEMPLATE.format(
        date_str=date_str,
        podcast_title=podcast_title,
        banned_str=banned_str,
        editor_plan_json=editor_plan_json,
    )
    if has_radar:
        prompt += WRITER_RADAR_SECTION
    return prompt
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_podcastwriter_prompt.py -v`
Expected: PASS(原有用例 + 新用例)

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/ tests/ && uv run ruff check src/ tests/ && git add src/ai_news_podcast/prompts.py tests/test_podcastwriter_prompt.py && git commit -m "feat(radar): add radar sections to editor/writer prompts"
```

---

### Task 7: podcastwriter 接线

**Files:**
- Modify: `src/ai_news_podcast/pipeline/podcastwriter.py`
- Test: `tests/test_podcastwriter.py`(追加)

- [ ] **Step 1: 写失败测试(追加到 tests/test_podcastwriter.py)**

```python
@pytest.mark.asyncio
async def test_generate_podcast_injects_radar_into_editor_prompt() -> None:
    """雷达素材必须进入 Editor prompt,writer prompt 收到 has_radar 标记。"""
    from unittest.mock import patch

    from ai_news_podcast.pipeline import podcastwriter

    brief = {"radar": {"projects": [
        {"repo": "owner/hot", "url": "u", "stars": 1500, "delta_stars": 1000,
         "language": "Python", "license": "MIT", "description": "hot harness"},
    ]}, "meta": {"deep_dive_repo": "owner/hot"}}
    captured: list[str] = []

    def fake_llm(prompt, cfg):
        captured.append(prompt)
        if len(captured) == 1:
            return (
                "# 今日播报大纲\n\n## 金句\nx\n\n## 头条 1\n- **标题**: a\n- **摘要**: b\n\n"
                "## 头条 2\n- **标题**: c\n- **摘要**: d\n\n## 项目雷达\n- **深评 [owner/hot]**"
            )
        return "[Host A] 我们聊聊刚过去的 radar-repo。\n[Host B] 好的,这个项目值得说说。"

    with patch.object(podcastwriter, "_call_llm", side_effect=fake_llm):
        podcastwriter.generate_podcast(brief, episode_date=_SOME_DATETIME)

    assert "owner/hot" in captured[0]          # editor prompt 含雷达素材
    assert "项目雷达栏目规范" in captured[1]    # writer prompt 含雷达规则
```

(测试里 `_SOME_DATETIME` 复用该文件已有的 datetime 常量或 `datetime(2026, 9, 9)`;`_call_llm` 的真实签名在该文件顶部,若为 `prompt, cfg` 两个位置参数则按此写,执行者先读 `podcastwriter.py` 中 `_call_llm` 定义再落笔——其参数名以实际代码为准。)

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_podcastwriter.py -k radar -v`
Expected: FAIL —— `captured[0]` 中不含 `owner/hot`

- [ ] **Step 3: 实现**

`src/ai_news_podcast/pipeline/podcastwriter.py`:

import 区增加(与现有 `_build_material_text` 并列):

```python
from ai_news_podcast.pipeline.material import build_radar_text as _build_radar_text
```

`generate_podcast` 中,`material = _build_material_text(brief, max_stories=5)` 之后增加:

```python
    radar_material = _build_radar_text(brief.get("radar"))
```

`build_editor_prompt(material, episode_date)` 改为:

```python
        editor_prompt = build_editor_prompt(
            material, episode_date, radar_material=radar_material
        )
```

`build_writer_prompt(raw_editor, episode_date, podcast_title, style_cfg)` 改为:

```python
            writer_prompt = build_writer_prompt(
                raw_editor, episode_date, podcast_title, style_cfg, has_radar=bool(radar_material)
            )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_podcastwriter.py -v`
Expected: PASS(原有用例 + 新用例;原用例 brief 无 radar → 行为不变)

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/ tests/ && uv run ruff check src/ tests/ && git add src/ai_news_podcast/pipeline/podcastwriter.py tests/test_podcastwriter.py && git commit -m "feat(radar): wire radar into podcast writer agents"
```

---

### Task 8: 日报雷达章节(代码生成,零幻觉)

**Files:**
- Modify: `src/ai_news_podcast/cli/podcast_report.py`
- Test: `tests/test_daily_report.py`(追加)

- [ ] **Step 1: 写失败测试(追加到 tests/test_daily_report.py)**

```python
from ai_news_podcast.cli.podcast_report import build_radar_report_section


class TestRadarReportSection:
    def test_empty_radar_returns_empty(self) -> None:
        assert build_radar_report_section(None, "2026年9月9日") == ""
        assert build_radar_report_section({"projects": []}, "2026年9月9日") == ""

    def test_renders_links_and_numbers(self) -> None:
        radar = {
            "projects": [
                {"repo": "owner/hot", "url": "https://github.com/owner/hot", "stars": 1500,
                 "delta_stars": 1000, "language": "Python", "license": "MIT",
                 "description": "Fast LLM harness"},
            ],
            "meta": {"deep_dive_repo": "owner/hot"},
        }
        section = build_radar_report_section(radar, "2026年9月9日")
        assert "## 📡 项目雷达 | 2026年9月9日" in section
        assert "https://github.com/owner/hot" in section
        assert "⭐ 1500" in section and "+1000/天" in section
        assert "🔬 深评" in section
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_daily_report.py -k radar -v`
Expected: FAIL —— `ImportError: cannot import name 'build_radar_report_section'`

- [ ] **Step 3: 实现**

`src/ai_news_podcast/cli/podcast_report.py` 中,`build_report_prompt` 之后追加:

```python
def build_radar_report_section(radar: dict[str, Any] | None, date_display: str) -> str:
    """雷达章节由代码生成(非 LLM),保证数字与链接零幻觉。"""
    if not radar or not radar.get("projects"):
        return ""
    meta = radar.get("meta", {})
    lines = [f"\n## 📡 项目雷达 | {date_display}\n", "> 数字为 GitHub 实测,链接可直接上手。\n"]
    for p in radar.get("projects", []):
        role = "🔬 深评" if p.get("repo") == meta.get("deep_dive_repo") else "⚡ 快讯"
        delta = p.get("delta_stars")
        delta_str = f"+{delta}/天" if isinstance(delta, int) else "首日"
        lines.append(
            f"- **{role} [{p.get('repo')}]({p.get('url')})**"
            f" ⭐ {p.get('stars')}({delta_str})"
            f" · {p.get('language') or '—'} · License: {p.get('license') or '无'}\n"
            f"  {str(p.get('description') or '').strip()}\n"
        )
    return "\n".join(lines) + "\n"
```

顶部 import 区确认有 `from typing import Any`(现有文件若无则补)。

`execute_async` 中,`report_path.write_text(...)` 之前插入:

```python
        report_md += build_radar_report_section(brief.get("radar"), date_display)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_daily_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/ tests/ && uv run ruff check src/ tests/ && git add src/ai_news_podcast/cli/podcast_report.py tests/test_daily_report.py && git commit -m "feat(radar): append code-generated radar chapter to daily report"
```

---

### Task 9: workflow 与文档

**Files:**
- Modify: `.github/workflows/daily.yml`
- Modify: `AGENTS.md`
- 验证: workflow YAML 解析

- [ ] **Step 1: daily.yml stage1 传 token 并提交雷达产物**

`stage1` job 的 `Run pipeline` step,env 增加一行:

```yaml
      - name: Run pipeline
        env:
          TZ: Asia/Shanghai
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          EPISODE="${{ steps.episode.outputs.id }}"
          uv run podcast-pipeline --date "$EPISODE"
```

`Commit brief` step 的 `git add data/briefs/` 改为:

```yaml
          git add data/briefs/ data/gh_radar/ data/gh_snapshots/
```

- [ ] **Step 2: 验证 YAML 可解析**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/daily.yml')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: AGENTS.md 同步**

`## Data flow and dates` 一节,在 `- Other artifacts:` 列表中追加:

```markdown
- 项目雷达: `data/gh_radar/radar_{date}.json`(当期推荐)与
  `data/gh_snapshots/snap_{date}.json`(星数快照,供次日差分算增速)。
  两者都由 stage1 提交到 main。雷达作为独立「项目轨」与新闻管线完全隔离,
  结果挂在 brief 的 `radar` 键上,播客栏目「项目雷达」与日报章节均由此生成。
```

`## Gotchas` 一节末尾追加:

```markdown
- **项目雷达必须是可失败环节**:`runner` 用 try/except 包裹 `build_radar`,失败只发
  `StageFailed` 事件、正片照常。LLM 禁止自报 stars/增速等数字——全部由
  `material.build_radar_text` 从结构化数据注入;日报雷达章节完全由代码生成。
  `GITHUB_TOKEN` 匿名时走匿名限流(每日一次扫描足够),不要在雷达里加需要
  更高限流的调用。
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/daily.yml AGENTS.md && git commit -m "chore(radar): wire GITHUB_TOKEN into stage1 and sync docs"
```

---

### Task 10: 全量验证 + 真网冒烟

**Files:** 无新改动(验证任务)

- [ ] **Step 1: 全量质量门**

Run: `uv run ruff check src/ tests/ scripts/ && uv run ruff format --check src/ tests/ scripts/ && uv run lint-imports && uv run pytest tests/ -q`
Expected: 全部通过(测试数从 271 增加约 12-14 个)

- [ ] **Step 2: pre-commit 全量**

Run: `uv run pre-commit run --all-files`
Expected: 全部 Passed

- [ ] **Step 3: 真网冒烟(本机可访问 GitHub API,下载不受限)**

```bash
uv run python - <<'EOF'
import asyncio
from datetime import UTC, datetime
from pathlib import Path
import tempfile

from ai_news_podcast.pipeline.gh_radar import build_radar

async def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    cfg = {
        "enabled": True, "min_stars": 500, "created_window_days": 30,
        "recent_push_days": 21, "top_n": 30, "quick_count": 3, "deep_dive_count": 1,
        "readme_probe_limit": 12, "excluded_name_patterns": ["awesome", "list"],
        "preferred_topics": ["llm", "agents", "rag"], "snapshot_dir": "gh_snapshots",
        "output_dir": "gh_radar",
    }
    radar = await build_radar(cfg, "2026-09-09", tmp, [], now=datetime.now(tz=UTC))
    print("degraded:", radar["meta"]["degraded"])
    for p in radar["projects"]:
        print(f"{p['repo']} ⭐{p['stars']} Δ{p['delta_stars']} score={p['score']}")

asyncio.run(main())
EOF
```

Expected: 打印 4 个真实仓库(或空列表但 `degraded: False`),`data` 临时目录写出 snap/radar 两个 JSON。

- [ ] **Step 4: 收尾报告**

向用户报告:测试增量、冒烟结果、(可选)按拆分提交历史。

---

## Self-Review 记录

1. **Spec 覆盖**:双轨隔离(雷达不进新闻池——`build_radar` 独立数据源 ✓)、可上手硬过滤(Task 3 `_hard_filter` ✓)、增速信号(快照差分 ✓)、交叉提及(`count_news_mentions` ✓)、播客栏目(Task 6/7 ✓)、日报章节(Task 8 ✓)、可失败纪律(Task 4 ✓)、数字防幻觉(素材由代码注入 + 日报代码生成 ✓)、AI 工具链优先(`preferred_topics` 加分 ✓)、快照提交纪律(Task 9 ✓)、L3 网页不在本期(范围已定 L1+L2 ✓)。
2. **占位符**:无 TBD/TODO;Task 7 测试中 `_call_llm` 签名以源码为准已显式标注为执行前确认项(这是"读源码再落笔"的指令,非占位符)。
3. **类型一致性**:`build_radar(gcfg, date_str, data_dir, news_titles, *, client, now)` 在 Task 3 定义、Task 4 调用一致;`build_radar_text(radar) -> str` Task 5 定义、Task 7 调用一致;`build_editor_prompt(..., radar_material, quick_count, deep_dive_count)` 与 `build_writer_prompt(..., has_radar)` Task 6 定义、Task 7 调用一致;`build_radar_report_section(radar, date_display)` Task 8 内定义与测试一致。
