"""Shared helpers for daily episode CLI commands."""

from __future__ import annotations

import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_news_podcast.config.models import AppConfig


def get_base_url(cfg: dict[str, Any] | AppConfig, cli_base_url: str | None) -> str:
    env = os.environ
    if cli_base_url:
        return cli_base_url.rstrip("/")
    if env.get("PODCAST_BASE_URL"):
        return env["PODCAST_BASE_URL"].rstrip("/")

    # Handle both dict and AppConfig
    if isinstance(cfg, dict):
        base_url = str(cfg.get("podcast", {}).get("base_url") or "").strip()
    else:
        base_url = str(getattr(cfg.podcast, "base_url", "") or "").strip()

    if base_url:
        return base_url.rstrip("/")
    owner = (env.get("GITHUB_REPOSITORY_OWNER") or "").strip()
    repo_full = (env.get("GITHUB_REPOSITORY") or "").strip()
    if owner and repo_full and "/" in repo_full:
        repo = repo_full.split("/", 1)[1]
        if repo == f"{owner}.github.io":
            return f"https://{owner}.github.io".rstrip("/")
        return f"https://{owner}.github.io/{repo}".rstrip("/")
    return "http://localhost"


def episode_id(day: datetime) -> str:
    return day.strftime("%Y-%m-%d")


def coerce_episode_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


# 只匹配剧集形状的名字:YYYY-MM-DD 切片目录和 YYYY-MM-DD.(mp3|html|txt) 文件,
# episodes_dir 里的其他文件一律不动。
_EPISODE_ITEM_RE = re.compile(r"^(?P<id>\d{4}-\d{2}-\d{2})(?:\.(?:mp3|html|txt))?$")


def _sweep_orphan_files(episodes_dir: Path, keep_ids: set[str]) -> None:
    """删除索引里已不存在的日期残留文件。

    覆盖上游某阶段成功但当期从未发布的日子(例如脚本已提交但 TTS 失败),
    这些文件不在 episodes.json 里,keep_last 轮转不会碰它们。调用方必须先把
    当前期目加入 keep_ids,否则当期文件会被误删。
    """
    if not episodes_dir.exists():
        return
    # 索引里的 id 不是日期形状时放弃清扫,避免按错误规则删文件。
    newest = max((eid for eid in keep_ids if _EPISODE_ITEM_RE.match(eid)), default=None)
    if newest is None:
        return
    for item in episodes_dir.iterdir():
        match = _EPISODE_ITEM_RE.match(item.name)
        if not match or match["id"] in keep_ids:
            continue
        # 比索引最新一期还新的日期是"在途"文件(脚本已提交、还没发布),
        # 不能当作孤儿删除,否则会毁掉尚未发布的剧集。
        if match["id"] > newest:
            continue
        print(f"Removing orphan episode file: {item.name}")
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def prune_episodes(
    episodes: list[dict[str, Any]],
    *,
    keep_last: int,
    episodes_dir: Path,
) -> list[dict[str, Any]]:
    def parse_pubdate(ep: dict[str, Any]) -> datetime:
        try:
            return datetime.fromisoformat(str(ep["published_at_iso"]))
        except (ValueError, KeyError):
            return datetime(1970, 1, 1, tzinfo=UTC)

    sorted_eps = sorted(episodes, key=parse_pubdate, reverse=True)
    keep = sorted_eps[:keep_last]
    keep_ids = {str(ep.get("id") or "") for ep in keep}

    for ep in sorted_eps[keep_last:]:
        eid = str(ep.get("id") or "")
        if not eid:
            continue
        for suffix in (".mp3", ".html", ".txt"):
            path = episodes_dir / f"{eid}{suffix}"
            if path.exists():
                path.unlink()

        # 删除对应的音频切片文件夹
        chunks_dir = episodes_dir / eid
        if chunks_dir.exists() and chunks_dir.is_dir():
            shutil.rmtree(chunks_dir)

    _sweep_orphan_files(episodes_dir, keep_ids)

    return [ep for ep in keep if str(ep.get("id") or "") in keep_ids]
