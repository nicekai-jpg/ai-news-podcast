"""Shared helpers for daily episode CLI commands."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def get_base_url(cfg: dict[str, Any], cli_base_url: Optional[str]) -> str:
    env = os.environ
    if cli_base_url:
        return cli_base_url.rstrip("/")
    if env.get("PODCAST_BASE_URL"):
        return env["PODCAST_BASE_URL"].rstrip("/")
    base_url = str(cfg.get("podcast", {}).get("base_url") or "").strip()
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
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

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

    return [ep for ep in keep if str(ep.get("id") or "") in keep_ids]
