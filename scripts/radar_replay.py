"""离线回放:用仓库内现有 data/gh_radar 与 data/gh_snapshots 复盘 v4 雷达策略。

不发出任何网络请求:星数取自当日快照,候选元数据(描述/topic/README 探测结果)
取自历史 radar JSON。离线局限:D1(trending 镜像)/D3(主题轮换)/D4(新闻挂载)
三个发现源无法离线还原,回放反映的是「评分场 + 考期 + 多样性 + 排重」的净效果
(不含新增发现源的覆盖增益)。

用法::

    uv run python scripts/radar_replay.py [--since YYYY-MM-DD] [--data data]
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_news_podcast.business.pipeline.gh_radar_service import (
    RadarProject,
    _days_in_pool,
    _load_recent_pick_dates,
    _load_recent_snapshots,
    _select_probe_list,
    _vel_per_day,
    ai_affinity,
    score_project,
)
from ai_news_podcast.data.config.loader_dao import load_config
from ai_news_podcast.data.utils_dao import read_json


def _load_snapshots(snapshot_dir: Path) -> list[tuple[str, dict[str, int]]]:
    """按日期升序返回全部快照 [(date, stars)]。"""
    out: list[tuple[str, dict[str, int]]] = []
    for f in sorted(snapshot_dir.glob("snap_*.json")):
        m = re.search(r"snap_(\d{4}-\d{2}-\d{2})\.json$", f.name)
        if not m:
            continue
        try:
            data = read_json(f)
            stars = data.get("stars", {})
            if isinstance(stars, dict):
                out.append((m.group(1), {str(k): int(v) for k, v in stars.items()}))
        except Exception:
            continue
    return out


def _radars_by_date(root: Path) -> dict[str, dict[str, Any]]:
    by_date: dict[str, dict[str, Any]] = {}
    for f in sorted(root.glob("radar_*.json")):
        m = re.search(r"radar_(\d{4}-\d{2}-\d{2})\.json$", f.name)
        if not m:
            continue
        try:
            data = read_json(f)
            if isinstance(data, dict):
                by_date[m.group(1)] = data
        except Exception:
            continue
    return by_date


def _load_project_meta(radars: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """历史 radar 里的 project 记录是唯一离线可用的 repo 元数据来源。"""
    meta: dict[str, dict[str, Any]] = {}
    for r in radars:
        for p in r.get("projects") or []:
            if isinstance(p, dict) and p.get("repo"):
                meta[str(p["repo"])] = p
    return meta


def _synth_item(repo: str, meta: dict[str, Any], stars_today: int) -> dict[str, Any]:
    return {
        "full_name": repo,
        "html_url": str(meta.get("url") or f"https://github.com/{repo}"),
        "description": str(meta.get("description") or ""),
        "topics": [str(t) for t in (meta.get("topics") or [])],
        "stargazers_count": stars_today,
        "created_at": str(meta.get("created_at") or ""),
        "pushed_at": str(meta.get("pushed_at") or ""),
        "license": {"spdx_id": str(meta.get("license") or "")},
    }


def _yesterday_language(radars_by_date: dict[str, dict[str, Any]], date_str: str) -> str:
    best: dict[str, Any] = {}
    best_date = ""
    for d, r in radars_by_date.items():
        if d >= date_str or d <= best_date:
            continue
        best, best_date = r, d
    pick = str(best.get("meta", {}).get("pick_repo") or "")
    for p in best.get("projects") or []:
        if isinstance(p, dict) and p.get("repo") == pick:
            return str(p.get("language") or "")
    return ""


def _replay_day(
    date_str: str,
    snap_stars: dict[str, int],
    prev_snap: tuple[str, dict[str, int]],
    project_meta: dict[str, dict[str, Any]],
    output_dir: Path,
    gcfg: dict[str, Any],
    radars_by_date: dict[str, dict[str, Any]],
    snapshot_dir: Path,
) -> dict[str, Any] | None:
    """重放单日的选型:返回 {pick, affinity};无任何可选候选时返回 None。"""
    prev_stars = dict(prev_snap[1])
    snap_gap_days = (datetime.fromisoformat(date_str) - datetime.fromisoformat(prev_snap[0])).days
    items = [
        _synth_item(repo, project_meta[repo], stars)
        for repo, stars in snap_stars.items()
        if repo in project_meta
    ]
    items.sort(key=lambda it: int(it.get("stargazers_count", 0)), reverse=True)

    recent = _load_recent_pick_dates(output_dir, int(gcfg.get("repeat_window_days", 30)), date_str)
    _fresh, probe_list = _select_probe_list(
        items,
        prev_stars,
        recent,
        date_str=date_str,
        pick_count=1,
        probe_limit=int(gcfg.get("readme_probe_limit", 12)),
        repeat_gap_days=int(gcfg.get("repeat_gap_days", 7)),
    )
    if not probe_list:
        return None

    lexicon = dict(gcfg.get("ai_lexicon") or {})
    snaps = _load_recent_snapshots(snapshot_dir, date_str, 3)
    now = datetime.fromisoformat(date_str).replace(tzinfo=UTC)
    pool_ctx: dict[str, Any] = {
        "prev_stars": prev_stars,
        "velocity_cap": float(gcfg.get("velocity_cap", 1500)),
        "vel_rel": float(gcfg.get("vel_rel", 0.6)),
        "vel_abs": float(gcfg.get("vel_abs", 0.4)),
        "fresh_penalty": float(gcfg.get("fresh_penalty", 0.85)),
        "same_language_penalty": float(gcfg.get("same_language_penalty", 0.05)),
        "snap_gap_days": snap_gap_days,
        "vel_by_repo": {
            str(it.get("full_name", "")): _vel_per_day(it, prev_stars, snap_gap_days, now)
            for it in probe_list
        },
        "days_in_pool": {
            fn: _days_in_pool(snaps, fn, date_str)
            for fn in (str(it.get("full_name", "")) for it in probe_list)
        },
        "last_pick_language": _yesterday_language(radars_by_date, date_str),
    }

    graded: list[tuple[float, str, float]] = []
    for it in probe_list:
        fn = str(it.get("full_name", ""))
        p = RadarProject(
            repo=fn,
            url=str(it.get("html_url") or f"https://github.com/{fn}"),
            description=str(it.get("description") or ""),
            language="",
            topics=[],
            stars=int(it.get("stargazers_count", 0)),
            delta_stars=None,
            is_new=True,
            created_at=str(it.get("created_at") or ""),
            pushed_at=str(it.get("pushed_at") or ""),
            license="MIT",
            has_install_docs=False,
            mentions=0,
        )
        p.affinity = ai_affinity(it, lexicon)
        score_project(p, now=now, pool_ctx=pool_ctx)
        graded.append((p.score, fn, p.affinity))
    graded.sort(key=lambda x: x[0], reverse=True)
    _, pick, aff = graded[0]
    return {"pick": pick, "affinity": aff}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="项目雷达 v4 策略离线回放")
    parser.add_argument("--since", default="", help="只回放该日期及之后(YYYY-MM-DD)")
    parser.add_argument("--data", default="data", help="数据目录(默认 data)")
    args = parser.parse_args(argv)

    data_dir = Path(args.data)
    gcfg = asdict(load_config(Path("config/config.yaml")).gh_radar)
    snapshot_dir = data_dir / "gh_snapshots"
    snapshots = _load_snapshots(snapshot_dir)
    if len(snapshots) < 2:
        print("快照不足(需 ≥2 份),无法回放")
        return 1

    radars_by_date = _radars_by_date(data_dir / "gh_radar")
    project_meta = _load_project_meta(list(radars_by_date.values()))
    output_dir = data_dir / "gh_radar"

    rows: list[tuple[str, str, str, str]] = []
    blank_old = blank_new = 0
    new_picks: list[tuple[str, float]] = []
    for i in range(1, len(snapshots)):
        date_str, snap_stars = snapshots[i]
        if args.since and date_str < args.since:
            continue
        old_pick = str(radars_by_date.get(date_str, {}).get("meta", {}).get("pick_repo") or "")
        replay = _replay_day(
            date_str,
            snap_stars,
            snapshots[i - 1],
            project_meta,
            output_dir,
            gcfg,
            radars_by_date,
            snapshot_dir,
        )
        new_pick = replay["pick"] if replay else ""
        new_aff = replay["affinity"] if replay else 0.0
        if not old_pick:
            blank_old += 1
        if not new_pick:
            blank_new += 1
        if new_pick:
            new_picks.append((new_pick, new_aff))
        rows.append((date_str, old_pick or "—", new_pick or "—", f"{new_aff:.1f}"))

    print(f"{'日期':<12} {'改前主推':<38} {'改后主推(离线)':<38} AI契约度")
    for date_str, old_pick, new_pick, aff in rows:
        print(f"{date_str:<12} {old_pick:<38} {new_pick:<38} {aff}")
    distinct_old = {r[1] for r in rows if r[1] != "—"}
    distinct_new = {p for p, _ in new_picks}
    ai_new = sum(1 for _, a in new_picks if a > 0)
    print("\n== 指标汇总 ==")
    print(f"空白天: 改前 {blank_old} / 改后(离线) {blank_new}")
    print(f"distinct 主推: 改前 {len(distinct_old)} / 改后(离线) {len(distinct_new)}")
    print(f"主推 AI 契约度>0 占比(离线): {ai_new}/{len(new_picks)}")
    if new_picks:
        print("改后主推频次 Top:", Counter(p for p, _ in new_picks).most_common(5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
