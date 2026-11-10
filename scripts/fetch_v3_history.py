#!/usr/bin/env python3
"""下载最近 N 天 v3 流水线 artifact，作为 Stage 2 的历史输入。

设计：GHA 每次运行时，用当前 run 的 GITHUB_TOKEN 拉取历史 `v3-output-<date>`
artifact（默认 14 天），把其中的 `data/ocean/<date>/` 合并到工作区 `data/ocean/`，
让 Stage 2 的跨日去重/连续性在 CI 里也能拿到历史事件。

用法：
    python scripts/fetch_v3_history.py --days 14 --exclude-run-id "$GITHUB_RUN_ID"

依赖：gh CLI + GH_TOKEN（GITHUB_TOKEN 需要 actions: read 权限）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

ARTIFACT_RE = re.compile(r"^v3-output-(\d{4}-\d{2}-\d{2})$")


def _gh(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {result.stderr.strip()[:300]}")
    return result.stdout


def select_runs(
    runs: list[dict],
    *,
    days: int,
    exclude_run_id: str = "",
    now: datetime | None = None,
) -> list[dict]:
    """从 gh run list JSON 里筛出最近 N 天成功完成的 run（新→旧）。"""
    now = now or datetime.now(tz=UTC)
    cutoff = now - timedelta(days=days)
    picked: list[dict] = []
    for run in runs:
        if str(run.get("databaseId")) == str(exclude_run_id):
            continue
        if run.get("status") != "completed" or run.get("conclusion") != "success":
            continue
        created = str(run.get("createdAt") or "")
        try:
            created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            continue
        if created_at < cutoff:
            continue
        picked.append(run)
    return picked


def artifact_dates(artifacts: list[dict]) -> dict[str, str]:
    """artifact 名 -> 日期；只保留 v3-output-YYYY-MM-DD。"""
    result: dict[str, str] = {}
    for art in artifacts:
        name = str(art.get("name") or "")
        match = ARTIFACT_RE.match(name)
        if match:
            result[name] = match.group(1)
    return result


def fetch_history(
    *,
    workflow: str,
    days: int,
    out_dir: Path,
    exclude_run_id: str,
    repo: str,
    max_runs: int = 40,
) -> int:
    """下载历史 artifact，返回成功下载的天数。"""
    out_dir = Path(out_dir)
    ocean_dir = out_dir / "ocean"
    ocean_dir.mkdir(parents=True, exist_ok=True)

    raw = _gh(
        "run",
        "list",
        "--workflow",
        workflow,
        "--limit",
        str(max_runs),
        "--json",
        "databaseId,status,conclusion,createdAt,event",
    )
    runs = select_runs(json.loads(raw or "[]"), days=days, exclude_run_id=exclude_run_id)
    print(f"候选历史 run：{len(runs)} 个（最近 {days} 天）", flush=True)

    seen_dates: set[str] = set()
    fetched = 0
    for run in runs:
        run_id = str(run["databaseId"])
        try:
            artifacts = json.loads(
                _gh(
                    "api",
                    f"repos/{repo}/actions/runs/{run_id}/artifacts",
                    "--jq",
                    ".artifacts",
                )
                or "[]"
            )
        except RuntimeError as exc:
            print(f"  run {run_id}: 列 artifact 失败：{exc}", flush=True)
            continue
        for name, date_str in sorted(artifact_dates(artifacts).items()):
            if date_str in seen_dates:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                try:
                    _gh("run", "download", run_id, "--name", name, "--dir", str(tmp_path))
                except RuntimeError as exc:
                    print(f"  {date_str}: 下载失败：{exc}", flush=True)
                    continue
                src = tmp_path / "data" / "ocean"
                if not src.is_dir():
                    print(f"  {date_str}: artifact 缺少 data/ocean，跳过", flush=True)
                    continue
                for date_dir in sorted(p for p in src.iterdir() if p.is_dir()):
                    target = ocean_dir / date_dir.name
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(date_dir, target)
            seen_dates.add(date_str)
            fetched += 1
            print(f"  {date_str}: 已下载（run {run_id}）", flush=True)
    print(f"历史 artifact 合并完成：{fetched} 天 -> {ocean_dir}", flush=True)
    return fetched


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 v3 历史 artifact 作为 Stage 2 历史输入")
    parser.add_argument("--workflow", default="v3_daily.yml")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--exclude-run-id", default="")
    parser.add_argument("--max-runs", type=int, default=40)
    args = parser.parse_args()

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("未设置 GH_TOKEN/GITHUB_TOKEN，跳过历史 artifact 下载。", file=sys.stderr)
        return 0
    repo = (
        os.environ.get("GITHUB_REPOSITORY")
        or _gh("repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner").strip()
    )
    if not repo:
        print("无法确定仓库名，跳过。", file=sys.stderr)
        return 0
    try:
        fetch_history(
            workflow=args.workflow,
            days=args.days,
            out_dir=Path(args.out_dir),
            exclude_run_id=args.exclude_run_id,
            repo=repo,
            max_runs=args.max_runs,
        )
    except Exception as exc:  # 历史缺失不应阻断当天流水线
        print(f"历史 artifact 下载失败（继续无历史运行）：{exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
