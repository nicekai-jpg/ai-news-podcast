#!/usr/bin/env python3
"""文稿事实核对 CLI：用当日产物重建事实底稿，核对文稿里的数字。

为什么需要：GHA 的 "Check script quality" 步骤只传文稿文本给 `quality()`，
没有底稿 → 数字核对被静默跳过（`fact_checked=False`）。这里补上底稿重建，
让线上也能真正核对。

用法：
    uv run python scripts/script_factcheck.py --date 2026-11-25
    uv run python scripts/script_factcheck.py --date 2026-11-25 --script site/episodes/x.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_news_podcast.business.contracts import Event  # noqa: E402
from ai_news_podcast.business.editorial.config import load_config  # noqa: E402
from ai_news_podcast.business.editorial.quality import check_script  # noqa: E402
from ai_news_podcast.business.editorial.script import script_evidence_text  # noqa: E402


def _load_json(path: Path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def rebuild_evidence(data_dir: Path, date_str: str) -> str:
    """按 S6 的口径重建事实底稿（标题/摘要/正文 + 雷达素材）。"""
    ocean = data_dir / "ocean" / date_str
    events = [Event.from_dict(row) for row in _load_json(ocean / "events.json", [])]
    shortlist_rows = _load_json(data_dir / "selection" / date_str / "shortlist.json", [])
    items = _load_json(ocean / "items.json", [])
    evidence = {
        str(row.get("item_id")): str(row.get("full_text") or row.get("summary") or "")
        for row in items
        if isinstance(row, dict)
    }

    class _Entry:
        def __init__(self, event_id: str) -> None:
            self.event_id = event_id

    shortlist = [
        _Entry(str(row.get("event_id"))) for row in shortlist_rows if isinstance(row, dict)
    ]
    config = load_config(ROOT / "config" / "editorial_v3.yaml")
    text = script_evidence_text(shortlist, events, evidence, config)

    # 雷达素材：brief 里有就用
    brief = _load_json(data_dir / "briefs" / f"brief_{date_str}.json", {})
    radar = brief.get("radar") if isinstance(brief, dict) else None
    if radar and radar.get("projects"):
        from ai_news_podcast.business.pipeline.material_service import build_radar_text

        radar_text = build_radar_text(radar)
        if radar_text:
            text = f"{text}\n{radar_text}"
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="文稿事实核对")
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--script", default="")
    parser.add_argument("--strict", action="store_true", help="有未核对数字时返回非 0")
    args = parser.parse_args(argv)

    script_path = Path(args.script or f"site/episodes/{args.date}.txt")
    if not script_path.is_file():
        print(f"::error::文稿不存在：{script_path}")
        return 2
    script = script_path.read_text(encoding="utf-8")
    evidence = rebuild_evidence(Path(args.data_dir), args.date)
    if not evidence.strip():
        print("::warning::无法重建事实底稿，跳过数字核对")
        return 0

    config = load_config(ROOT / "config" / "editorial_v3.yaml")
    report = check_script(script, config, evidence_text=evidence)
    numbers = report["claim_numbers"]
    unverified = report["unverified_numbers"]
    print(
        f"script factcheck: 数字 {len(numbers)} 个，未核对 {len(unverified)} 个，ok={report['ok']}"
    )
    if unverified:
        print("未核对的数字：" + "、".join(unverified[:12]))
        print("::warning::文稿存在无法在事实底稿中核对的数字")
        if args.strict:
            return 1
    other = [issue for issue in report["issues"] if "数字" not in issue]
    if other:
        print("其他问题：" + "；".join(other))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
