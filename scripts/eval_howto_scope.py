#!/usr/bin/env python3
"""T4 影响量化：narrative_angle 只在标题命中后，howto 标记质量与 P@8 的变化。

用法：python scripts/eval_howto_scope.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GOLD_FILES = (
    ROOT / "data/eval/recommend/gold_v1_llm.jsonl",
    ROOT / "data/eval/recommend/gold_8day_pool4_llm.jsonl",
    ROOT / "data/eval/recommend/gold_coarse_grid_llm.jsonl",
    ROOT / "data/eval/recommend/gold_17day_llm.jsonl",
)
HOWTO_TITLE_HINTS = ("如何", "指南", "教程", "怎么", "实战", "入门", "how to", "guide")


def raw_paths() -> dict[str, Path]:
    dirs = [ROOT / "data" / "raw", Path("/tmp/s1check/data/raw"), Path("/tmp/s1check24/data/raw")]
    dirs.extend(Path("/tmp/hist").glob("*/data/raw"))
    found: dict[str, Path] = {}
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in directory.glob("raw-*.json.gz"):
            date = path.name[len("raw-") : -len(".json.gz")]
            found.setdefault(date, path)
    return dict(sorted(found.items()))


def build_all(raw: dict[str, Path]) -> dict[str, list[dict]]:
    from ai_news_podcast.business.ocean.build import build_ocean

    out: dict[str, list[dict]] = {}
    for date, path in raw.items():
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = build_ocean(
                str(path),
                date_str=date,
                history_dir=Path(tmp) / "ocean",
                ledger_path=Path(tmp) / "ledger.json",
                config_path=str(ROOT / "config" / "ocean_v3.yaml"),
            )
        out[date] = [
            event if isinstance(event, dict) else event.to_dict() for event in snapshot.events
        ]
    return out


def load_gold() -> dict[str, dict[str, dict]]:
    from ai_news_podcast.business.eval.recommend_eval import normalize_title

    gold: dict[str, dict[str, dict]] = {}
    for path in GOLD_FILES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            gold.setdefault(str(row.get("date")), {})[normalize_title(row.get("title"))] = row
    return gold


def evaluate(events_by_day: dict[str, list[dict]], gold: dict[str, dict[str, dict]]) -> tuple:
    from ai_news_podcast.business.contracts import Event
    from ai_news_podcast.business.eval.recommend_eval import normalize_title
    from ai_news_podcast.business.recommend.service import recommend

    judged = positive = 0
    for date, rows in events_by_day.items():
        events = [Event.from_dict(r) for r in rows]
        by_id = {str(e.event_id): e for e in events}
        shortlist = recommend(
            events,
            config_path=ROOT / "config" / "recommend_v3.yaml",
            reference_time=datetime.strptime(date, "%Y-%m-%d").replace(hour=12, tzinfo=UTC),
        )
        for entry in shortlist:
            event = by_id[str(entry.event_id)]
            row = gold.get(date, {}).get(normalize_title(event.representative_title))
            if not row or row.get("worth_publishing") is None:
                continue
            judged += 1
            positive += int(bool(row["worth_publishing"]))
    return judged, positive


def howto_stats(events_by_day: dict[str, list[dict]]) -> tuple[int, int, int]:
    total = marked = titled = 0
    for rows in events_by_day.values():
        for row in rows:
            total += 1
            angles = (row.get("labels") or {}).get("narrative_angle") or []
            if "howto" not in angles:
                continue
            marked += 1
            title = str(row.get("representative_title") or "").lower()
            if any(hint in title for hint in HOWTO_TITLE_HINTS):
                titled += 1
    return total, marked, titled


def main() -> int:
    raw = raw_paths()
    if not raw:
        print("没有可用的 raw 数据")
        return 1
    print(f"raw {len(raw)} 天（{min(raw)} ~ {max(raw)}），重建 ocean 中…", flush=True)
    events_by_day = build_all(raw)

    total, marked, titled = howto_stats(events_by_day)
    print(f"\n事件 {total}")
    print(f"  howto 标记 {marked}（{marked / max(1, total):.1%}）")
    print(f"  标题确含教程句式 {titled} → 精确率 {titled / max(1, marked):.2f}")

    gold = load_gold()
    judged, positive = evaluate(events_by_day, gold)
    rate = positive / judged if judged else 0
    print(f"\nP@8 {positive}/{judged} = {rate:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
