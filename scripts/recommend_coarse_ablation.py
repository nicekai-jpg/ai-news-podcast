# ruff: noqa: PERF401
#!/usr/bin/env python3
"""粗排门槛消融：subject_cap / max_candidates / 单源传闻门槛。


同时输出两个指标：
- 正例留存：已知金标正例在粗排后还剩多少（衡量 S2/S3 漏斗损失）
- P@8：最终 shortlist 里值得播的比例（衡量成品质量）

用法：uv run python scripts/recommend_coarse_ablation.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_news_podcast.business.contracts import Event  # noqa: E402
from ai_news_podcast.business.eval.recommend_eval import normalize_title  # noqa: E402
from ai_news_podcast.business.recommend.config import RecommendConfig  # noqa: E402
from ai_news_podcast.business.recommend.service import recommend_with_audit  # noqa: E402

CONFIG = ROOT / "config" / "recommend_v3.yaml"
OCEAN = ROOT / "data" / "ocean"
GOLD = [
    ROOT / "data/eval/recommend/gold_v1_llm.jsonl",
    ROOT / "data/eval/recommend/gold_8day_pool4_llm.jsonl",
]


def load_gold():
    out: dict[str, dict[str, dict]] = {}
    for path in GOLD:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out.setdefault(str(row.get("date")), {})[normalize_title(row.get("title"))] = row
    return out


def load_events(day: str) -> list[Event]:
    return [
        Event.from_dict(row)
        for row in json.loads((OCEAN / day / "events.json").read_text(encoding="utf-8"))
    ]


def run(gold, cfg, days) -> dict:
    total = recalled = coarse = short = judged = positive = 0
    for day in days:
        events = load_events(day)
        reference = datetime.strptime(day, "%Y-%m-%d").replace(hour=12, tzinfo=UTC)
        result = recommend_with_audit(
            events, config_path=CONFIG, reference_time=reference, config=cfg
        )
        rec = {str(c.event.event_id) for c in result.recalled}
        co = {str(c.event.event_id) for c in result.coarse.candidates}
        sh = {str(e.event_id) for e in result.shortlist}
        by_id = {str(e.event_id): e for e in events}
        pos = [
            e
            for e in events
            if (g := gold.get(day, {}).get(normalize_title(e.representative_title)))
            and g.get("worth_publishing") is True
        ]
        total += len(pos)
        recalled += sum(1 for e in pos if e.event_id in rec)
        coarse += sum(1 for e in pos if e.event_id in co)
        short += sum(1 for e in pos if e.event_id in sh)
        for entry in result.shortlist:
            event = by_id[str(entry.event_id)]
            g = gold.get(day, {}).get(normalize_title(event.representative_title))
            if not g or g.get("worth_publishing") is None:
                continue
            judged += 1
            positive += int(bool(g["worth_publishing"]))
    return {
        "total": total,
        "recalled": recalled,
        "coarse": coarse,
        "short": short,
        "judged": judged,
        "positive": positive,
    }


def main() -> int:
    gold = load_gold()
    days = sorted(d for d in gold if (OCEAN / d / "events.json").exists())
    base = RecommendConfig.load(CONFIG)

    variants: list[tuple[str, RecommendConfig]] = [("baseline", base)]
    for cap in (4, 6, 99):
        variants.append(
            (
                f"subject_cap={cap}",
                dataclasses.replace(base, coarse=dataclasses.replace(base.coarse, subject_cap=cap)),
            )
        )
    for cap in (40,):
        variants.append(
            (
                f"max_candidates={cap}",
                dataclasses.replace(
                    base, coarse=dataclasses.replace(base.coarse, max_candidates=cap)
                ),
            )
        )
    variants.append(
        (
            "no_rumor_gate",
            dataclasses.replace(base, coarse=dataclasses.replace(base.coarse, rumor_tiers=())),
        )
    )
    variants.append(
        (
            "cap=6+mc=40",
            dataclasses.replace(
                base,
                coarse=dataclasses.replace(base.coarse, subject_cap=6, max_candidates=40),
            ),
        )
    )

    print(f"金标日 {len(days)} 天")
    print(f"{'配置':22}{'正例':>6}{'召回':>6}{'粗排':>6}{'shortlist':>10}{'P@8':>8}")
    for label, cfg in variants:
        r = run(gold, cfg, days)
        p8 = (
            f"{r['positive']}/{r['judged']}={r['positive'] / r['judged']:.2f}"
            if r["judged"]
            else "-"
        )
        print(f"{label:22}{r['total']:>6}{r['recalled']:>6}{r['coarse']:>6}{r['short']:>10}{p8:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
