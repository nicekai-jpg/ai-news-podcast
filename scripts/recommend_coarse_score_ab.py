# ruff: noqa: SLF001
#!/usr/bin/env python3
"""粗排打分公式 A/B：现公式 vs 去掉无区分度特征的新公式。


依据（239 正例 / 485 负例，全部已标注事件）：
| 特征 | 正例 | 负例 | 差 |
|---|---|---|---|
| novelty_numbers 数 | 5.05 | 3.26 | +1.79 |
| tech_stack 数 | 3.09 | 1.91 | +1.18 |
| actions 数 | 2.41 | 1.69 | +0.72 |
| 主体显著度 | 0.81 | 0.41 | +0.41 |
| 独立来源 | 1.07 | 1.01 | +0.06 |  ← 无区分度
| 来源等级 | 2.87 | 2.90 | -0.03 |  ← 无区分度

现公式把 1.8 分压在「多源/来源数」上，但 96.8% 事件单源，这两项几乎恒为 0；
再统一扣 0.4 单源惩罚，把整个分布压平，导致粗排分几乎没有排序能力。
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_news_podcast.business.recommend import coarse as coarse_mod  # noqa: E402
from ai_news_podcast.business.recommend.base import (  # noqa: E402
    event_age_hours,
    max_subject_significance,
)
from ai_news_podcast.business.recommend.coarse import (  # noqa: E402
    _EVIDENCE_BONUS,
    _LIFECYCLE_BONUS,
)

# 新公式：只保留有区分度的特征
NEW_WEIGHTS = {
    "significance": 1.2,
    "numbers": 1.0,
    "tech": 0.6,
    "actions": 0.4,
}


def new_score(candidate, settings, reference_time):
    event = candidate.event
    labels = event.labels
    reasons: list[str] = []
    score = 0.0

    significance = max_subject_significance(labels)
    score += NEW_WEIGHTS["significance"] * significance
    if significance > 0:
        reasons.append(f"重要主体 {significance:.2f}")

    numbers = min(len(labels.novelty_numbers), 6) / 6
    score += NEW_WEIGHTS["numbers"] * numbers
    if numbers > 0:
        reasons.append(f"数字 {len(labels.novelty_numbers)}")

    tech = min(len(labels.tech_stack), 5) / 5
    score += NEW_WEIGHTS["tech"] * tech

    actions = min(len(labels.actions), 4) / 4
    score += NEW_WEIGHTS["actions"] * actions

    age = event_age_hours(event, reference_time)
    if age is None:
        score += 0.5
    elif age <= 48:
        score += 1.0
        reasons.append("48h 内")
    elif age <= 168:
        score += 0.6
    else:
        score += 0.2

    score += _LIFECYCLE_BONUS.get(labels.lifecycle, 0.0)
    score += _EVIDENCE_BONUS.get(labels.evidence_tier, 0.0)
    score += coarse_mod._priority_action_bonus(labels, settings.priority_actions)
    penalty, pen_reasons = coarse_mod._demote_penalties(event, labels, settings)
    if penalty:
        score -= penalty
        reasons.extend(pen_reasons)
    return max(score, 0.0), reasons or ["通过粗排门槛"]


CONFIG = ROOT / "config" / "recommend_v3.yaml"
OCEAN = ROOT / "data" / "ocean"
GOLD = [
    ROOT / "data/eval/recommend/gold_v1_llm.jsonl",
    ROOT / "data/eval/recommend/gold_8day_pool4_llm.jsonl",
]


def load_gold():
    from ai_news_podcast.business.eval.recommend_eval import normalize_title

    out: dict[str, dict[str, dict]] = {}
    for path in GOLD:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out.setdefault(str(row.get("date")), {})[normalize_title(row.get("title"))] = row
    return out


def run(gold, days):
    from ai_news_podcast.business.contracts import Event
    from ai_news_podcast.business.eval.recommend_eval import normalize_title
    from ai_news_podcast.business.recommend.config import RecommendConfig
    from ai_news_podcast.business.recommend.service import recommend_with_audit

    cfg = RecommendConfig.load(CONFIG)
    total = recalled = coarse_n = judged = positives = 0
    missing = 0
    for day in days:
        events = [
            Event.from_dict(row)
            for row in json.loads((OCEAN / day / "events.json").read_text(encoding="utf-8"))
        ]
        reference = datetime.strptime(day, "%Y-%m-%d").replace(hour=12, tzinfo=UTC)
        result = recommend_with_audit(
            events, config_path=CONFIG, reference_time=reference, config=cfg
        )
        rec = {str(c.event.event_id) for c in result.recalled}
        co = {str(c.event.event_id) for c in result.coarse.candidates}
        pos = [
            e
            for e in events
            if (g := gold.get(day, {}).get(normalize_title(e.representative_title)))
            and g.get("worth_publishing") is True
        ]
        total += len(pos)
        recalled += sum(1 for e in pos if e.event_id in rec)
        coarse_n += sum(1 for e in pos if e.event_id in co)
        by_id = {str(e.event_id): e for e in events}
        for entry in result.shortlist:
            event = by_id[str(entry.event_id)]
            g = gold.get(day, {}).get(normalize_title(event.representative_title))
            if not g or g.get("worth_publishing") is None:
                missing += 1
                continue
            judged += 1
            positives += int(bool(g["worth_publishing"]))
    return {
        "total": total,
        "recalled": recalled,
        "coarse": coarse_n,
        "judged": judged,
        "positive": positives,
        "missing": missing,
    }


def main() -> int:
    gold = load_gold()
    days = sorted(d for d in gold if (OCEAN / d / "events.json").exists())

    print("=== 现公式 ===")
    base = run(gold, days)
    print(base)

    print("\n=== 新公式 ===")
    original = coarse_mod._coarse_score
    coarse_mod._coarse_score = new_score
    try:
        new = run(gold, days)
    finally:
        coarse_mod._coarse_score = original
    print(new)

    def p8(r):
        return (
            f"{r['positive']}/{r['judged']} = {r['positive'] / r['judged']:.2f}"
            if r["judged"]
            else "-"
        )

    print("\n| 配置 | 正例总数 | 召回 | 粗排 | P@8 | 未标注 |")
    print("|---|---|---|---|---|---|")
    print(
        f"| 现公式 | {base['total']} | {base['recalled']} | {base['coarse']} | {p8(base)} | {base['missing']} |"
    )
    print(
        f"| 新公式 | {new['total']} | {new['recalled']} | {new['coarse']} | {p8(new)} | {new['missing']} |"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
