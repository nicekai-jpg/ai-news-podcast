# ruff: noqa: SLF001
#!/usr/bin/env python3
"""粗排公式变体网格 + 未标注条目收集。


一次跑多个权重变体，收集所有变体 shortlist 里缺金标的标题，
统一补标后即可公平对比（避免「先跑先标」的偏置）。

用法：
    uv run python scripts/recommend_coarse_grid.py --queue-out /tmp/q.jsonl
"""

from __future__ import annotations

import argparse
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

CONFIG = ROOT / "config" / "recommend_v3.yaml"
OCEAN = ROOT / "data" / "ocean"
GOLD = [
    ROOT / "data/eval/recommend/gold_v1_llm.jsonl",
    ROOT / "data/eval/recommend/gold_8day_pool4_llm.jsonl",
    ROOT / "data/eval/recommend/gold_coarse_grid_llm.jsonl",
]

# 变体：significance / numbers / tech / actions / 是否保留来源项 / 是否保留单源惩罚
VARIANTS: dict[str, dict] = {
    "baseline": {},
    "drop_single_source_penalty": {"single_source_penalty": 0.0},
    "novelty_heavy": {"w": (1.2, 1.2, 0.6, 0.4), "drop_source": True, "no_ss": True},
    "novelty_mid": {"w": (1.2, 0.8, 0.5, 0.4), "drop_source": True, "no_ss": True},
    "keep_source_novelty": {"w": (1.2, 0.8, 0.5, 0.4), "drop_source": False, "no_ss": True},
    "sig_only_boost": {"w": (1.8, 0.0, 0.0, 0.0), "drop_source": False, "no_ss": True},
}


def make_scorer(spec: dict):
    def scorer(candidate, settings, reference_time):
        event = candidate.event
        labels = event.labels
        reasons: list[str] = []
        score = 0.0

        if spec.get("w"):
            ws, wn, wt, wa = spec["w"]
            significance = max_subject_significance(labels)
            score += ws * significance
            score += wn * (min(len(labels.novelty_numbers), 6) / 6)
            score += wt * (min(len(labels.tech_stack), 5) / 5)
            score += wa * (min(len(labels.actions), 4) / 4)
            if not spec.get("drop_source"):
                score += 1.2 * (min(event.independent_sources, 5) / 5)
                score += 0.6 * (min(event.source_count, 8) / 8)
            if not spec.get("no_ss") and event.independent_sources <= 1:
                score -= settings.single_source_penalty
        elif "single_source_penalty" in spec:
            import dataclasses as _dc

            return original_scorer(
                candidate,
                _dc.replace(settings, single_source_penalty=spec["single_source_penalty"]),
                reference_time,
            )
        elif not spec:
            # baseline 直接复用原实现
            return original_scorer(candidate, settings, reference_time)

        age = event_age_hours(event, reference_time)
        if age is None:
            score += 0.5
        elif age <= 48:
            score += 1.0
        elif age <= 168:
            score += 0.6
        else:
            score += 0.2

        score += _LIFECYCLE_BONUS.get(labels.lifecycle, 0.0)
        score += _EVIDENCE_BONUS.get(labels.evidence_tier, 0.0)
        # 复用生产实现，保证只有权重差异
        score += coarse_mod._priority_action_bonus(labels, settings.priority_actions)
        penalty, pen_reasons = coarse_mod._demote_penalties(event, labels, settings)
        if penalty:
            score -= penalty
            reasons.extend(pen_reasons)
        return max(score, 0.0), reasons or ["通过粗排门槛"]

    return scorer


original_scorer = coarse_mod._coarse_score


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


def evaluate(gold, days) -> tuple[int, int, int]:
    from ai_news_podcast.business.contracts import Event
    from ai_news_podcast.business.eval.recommend_eval import normalize_title
    from ai_news_podcast.business.recommend.config import RecommendConfig
    from ai_news_podcast.business.recommend.service import recommend_with_audit

    cfg = RecommendConfig.load(CONFIG)
    judged = positive = missing = 0
    for day in days:
        events = [
            Event.from_dict(row)
            for row in json.loads((OCEAN / day / "events.json").read_text(encoding="utf-8"))
        ]
        reference = datetime.strptime(day, "%Y-%m-%d").replace(hour=12, tzinfo=UTC)
        result = recommend_with_audit(
            events, config_path=CONFIG, reference_time=reference, config=cfg
        )
        by_id = {str(e.event_id): e for e in events}
        for entry in result.shortlist:
            event = by_id[str(entry.event_id)]
            g = gold.get(day, {}).get(normalize_title(event.representative_title))
            if not g or g.get("worth_publishing") is None:
                missing += 1
                continue
            judged += 1
            positive += int(bool(g["worth_publishing"]))
    return judged, positive, missing


def collect_missing(gold, days) -> dict[tuple[str, str], dict]:
    from ai_news_podcast.business.contracts import Event
    from ai_news_podcast.business.eval.recommend_eval import normalize_title
    from ai_news_podcast.business.recommend.config import RecommendConfig
    from ai_news_podcast.business.recommend.service import recommend_with_audit

    cfg = RecommendConfig.load(CONFIG)
    out: dict[tuple[str, str], dict] = {}
    for day in days:
        events = [
            Event.from_dict(row)
            for row in json.loads((OCEAN / day / "events.json").read_text(encoding="utf-8"))
        ]
        reference = datetime.strptime(day, "%Y-%m-%d").replace(hour=12, tzinfo=UTC)
        result = recommend_with_audit(
            events, config_path=CONFIG, reference_time=reference, config=cfg
        )
        by_id = {str(e.event_id): e for e in events}
        for entry in result.shortlist:
            event = by_id[str(entry.event_id)]
            if gold.get(day, {}).get(normalize_title(event.representative_title)):
                continue
            out[(day, event.representative_title)] = {
                "date": day,
                "title": event.representative_title,
                "summary": event.summary or "",
                "suggested_ai_relevance": "unknown",
                "suggested_worth_publishing": None,
            }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="粗排公式网格")
    parser.add_argument("--queue-out", default="")
    args = parser.parse_args(argv)

    gold = load_gold()
    days = sorted(d for d in gold if (OCEAN / d / "events.json").exists())

    print(f"{'变体':28}{'已判':>6}{'正例':>6}{'P@8':>8}{'缺失':>6}")
    all_missing: dict[tuple[str, str], dict] = {}
    for name, spec in VARIANTS.items():
        coarse_mod._coarse_score = original_scorer if not spec else make_scorer(spec)
        try:
            judged, positive, missing = evaluate(gold, days)
            for key, row in collect_missing(gold, days).items():
                all_missing.setdefault(key, row)
        finally:
            coarse_mod._coarse_score = original_scorer
        rate = f"{positive / judged:.2f}" if judged else "-"
        print(f"{name:28}{judged:>6}{positive:>6}{rate:>8}{missing:>6}")

    if args.queue_out and all_missing:
        out = Path(args.queue_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in all_missing.values()) + "\n",
            encoding="utf-8",
        )
        print(f"\n所有变体缺失并集 -> {out}（{len(all_missing)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
