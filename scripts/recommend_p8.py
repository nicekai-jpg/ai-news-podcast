#!/usr/bin/env python3
"""推荐 P@k 离线评测（用 LLM 初标金标的 worth_publishing）。

与 ``scripts/eval_recommend.py`` 的区别：
- 那个按 ``ai_relevance`` 判正样本（等于评“是不是 AI 新闻”）；
- 这个按 ``worth_publishing`` 判（等于评“是不是值得播”），更接近真实目标。

用法：
    uv run python scripts/recommend_p8.py
    uv run python scripts/recommend_p8.py --days 2026-11-09 2026-11-10 --k 8
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_news_podcast.business.contracts import Event  # noqa: E402
from ai_news_podcast.business.eval.recommend_eval import normalize_title  # noqa: E402
from ai_news_podcast.business.recommend.service import recommend  # noqa: E402

GOLD = ROOT / "data" / "eval" / "recommend" / "gold_v1_llm.jsonl"
OCEAN = ROOT / "data" / "ocean"


def load_gold(paths: list[Path] | None = None) -> dict[str, dict[str, dict]]:
    """date -> normalized title -> gold row（多个文件合并，后者覆盖前者）。"""
    by_date: dict[str, dict[str, dict]] = {}
    for path in paths or [GOLD]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            by_date.setdefault(str(row.get("date")), {})[normalize_title(row.get("title"))] = row
    return by_date


def build_llm_judge() -> object:
    """按生产配置构造 Stage 3 的 LLM rubric judge（需要 MINIMAX_API_KEY）。"""
    import dataclasses

    import yaml

    from ai_news_podcast.business.pipeline.v3_runner_service import (
        make_llm_call,
        make_llm_judge,
    )
    from ai_news_podcast.data.config.loader_dao import load_config as load_app_config

    cfg = load_app_config(ROOT / "config" / "config.yaml")
    v3 = yaml.safe_load((ROOT / "config" / "pipeline_v3.yaml").read_text(encoding="utf-8")) or {}
    llm_section = v3.get("llm") or {}
    stage_models = llm_section.get("stage_models") or {}
    stage_options = llm_section.get("stage_options") or {}
    llm_cfg = cfg.llm
    if stage_models:
        llm_cfg = dataclasses.replace(
            cfg.llm, stage_models={**cfg.llm.stage_models, **stage_models}
        )
    return make_llm_judge(make_llm_call(llm_cfg, "stage3", stage_options.get("stage3")))


def eval_day(day: str, gold: dict[str, dict], k: int, llm_judge: object = None) -> dict:
    events_path = OCEAN / day / "events.json"
    if not events_path.exists():
        return {"day": day, "error": "no events"}
    events = [Event.from_dict(row) for row in json.loads(events_path.read_text(encoding="utf-8"))]
    reference = datetime.strptime(day, "%Y-%m-%d").replace(hour=12, tzinfo=UTC)
    shortlist = recommend(
        events,
        config_path=ROOT / "config" / "recommend_v3.yaml",
        reference_time=reference,
        llm_judge=llm_judge,
    )
    judged = 0
    positive = 0
    picks: list[dict] = []
    for entry in shortlist[:k]:
        event = next((e for e in events if str(e.event_id) == str(entry.event_id)), None)
        row = gold.get(normalize_title(event.representative_title)) if event else None
        if row is None or row.get("worth_publishing") is None:
            picks.append({"title": event.representative_title[:30] if event else "?", "gold": None})
            continue
        judged += 1
        worth = bool(row["worth_publishing"])
        positive += int(worth)
        picks.append({"title": event.representative_title[:30], "gold": worth})
    return {
        "day": day,
        "shortlist": len(shortlist[:k]),
        "judged": judged,
        "positive": positive,
        "p_at_k": (positive / judged) if judged else None,
        "picks": picks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="推荐 P@k 评测")
    parser.add_argument("--days", nargs="+", default=["2026-11-09", "2026-11-10", "2026-11-11"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--show-picks", action="store_true")
    parser.add_argument(
        "--llm", action="store_true", help="启用生产同款 Stage 3 LLM 精排（需 MINIMAX_API_KEY）"
    )
    parser.add_argument(
        "--gold",
        nargs="+",
        default=[str(GOLD)],
        help="金标 JSONL（可多个；默认 3 天全量金标）",
    )
    args = parser.parse_args(argv)

    gold = load_gold([Path(path) for path in args.gold])
    judge = build_llm_judge() if args.llm else None
    rows = [eval_day(day, gold.get(day, {}), args.k, judge) for day in args.days]
    total_judged = sum(r.get("judged", 0) for r in rows)
    total_positive = sum(r.get("positive", 0) for r in rows)
    print(f"{'日期':12}{'shortlist':>10}{'已判':>6}{'正例':>6}{'P@k':>8}")
    for r in rows:
        if r.get("error"):
            print(f"{r['day']:12}  {r['error']}")
            continue
        p = f"{r['p_at_k']:.2f}" if r["p_at_k"] is not None else "-"
        print(f"{r['day']:12}{r['shortlist']:>10}{r['judged']:>6}{r['positive']:>6}{p:>8}")
        if args.show_picks:
            for pick in r["picks"]:
                mark = {True: "✓", False: "✗", None: "?"}[pick["gold"]]
                print(f"    {mark} {pick['title']}")
    if total_judged:
        print(
            f"{'合计':12}{'':>10}{total_judged:>6}{total_positive:>6}{total_positive / total_judged:>8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
