#!/usr/bin/env python3
"""规则分 vs LLM 精排：8 天 P@8 对比（LLM 打分磁盘缓存）。

流程：
1. 每天按生产逻辑取规则分前 llm_top_k 条，调用一次 LLM rubric；
2. 结果写入缓存 JSON，之后重跑不再调用 LLM；
3. 分别用「纯规则」和「规则+LLM」重排，算 P@8；
4. --queue-out 写出短名单里金标缺失的标题，便于补标后再对比。

用法：
    MINIMAX_API_KEY=... uv run python scripts/recommend_llm_vs_rules.py --queue-out /tmp/q.jsonl
    uv run python scripts/recommend_llm_vs_rules.py      # 命中缓存，秒出
"""

from __future__ import annotations

import argparse
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
from ai_news_podcast.business.recommend.fine import score_events  # noqa: E402
from ai_news_podcast.business.recommend.service import recommend_with_audit  # noqa: E402

DAYS = [
    "2026-11-09",
    "2026-11-10",
    "2026-11-11",
    "2026-11-12",
    "2026-11-13",
    "2026-11-14",
    "2026-11-15",
    "2026-11-16",
]
OCEAN = ROOT / "data" / "ocean"
CONFIG = ROOT / "config" / "recommend_v3.yaml"
CACHE = ROOT / "data" / "eval" / "recommend" / "llm_judgements_8day.json"
GOLD = [
    ROOT / "data/eval/recommend/gold_v1_llm.jsonl",
    ROOT / "data/eval/recommend/gold_8day_pool4_llm.jsonl",
]


def load_gold() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for path in GOLD:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out.setdefault(str(row.get("date")), {})[normalize_title(row.get("title"))] = row
    return out


def load_events(day: str) -> list[Event]:
    path = OCEAN / day / "events.json"
    return [Event.from_dict(row) for row in json.loads(path.read_text(encoding="utf-8"))]


def build_judge_call():
    import yaml

    from ai_news_podcast.business.pipeline.v3_runner_service import (
        make_llm_call,
        make_llm_judge,
    )
    from ai_news_podcast.data.config.loader_dao import load_config as load_app_config

    cfg = load_app_config(ROOT / "config" / "config.yaml")
    v3 = yaml.safe_load((ROOT / "config" / "pipeline_v3.yaml").read_text(encoding="utf-8")) or {}
    section = v3.get("llm") or {}
    models = section.get("stage_models") or {}
    options = section.get("stage_options") or {}
    llm_cfg = cfg.llm
    if models:
        llm_cfg = dataclasses.replace(cfg.llm, stage_models={**cfg.llm.stage_models, **models})
    return make_llm_judge(make_llm_call(llm_cfg, "stage3", options.get("stage3")))


def judgements_for(day: str, cache: dict, cfg: RecommendConfig, use_llm: bool) -> dict:
    """取当天的 LLM 打分：优先缓存，缺失时调用一次并写回。"""
    if not use_llm:
        return {}
    if day in cache:
        return cache[day]
    events = load_events(day)
    scored = score_events(events, cfg.fine)[: cfg.fine.llm_top_k]
    judge = build_judge_call()
    cache[day] = judge([item.event for item in scored])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return cache[day]


def evaluate(gold, cfg, cache, use_llm: bool):
    judged = positive = 0
    missing: list[dict] = []
    repeats: list[str] = []
    seen: set[str] = set()
    for day in DAYS:
        events = load_events(day)
        judgements = judgements_for(day, cache, cfg, use_llm)
        reference = datetime.strptime(day, "%Y-%m-%d").replace(hour=12, tzinfo=UTC)
        result = recommend_with_audit(
            events,
            config_path=CONFIG,
            llm_judge=(lambda _events, j=judgements: j) if use_llm else None,
            reference_time=reference,
            config=cfg,
        )
        by_id = {str(event.event_id): event for event in events}
        for entry in result.shortlist:
            event = by_id[str(entry.event_id)]
            if event.event_id in seen:
                repeats.append(event.representative_title[:30])
            seen.add(event.event_id)
            row = gold.get(day, {}).get(normalize_title(event.representative_title))
            if not row or row.get("worth_publishing") is None:
                missing.append(
                    {
                        "date": day,
                        "title": event.representative_title,
                        "summary": event.summary or "",
                        "suggested_ai_relevance": "unknown",
                        "suggested_worth_publishing": None,
                    }
                )
                continue
            judged += 1
            positive += int(bool(row["worth_publishing"]))
    return judged, positive, missing, repeats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="规则 vs LLM 精排 P@8")
    parser.add_argument("--queue-out", default="", help="写出金标缺失标题的 JSONL 路径")
    parser.add_argument("--no-llm", action="store_true", help="不调用/不使用 LLM（纯规则）")
    args = parser.parse_args(argv)

    gold = load_gold()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    cfg = RecommendConfig.load(CONFIG)

    rows = {}
    all_missing: dict[tuple[str, str], dict] = {}
    for label, use_llm in (("规则", False), ("规则+LLM", True)):
        if args.no_llm and use_llm:
            continue
        judged, positive, missing, repeats = evaluate(gold, cfg, cache, use_llm)
        rows[label] = (judged, positive)
        rate = f"{positive / judged:.2f}" if judged else "-"
        print(
            f"{label:10} P@8 {positive}/{judged} = {rate}"
            f" | 金标缺失 {len(missing)} | 跨天重复 {len(repeats)}"
        )
        for row in missing:
            all_missing[(row["date"], row["title"])] = row

    if args.queue_out and all_missing:
        out = Path(args.queue_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(row, ensure_ascii=False) for row in all_missing.values()]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"两组缺失并集 -> {out}（{len(lines)} 条，补标后可公平对比）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
