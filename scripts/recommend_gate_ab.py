#!/usr/bin/env python3
"""事件性闸门 A/B：同一批 LLM 打分，只切换 eventness_gate。

为什么要单独写：`recommend_p8.py --llm` 每次都会重新调 LLM（temperature 0.2），
两次运行之间既有「闸门」差异也有「采样」差异，不能归因。这里每天只调一次
LLM，把打分结果缓存下来，再用两份配置分别重排，差异只来自闸门。
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
from ai_news_podcast.business.recommend.config import RecommendConfig  # noqa: E402
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
GOLD = [
    ROOT / "data/eval/recommend/gold_v1_llm.jsonl",
    ROOT / "data/eval/recommend/gold_8day_pool3_llm.jsonl",
]
OCEAN = ROOT / "data/ocean"
CONFIG = ROOT / "config/recommend_v3.yaml"


def load_gold() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for path in GOLD:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out.setdefault(str(row.get("date")), {})[normalize_title(row.get("title"))] = row
    return out


def build_judge():
    import dataclasses

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


def score(gold, day, judge, gate: bool) -> tuple[int, int, int]:
    """用**同一份** LLM 打分重排（judge 带缓存），差异只来自事件性闸门。"""
    events = [
        Event.from_dict(row)
        for row in json.loads((OCEAN / day / "events.json").read_text(encoding="utf-8"))
    ]
    import dataclasses

    cfg = RecommendConfig.load(CONFIG)
    cfg = dataclasses.replace(cfg, fine=dataclasses.replace(cfg.fine, eventness_gate=gate))
    reference = datetime.strptime(day, "%Y-%m-%d").replace(hour=12, tzinfo=UTC)
    result = recommend_with_audit(
        events,
        config_path=CONFIG,
        llm_judge=judge,
        reference_time=reference,
        config=cfg,
    )
    judged = positive = 0
    for entry in result.shortlist:
        event = next(e for e in events if str(e.event_id) == str(entry.event_id))
        row = gold.get(day, {}).get(normalize_title(event.representative_title))
        if not row or row.get("worth_publishing") is None:
            continue
        judged += 1
        positive += int(bool(row["worth_publishing"]))
    excluded = sum(1 for item in result.fine_scores if item.is_event is False)
    if gate:
        # 安全检查：闸门误杀的正例（金标认为值得播，却被判非事件）
        for item in result.fine_scores:
            if item.is_event is not False:
                continue
            row = gold.get(day, {}).get(normalize_title(item.event.representative_title))
            if row and row.get("worth_publishing"):
                print(
                    f"    误杀正例：{item.event.representative_title[:36]}",
                    flush=True,
                )
    return judged, positive, excluded


def _memo_judge(judge):
    """同一份输入只调用一次 LLM（A/B 两组共用打分）。

    service 会按生产逻辑（规则分前 llm_top_k 条）传入事件，这里以 event_id
    序列为键缓存；两次调用的入参一致时直接复用结果。
    """
    cache: dict[tuple[str, ...], dict] = {}

    def wrapped(events):
        key = tuple(str(event.event_id) for event in events)
        if key not in cache:
            cache[key] = judge(events)
        return cache[key]

    return wrapped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="事件性闸门 A/B")
    parser.parse_args(argv)
    gold = load_gold()
    judge = build_judge()
    tot = {"off": [0, 0], "on": [0, 0]}
    print(f"{'日期':12}{'关闸门':>12}{'开闸门':>12}{'剔除':>6}")
    for day in DAYS:
        # 每天只调一次 LLM：两层都走同一个缓存 judge，命中同一批 event_id
        cached = _memo_judge(judge)
        off_j, off_p, _ = score(gold, day, cached, gate=False)
        on_j, on_p, excluded = score(gold, day, cached, gate=True)
        tot["off"][0] += off_j
        tot["off"][1] += off_p
        tot["on"][0] += on_j
        tot["on"][1] += on_p
        f = lambda p, j: f"{p}/{j}={p / j:.2f}" if j else "-"  # noqa: E731
        print(f"{day:12}{f(off_p, off_j):>12}{f(on_p, on_j):>12}{excluded:>6}")
    for key, label in (("off", "关闸门"), ("on", "开闸门")):
        p, j = tot[key][1], tot[key][0]
        print(f"{label}合计 {p}/{j} = {p / j:.2f}" if j else f"{label} 无样本")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
