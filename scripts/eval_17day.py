#!/usr/bin/env python3
"""T1/T2 评测脚本：17 天推荐 P@8 + 标注 P/R。

用法：
    python scripts/eval_17day.py            # 只算推荐 P@8
    python scripts/eval_17day.py --tags     # 同时算标注 P/R（需 label_gold_llm.jsonl）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OCEAN = ROOT / "data" / "ocean"
RECO_GOLD = [
    ROOT / "data/eval/recommend/gold_v1_llm.jsonl",
    ROOT / "data/eval/recommend/gold_8day_pool4_llm.jsonl",
    ROOT / "data/eval/recommend/gold_coarse_grid_llm.jsonl",
    ROOT / "data/eval/recommend/gold_17day_llm.jsonl",
]
LABEL_GOLD = ROOT / "data/eval/label_gold_llm.jsonl"

# 金标（LLM 自由标注）→ 规则受控枚举 的映射
TECH_MAP = {
    "llm": ["大语言模型", "语言模型", "llm"],
    "agent": ["智能体", "agent"],
    "multimodal": ["多模态", "multimodal"],
    "reasoning": ["推理", "reasoning"],
    "training": ["训练"],
    "inference": ["推理服务", "推理引擎", "inference"],
    "chip": ["芯片", "gpu", "tpu", "算力"],
    "robot": ["机器人", "具身"],
    "autonomous_driving": ["自动驾驶", "智驾", "fsd"],
    "speech": ["语音", "音频", "speech"],
    "vision": ["视觉", "图像", "视频", "vision"],
    "coding": ["代码", "编程", "coding"],
    "rag": ["检索增强", "rag"],
    "safety": ["安全", "隐私", "越狱", "对齐"],
}
TOPIC_MAP = {
    "policy": ["政策", "监管", "法规"],
    "safety": ["安全", "隐私"],
    "business": ["商业", "营收", "市场"],
    "capital": ["融资", "投资", "估值", "ipo", "收购"],
    "research": ["研究", "论文", "成果", "基准"],
    "product": ["发布", "产品", "上线", "开源"],
    "technology": ["技术"],
    "industry": ["行业", "产业"],
    "hardware": ["硬件", "设备", "芯片"],
    "open_source": ["开源"],
    "infrastructure": ["基础设施", "算力", "云"],
    "application": ["应用", "落地"],
}


def _to_enum(values, mapping) -> set[str]:
    out: set[str] = set()
    for value in values or []:
        low = str(value).lower()
        for key, triggers in mapping.items():
            if any(trigger in low for trigger in triggers):
                out.add(key)
    return out


def _load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _ocean_days() -> list[str]:
    return sorted(p.name for p in OCEAN.iterdir() if p.is_dir())


def eval_recommend(days: list[str]) -> dict:
    from ai_news_podcast.business.contracts import Event
    from ai_news_podcast.business.eval.recommend_eval import normalize_title
    from ai_news_podcast.business.recommend.service import recommend

    gold: dict[str, dict[str, dict]] = {}
    for path in RECO_GOLD:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            gold.setdefault(str(row.get("date")), {})[normalize_title(row.get("title"))] = row

    rows = []
    total_j = total_p = 0
    for day in days:
        path = OCEAN / day / "events.json"
        if not path.is_file():
            continue
        events = [Event.from_dict(r) for r in json.loads(path.read_text(encoding="utf-8"))]
        by_id = {str(e.event_id): e for e in events}
        shortlist = recommend(
            events,
            config_path=ROOT / "config" / "recommend_v3.yaml",
            reference_time=datetime.strptime(day, "%Y-%m-%d").replace(hour=12, tzinfo=UTC),
        )
        judged = positive = 0
        for entry in shortlist:
            event = by_id[str(entry.event_id)]
            row = gold.get(day, {}).get(normalize_title(event.representative_title))
            if not row or row.get("worth_publishing") is None:
                continue
            judged += 1
            positive += int(bool(row["worth_publishing"]))
        total_j += judged
        total_p += positive
        rows.append((day, judged, positive))
    return {"rows": rows, "judged": total_j, "positive": total_p}


def eval_tags() -> dict | None:
    from ai_news_podcast.business.contracts import Event
    from ai_news_podcast.business.eval.tagging_eval import _multi_metrics, load_gold

    if not LABEL_GOLD.is_file():
        return None
    gold = load_gold(LABEL_GOLD)
    events: dict[str, Event] = {}
    for day in _ocean_days():
        path = OCEAN / day / "events.json"
        if not path.is_file():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            events[str(row["event_id"])] = Event.from_dict(row)
    matched = [r for r in gold if r.event_id in events]
    out: dict[str, dict] = {}
    for field, mapping in (("tech_stack", TECH_MAP), ("topics", TOPIC_MAP)):
        gold_sets = [_to_enum(r.fields.get(field), mapping) for r in matched]
        pred_sets = [set(getattr(events[r.event_id].labels, field) or []) for r in matched]
        out[field] = _multi_metrics([sorted(g) for g in gold_sets], [sorted(p) for p in pred_sets])
        # 精度损失来源
        fp: Counter[str] = Counter()
        for g, p in zip(gold_sets, pred_sets, strict=True):
            for item in p - g:
                fp[item] += 1
        out[field]["top_false_positives"] = fp.most_common(6)
    return {"gold": len(gold), "matched": len(matched), "fields": out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="17 天推荐与标注评测")
    parser.add_argument("--tags", action="store_true", help="同时计算标注 P/R")
    args = parser.parse_args(argv)

    reco = eval_recommend(_ocean_days())
    print("## 推荐 P@8\n")
    print("| 日期 | 已判 | 正例 | P@8 |")
    print("|---|---|---|---|")
    for day, judged, positive in reco["rows"]:
        rate = f"{positive / judged:.2f}" if judged else "-"
        print(f"| {day} | {judged} | {positive} | {rate} |")
    rate = reco["positive"] / reco["judged"] if reco["judged"] else 0
    print(f"| **合计** | **{reco['judged']}** | **{reco['positive']}** | **{rate:.2f}** |")

    if args.tags:
        tags = eval_tags()
        if not tags:
            print("\n标注金标缺失，跳过")
            return 0
        print(f"\n## 标注 P/R（金标 {tags['gold']}，匹配 {tags['matched']}）\n")
        print("| 字段 | 精度 | 召回 | F1 | Jaccard |")
        print("|---|---|---|---|---|")
        for field, metrics in tags["fields"].items():
            print(
                f"| {field} | {metrics['micro_precision']:.2f} | "
                f"{metrics['micro_recall']:.2f} | {metrics['micro_f1']:.2f} | "
                f"{metrics['mean_jaccard']:.2f} |"
            )
        print("\n精度损失主要来源（规则标出但金标没有）：\n")
        for field, metrics in tags["fields"].items():
            top = "、".join(f"{k}({v})" for k, v in metrics["top_false_positives"])
            print(f"- **{field}**：{top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
