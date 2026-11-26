#!/usr/bin/env python3
"""标注金标抽检清单：按「规则与金标分歧度」挑 30 条，供人工核对。

为什么按分歧度挑：随机抽样大部分是一致的，看不出问题；分歧大的样本
才暴露「到底是规则错了还是 LLM 金标错了」。

用法：
    uv run python scripts/label_spotcheck.py --per-class 15 --out data/eval/label_spotcheck.md
    uv run python scripts/label_spotcheck.py --apply   # 按清单里的判定回写 reviewed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GOLD = ROOT / "data/eval/label_gold_llm.jsonl"
OCEAN = ROOT / "data" / "ocean"

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


def to_enum(values, mapping) -> set[str]:
    out: set[str] = set()
    for value in values or []:
        low = str(value).lower()
        for key, triggers in mapping.items():
            if any(trigger in low for trigger in triggers):
                out.add(key)
    return out


def load_events() -> dict:
    from ai_news_podcast.business.contracts import Event

    events = {}
    for day_dir in sorted(OCEAN.iterdir()):
        path = day_dir / "events.json"
        if not path.is_file():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            events[str(row["event_id"])] = Event.from_dict(row)
    return events


def divergence(gold_row, event) -> int:
    """分歧度：金标与规则在 tech/topics 上的对称差大小。"""
    gold_tech = to_enum(gold_row["fields"].get("tech_stack"), TECH_MAP)
    gold_topics = to_enum(gold_row["fields"].get("topics"), TOPIC_MAP)
    pred_tech = set(event.labels.tech_stack or [])
    pred_topics = set(event.labels.topics or [])
    return len(gold_tech ^ pred_tech) + len(gold_topics ^ pred_topics)


def classify(item: dict) -> str:  # noqa: PLR0911 - 分类规则本身就是多分支
    """把分歧归到可读的类别，方便分组审阅。"""
    event = item["event"]
    gold = item["gold"]["fields"]
    gold_subs = {str(s).lower() for s in (gold.get("subjects") or [])}
    pred_subs = {
        str(s.get("name") or s.get("id") or "").lower() for s in (event.labels.subjects or [])
    }
    gold_tech = to_enum(gold.get("tech_stack"), TECH_MAP)
    pred_tech = set(event.labels.tech_stack or [])
    gold_topics = to_enum(gold.get("topics"), TOPIC_MAP)
    pred_topics = set(event.labels.topics or [])

    if pred_subs and not (pred_subs & gold_subs) and gold_subs:
        return "主体完全错配"
    if not pred_subs and gold_subs:
        return "主体漏标"
    if len(pred_tech - gold_tech) >= 3:
        return "技术过度标注"
    if len(gold_tech - pred_tech) >= 3:
        return "技术漏标"
    if len(pred_topics - gold_topics) >= 3:
        return "类别过度标注"
    if len(gold_topics - pred_topics) >= 3:
        return "类别漏标"
    return "其他分歧"


def _snippet(event, limit: int = 180) -> str:
    """取事件正文片段，供人工判断标注是否合理。"""
    text = str(event.summary or "").strip()
    if not text:
        return "（无摘要）"
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def render(items: list[dict]) -> str:
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(classify(item), []).append(item)

    order = [
        "主体完全错配",
        "主体漏标",
        "技术过度标注",
        "技术漏标",
        "类别过度标注",
        "类别漏标",
        "其他分歧",
    ]
    lines = [
        "# 标注金标抽检（30 条）",
        "",
        "> 按「规则与 LLM 金标的分歧」分组，每组内按分歧度降序。",
        "> 每条在「判定」处填 `对`（金标正确）/ `错`（金标有误）/ `部分`，并写一句备注。",
        "> **你只需判断「LLM 金标这一栏是否合理」，不必逐字校对规则。**",
        "> 填完运行 `uv run python scripts/label_spotcheck.py --apply` 回写。",
        "",
        "## 分组概览",
        "",
        "| 分组 | 条数 | 含义 |",
        "|---|---|---|",
    ]
    meaning = {
        "主体完全错配": "规则标的主体与金标完全不重合，通常是规则误命中",
        "主体漏标": "规则没标出主体",
        "技术过度标注": "规则标的技术比金标多 3 个以上，多为裸词误命中",
        "技术漏标": "金标有而规则缺",
        "类别过度标注": "规则类别比金标多 3 个以上",
        "类别漏标": "金标类别有而规则缺",
        "其他分歧": "差异较小或方向混合",
    }
    lines.extend(
        f"| {key} | {len(groups[key])} | {meaning[key]} |" for key in order if key in groups
    )
    lines.append("")

    index = 0
    for key in order:
        if key not in groups:
            continue
        lines.append(f"## {key}（{len(groups[key])} 条）")
        lines.append("")
        for item in groups[key]:
            index += 1
            event = item["event"]
            gold = item["gold"]["fields"]
            pred_subs = [s.get("name") for s in (event.labels.subjects or [])]
            lines.append(f"### {index}. {event.representative_title}")
            lines.append("")
            lines.append(
                f"- **ID**：`{item['gold']['event_id'][:8]}`　**分歧度**：{item['divergence']}"
            )
            lines.append(f"- **摘要**：{_snippet(event)}")
            lines.append("")
            lines.append("| | 主体 | 技术 | 类别 |")
            lines.append("|---|---|---|---|")
            lines.append(
                f"| **LLM 金标** | {', '.join(map(str, gold.get('subjects') or [])) or '（空）'} "
                f"| {', '.join(map(str, gold.get('tech_stack') or [])) or '（空）'} "
                f"| {', '.join(map(str, gold.get('topics') or [])) or '（空）'} |"
            )
            lines.append(
                f"| **规则产出** | {', '.join(str(s) for s in pred_subs if s) or '（空）'} "
                f"| {', '.join(event.labels.tech_stack or []) or '（空）'} "
                f"| {', '.join(event.labels.topics or []) or '（空）'} |"
            )
            lines.append("")
            lines.append("- **判定**：　　（填 `对` / `错` / `部分`）")
            lines.append("- **备注**：")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="标注金标抽检")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--out", default=str(ROOT / "data/eval/label_spotcheck.md"))
    parser.add_argument("--apply", action="store_true", help="按清单判定回写 reviewed")
    args = parser.parse_args(argv)

    # 同一 event_id 可能被标注多次（跨天重复），抽检只保留第一条
    seen: set[str] = set()
    gold: list[dict] = []
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        event_id = str(row.get("event_id") or "")
        if not event_id or event_id in seen:
            continue
        seen.add(event_id)
        gold.append(row)
    events = load_events()
    scored = []
    for row in gold:
        event = events.get(str(row["event_id"]))
        if event is None:
            continue
        scored.append({"gold": row, "event": event, "divergence": divergence(row, event)})
    scored.sort(key=lambda item: (-item["divergence"], item["gold"]["event_id"]))
    picked = scored[: args.count]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(picked), encoding="utf-8")
    print(f"抽检清单 -> {out}（{len(picked)} 条）")
    print(f"分歧度范围：{picked[0]['divergence']} ~ {picked[-1]['divergence']}")
    if args.apply:
        print("--apply 需先人工填入「判定」列，本次未回写")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
