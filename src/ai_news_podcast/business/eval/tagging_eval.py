"""金标准标签评测：按字段算 precision/recall/F1，并报告样本量与置信区间。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_news_podcast.business.eval.stats import wilson_interval

SINGLE_FIELDS = (
    "ai_relevance",
    "primary_domain",
    "industry",
    "primary_action",
    "lifecycle",
)
MULTI_FIELDS = (
    "topics",
    "tech_stack",
    "actions",
    "narrative_angle",
    "audience_persona",
    "sensitivity",
)


@dataclass
class GoldRow:
    event_id: str
    fields: dict[str, Any]
    annotator: str = "human"
    reviewed: bool = True


def load_gold(path: str | Path) -> list[GoldRow]:
    rows: list[GoldRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        rows.append(
            GoldRow(
                event_id=str(data["event_id"]),
                fields=dict(data.get("fields") or {}),
                annotator=str(data.get("annotator") or "human"),
                reviewed=bool(data.get("reviewed", True)),
            )
        )
    return rows


def _predicted(event: Any, field: str) -> Any:
    labels = event.labels if hasattr(event, "labels") else (event.get("labels") or {})
    return getattr(labels, field, None) if not isinstance(labels, dict) else labels.get(field)


def _single_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(gold) | set(pred))
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred, strict=True) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    accuracy = sum(1 for g, p in zip(gold, pred, strict=True) if g == p) / len(gold)
    low, high = wilson_interval(
        sum(1 for g, p in zip(gold, pred, strict=True) if g == p), len(gold)
    )
    macro_f1 = sum(row["f1"] for row in per_class.values()) / len(per_class) if per_class else 0.0
    return {
        "n": len(gold),
        "accuracy": round(accuracy, 4),
        "accuracy_ci": [low, high],
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


def _multi_metrics(gold: list[list[str]], pred: list[list[str]]) -> dict[str, Any]:
    tp = fp = fn = 0
    jaccard = 0.0
    for gold_set, pred_set in zip(gold, pred, strict=True):
        gold_values, pred_values = set(gold_set), set(pred_set)
        tp += len(gold_values & pred_values)
        fp += len(pred_values - gold_values)
        fn += len(gold_values - pred_values)
        union = gold_values | pred_values
        jaccard += len(gold_values & pred_values) / len(union) if union else 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(gold),
        "micro_precision": round(precision, 4),
        "micro_recall": round(recall, 4),
        "micro_f1": round(f1, 4),
        "mean_jaccard": round(jaccard / len(gold), 4) if gold else 0.0,
    }


def evaluate_tagging(
    gold: list[GoldRow],
    events: dict[str, Any],
    *,
    fields: tuple[str, ...] = SINGLE_FIELDS + MULTI_FIELDS,
) -> dict[str, Any]:
    matched = [row for row in gold if row.event_id in events]
    report: dict[str, Any] = {
        "gold": len(gold),
        "matched": len(matched),
        "unmatched": len(gold) - len(matched),
        "fields": {},
    }
    for field in fields:
        rows = [row for row in matched if field in row.fields]
        if not rows:
            continue
        predicted = [_predicted(events[row.event_id], field) for row in rows]
        if field in SINGLE_FIELDS:
            report["fields"][field] = _single_metrics(
                [str(row.fields[field]) for row in rows],
                [str(value or "unknown") for value in predicted],
            )
        else:
            report["fields"][field] = _multi_metrics(
                [list(row.fields[field]) for row in rows],
                [list(value or []) for value in predicted],
            )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 标签金标准评测",
        "",
        f"- 金标准：{report['gold']} 条",
        f"- 匹配：{report['matched']} 条",
        f"- 未匹配：{report['unmatched']} 条（未匹配不计入指标）",
        "",
        "| 字段 | n | 指标 | 值 |",
        "|---|---|---|---|",
    ]
    for field, metrics in report["fields"].items():
        if "accuracy" in metrics:
            lines.append(f"| {field} | {metrics['n']} | accuracy | {metrics['accuracy']:.2%} |")
            low, high = metrics["accuracy_ci"]
            lines.append(f"| {field} | {metrics['n']} | 95% CI | {low:.2%}–{high:.2%} |")
            lines.append(f"| {field} | {metrics['n']} | macro F1 | {metrics['macro_f1']:.3f} |")
        else:
            lines.append(f"| {field} | {metrics['n']} | micro F1 | {metrics['micro_f1']:.3f} |")
            lines.append(
                f"| {field} | {metrics['n']} | 平均 Jaccard | {metrics['mean_jaccard']:.3f} |"
            )
    return "\n".join(lines) + "\n"


__all__ = ["GoldRow", "evaluate_tagging", "load_gold", "render_markdown"]
