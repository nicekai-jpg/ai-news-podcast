"""标签覆盖率审计：每个字段的空白率/数量分布，暴露“词表没覆盖到”的地方。"""

from __future__ import annotations

from collections import Counter
from typing import Any

DEFAULT_FIELDS = (
    "topics",
    "tech_stack",
    "subjects",
    "actions",
    "narrative_angle",
    "audience_persona",
    "sensitivity",
)

# 这些字段缺省是正常的（多数新闻不涉及敏感/叙事角度），不按同一阈值告警。
LOW_EXPECTATION = {"sensitivity"}


def _labels(event: Any) -> Any:
    return event.labels if hasattr(event, "labels") else (event.get("labels") or {})


def _label(event: Any, field: str) -> list[Any]:
    labels = _labels(event)
    value = getattr(labels, field, None) if not isinstance(labels, dict) else labels.get(field)
    return list(value or [])


def _has_evidence(event: Any, field: str) -> bool:
    labels = _labels(event)
    evidence = (
        getattr(labels, "label_evidence", None)
        if not isinstance(labels, dict)
        else labels.get("label_evidence")
    )
    return bool(evidence and evidence.get(field))


def label_coverage(
    events: list[Any],
    *,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
    warn_threshold: float = 0.3,
) -> dict[str, Any]:
    """返回每个字段的空白率、平均标签数、告警。"""
    total = len(events) or 1
    rows: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for field in fields:
        empty = sum(1 for event in events if not _label(event, field))
        counts = Counter(len(_label(event, field)) for event in events)
        rate = round(empty / total, 4)
        with_evidence = sum(1 for event in events if _has_evidence(event, field))
        rows[field] = {
            "empty": empty,
            "total": len(events),
            "empty_rate": rate,
            "evidence_rate": round(with_evidence / total, 4),
            "distribution": dict(sorted(counts.items())),
        }
        if field not in LOW_EXPECTATION and rate > warn_threshold:
            warnings.append(f"{field} 空白率 {rate:.0%} > {warn_threshold:.0%}，词表可能没覆盖")
    return {"events": len(events), "fields": rows, "warnings": warnings}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 标签覆盖率",
        "",
        f"- 事件数：{report['events']}",
        "",
        "> 空白率只是词表覆盖诊断，不等于标签准确率；准确率需要人工金标准。",
        "",
        "| 字段 | 空白数 | 空白率 | 证据覆盖率 | 标签数分布 |",
        "|---|---|---|---|---|",
    ]
    for field, row in report["fields"].items():
        lines.append(
            f"| {field} | {row['empty']} | {row['empty_rate']:.0%} | "
            f"{row['evidence_rate']:.0%} | {row['distribution']} |"
        )
    if report["warnings"]:
        lines += ["", "## 告警", ""]
        lines += [f"- {warning}" for warning in report["warnings"]]
    return "\n".join(lines) + "\n"


__all__ = ["DEFAULT_FIELDS", "label_coverage", "render_markdown"]
