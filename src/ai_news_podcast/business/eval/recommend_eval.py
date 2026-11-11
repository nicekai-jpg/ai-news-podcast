"""推荐流程离线评测：用人工标注的标题回归召回/粗排/精排/top8。

标注文件是 JSONL，每行一条：
    {"title": "...", "ai_relevance": "core|adjacent|none|unknown",
     "worth_publishing": true|false|null, "label_source": "human|assistant_seed",
     "reviewed": true|false, "note": ""}

匹配方式：按标题归一化精确匹配事件 representative_title；匹配不到的在报告里列出，
提醒标注/事件生成有问题（例如合并事件标题被改写）。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_news_podcast.business.recommend.service import recommend_with_audit

VALID_RELEVANCE = ("core", "adjacent", "none", "unknown")
POSITIVE_RELEVANCE = ("core", "adjacent")
DEFAULT_CONFIG_PATH = "config/recommend_v3.yaml"


@dataclass(frozen=True)
class EvalLabel:
    """一条人工/助审标注。"""

    title: str
    ai_relevance: str = "unknown"
    worth_publishing: bool | None = None
    label_source: str = "human"
    reviewed: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if self.ai_relevance not in VALID_RELEVANCE:
            raise ValueError(f"非法 ai_relevance: {self.ai_relevance!r}")
        if not self.title.strip():
            raise ValueError("标注 title 不能为空")


def normalize_title(text: str) -> str:
    """标题归一化：去空白/标点差异，便于事件与标注匹配。"""
    text = str(text or "").strip().lower()
    return re.sub(r"[\s\-—_·:：,，。.!！?？\"'“”‘’()（）\[\]【】|]+", "", text)


def load_labels(path: str | Path) -> list[EvalLabel]:
    """读取 JSONL 标注文件。"""
    labels: list[EvalLabel] = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} 不是合法 JSON: {exc}") from exc
        labels.append(
            EvalLabel(
                title=str(row.get("title") or ""),
                ai_relevance=str(row.get("ai_relevance") or "unknown"),
                worth_publishing=row.get("worth_publishing"),
                label_source=str(row.get("label_source") or "human"),
                reviewed=bool(row.get("reviewed", True)),
                note=str(row.get("note") or ""),
            )
        )
    return labels


def match_labels(
    events: list[Any], labels: list[EvalLabel]
) -> tuple[dict[str, EvalLabel], list[str]]:
    """事件标题 ↔ 标注标题精确匹配；返回 event_id→label 与未匹配标题。"""
    by_title = {normalize_title(event.representative_title): event for event in events}
    matched: dict[str, EvalLabel] = {}
    unmatched: list[str] = []
    for label in labels:
        event = by_title.get(normalize_title(label.title))
        if event is None:
            unmatched.append(label.title)
        else:
            matched[str(event.event_id)] = label
    return matched, unmatched


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def evaluate(
    events: list[Any],
    labels: list[EvalLabel],
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """跑完整推荐链路并计算分阶段指标。"""
    reference = reference_time or datetime.now(tz=UTC)
    result = recommend_with_audit(
        events,
        config_path=config_path,
        llm_judge=None,
        reference_time=reference,
    )
    matched, unmatched = match_labels(events, labels)
    recalled = {str(item.event.event_id) for item in result.recalled}
    coarse_kept = {str(item.event.event_id) for item in result.coarse.candidates}
    shortlist = {str(entry.event_id) for entry in result.shortlist}

    positives = {eid for eid, label in matched.items() if label.ai_relevance in POSITIVE_RELEVANCE}
    negatives = {eid for eid, label in matched.items() if label.ai_relevance == "none"}
    unknown = {eid for eid, label in matched.items() if label.ai_relevance == "unknown"}

    def _stage_filter(ids: set[str], keep: set[str]) -> set[str]:
        return ids & keep

    shortlist_positive = _stage_filter(positives, shortlist)
    shortlist_negative = _stage_filter(negatives, shortlist)
    shortlist_unknown = _stage_filter(unknown, shortlist)

    stages = {
        "recall": {
            "positives_recalled": len(_stage_filter(positives, recalled)),
            "positives_total": len(positives),
            "negative_leaked": len(_stage_filter(negatives, recalled)),
            "negative_total": len(negatives),
        },
        "coarse": {
            "positives_kept": len(_stage_filter(positives, coarse_kept)),
            "positives_from_recall": len(_stage_filter(positives, recalled)),
            "negative_leaked": len(_stage_filter(negatives, coarse_kept)),
            "negative_from_recall": len(_stage_filter(negatives, recalled)),
        },
        "shortlist": {
            "total": len(shortlist),
            "positives": len(shortlist_positive),
            "negatives": len(shortlist_negative),
            "unknown": len(shortlist_unknown),
        },
    }

    metrics = {
        "label_coverage": {
            "labeled": len(labels),
            "matched": len(matched),
            "unmatched": len(unmatched),
            "reviewed": sum(1 for label in labels if label.reviewed),
        },
        "recall": {
            "positive_recall": _ratio(
                stages["recall"]["positives_recalled"], stages["recall"]["positives_total"]
            ),
            "negative_leak_rate": _ratio(
                stages["recall"]["negative_leaked"], stages["recall"]["negative_total"]
            ),
        },
        "coarse": {
            "positive_keep_rate": _ratio(
                stages["coarse"]["positives_kept"],
                stages["coarse"]["positives_from_recall"],
            ),
            "negative_leak_rate": _ratio(
                stages["coarse"]["negative_leaked"],
                stages["coarse"]["negative_from_recall"],
            ),
        },
        "shortlist": {
            "precision": _ratio(stages["shortlist"]["positives"], stages["shortlist"]["total"]),
            "positive_recall": _ratio(stages["shortlist"]["positives"], len(positives)),
            "negative_rate": _ratio(stages["shortlist"]["negatives"], stages["shortlist"]["total"]),
        },
    }

    misses = {
        "positive_not_recalled": sorted(
            matched[eid].title for eid in positives - recalled if eid in matched
        ),
        "positive_not_shortlisted": sorted(
            matched[eid].title for eid in positives - shortlist if eid in matched
        ),
        "negative_in_shortlist": sorted(
            matched[eid].title for eid in negatives & shortlist if eid in matched
        ),
        "unmatched_labels": unmatched,
    }

    return {
        "generated_at": reference.isoformat(),
        "config_path": str(config_path),
        "events": len(events),
        "labels": [asdict(label) for label in labels],
        "stages": stages,
        "metrics": metrics,
        "misses": misses,
        "relevance_distribution": dict(Counter(label.ai_relevance for label in labels)),
        "warnings": _warnings(matched, labels),
    }


def _warnings(matched: dict[str, EvalLabel], labels: list[EvalLabel]) -> list[str]:
    warnings: list[str] = []
    if len(matched) < 30:
        warnings.append(f"matched labels 只有 {len(matched)} 条，指标波动大，建议 ≥50 条")
    if any(not label.reviewed for label in labels):
        warnings.append("存在未人工复核的标注（reviewed=false）")
    if not any(label.ai_relevance == "none" for label in labels):
        warnings.append("缺少 none 负样本，无法评估误放率")
    return warnings


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """多天评测结果按分阶段计数汇总，重算比率。"""
    if not reports:
        raise ValueError("reports 不能为空")
    stage_sums = {
        "recall": {
            "positives_recalled": 0,
            "positives_total": 0,
            "negative_leaked": 0,
            "negative_total": 0,
        },
        "coarse": {
            "positives_kept": 0,
            "positives_from_recall": 0,
            "negative_leaked": 0,
            "negative_from_recall": 0,
        },
        "shortlist": {"total": 0, "positives": 0, "negatives": 0, "unknown": 0},
    }
    matched = 0
    labeled = 0
    reviewed = 0
    unmatched: list[str] = []
    misses = {
        "positive_not_recalled": [],
        "positive_not_shortlisted": [],
        "negative_in_shortlist": [],
        "unmatched_labels": [],
    }
    for report in reports:
        for stage, values in report.get("stages", {}).items():
            for key, value in values.items():
                stage_sums.setdefault(stage, {})[key] = stage_sums.setdefault(stage, {}).get(
                    key, 0
                ) + int(value)
        coverage = report.get("metrics", {}).get("label_coverage", {})
        matched += int(coverage.get("matched", 0))
        labeled += int(coverage.get("labeled", 0))
        reviewed += int(coverage.get("reviewed", 0))
        for key, bucket in misses.items():
            bucket.extend(report.get("misses", {}).get(key, []))
        unmatched.extend(report.get("misses", {}).get("unmatched_labels", []))

    recall = stage_sums["recall"]
    coarse = stage_sums["coarse"]
    shortlist = stage_sums["shortlist"]
    metrics = {
        "label_coverage": {
            "labeled": labeled,
            "matched": matched,
            "unmatched": len(set(unmatched)),
            "reviewed": reviewed,
        },
        "recall": {
            "positive_recall": _ratio(recall["positives_recalled"], recall["positives_total"]),
            "negative_leak_rate": _ratio(recall["negative_leaked"], recall["negative_total"]),
        },
        "coarse": {
            "positive_keep_rate": _ratio(coarse["positives_kept"], coarse["positives_from_recall"]),
            "negative_leak_rate": _ratio(coarse["negative_leaked"], coarse["negative_from_recall"]),
        },
        "shortlist": {
            "precision": _ratio(shortlist["positives"], shortlist["total"]),
            "positive_recall": _ratio(shortlist["positives"], recall["positives_total"]),
            "negative_rate": _ratio(shortlist["negatives"], shortlist["total"]),
        },
    }
    unmatched_unique = sorted(set(unmatched))
    warnings: list[str] = []
    if matched < 100:
        warnings.append(f"matched labels 只有 {matched} 条，建议 ≥200 条")
    if any(report.get("warnings") for report in reports):
        warnings.extend(w for report in reports for w in report.get("warnings", []))
    return {
        "generated_at": reports[0].get("generated_at", ""),
        "days": len(reports),
        "event_days": [report.get("events", 0) for report in reports],
        "events": sum(int(report.get("events", 0)) for report in reports),
        "stages": stage_sums,
        "metrics": metrics,
        "misses": {key: sorted(set(values)) for key, values in misses.items()},
        "warnings": sorted(set(warnings)),
        "unmatched_labels": unmatched_unique,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """把评测报告渲染成便于人看的 Markdown。"""
    metrics = report["metrics"]
    stages = report["stages"]
    coverage = metrics["label_coverage"]
    day_note = f"- 评测天数：{report['days']}" if "days" in report else ""
    lines = [
        "# 推荐流程评测报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        day_note,
        f"- 事件数：{report['events']}",
        f"- 标注：{coverage['labeled']} 条，匹配 {coverage['matched']} 条，未匹配 {coverage['unmatched']} 条",
        "",
        "## 分阶段指标",
        "",
        "| 阶段 | 指标 | 值 |",
        "|---|---|---|",
        f"| 召回 | 正样本召回率 | {_fmt(metrics['recall']['positive_recall'])} |",
        f"| 召回 | 负样本漏入率 | {_fmt(metrics['recall']['negative_leak_rate'])} |",
        f"| 粗排 | 正样本保留率 | {_fmt(metrics['coarse']['positive_keep_rate'])} |",
        f"| 粗排 | 负样本漏入率 | {_fmt(metrics['coarse']['negative_leak_rate'])} |",
        f"| top8 | 精确率 | {_fmt(metrics['shortlist']['precision'])} |",
        f"| top8 | 正样本覆盖 | {_fmt(metrics['shortlist']['positive_recall'])} |",
        f"| top8 | 负样本占比 | {_fmt(metrics['shortlist']['negative_rate'])} |",
        "",
        "## 各阶段数量",
        "",
        "| 阶段 | 正样本 | 负样本 |",
        "|---|---|---|",
        f"| 召回 | {stages['recall']['positives_recalled']} / {stages['recall']['positives_total']} | {stages['recall']['negative_leaked']} / {stages['recall']['negative_total']} |",
        f"| 粗排 | {stages['coarse']['positives_kept']} / {stages['coarse']['positives_from_recall']} | {stages['coarse']['negative_leaked']} / {stages['coarse']['negative_from_recall']} |",
        f"| top8 | {stages['shortlist']['positives']} | {stages['shortlist']['negatives']} |",
    ]
    misses = report["misses"]
    if misses["positive_not_recalled"]:
        lines += ["", "## 漏召回的正样本", ""]
        lines += [f"- {title}" for title in misses["positive_not_recalled"]]
    if misses["negative_in_shortlist"]:
        lines += ["", "## 进入 top8 的负样本", ""]
        lines += [f"- {title}" for title in misses["negative_in_shortlist"]]
    if misses["unmatched_labels"]:
        lines += ["", "## 未匹配的标注（检查标题是否被事件合并改写）", ""]
        lines += [f"- {title}" for title in misses["unmatched_labels"]]
    if report["warnings"]:
        lines += ["", "## 警告", ""]
        lines += [f"- {warning}" for warning in report["warnings"]]
    return "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"
