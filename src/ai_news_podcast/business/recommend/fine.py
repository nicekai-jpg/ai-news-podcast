"""Stage 3 精排：规则特征分 + 可注入的 LLM rubric 打分。

七个规则特征（0–5）：

- impact        影响力：主体显著度（weight × confidence）；
- novelty       新颖度：新版本/新数字/新主体数量与类型；
- momentum      势头：独立来源数、来源数、6h 增量、序号、重现标记；
- audience_fit  受众相关：audience_persona + tech_stack + topics；
- narratability 可讲性：narrative_angle + 动作多标签 + 数字 + 引语 + 对象；
- evidence      证据：独立来源数/来源总数/引语（**不使用 evidence_tier**）；
- risk          风险安全度：5 − 风险惩罚（单一来源/敏感标签/低置信度/标题党）。

硬约束（docs/pipeline_redesign_v5.md §3.3A）：`lifecycle` 与 `evidence_tier`
只用于召回/粗排，**禁止**进入精排；`evidence_score` 是 evidence_tier 的公式来源，
同样不使用，避免与粗排证据门槛重复计数。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from ai_news_podcast.business.contracts import Event, Labels
from ai_news_podcast.business.recommend.base import (
    DIMENSIONS,
    JUDGE_DIMENSIONS,
    clamp,
    matched_keywords,
    subject_significance,
    to_float,
)
from ai_news_podcast.business.recommend.config import (
    FineSettings,
    judge_dimension_weights,
)

logger = logging.getLogger(__name__)

# LLM judge 输出约定：{event_id: {"impact": 0-5, ..., "evidence_risk": 0-5, "reason": str}}
JudgeOutput = dict[str, dict[str, Any]]
LLMJudge = Callable[[list[Event]], JudgeOutput]


@dataclass
class FineScore:
    """单个事件的精排结果（规则分 / LLM 分 / 融合分）。"""

    event: Event
    components: dict[str, float] = field(default_factory=dict)
    rule_score: float = 0.0
    llm_score: float | None = None
    final_score: float = 0.0
    llm_reason: str = ""
    reasons: list[str] = field(default_factory=list)


def score_events(events: list[Event], settings: FineSettings) -> list[FineScore]:
    """对事件计算规则特征分，按 rule_score 降序返回（同分按 event_id 稳定）。"""
    scored = [_score_event(event, settings) for event in events or []]
    scored.sort(key=lambda item: (-item.rule_score, item.event.event_id))
    logger.debug("规则精排完成: %d 条", len(scored))
    return scored


def apply_judgements(
    scores: list[FineScore],
    judgements: JudgeOutput | None,
    settings: FineSettings,
) -> list[FineScore]:
    """把 LLM rubric 评分融合进规则分：final = rule_w * rule + llm_w * llm。

    LLM 缺该事件 / 维度非法时回退纯规则分（llm_score 保持 None），保证不丢候选。
    """
    if not judgements:
        return list(scores)
    weights = judge_dimension_weights(settings)
    updated: list[FineScore] = []
    for item in scores:
        raw = judgements.get(item.event.event_id)
        if not isinstance(raw, dict):
            updated.append(item)
            continue
        normalized, reason = _normalize_judgement(raw)
        llm_score = _weighted_judge_score(normalized, weights)
        if llm_score is None:
            updated.append(item)
            continue
        final = clamp(settings.blend_rule * item.rule_score + settings.blend_llm * llm_score)
        reasons = [*item.reasons, f"LLM 复核分 {llm_score:.2f}/5"]
        updated.append(
            replace(
                item,
                llm_score=llm_score,
                final_score=final,
                llm_reason=reason,
                reasons=reasons,
            )
        )
    return updated


def _score_event(event: Event, settings: FineSettings) -> FineScore:
    labels = event.labels
    components = {
        "impact": _impact_score(labels),
        "novelty": _novelty_score(labels),
        "momentum": _momentum_score(event),
        "audience_fit": _audience_fit_score(labels),
        "narratability": _narratability_score(labels),
        "evidence": _evidence_score(event, labels),
        "risk": 5.0 - _risk_penalty(event, labels, settings),
    }
    risk_penalty = 5.0 - components["risk"]
    components["risk_penalty"] = risk_penalty
    rule_score = _weighted_rule_score(components, settings)
    return FineScore(
        event=event,
        components=components,
        rule_score=rule_score,
        final_score=rule_score,
        reasons=_build_reasons(event, components),
    )


def _weighted_rule_score(components: dict[str, float], settings: FineSettings) -> float:
    total_weight = sum(settings.weights.get(dim, 0.0) for dim in DIMENSIONS)
    if total_weight <= 0:
        return 0.0
    weighted = sum(settings.weights.get(dim, 0.0) * components.get(dim, 0.0) for dim in DIMENSIONS)
    return clamp(weighted / total_weight)


def _impact_score(labels: Labels) -> float:
    values = sorted(
        (subject_significance(subject) for subject in labels.subjects if isinstance(subject, dict)),
        reverse=True,
    )
    if not values:
        return 0.0
    top = values[:2]
    mean_top = sum(top) / len(top)
    return clamp(5.0 * (0.7 * top[0] + 0.3 * mean_top))


def _novelty_score(labels: Labels) -> float:
    numbers = [item for item in labels.novelty_numbers if isinstance(item, dict)]
    if not numbers:
        return 0.0
    types = {str(item.get("type") or "").strip().casefold() for item in numbers} - {""}
    score = 2.0 + 0.6 * min(len(types), 3)
    if len(numbers) >= 2:
        score += 0.4
    if any(_has_digits(item.get("value")) for item in numbers):
        score += 0.4
    return clamp(score)


def _momentum_score(event: Event) -> float:
    source_signal = min(max(event.independent_sources, 0), 5) / 5
    growth_signal = min(max(event.momentum_growth, 0.0), 4.0) / 4.0
    sequence_signal = min(max(event.sequence - 1, 0), 3) / 3
    count_signal = min(max(event.source_count, 0), 8) / 8
    reappear_signal = 1.0 if event.reappeared else 0.0
    score = (
        2.0 * source_signal
        + 1.2 * growth_signal
        + 0.8 * sequence_signal
        + 0.6 * count_signal
        + 0.4 * reappear_signal
    )
    return clamp(score)


def _audience_fit_score(labels: Labels) -> float:
    personas = _unique(labels.audience_persona)
    tech_stack = _unique(labels.tech_stack)
    topics = _unique(labels.topics)
    score = 0.7 * min(len(personas), 4) + 0.4 * min(len(tech_stack), 3)
    if topics:
        score += 0.3
    if labels.primary_topic:
        score += 0.2
    return clamp(score)


def _narratability_score(labels: Labels) -> float:
    angles = _unique(labels.narrative_angle)
    actions = _unique([labels.primary_action, *labels.actions])
    numbers = [item for item in labels.novelty_numbers if isinstance(item, dict)]
    score = 0.9 * min(len(angles), 3) + 0.4 * min(len(actions), 3) + 0.3 * min(len(numbers), 3)
    if labels.quote_availability:
        score += 0.4
    if _unique(labels.object_type):
        score += 0.3
    return clamp(score)


def _evidence_score(event: Event, labels: Labels) -> float:
    """证据特征只用独立信号，不读 evidence_tier / evidence_score（防重复计数）。"""
    score = 0.9 * min(max(event.independent_sources, 0), 4)
    score += 0.2 * min(max(event.source_count, 0), 5)
    if event.independent_sources >= 2:
        score += 0.4
    if labels.quote_availability:
        score += 0.3
    return clamp(score)


def _risk_penalty(event: Event, labels: Labels, settings: FineSettings) -> float:
    """风险惩罚（0–5）：单一来源/敏感标签/低置信度/标题党。"""
    penalty = 0.0
    if event.independent_sources < 2:
        penalty += 1.2
    if event.source_count < 2:
        penalty += 0.4
    penalty += 0.8 * min(len(_unique(labels.sensitivity)), 2)
    penalty += 0.3 * min(_low_confidence_count(labels), 3)
    clickbait = matched_keywords(event.representative_title, settings.clickbait_keywords)
    penalty += 0.6 * min(len(clickbait), 2)
    return clamp(penalty)


def _low_confidence_count(labels: Labels) -> int:
    count = 0
    for value in labels.confidence.values():
        if isinstance(value, bool):
            continue
        if to_float(value, 1.0) < 0.4:
            count += 1
    return count


def _normalize_judgement(raw: dict[str, Any]) -> tuple[dict[str, float], str]:
    scores: dict[str, float] = {}
    for dim in JUDGE_DIMENSIONS:
        if dim not in raw:
            continue
        value = raw.get(dim)
        if isinstance(value, bool) or value is None:
            continue
        try:
            scores[dim] = clamp(float(value))
        except (TypeError, ValueError):
            continue
    reason = raw.get("reason") or raw.get("rationale") or ""
    if isinstance(reason, (list, tuple)):
        reason = "；".join(str(part) for part in reason)
    return scores, str(reason).strip()


def _weighted_judge_score(scores: dict[str, float], weights: dict[str, float]) -> float | None:
    total_weight = sum(weights.get(dim, 0.0) for dim in scores)
    if total_weight <= 0:
        return None
    weighted = sum(weights.get(dim, 0.0) * value for dim, value in scores.items())
    return clamp(weighted / total_weight)


def _unique(values: list[str]) -> list[str]:
    return [value for value in dict.fromkeys(str(item).strip() for item in values) if value]


def _has_digits(value: Any) -> bool:
    return any(char.isdigit() for char in str(value or ""))


def _build_reasons(event: Event, components: dict[str, float]) -> list[str]:
    """生成可审计的推荐理由（每个维度一句，附带关键事实）。"""
    labels = event.labels
    return [
        _impact_reason(labels, components["impact"]),
        _novelty_reason(labels, components["novelty"]),
        _momentum_reason(event, components["momentum"]),
        _audience_reason(labels, components["audience_fit"]),
        _narratability_reason(labels, components["narratability"]),
        _evidence_reason(event, labels, components["evidence"]),
        _risk_reason(labels, components["risk"]),
    ]


def _impact_reason(labels: Labels, score: float) -> str:
    subjects = [item for item in labels.subjects if isinstance(item, dict)]
    if not subjects:
        return f"影响力 {score:.1f}/5（未标注主体）"
    top = max(subjects, key=subject_significance)
    name = str(top.get("name") or top.get("id") or "未知主体")
    return f"影响力 {score:.1f}/5（主体 {name}，显著度 {subject_significance(top):.2f}）"


def _novelty_reason(labels: Labels, score: float) -> str:
    numbers = [item for item in labels.novelty_numbers if isinstance(item, dict)]
    if not numbers:
        return f"新颖度 {score:.1f}/5（无新数字/版本）"
    types = sorted({str(item.get("type") or "数字") for item in numbers})
    return f"新颖度 {score:.1f}/5（{len(numbers)} 个新数字/版本，类型：{'、'.join(types)}）"


def _momentum_reason(event: Event, score: float) -> str:
    growth = f"，6h 增量 {event.momentum_growth:.1f}" if event.momentum_growth else ""
    return (
        f"势头 {score:.1f}/5（独立来源 {event.independent_sources}，"
        f"来源总数 {event.source_count}{growth}）"
    )


def _audience_reason(labels: Labels, score: float) -> str:
    personas = _unique(labels.audience_persona)
    tech_stack = _unique(labels.tech_stack)
    persona_text = "、".join(personas[:3]) or "未标注"
    tech_text = "、".join(tech_stack[:3]) or "未标注"
    return f"受众相关 {score:.1f}/5（受众：{persona_text}；技术栈：{tech_text}）"


def _narratability_reason(labels: Labels, score: float) -> str:
    angles = _unique(labels.narrative_angle)
    angle_text = "、".join(angles[:3]) or "无明确叙事角度"
    actions = _unique([labels.primary_action, *labels.actions])
    action_text = "、".join(actions[:3]) or "未标注动作"
    return f"可讲性 {score:.1f}/5（角度：{angle_text}；动作：{action_text}）"


def _evidence_reason(event: Event, labels: Labels, score: float) -> str:
    quote = "有官方引语" if labels.quote_availability else "无引语"
    return (
        f"证据 {score:.1f}/5（独立来源 {event.independent_sources}，"
        f"来源总数 {event.source_count}，{quote}）"
    )


def _risk_reason(labels: Labels, score: float) -> str:
    risks = _unique(labels.sensitivity)
    risk_text = "、".join(risks[:3]) if risks else "无敏感标签"
    return f"风险 {score:.1f}/5（{risk_text}）"


__all__ = [
    "FineScore",
    "JudgeOutput",
    "LLMJudge",
    "apply_judgements",
    "score_events",
]
