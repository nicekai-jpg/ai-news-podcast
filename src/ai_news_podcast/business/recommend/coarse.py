"""Stage 3 规则粗排：淘汰明显不合格的候选，并给出可审计的淘汰原因。

五类硬门槛（与 docs/pipeline_redesign_v5.md §7.3 对齐）：

1. `covered` / `covered_before`：已播报过（覆盖账本命中）；
2. `junk`：招聘/课程/广告/榜单等非新闻内容；
3. `single_source_rumor`：单一来源且证据分级为传闻；
4. `fact_conflict`：跨源事实一致性 < 阈值（标记待核，暂不推荐）；
5. `subject_cap`：同一主体超过上限（默认先留 2 条）。

另外按「独立来源数/主体权重/时效/生命周期/证据分级/动作类型」算一个可解释的
粗排分，用于同主体截断和候选截断（`coarse_cut`）。lifecycle / evidence_tier
只在本阶段使用，不进入精排（防循环/重复计数）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from ai_news_podcast.business.contracts import Event, Labels
from ai_news_podcast.business.recommend.base import (
    contains_keyword,
    event_age_hours,
    matched_keywords,
    max_subject_significance,
)
from ai_news_podcast.business.recommend.config import CoarseSettings
from ai_news_podcast.business.recommend.recall import RecallCandidate

logger = logging.getLogger(__name__)

# 粗排硬门槛原因 → 便于审计的说明。
REJECT_COVERED = "covered"
REJECT_COVERED_BEFORE = "covered_before"
REJECT_JUNK = "junk"
REJECT_SINGLE_SOURCE_RUMOR = "single_source_rumor"
REJECT_FACT_CONFLICT = "fact_conflict"
REJECT_NOT_AI = "not_ai"
REJECT_SUBJECT_CAP = "subject_cap"
REJECT_COARSE_CUT = "coarse_cut"

_PRIORITY_ACTION_BONUS = 0.4
_FADING_PENALTY = 0.6
_LIFECYCLE_BONUS = {"burst": 0.6, "evolving": 0.6, "long_tail": 0.2, "fading": -_FADING_PENALTY}
_EVIDENCE_BONUS = {"strong": 0.5, "medium": 0.15, "weak": -0.1, "rumor": -0.8}


@dataclass(frozen=True)
class CoarseElimination:
    """一条被粗排淘汰的候选及原因。"""

    event_id: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class CoarseCandidate:
    """通过粗排的候选：保留命中通道、粗排分与「为什么留下」。"""

    event: Event
    channels: tuple[str, ...]
    coarse_score: float
    reasons: tuple[str, ...] = ()


@dataclass
class CoarseOutcome:
    """粗排结果：候选 + 淘汰审计。"""

    candidates: list[CoarseCandidate] = field(default_factory=list)
    eliminated: list[CoarseElimination] = field(default_factory=list)
    considered: int = 0


def coarse_rank(
    candidates: list[RecallCandidate],
    settings: CoarseSettings,
    *,
    covered_event_ids: set[str] | None = None,
    reference_time: datetime | None = None,
) -> CoarseOutcome:
    """对召回候选做规则粗排，返回候选与完整淘汰原因。"""
    covered = covered_event_ids or set()
    outcome = CoarseOutcome(considered=len(candidates))
    scored: list[CoarseCandidate] = []

    for candidate in candidates:
        rejection = _hard_rejection(candidate.event, settings, covered)
        if rejection is not None:
            outcome.eliminated.append(rejection)
            continue
        score, reasons = _coarse_score(candidate, settings, reference_time)
        scored.append(
            CoarseCandidate(
                event=candidate.event,
                channels=candidate.channels,
                coarse_score=score,
                reasons=tuple(reasons),
            )
        )

    # 先按粗排分降序（同分按 event_id 稳定排序），再做同主体截断。
    scored.sort(key=lambda item: (-item.coarse_score, item.event.event_id))
    kept: list[CoarseCandidate] = []
    subject_counts: dict[str, int] = {}
    for candidate in scored:
        subject_id = _capped_subject(candidate.event, subject_counts, settings.subject_cap)
        if subject_id is not None:
            outcome.eliminated.append(
                CoarseElimination(
                    event_id=candidate.event.event_id,
                    reason=REJECT_SUBJECT_CAP,
                    detail=f"主体 {subject_id} 已达上限 {settings.subject_cap}",
                )
            )
            continue
        for sid in _subject_ids(candidate.event):
            subject_counts[sid] = subject_counts.get(sid, 0) + 1
        kept.append(candidate)

    if len(kept) > settings.max_candidates:
        for candidate in kept[settings.max_candidates :]:
            outcome.eliminated.append(
                CoarseElimination(
                    event_id=candidate.event.event_id,
                    reason=REJECT_COARSE_CUT,
                    detail=f"粗排候选超过 max_candidates={settings.max_candidates}",
                )
            )
        kept = kept[: settings.max_candidates]

    outcome.candidates = kept
    logger.debug(
        "粗排完成: 输入 %d 条，保留 %d 条，淘汰 %d 条",
        outcome.considered,
        len(outcome.candidates),
        len(outcome.eliminated),
    )
    return outcome


def _hard_rejection(
    event: Event, settings: CoarseSettings, covered: set[str]
) -> CoarseElimination | None:
    """按优先级依次检查硬门槛，命中第一条即淘汰。"""
    for rejection in (
        _coverage_rejection(event, covered),
        _content_rejection(event, settings),
        _consistency_rejection(event, settings),
    ):
        if rejection is not None:
            return rejection
    return None


def _coverage_rejection(event: Event, covered: set[str]) -> CoarseElimination | None:
    if event.event_id in covered:
        return CoarseElimination(event.event_id, REJECT_COVERED, "覆盖账本已选中")
    if event.covered_before:
        return CoarseElimination(event.event_id, REJECT_COVERED_BEFORE, "事件标记为已播报")
    return None


def _content_rejection(event: Event, settings: CoarseSettings) -> CoarseElimination | None:
    junk = _matched_junk(event, settings)
    if junk:
        return CoarseElimination(event.event_id, REJECT_JUNK, f"命中非新闻关键词: {junk}")
    if settings.require_ai_relevance and not _is_ai_related(event, settings):
        return CoarseElimination(
            event.event_id,
            REJECT_NOT_AI,
            f"与 AI 无关（未命中 AI 关键词；主体={_subject_names(event)}）",
        )
    if _is_single_source_rumor(event, settings):
        return CoarseElimination(
            event.event_id,
            REJECT_SINGLE_SOURCE_RUMOR,
            "独立来源不足且证据分级为传闻",
        )
    return None


def _consistency_rejection(event: Event, settings: CoarseSettings) -> CoarseElimination | None:
    consistency = event.labels.confidence.get("fact_consistency")
    if consistency is None:
        return None
    try:
        value = float(consistency)
    except (TypeError, ValueError):
        value = 1.0
    if value < settings.fact_consistency_min:
        return CoarseElimination(
            event.event_id,
            REJECT_FACT_CONFLICT,
            f"跨源事实一致性 {value:.2f} < {settings.fact_consistency_min}",
        )
    return None


def _subject_names(event: Event) -> str:
    names = [str(item.get("name") or item.get("id") or "") for item in event.labels.subjects[:3]]
    return "、".join(name for name in names if name) or "未标注"


def _is_ai_related(event: Event, settings: CoarseSettings) -> bool:
    """事件必须带 AI 信号（标题/摘要命中 AI 关键词）。

    汽车/消费/社会新闻常因主体（如华为、特斯拉）命中所属主体通道而进入候选；
    这里只在标题/摘要上做 AI 关键词硬门槛，避免被正文里的偶然词或宽泛主题标签误放行。
    """
    grade = str(getattr(event.labels, "ai_relevance", "unknown") or "unknown")
    if grade in ("core", "adjacent"):
        return True
    if grade == "none":
        return False
    text = f"{event.representative_title} {event.summary}"
    return bool(matched_keywords(text, settings.ai_text_keywords))


def _matched_junk(event: Event, settings: CoarseSettings) -> str:
    text = f"{event.representative_title} {event.summary}"
    matches = matched_keywords(text, settings.junk_keywords)
    return "、".join(matches)


def _is_single_source_rumor(event: Event, settings: CoarseSettings) -> bool:
    return event.independent_sources <= 1 and event.labels.evidence_tier in settings.rumor_tiers


def _subject_ids(event: Event) -> list[str]:
    """事件里参与同主体截断的主体 id（按出现顺序去重）。"""
    ids: list[str] = []
    for subject in event.labels.subjects:
        if not isinstance(subject, dict):
            continue
        subject_id = str(subject.get("id") or subject.get("name") or "").strip()
        if subject_id and subject_id not in ids:
            ids.append(subject_id)
    return ids


def _capped_subject(event: Event, counts: dict[str, int], cap: int) -> str | None:
    """若事件任一主体已达上限，返回该主体 id。"""
    for subject_id in _subject_ids(event):
        if counts.get(subject_id, 0) >= cap:
            return subject_id
    return None


def _coarse_score(
    candidate: RecallCandidate, settings: CoarseSettings, reference_time: datetime | None
) -> tuple[float, list[str]]:
    """可解释粗排分（约 0–5），只用于排序/截断，不作为最终推荐分。"""
    event = candidate.event
    labels = event.labels
    score = 0.0
    reasons: list[str] = []

    source_signal = min(event.independent_sources, 5) / 5
    count_signal = min(event.source_count, 8) / 8
    score += 1.2 * source_signal + 0.6 * count_signal
    if event.independent_sources >= 2:
        reasons.append(f"多源（{event.independent_sources} 个独立来源）")

    significance = max_subject_significance(labels)
    score += 1.2 * significance
    if significance > 0:
        reasons.append(f"重要主体（权重 {significance:.2f}）")

    age = event_age_hours(event, reference_time)
    if age is None:
        score += 0.5
    elif age <= 48:
        score += 1.0
        reasons.append("48 小时内")
    elif age <= 168:
        score += 0.6
        reasons.append("7 天内持续事件")
    else:
        score += 0.2
        reasons.append("超过 7 天")

    lifecycle_bonus = _LIFECYCLE_BONUS.get(labels.lifecycle, 0.0)
    score += lifecycle_bonus
    if labels.lifecycle in {"burst", "evolving"}:
        reasons.append(f"生命周期 {labels.lifecycle}")
    elif labels.lifecycle == "fading":
        reasons.append("消退期降权")

    score += _EVIDENCE_BONUS.get(labels.evidence_tier, 0.0)
    if labels.evidence_tier == "strong":
        reasons.append("证据分级强")

    score += _priority_action_bonus(labels, settings.priority_actions)
    if _has_demote_keyword(event, settings):
        score -= settings.demote_penalty
        reasons.append("软文/评测类降权")

    if event.independent_sources <= 1:
        score -= settings.single_source_penalty
        reasons.append("单一来源降权")

    if not reasons:
        reasons.append("通过粗排门槛")
    return max(score, 0.0), reasons


def _priority_action_bonus(labels: Labels, priority_actions: tuple[str, ...]) -> float:
    """发布/开源/融资/政策等主动作加分（动作多标签，任一命中即可）。"""
    actions = [labels.primary_action, *labels.actions]
    for action in actions:
        normalized = str(action).casefold()
        if not normalized:
            continue
        for keyword in priority_actions:
            if keyword.casefold() in normalized:
                return _PRIORITY_ACTION_BONUS
    return 0.0


def _has_demote_keyword(event: Event, settings: CoarseSettings) -> bool:
    text = f"{event.representative_title} {event.summary}"
    return any(contains_keyword(text, keyword) for keyword in settings.demote_keywords)


__all__ = [
    "CoarseCandidate",
    "CoarseElimination",
    "CoarseOutcome",
    "coarse_rank",
]
