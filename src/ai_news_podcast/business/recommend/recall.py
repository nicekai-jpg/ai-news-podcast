"""Stage 3 多路召回：只判断「有没有资格」，不打分。

六路通道（与 docs/pipeline_redesign_v5.md §7.2 对齐）：

- freshness  时效：新事件 / 持续事件 / 长尾新进展；
- topic      主题：命中 AI 主题词；
- tech_stack 技术栈：命中 LLM/Agent/芯片/机器人等；
- subject    主体：命中重要公司/实验室（按 weight × confidence）；
- evidence   证据：官方证据分达标，或独立来源 ≥2；
- continuity 连续性：历史事件出现新进展。

召回允许使用 lifecycle（由原始统计量推导）作为时效分层依据；精排禁止使用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from ai_news_podcast.business.contracts import Event
from ai_news_podcast.business.recommend.base import (
    event_age_hours,
    matched_keywords,
    max_subject_significance,
)
from ai_news_podcast.business.recommend.config import RecallSettings

logger = logging.getLogger(__name__)

# 通道顺序固定，保证输出可复现。
RECALL_CHANNELS = (
    "freshness",
    "topic",
    "tech_stack",
    "subject",
    "evidence",
    "continuity",
)

CHANNEL_LABELS: dict[str, str] = {
    "freshness": "时效",
    "topic": "主题",
    "tech_stack": "技术栈",
    "subject": "主体",
    "evidence": "证据",
    "continuity": "连续性",
}


@dataclass(frozen=True)
class RecallCandidate:
    """多路召回的一个候选事件及其命中通道。"""

    event: Event
    channels: tuple[str, ...]
    age_hours: float | None = None


def recall_events(
    events: list[Event],
    settings: RecallSettings,
    *,
    reference_time: datetime | None = None,
) -> list[RecallCandidate]:
    """对事件做多路召回，返回命中至少一个通道的候选（按输入顺序去重）。"""
    reference = reference_time or datetime.now(tz=UTC)
    event_order: list[str] = []
    events_by_id: dict[str, Event] = {}
    channels_by_id: dict[str, set[str]] = {}
    ages: dict[str, float | None] = {}

    for event in events or []:
        if event is None:
            continue
        event_id = event.event_id or _fallback_event_id(event)
        if settings.require_ai_relevance and not _has_ai_signal(event, settings):
            continue
        if event_id not in events_by_id:
            event_order.append(event_id)
            events_by_id[event_id] = event
            channels_by_id[event_id] = set()
            ages[event_id] = event_age_hours(event, reference)
        channels_by_id[event_id].update(_hit_channels(event, settings, ages[event_id]))

    candidates = [
        RecallCandidate(
            event=events_by_id[event_id],
            channels=tuple(
                channel for channel in RECALL_CHANNELS if channel in channels_by_id[event_id]
            ),
            age_hours=ages[event_id],
        )
        for event_id in event_order
        if channels_by_id[event_id]
    ]
    logger.debug(
        "召回完成: 输入 %d 条，候选 %d 条，通道分布 %s",
        len(events or []),
        len(candidates),
        _channel_counts(candidates),
    )
    return candidates


def _has_ai_signal(event: Event, settings: RecallSettings) -> bool:
    """召回准入前提：事件标题/摘要必须带 AI 信号。

    多路召回的各通道（时效/主体/连续性）只回答“是否值得关注”，不回答“是不是 AI”；
    因此 AI 相关性必须作为前置标准，避免汽车/消费/社会新闻靠时效或主体混进来。
    """
    head_signal = bool(matched_keywords(event.representative_title, settings.ai_text_keywords))
    grade = str(getattr(event.labels, "ai_relevance", "unknown") or "unknown")
    if grade in ("core", "adjacent"):
        # 只有一手/研究来源才允许“正文里提 AI”算相关；新闻/社区必须标题或摘要带 AI 信号
        return head_signal or int(getattr(event, "best_source_tier", 3) or 3) == 1
    if grade == "none":
        return False
    return head_signal


def _fallback_event_id(event: Event) -> str:
    """事件缺 event_id 时用标题+时间兜底，保证去重键稳定。"""
    return f"{event.representative_title}|{event.first_seen_at}"


def _hit_channels(event: Event, settings: RecallSettings, age_hours: float | None) -> list[str]:
    hits: list[str] = []
    if _hits_freshness(event, settings, age_hours):
        hits.append("freshness")
    if _hits_topic(event, settings):
        hits.append("topic")
    if _hits_tech_stack(event, settings):
        hits.append("tech_stack")
    if _hits_subject(event, settings):
        hits.append("subject")
    if _hits_evidence(event, settings):
        hits.append("evidence")
    if _hits_continuity(event):
        hits.append("continuity")
    return hits


def _hits_freshness(event: Event, settings: RecallSettings, age_hours: float | None) -> bool:
    """时效召回：新事件 ≤48h；持续事件 ≤7 天；长尾 ≤30 天且须有新进展。"""
    if age_hours is None:
        return settings.unknown_age_as_fresh
    if age_hours <= settings.new_hours:
        return True
    lifecycle = event.labels.lifecycle
    if age_hours <= settings.evolving_hours and lifecycle in {"burst", "evolving"}:
        return True
    return (
        age_hours <= settings.long_tail_hours
        and lifecycle == "long_tail"
        and _has_new_progress(event)
    )


def _has_new_progress(event: Event) -> bool:
    """是否存在可验证的新进展（新版本/新数字/新动作/更新标记）。"""
    return bool(
        event.is_update or event.reappeared or event.sequence > 1 or event.labels.novelty_numbers
    )


def _hits_topic(event: Event, settings: RecallSettings) -> bool:
    labels = event.labels
    text = " ".join([label for label in labels.topics if label] + [labels.primary_topic])
    return bool(matched_keywords(text, settings.ai_topics))


def _hits_tech_stack(event: Event, settings: RecallSettings) -> bool:
    text = " ".join(label for label in event.labels.tech_stack if label)
    return bool(matched_keywords(text, settings.ai_tech_stack))


def _hits_subject(event: Event, settings: RecallSettings) -> bool:
    return max_subject_significance(event.labels) >= settings.min_subject_weight


def _hits_evidence(event: Event, settings: RecallSettings) -> bool:
    """证据召回：官方证据分达标，或独立来源 ≥2，或来源数达到兜底阈值。"""
    return bool(
        event.evidence_score >= settings.min_evidence_score
        or event.independent_sources >= settings.min_independent_sources
        or event.source_count >= settings.min_source_count
    )


def _hits_continuity(event: Event) -> bool:
    """连续性召回：历史事件有新进展/新动作/再次出现。"""
    return bool(
        event.is_update
        or event.reappeared
        or event.sequence > 1
        or event.continuity_id
        or event.labels.novelty_numbers
    )


def _channel_counts(candidates: list[RecallCandidate]) -> dict[str, int]:
    counts = dict.fromkeys(RECALL_CHANNELS, 0)
    for candidate in candidates:
        for channel in candidate.channels:
            counts[channel] = counts.get(channel, 0) + 1
    return counts


def channel_label(channel: str) -> str:
    """通道 key 转中文标签（用于推荐理由）。"""
    return CHANNEL_LABELS.get(channel, channel)


def format_channels(channels: tuple[str, ...]) -> str:
    """把通道列表格式化为中文理由片段。"""
    return "、".join(channel_label(channel) for channel in channels) if channels else "无"


__all__ = [
    "CHANNEL_LABELS",
    "RECALL_CHANNELS",
    "RecallCandidate",
    "channel_label",
    "format_channels",
    "recall_events",
]
