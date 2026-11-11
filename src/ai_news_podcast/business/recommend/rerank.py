"""Stage 3 重排：MMR 多样性 + 去重 + 时效衰减 + top N。

- 去重：覆盖账本 / `covered_before` / 标题近重复（默认相似度 ≥0.92 只留高分一条）；
- 时效衰减：≤24h ×1.0，≤48h ×0.95，≤7d ×0.85，≤30d ×0.70，更旧 ×0.50；
- MMR：`λ × 归一化相关度 - (1-λ) × 与已选集合的最大相似度`；
- 硬约束：同一主体、同一技术栈各 ≤2（与 docs/pipeline_redesign_v5.md §7.5 一致）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from ai_news_podcast.business.contracts import Event
from ai_news_podcast.business.recommend.base import (
    event_age_hours,
    normalize_text,
    title_similarity,
)
from ai_news_podcast.business.recommend.config import RerankSettings
from ai_news_podcast.business.recommend.fine import FineScore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankedItem:
    """重排后的候选：融合分 + 时效衰减 + MMR 结果 + 理由。"""

    fine: FineScore
    relevance: float
    freshness: float
    final_score: float
    mmr_utility: float
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def event(self) -> Event:
        return self.fine.event


@dataclass(frozen=True)
class _Features:
    """MMR 相似度用的离散特征集合。"""

    subjects: frozenset[str]
    tech_stack: frozenset[str]
    topics: frozenset[str]
    categories: frozenset[str]


def rerank(
    scored: list[FineScore],
    settings: RerankSettings,
    *,
    covered_event_ids: set[str] | None = None,
    reference_time: datetime | None = None,
    shortlist_size: int | None = None,
) -> list[RerankedItem]:
    """对精排结果做去重/衰减/MMR，返回最终 top N。"""
    covered = covered_event_ids or set()
    candidates = [
        item
        for item in scored or []
        if item.event.event_id not in covered and not item.event.covered_before
    ]
    deduped = _dedupe_by_title(candidates, settings.title_dedupe_ratio)
    items = [_build_item(item, settings, reference_time) for item in deduped]
    limit = shortlist_size if shortlist_size is not None else settings.shortlist_size
    selected = _mmr_select(items, settings, limit)
    logger.debug(
        "重排完成: 输入 %d 条，去重后 %d 条，输出 %d 条",
        len(scored or []),
        len(items),
        len(selected),
    )
    return selected


def _dedupe_by_title(candidates: list[FineScore], threshold: float) -> list[FineScore]:
    """标题近重复去重：保留融合分更高的一条。"""
    ordered = sorted(candidates, key=lambda item: (-item.final_score, item.event.event_id))
    kept: list[FineScore] = []
    for item in ordered:
        duplicate = any(
            title_similarity(item.event.representative_title, other.event.representative_title)
            >= threshold
            for other in kept
        )
        if duplicate:
            logger.debug("去重丢弃近重复事件: %s", item.event.event_id)
            continue
        kept.append(item)
    return kept


def _build_item(
    fine: FineScore, settings: RerankSettings, reference_time: datetime | None
) -> RerankedItem:
    age = event_age_hours(fine.event, reference_time)
    freshness = _freshness_factor(age, settings)
    relevance = fine.final_score
    reasons: list[str] = []
    if freshness < 1.0:
        reasons.append(f"时效衰减 ×{freshness:.2f}")
    return RerankedItem(
        fine=fine,
        relevance=relevance,
        freshness=freshness,
        final_score=max(0.0, relevance * freshness),
        mmr_utility=0.0,
        reasons=tuple(reasons),
    )


def _freshness_factor(age_hours: float | None, settings: RerankSettings) -> float:
    """按时效返回衰减系数；时间未知时不衰减。"""
    if age_hours is None:
        return 1.0
    for max_hours, factor in settings.decay_points:
        if age_hours <= max_hours:
            return factor
    return settings.stale_factor


def _mmr_select(
    items: list[RerankedItem], settings: RerankSettings, limit: int
) -> list[RerankedItem]:
    """贪心 MMR 选择，同时执行同主体/同技术栈硬上限。"""
    remaining = list(items)
    selected: list[RerankedItem] = []
    max_relevance = max((item.final_score for item in remaining), default=0.0) or 1.0
    subject_counts: dict[str, int] = {}
    tech_counts: dict[str, int] = {}
    topic_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}

    while remaining and len(selected) < limit:
        picked = _pick_best(
            remaining,
            selected,
            settings,
            max_relevance,
            subject_counts,
            tech_counts,
            topic_counts,
            category_counts,
        )
        if picked is None:
            logger.debug("MMR 严格上限下已无可用候选，提前结束")
            break
        chosen, utility, penalty = picked
        selected.append(
            RerankedItem(
                fine=chosen.fine,
                relevance=chosen.relevance,
                freshness=chosen.freshness,
                final_score=chosen.final_score,
                mmr_utility=utility,
                reasons=chosen.reasons
                + ((f"MMR 多样性惩罚 {penalty:.2f}",) if penalty > 0.01 else ()),
            )
        )
        features = _features(chosen.event, settings)
        for subject in features.subjects:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        for tech in features.tech_stack:
            tech_counts[tech] = tech_counts.get(tech, 0) + 1
        for topic in features.topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        for category in features.categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        remaining.remove(chosen)
    return selected


def _pick_best(
    remaining: list[RerankedItem],
    selected: list[RerankedItem],
    settings: RerankSettings,
    max_relevance: float,
    subject_counts: dict[str, int],
    tech_counts: dict[str, int],
    topic_counts: dict[str, int],
    category_counts: dict[str, int],
) -> tuple[RerankedItem, float, float] | None:
    """返回 (原始候选, MMR utility, 多样性惩罚)，保持原对象以便从 remaining 移除。"""
    best: tuple[RerankedItem, float, float] | None = None
    best_key: tuple[float, float, int, str] | None = None
    for item in remaining:
        features = _features(item.event, settings)
        if _capped(
            features,
            settings,
            subject_counts,
            tech_counts,
            topic_counts,
            category_counts,
        ):
            continue
        relevance = item.final_score / max_relevance
        similarity = max(
            (
                _similarity(features, _features(chosen.event, settings), settings)
                for chosen in selected
            ),
            default=0.0,
        )
        utility = settings.mmr_lambda * relevance - (1.0 - settings.mmr_lambda) * similarity
        penalty = (1.0 - settings.mmr_lambda) * similarity
        # 同分时优先更可信来源（tier 数字越小越好）
        key = (utility, item.final_score, -item.event.best_source_tier, item.event.event_id)
        if best_key is None or key > best_key:
            best_key = key
            best = (item, utility, penalty)
    return best


def _capped(
    features: _Features,
    settings: RerankSettings,
    subject_counts: dict[str, int],
    tech_counts: dict[str, int],
    topic_counts: dict[str, int],
    category_counts: dict[str, int],
) -> bool:
    if any(
        subject_counts.get(subject, 0) >= settings.max_per_subject for subject in features.subjects
    ):
        return True
    if any(tech_counts.get(tech, 0) >= settings.max_per_tech_stack for tech in features.tech_stack):
        return True
    if settings.max_per_topic > 0 and any(
        topic_counts.get(topic, 0) >= settings.max_per_topic for topic in features.topics
    ):
        return True
    return settings.max_per_category > 0 and any(
        category_counts.get(category, 0) >= settings.max_per_category
        for category in features.categories
    )


def _features(event: Event, settings: RerankSettings) -> _Features:
    labels = event.labels
    subjects = {
        normalize_text(str(subject.get("id") or subject.get("name") or ""))
        for subject in labels.subjects
        if isinstance(subject, dict)
    }
    generic = {normalize_text(tag) for tag in settings.generic_tech_tags}
    tech_stack = {normalize_text(tech) for tech in labels.tech_stack if normalize_text(tech)}
    return _Features(
        subjects=frozenset(subject for subject in subjects if subject),
        tech_stack=frozenset(tech for tech in tech_stack if tech and tech not in generic),
        topics=frozenset(normalize_text(topic) for topic in labels.topics if normalize_text(topic)),
        categories=frozenset(
            normalize_text(category)
            for category in event.source_categories
            if normalize_text(category)
        ),
    )


def _similarity(left: _Features, right: _Features, settings: RerankSettings) -> float:
    return (
        settings.subject_similarity_weight * _jaccard(left.subjects, right.subjects)
        + settings.tech_similarity_weight * _jaccard(left.tech_stack, right.tech_stack)
        + settings.topic_similarity_weight * _jaccard(left.topics, right.topics)
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


__all__ = ["RerankedItem", "rerank"]
