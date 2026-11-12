"""事件级证据分与证据分级。

公式（v4 §1.6，权重全部配置化）::

    evidence_score = 2.0 * official_source
                   + 1.0 * min(independent_sources, 5)
                   + 1.0 * fact_consistency
                   + 0.5 * full_text_ratio
                   - 1.0 * single_source
                   - 1.5 * rumor_flag

evidence_tier 只用于 Stage 3 粗排门槛，不参与精排打分（v5 去重复结论）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ai_news_podcast.business.contracts import Item
from ai_news_podcast.business.ocean.config import cfg_get
from ai_news_podcast.business.ocean.textutils import (
    clean_text,
    extract_fact_tokens,
    hours_between,
    is_official_category,
    parse_time,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceResult:
    """事件证据分及组成（分量保留给 Stage 3/4 审计与风险判断）。"""

    score: float
    tier: str
    official_source: bool
    source_count: int
    independent_sources: int
    fact_consistency: float
    full_text_ratio: float
    rumor_flag: bool
    momentum_growth: float = 0.0
    components: dict[str, float] = field(default_factory=dict)


def _source_key(item: Item) -> str:
    extra = item.extra or {}
    group = extra.get("source_group")
    if group:
        return str(group)
    return str(item.source or item.source_label or "unknown")


def _normalized_title(title: str) -> str:
    return "".join(
        ch for ch in str(title or "").lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"
    )


def _independent_sources(items: list[Item], config: dict[str, Any] | None = None) -> int:
    cfg = config or {}
    if not bool(cfg_get(cfg, "evidence.dedup_syndicated_titles", False)):
        return len({_source_key(item) for item in items})
    seen_sources: set[str] = set()
    seen_titles: set[str] = set()
    independent = 0
    for item in items:
        source = _source_key(item)
        title = _normalized_title(item.title)
        if source in seen_sources or (title and title in seen_titles):
            continue
        independent += 1
        seen_sources.add(source)
        if title:
            seen_titles.add(title)
    return independent


def _full_text_ratio(items: list[Item], min_chars: int) -> float:
    if not items:
        return 0.0
    ok = sum(
        1
        for item in items
        if (item.full_text_status or "").lower() == "ok" and len(item.full_text or "") >= min_chars
    )
    return ok / len(items)


def _item_text(item: Item) -> str:
    return clean_text(f"{item.title} {item.summary} {item.full_text}")


def _fact_consistency(items: list[Item]) -> float:
    """跨源数字/版本一致率：同一 token 出现在 ≥2 个来源的比例（无法验证给 0.5）。"""
    if len(items) < 2:
        return 0.5
    token_sets = [extract_fact_tokens(_item_text(item)) for item in items]
    token_sets = [tokens for tokens in token_sets if tokens]
    if not token_sets:
        return 0.5
    counts: dict[str, int] = {}
    for tokens in token_sets:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    agreed = sum(1 for count in counts.values() if count >= 2)
    return agreed / len(counts)


def _rumor_flag(items: list[Item], config: dict[str, Any]) -> bool:
    independent = _independent_sources(items, config)
    if independent >= int(cfg_get(config, "evidence.rumor_max_sources", 2)):
        return False
    keywords = [str(word).lower() for word in cfg_get(config, "evidence.rumor_keywords", [])]
    text = " ".join(clean_text(f"{item.title} {item.summary}").lower() for item in items)
    return any(word and word in text for word in keywords)


def _momentum_growth(items: list[Item], config: dict[str, Any]) -> float:
    """近 N 小时新增来源占全部来源比例（无时间信息时为 0）。"""
    window_hours = float(cfg_get(config, "evidence.momentum_window_hours", 6))
    parsed = [
        parsed for parsed in (parse_time(item.published_at) for item in items) if parsed is not None
    ]
    if not parsed:
        return 0.0
    reference = max(parsed)
    recent = sum(1 for moment in parsed if hours_between(moment, reference) <= window_hours)
    return recent / max(1, len({_source_key(item) for item in items}))


def tier_for(score: float, rumor_flag: bool, config: dict[str, Any]) -> str:
    """证据分级：传闻 < 弱 < 中 < 强。"""
    weak = float(cfg_get(config, "evidence.tiers.weak", 1.0))
    medium = float(cfg_get(config, "evidence.tiers.medium", 2.5))
    strong = float(cfg_get(config, "evidence.tiers.strong", 4.0))
    if rumor_flag or score < weak:
        return "rumor"
    if score >= strong:
        return "strong"
    if score >= medium:
        return "medium"
    return "weak"


def score_event(items: list[Item], *, config: dict[str, Any]) -> EvidenceResult:
    """按公式计算事件证据分与分级。"""
    if not items:
        return EvidenceResult(
            score=0.0,
            tier="rumor",
            official_source=False,
            source_count=0,
            independent_sources=0,
            fact_consistency=0.0,
            full_text_ratio=0.0,
            rumor_flag=False,
        )
    official_categories = [
        str(category) for category in cfg_get(config, "evidence.official_categories", ["official"])
    ]
    official_source = any(
        is_official_category(item.category, official_categories) for item in items
    )
    source_count = len({str(item.source or item.source_label) for item in items})
    independent_sources = _independent_sources(items, config)
    cap = int(cfg_get(config, "evidence.source_count_cap", 5))
    min_full_text = int(cfg_get(config, "evidence.min_full_text_chars", 300))
    full_text_ratio = _full_text_ratio(items, min_full_text)
    fact_consistency = _fact_consistency(items)
    rumor_flag = _rumor_flag(items, config)
    momentum_growth = _momentum_growth(items, config)

    weights = cfg_get(config, "evidence.weights", {}) or {}
    components = {
        "official_source": float(weights.get("official_source", 2.0)) * int(official_source),
        "source_count": float(weights.get("source_count", 1.0)) * min(independent_sources, cap),
        "fact_consistency": float(weights.get("fact_consistency", 1.0)) * fact_consistency,
        "full_text_ratio": float(weights.get("full_text_ratio", 0.5)) * full_text_ratio,
        "single_source": float(weights.get("single_source", -1.0)) * int(independent_sources <= 1),
        "rumor": float(weights.get("rumor", -1.5)) * int(rumor_flag),
    }
    score = round(sum(components.values()), 4)
    return EvidenceResult(
        score=score,
        tier=tier_for(score, rumor_flag, config),
        official_source=official_source,
        source_count=source_count,
        independent_sources=independent_sources,
        fact_consistency=round(fact_consistency, 4),
        full_text_ratio=round(full_text_ratio, 4),
        rumor_flag=rumor_flag,
        momentum_growth=round(momentum_growth, 4),
        components={key: round(value, 4) for key, value in components.items()},
    )
