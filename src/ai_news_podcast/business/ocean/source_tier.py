"""来源可信度分级：1=一手/研究，2=媒体，3=社区/聚合。

分级只依据 Item.category，先保证可解释、零维护；后续可按源名单加 overrides。
数字越小越可信，仅用于同分时的优先顺序与审计，不直接改分数。
"""

from __future__ import annotations

from ai_news_podcast.business.contracts import Item

TIER_OFFICIAL = 1
TIER_RESEARCH = 1
TIER_NEWS = 2
TIER_COMMUNITY = 3
TIER_UNKNOWN = 3

_CATEGORY_TIERS: dict[str, int] = {
    "official": TIER_OFFICIAL,
    "research": TIER_RESEARCH,
    "news": TIER_NEWS,
    "community": TIER_COMMUNITY,
    "video": TIER_COMMUNITY,
    "podcast": TIER_COMMUNITY,
}


def tier_for_category(category: str) -> int:
    return _CATEGORY_TIERS.get(str(category or "").strip().lower(), TIER_UNKNOWN)


def tier_for_item(item: Item) -> int:
    return tier_for_category(item.category)


def best_tier(items: list[Item]) -> int:
    """事件内最好的来源等级（数字最小）。"""
    if not items:
        return TIER_UNKNOWN
    return min(tier_for_item(item) for item in items)


def tier_name(tier: int) -> str:
    return {1: "一手/研究", 2: "媒体", 3: "社区/聚合"}.get(int(tier), "未知")


__all__ = [
    "TIER_COMMUNITY",
    "TIER_NEWS",
    "TIER_OFFICIAL",
    "TIER_RESEARCH",
    "TIER_UNKNOWN",
    "best_tier",
    "tier_for_category",
    "tier_for_item",
    "tier_name",
]
