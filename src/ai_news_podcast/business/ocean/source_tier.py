"""来源可信度分级：1=一手/研究，2=媒体，3=社区/聚合。

分级只依据 Item.category，先保证可解释、零维护；后续可按源名单加 overrides。
数字越小越可信，仅用于同分时的优先顺序与审计，不直接改分数。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from ai_news_podcast.business.contracts import Item

_REGISTRY_PATH = Path(__file__).resolve().parents[4] / "config" / "source_registry.yaml"

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


@lru_cache(maxsize=1)
def load_source_registry(path: str | Path | None = None) -> dict[str, int]:
    target = Path(path) if path else _REGISTRY_PATH
    if not target.is_file():
        return {}
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return {
        str(name): int(tier)
        for name, tier in (data.get("tiers") or {}).items()
        if str(tier).isdigit()
    }


def tier_for_category(category: str) -> int:
    return _CATEGORY_TIERS.get(str(category or "").strip().lower(), TIER_UNKNOWN)


def tier_for_item(item: Item, registry: dict[str, int] | None = None) -> int:
    overrides = load_source_registry() if registry is None else registry
    key = str(item.source or item.source_label or "")
    if key in overrides:
        return int(overrides[key])
    return tier_for_category(item.category)


def best_tier(items: list[Item], registry: dict[str, int] | None = None) -> int:
    """事件内最好的来源等级（数字最小）。"""
    if not items:
        return TIER_UNKNOWN
    return min(tier_for_item(item, registry) for item in items)


def tier_name(tier: int) -> str:
    return {1: "一手/研究", 2: "媒体", 3: "社区/聚合"}.get(int(tier), "未知")


__all__ = [
    "TIER_COMMUNITY",
    "TIER_NEWS",
    "TIER_OFFICIAL",
    "TIER_RESEARCH",
    "TIER_UNKNOWN",
    "best_tier",
    "load_source_registry",
    "tier_for_category",
    "tier_for_item",
    "tier_name",
]
