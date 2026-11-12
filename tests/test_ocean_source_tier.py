"""Tests for ocean/source_tier.py — 来源可信度分级。"""

from __future__ import annotations

from ai_news_podcast.business.contracts import Item
from ai_news_podcast.business.ocean.source_tier import best_tier, tier_for_category
from tests.test_ocean_support import make_item


def _item(category: str) -> Item:
    return Item(
        item_id=f"i-{category}",
        channel="c1",
        source=f"src-{category}",
        source_label=category,
        category=category,
        title="t",
        link="https://example.com",
    )


class TestTierForCategory:
    def test_official_and_research_are_tier_one(self) -> None:
        assert tier_for_category("official") == 1
        assert tier_for_category("research") == 1

    def test_news_is_tier_two(self) -> None:
        assert tier_for_category("news") == 2

    def test_community_and_unknown_are_tier_three(self) -> None:
        assert tier_for_category("community") == 3
        assert tier_for_category("mystery") == 3


def test_best_tier_takes_minimum() -> None:
    items = [_item("community"), _item("official"), _item("news")]
    assert best_tier(items) == 1


def test_best_tier_empty_is_unknown() -> None:
    assert best_tier([]) == 3


def test_registry_overrides_category() -> None:
    from ai_news_podcast.business.ocean.source_tier import (
        TIER_NEWS,
        load_source_registry,
        tier_for_item,
    )

    registry = load_source_registry()
    official_but_news = make_item(title="t", source="openai_news", category="news")
    assert tier_for_item(official_but_news, registry) == 1
    media = make_item(title="t", source="jiqizhixin", category="news")
    assert tier_for_item(media, registry) == TIER_NEWS


def test_registry_falls_back_to_category() -> None:
    from ai_news_podcast.business.ocean.source_tier import tier_for_item

    unknown = make_item(title="t", source="nobody", category="official")
    assert tier_for_item(unknown, {}) == 1
