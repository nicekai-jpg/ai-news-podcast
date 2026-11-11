"""Tests for ocean/source_tier.py — 来源可信度分级。"""

from __future__ import annotations

from ai_news_podcast.business.contracts import Item
from ai_news_podcast.business.ocean.source_tier import best_tier, tier_for_category


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
