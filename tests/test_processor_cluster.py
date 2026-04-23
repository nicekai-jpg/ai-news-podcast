"""Tests for processor clustering and context memory."""

from __future__ import annotations

from ai_news_podcast.pipeline.processor import (
    Cluster,
    _build_context,
    cluster_stories,
)


class TestClusterStories:
    def test_empty_list(self) -> None:
        clusters = cluster_stories([])
        assert clusters == []

    def test_single_item(self, raw_item_factory) -> None:
        items = [raw_item_factory(title="Only One")]
        clusters = cluster_stories(items)
        assert len(clusters) == 1
        assert clusters[0].cluster_id == 0
        assert clusters[0].representative.title == "Only One"

    def test_multiple_items_form_clusters(self, raw_item_factory) -> None:
        # These titles are similar enough to potentially cluster
        items = [
            raw_item_factory(title="GPT-5 released today", summary="AI model"),
            raw_item_factory(title="GPT-5 launch event", summary="OpenAI event"),
            raw_item_factory(title="Weather is nice", summary="Sunny day"),
        ]
        clusters = cluster_stories(items, eps=0.5, min_samples=2)
        # With eps=0.5 and min_samples=2, first two might cluster, third becomes noise
        assert len(clusters) >= 2

    def test_noise_points_become_own_clusters(self, raw_item_factory) -> None:
        items = [
            raw_item_factory(title="A", summary="a"),
            raw_item_factory(title="B", summary="b"),
            raw_item_factory(title="C", summary="c"),
        ]
        # Very tight eps should make everything noise
        clusters = cluster_stories(items, eps=0.01, min_samples=2)
        # Each noise point becomes its own cluster
        assert len(clusters) == 3

    def test_representative_is_longest_fulltext(self, raw_item_factory) -> None:
        items = [
            raw_item_factory(title="Short", full_text_snippet="x"),
            raw_item_factory(title="Long", full_text_snippet="x" * 100),
        ]
        clusters = cluster_stories(items, eps=0.99, min_samples=1)
        rep = clusters[0].representative
        assert rep.title == "Long"


class TestBuildContextStoryMemory:
    def test_story_memory_match(self, raw_item_factory) -> None:
        items = [raw_item_factory(title="GPT-5 发布")]
        cluster = Cluster(cluster_id=0, items=items, representative=items[0])
        memory = {"GPT 模型": "Previous GPT history"}
        ctx = _build_context(cluster, story_memory=memory)
        assert ctx.historical_background == "Previous GPT history"

    def test_story_memory_no_match(self, raw_item_factory) -> None:
        items = [raw_item_factory(title=" unrelated title ")]
        cluster = Cluster(cluster_id=0, items=items, representative=items[0])
        memory = {"GPT": "History"}
        ctx = _build_context(cluster, story_memory=memory)
        assert ctx.historical_background == ""

    def test_custom_authority_order(self, raw_item_factory) -> None:
        items = [
            raw_item_factory(source_name="Blog", source_category="tools"),
            raw_item_factory(source_name="Gov", source_category="official"),
        ]
        cluster = Cluster(cluster_id=0, items=items, representative=items[0])
        custom_order = {"official": 1, "tools": 5}
        ctx = _build_context(cluster, source_authority=custom_order)
        names = [s["name"] for s in ctx.sources_ranked]
        assert names.index("Gov") < names.index("Blog")
