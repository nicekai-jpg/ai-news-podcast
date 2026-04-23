"""Tests for ai_news_podcast.pipeline.processor pure functions."""

from __future__ import annotations

from unittest.mock import patch

from ai_news_podcast.pipeline.processor import (
    Cluster,
    ContextBlock,
    ScoredStory,
    _assign_role,
    _build_context,
    _dedup_url,
    _extract_thesis,
    _score_cluster,
)


class TestDedupUrl:
    def test_removes_duplicates(self, raw_item_factory) -> None:
        a = raw_item_factory(title="A", link="https://a.com", normalized_link="https://a.com")
        b = raw_item_factory(title="B", link="https://b.com", normalized_link="https://b.com")
        c = raw_item_factory(title="A2", link="https://a.com", normalized_link="https://a.com")
        result = _dedup_url([a, b, c])
        assert len(result) == 2
        assert {item.title for item in result} == {"A", "B"}

    def test_empty_list(self) -> None:
        assert _dedup_url([]) == []


class TestAssignRole:
    def test_main(self) -> None:
        assert _assign_role(15, {"main": [12, 15], "supporting": [8, 11], "quick": [5, 7]}) == (
            "main",
            "🔴",
        )

    def test_supporting(self) -> None:
        assert _assign_role(10, {"main": [12, 15], "supporting": [8, 11], "quick": [5, 7]}) == (
            "supporting",
            "🟡",
        )

    def test_quick(self) -> None:
        assert _assign_role(6, {"main": [12, 15], "supporting": [8, 11], "quick": [5, 7]}) == (
            "quick",
            "🟢",
        )

    def test_skip(self) -> None:
        assert _assign_role(4, {"main": [12, 15], "supporting": [8, 11], "quick": [5, 7]}) == (
            "skip",
            "⚪",
        )


class TestScoreCluster:
    def test_multi_source_gets_high_impact(self, raw_item_factory) -> None:
        items = [
            raw_item_factory(source_name="A", source_category="official"),
            raw_item_factory(source_name="B", source_category="official"),
            raw_item_factory(source_name="C", source_category="official"),
        ]
        cluster = Cluster(cluster_id=0, items=items, representative=items[0])
        scores = _score_cluster(cluster)
        assert scores["impact_scope"] == 3
        assert scores["source_richness"] == 3

    def test_chinese_relevance_boost(self, raw_item_factory) -> None:
        items = [
            raw_item_factory(
                language="zh",
                title="ChatGPT 新功能",
                summary="大模型 LLM 突破",
                full_text_snippet="x" * 1200,
            )
        ]
        cluster = Cluster(cluster_id=0, items=items, representative=items[0])
        scores = _score_cluster(cluster)
        assert scores["listener_relevance"] == 3
        assert scores["explainability"] == 3

    def test_novelty_keywords(self, raw_item_factory) -> None:
        items = [
            raw_item_factory(
                title="Breakthrough in AI",
                summary="state-of-the-art model",
                full_text_snippet="x" * 600,
            )
        ]
        cluster = Cluster(cluster_id=0, items=items, representative=items[0])
        scores = _score_cluster(cluster)
        assert scores["novelty"] == 3


class TestBuildContext:
    def test_extracts_summary_sentences(self, raw_item_factory) -> None:
        items = [
            raw_item_factory(
                summary="First sentence. Second sentence.",
                full_text_snippet="",
            )
        ]
        cluster = Cluster(cluster_id=0, items=items, representative=items[0])
        ctx: ContextBlock = _build_context(cluster)
        assert ctx.factual_summary == ["First sentence"]

    def test_falls_back_to_title(self, raw_item_factory) -> None:
        items = [raw_item_factory(title="Fallback Title", summary="", full_text_snippet="")]
        cluster = Cluster(cluster_id=0, items=items, representative=items[0])
        ctx: ContextBlock = _build_context(cluster)
        assert ctx.factual_summary == ["Fallback Title"]

    def test_sorts_sources_by_authority(self, raw_item_factory) -> None:
        items = [
            raw_item_factory(source_name="News", source_category="news"),
            raw_item_factory(source_name="Official", source_category="official"),
        ]
        cluster = Cluster(cluster_id=0, items=items, representative=items[0])
        ctx: ContextBlock = _build_context(cluster)
        names = [s["name"] for s in ctx.sources_ranked]
        assert names.index("Official") < names.index("News")


class TestExtractThesis:
    def test_no_main_stories(self) -> None:
        stories: list[ScoredStory] = []
        assert _extract_thesis(stories) == "今天的AI领域动态丰富，多个方向齐头并进"

    def test_with_main_story_mocked_template(self) -> None:
        stories = [
            ScoredStory(
                cluster_id=0,
                representative_title="GPT-5发布",
                items=[],
                context=ContextBlock(factual_summary=[], historical_background="", sources_ranked=[]),
                scores={},
                total_score=14,
                role="main",
                role_emoji="🔴",
            )
        ]
        with patch(
            "random.choice",
            return_value="{main_topic}是最热话题",
        ):
            result = _extract_thesis(stories)
        assert result == "GPT-5发布是最热话题"
