"""Stage 2 事件聚类：候选对、相似度、版本保护、代表标题、时间窗。"""

from __future__ import annotations

from ai_news_podcast.business.ocean.cluster import cluster_items
from ai_news_podcast.business.ocean.labels import label_item
from tests.test_ocean_support import make_item, ocean_config, ocean_lexicon


def _labeled(items):
    lexicon = ocean_lexicon()
    config = ocean_config()
    return {item.item_id: label_item(item, lexicon, config) for item in items}


def _run(items, *, config=None):
    cfg = config or ocean_config()
    return cluster_items(items, labeled=_labeled(items), config=cfg)


def _titles(outcome) -> dict[str, int]:
    return {cluster.representative_title: cluster.size for cluster in outcome.clusters}


class TestClusterItems:
    def test_same_event_from_two_sources_is_one_cluster(self) -> None:
        items = [
            make_item(
                title="OpenAI 发布 GPT-5 模型",
                summary="OpenAI 推出 GPT-5 大模型，性能提升。",
                source="a",
            ),
            make_item(
                title="OpenAI GPT-5 正式发布，能力大幅提升",
                summary="OpenAI 发布 GPT-5，支持更长上下文。",
                source="b",
            ),
            make_item(
                title="英伟达财报超预期",
                summary="英伟达公布季度财报，营收增长。",
                source="c",
            ),
        ]
        outcome = _run(items)
        sizes = sorted(cluster.size for cluster in outcome.clusters)
        assert sizes == [1, 2]
        merged = next(cluster for cluster in outcome.clusters if cluster.size == 2)
        assert len(set(merged.item_ids)) == 2
        assert outcome.candidate_pairs >= 1

    def test_unrelated_events_stay_separate(self) -> None:
        items = [
            make_item(title="OpenAI 发布 GPT-5 模型", summary="发布新模型。", source="a"),
            make_item(title="英伟达财报超预期", summary="营收大涨。", source="b"),
            make_item(title="某大学发表机器人论文", summary="论文被接收。", source="c"),
        ]
        outcome = _run(items)
        assert len(outcome.clusters) == 3

    def test_distinct_model_versions_are_not_merged(self) -> None:
        items = [
            make_item(title="OpenAI 发布 GPT-5 模型", summary="性能提升。", source="a"),
            make_item(title="OpenAI 发布 GPT-5.1 模型", summary="性能再提升。", source="b"),
        ]
        outcome = _run(items)
        assert len(outcome.clusters) == 2

    def test_time_window_blocks_old_candidate(self) -> None:
        items = [
            make_item(
                title="OpenAI 发布 GPT-5 模型",
                summary="发布新模型。",
                source="a",
                published_at="2026-11-01T08:00:00+00:00",
            ),
            make_item(
                title="OpenAI GPT-5 正式发布，能力大幅提升",
                summary="发布新模型，支持更长上下文。",
                source="b",
                published_at="2026-11-06T08:00:00+00:00",
            ),
        ]
        outcome = _run(items)
        assert len(outcome.clusters) == 2

    def test_representative_prefers_short_title_with_entity_and_action(self) -> None:
        items = [
            make_item(
                title="OpenAI 正式发布全新一代 GPT-5 大模型，能力大幅提升",
                summary="OpenAI 发布 GPT-5，性能提升。",
                source="a",
            ),
            make_item(
                title="OpenAI 发布 GPT-5",
                summary="OpenAI 发布 GPT-5 模型。",
                source="b",
            ),
        ]
        outcome = _run(items)
        merged = max(outcome.clusters, key=lambda cluster: cluster.size)
        assert merged.representative_title == "OpenAI 发布 GPT-5"

    def test_single_item(self) -> None:
        item = make_item(title="OpenAI 发布 GPT-5", source="a")
        outcome = _run([item])
        assert len(outcome.clusters) == 1
        assert outcome.clusters[0].item_ids == [item.item_id]

    def test_empty(self) -> None:
        assert _run([]).clusters == []

    def test_hdbscan_method_available(self) -> None:
        config = ocean_config()
        config["cluster"]["method"] = "hdbscan"
        items = [
            make_item(
                title="OpenAI 发布 GPT-5 模型",
                summary="OpenAI 推出 GPT-5 大模型，性能提升。",
                source="a",
            ),
            make_item(
                title="OpenAI GPT-5 正式发布，能力大幅提升",
                summary="OpenAI 发布 GPT-5，支持更长上下文。",
                source="b",
            ),
            make_item(title="英伟达财报超预期", summary="营收大涨。", source="c"),
        ]
        outcome = _run(items, config=config)
        sizes = sorted(cluster.size for cluster in outcome.clusters)
        assert sizes == [1, 2]
