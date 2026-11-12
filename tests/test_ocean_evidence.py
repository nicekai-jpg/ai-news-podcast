"""Stage 2 证据分公式与分级。"""

from __future__ import annotations

from ai_news_podcast.business.ocean.evidence import score_event, tier_for
from tests.test_ocean_support import make_item, ocean_config

LONG_TEXT = (
    "OpenAI 今日发布 GPT-5.1 模型，性能提升 30%，上下文扩展到 200 万 tokens。"
    "官方博客详细介绍了训练方法、评测结果与安全措施，并给出了 API 价格。"
) * 6


class TestEvidenceScore:
    def test_strong_official_multi_source(self) -> None:
        items = [
            make_item(
                title="OpenAI 发布 GPT-5.1",
                summary="性能提升 30%",
                full_text=LONG_TEXT,
                source="openai_blog",
                category="official",
            ),
            make_item(
                title="OpenAI 发布 GPT-5.1 模型",
                summary="性能提升 30%",
                full_text=LONG_TEXT,
                source="techcrunch",
            ),
            make_item(
                title="OpenAI 发布 GPT-5.1 新模型",
                summary="性能提升 30%",
                full_text=LONG_TEXT,
                source="theverge",
            ),
        ]
        result = score_event(items, config=ocean_config())
        assert result.official_source is True
        assert result.independent_sources == 3
        assert result.full_text_ratio == 1.0
        assert result.fact_consistency == 1.0
        assert result.score >= 4.0
        assert result.tier == "strong"
        assert result.components["official_source"] == 2.0

    def test_single_anonymous_rumor(self) -> None:
        item = make_item(
            title="传闻：OpenAI 将裁员",
            summary="据悉，消息称 OpenAI 计划裁员。",
            full_text="",
            source="anonymous_blog",
            category="news",
        )
        result = score_event([item], config=ocean_config())
        assert result.rumor_flag is True
        assert result.independent_sources == 1
        assert result.components["single_source"] == -1.0
        assert result.components["rumor"] == -1.5
        assert result.tier == "rumor"

    def test_single_official_is_medium(self) -> None:
        item = make_item(
            title="OpenAI 发布 GPT-5",
            full_text=LONG_TEXT,
            source="openai_blog",
            category="official",
        )
        result = score_event([item], config=ocean_config())
        assert result.score == 3.0
        assert result.tier == "medium"

    def test_fact_conflict_lowers_consistency(self) -> None:
        left = make_item(
            title="OpenAI 发布 GPT-5",
            summary="性能提升 30%",
            full_text=LONG_TEXT,
            source="a",
        )
        right = make_item(
            title="OpenAI 发布 GPT-5",
            summary="性能提升 50%",
            full_text=LONG_TEXT.replace("30%", "50%"),
            source="b",
        )
        result = score_event([left, right], config=ocean_config())
        assert result.fact_consistency < 1.0
        assert "fact_consistency" in result.components

    def test_source_group_counts_as_one_independent_source(self) -> None:
        items = [
            make_item(
                title="OpenAI 发布 GPT-5",
                source="media_a",
                source_group="same_group",
                full_text=LONG_TEXT,
            ),
            make_item(
                title="OpenAI 发布 GPT-5",
                source="media_b",
                source_group="same_group",
                full_text=LONG_TEXT,
            ),
        ]
        result = score_event(items, config=ocean_config())
        assert result.source_count == 2
        assert result.independent_sources == 1

    def test_no_items(self) -> None:
        result = score_event([], config=ocean_config())
        assert result.score == 0.0
        assert result.tier == "rumor"

    def test_momentum_growth(self) -> None:
        items = [
            make_item(
                title="OpenAI 发布 GPT-5",
                source="a",
                full_text=LONG_TEXT,
                published_at="2026-11-09T07:00:00+00:00",
            ),
            make_item(
                title="OpenAI 发布 GPT-5",
                source="b",
                full_text=LONG_TEXT,
                published_at="2026-11-09T12:00:00+00:00",
            ),
        ]
        result = score_event(items, config=ocean_config())
        assert 0.0 < result.momentum_growth <= 1.0


class TestTierFor:
    def test_boundaries(self) -> None:
        config = ocean_config()
        assert tier_for(4.0, False, config) == "strong"
        assert tier_for(3.9, False, config) == "medium"
        assert tier_for(2.5, False, config) == "medium"
        assert tier_for(2.4, False, config) == "weak"
        assert tier_for(1.0, False, config) == "weak"
        assert tier_for(0.9, False, config) == "rumor"

    def test_rumor_flag_overrides_high_score(self) -> None:
        assert tier_for(9.0, True, ocean_config()) == "rumor"


def test_syndicated_titles_count_as_one_source() -> None:
    from ai_news_podcast.business.ocean.evidence import _independent_sources

    items = [
        make_item(title="OpenAI 发布 GPT-5.1", source="source_a"),
        make_item(title="OpenAI 发布 GPT-5.1", source="source_b"),
    ]
    off = _independent_sources(items, {"evidence": {"dedup_syndicated_titles": False}})
    on = _independent_sources(items, {"evidence": {"dedup_syndicated_titles": True}})
    assert off == 2
    assert on == 1
