"""Stage 2 规则标注：词典命中、多标签、新增 6 标签、生命周期。"""

from __future__ import annotations

from ai_news_podcast.business.ocean.labels import (
    LabeledItem,
    infer_lifecycle,
    label_text,
    merge_labels,
)
from tests.test_ocean_support import make_item, ocean_config, ocean_lexicon


def _label(title: str, summary: str = "", full_text: str = "") -> LabeledItem:
    return label_text(
        title,
        summary,
        full_text,
        lexicon=ocean_lexicon(),
        config=ocean_config(),
    )


class TestSubjects:
    def test_company_and_weight(self) -> None:
        labeled = _label("OpenAI 与 DeepSeek 同日发布新模型")
        ids = {subject["id"] for subject in labeled.labels.subjects}
        assert {"openai", "deepseek"} <= ids
        openai = next(s for s in labeled.labels.subjects if s["id"] == "openai")
        assert openai["weight"] == 1.0
        assert 0.0 <= openai["confidence"] <= 1.0

    def test_ambiguous_entity_needs_tech_context(self) -> None:
        apple_with_context = _label("苹果发布 M5 芯片，性能大幅提升")
        apple_without_context = _label("苹果价格上涨 20%")
        assert "apple" in {s["id"] for s in apple_with_context.labels.subjects}
        assert "apple" not in {s["id"] for s in apple_without_context.labels.subjects}

    def test_person_entity(self) -> None:
        labeled = _label("马斯克宣布 xAI 将开源新模型")
        ids = {subject["id"] for subject in labeled.labels.subjects}
        assert "elon_musk" in ids

    def test_entity_keys_include_objects(self) -> None:
        labeled = _label("OpenAI 发布 GPT-5.1 模型")
        assert {"openai", "gpt-5.1"} <= set(labeled.entity_keys)


class TestDimensions:
    def test_object_type_and_tech_stack(self) -> None:
        labeled = _label("OpenAI 发布 GPT-5 多模态模型，支持 Agent 能力")
        assert "model" in labeled.labels.object_type
        assert {"llm", "multimodal", "agent"} <= set(labeled.labels.tech_stack)

    def test_actions_are_multi_label_with_primary(self) -> None:
        labeled = _label("OpenAI 发布并开源 GPT-5 模型")
        assert {"release", "open_source"} <= set(labeled.labels.actions)
        assert labeled.labels.primary_action == "release"

    def test_priority_picks_primary_action(self) -> None:
        labeled = _label("OpenAI 遭起诉后又发布新模型")
        assert "lawsuit" in labeled.labels.actions
        assert "release" in labeled.labels.actions
        assert labeled.labels.primary_action == "lawsuit"

    def test_regions_explicit_and_headquarters(self) -> None:
        explicit = _label("OpenAI 在美国发布 GPT-5")
        assert "usa" in explicit.labels.regions
        inferred = _label("DeepSeek 发布新模型")
        assert "china" in inferred.labels.regions

    def test_topics(self) -> None:
        labeled = _label("OpenAI 完成 100 亿美元融资，估值创新高")
        assert "capital" in labeled.labels.topics
        assert labeled.labels.primary_topic == "capital"

    def test_novelty_numbers(self) -> None:
        labeled = _label("OpenAI 发布 GPT-5.1，性能提升 30%，融资 10 亿美元，2026-11-09")
        kinds = {number["type"] for number in labeled.labels.novelty_numbers}
        assert {"version", "percent", "money", "date"} <= kinds
        values = {number["value"] for number in labeled.labels.novelty_numbers}
        assert "gpt-5.1" in values

    def test_narrative_angles(self) -> None:
        labeled = _label("OpenAI 首次发布史上最大的模型，引发争议")
        angles = set(labeled.labels.narrative_angle)
        assert {"first", "record", "conflict"} <= angles

    def test_audience_personas(self) -> None:
        developer = _label("开源 Agent 框架，开发者可通过 API 调用")
        investor = _label("该公司完成 10 亿美元融资，估值翻倍")
        assert "developer" in developer.labels.audience_persona
        assert "investor" in investor.labels.audience_persona

    def test_sensitivity(self) -> None:
        labeled = _label("OpenAI 服务发生宕机事故，监管机构已介入调查")
        assert {"accident", "regulation"} <= set(labeled.labels.sensitivity)

    def test_quote_availability(self) -> None:
        quoted = _label("官方声明称：“我们会继续投入安全研究”")
        plain = _label("某公司发布新模型")
        assert quoted.labels.quote_availability is True
        assert plain.labels.quote_availability is False

    def test_confidence_metadata(self) -> None:
        labeled = _label("OpenAI 发布 GPT-5")
        assert labeled.labels.method == "rule"
        assert labeled.labels.confidence["subjects"] > 0
        assert "evidence_tier" in labeled.labels.confidence

    def test_empty_text_still_valid(self) -> None:
        labeled = _label("")
        assert labeled.labels.subjects == []
        assert labeled.labels.actions == []
        assert labeled.keywords == ()


class TestMergeLabels:
    def test_union_and_primary_vote(self) -> None:
        first = _label("OpenAI 发布 GPT-5 模型")
        second = _label("OpenAI 发布 GPT-5 模型，并开源权重")
        merged = merge_labels([first, second], evidence_tier="strong", config=ocean_config())
        assert {"release", "open_source"} <= set(merged.actions)
        assert merged.primary_action == "release"
        assert merged.evidence_tier == "strong"
        assert merged.confidence["evidence_tier"] > 0
        assert merged.quote_availability is False

    def test_subject_union_dedup(self) -> None:
        first = _label("OpenAI 发布新模型")
        second = _label("DeepSeek 发布新模型")
        merged = merge_labels([first, second], evidence_tier="weak", config=ocean_config())
        ids = [subject["id"] for subject in merged.subjects]
        assert ids.count("openai") == 1
        assert "deepseek" in ids

    def test_empty_members(self) -> None:
        merged = merge_labels([], evidence_tier="rumor", config=ocean_config())
        assert merged.evidence_tier == "rumor"
        assert merged.actions == []


class TestLifecycle:
    def test_burst(self) -> None:
        assert (
            infer_lifecycle(sequence=1, source_count=4, is_update=False, age_days=0, gap_days=0)
            == "burst"
        )

    def test_evolving(self) -> None:
        assert (
            infer_lifecycle(sequence=2, source_count=2, is_update=True, age_days=1, gap_days=1)
            == "evolving"
        )

    def test_long_tail(self) -> None:
        assert (
            infer_lifecycle(sequence=3, source_count=2, is_update=True, age_days=9, gap_days=1)
            == "long_tail"
        )

    def test_fading(self) -> None:
        assert (
            infer_lifecycle(sequence=2, source_count=1, is_update=False, age_days=10, gap_days=4)
            == "fading"
        )

    def test_unknown_small_new_event(self) -> None:
        assert (
            infer_lifecycle(sequence=1, source_count=1, is_update=False, age_days=0, gap_days=0)
            == "unknown"
        )


class TestItemLabeling:
    def test_label_item_from_contracts_item(self) -> None:
        item = make_item(
            title="OpenAI 发布 GPT-5 模型",
            summary="OpenAI 推出新模型。",
            full_text="OpenAI 今日发布 GPT-5 模型，支持多模态。",
            category="official",
        )
        from ai_news_podcast.business.ocean.labels import label_item

        labeled = label_item(item, ocean_lexicon(), ocean_config())
        assert labeled.labels.subjects[0]["id"] == "openai"
        assert "model" in labeled.labels.object_type
        assert labeled.entity_keys
