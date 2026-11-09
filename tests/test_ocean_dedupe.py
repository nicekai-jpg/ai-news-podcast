"""Stage 2 四层去重：URL / 标题 / 语义 / 跨日 + 保留策略。"""

from __future__ import annotations

from ai_news_podcast.business.ocean.dedupe import (
    DUP_CROSS_DAY,
    DUP_SEMANTIC,
    DUP_TITLE,
    DUP_URL,
    dedupe_items,
)
from ai_news_podcast.business.ocean.labels import label_item
from tests.test_ocean_support import (
    make_event,
    make_item,
    make_labels,
    ocean_config,
    ocean_lexicon,
)


def _labeled(items):
    lexicon = ocean_lexicon()
    config = ocean_config()
    return {item.item_id: label_item(item, lexicon, config) for item in items}


def _dedupe(items, **kwargs):
    return dedupe_items(items, labeled=_labeled(items), config=ocean_config(), **kwargs)


class TestLayer1Url:
    def test_same_normalized_url_merged(self) -> None:
        first = make_item(
            title="OpenAI 发布 GPT-5",
            link="https://example.com/gpt5",
            source="a",
            full_text="short",
        )
        second = make_item(
            title="OpenAI 发布 GPT-5 模型",
            link="https://example.com/gpt5?utm_source=x",
            source="b",
            full_text="a much longer body " * 20,
        )
        result = _dedupe([first, second])
        assert len(result.survivors) == 1
        assert result.survivors[0].source == "b"  # 正文更长者保留
        duplicate = result.duplicates[first.item_id]
        assert duplicate.reason == DUP_URL
        assert duplicate.dup_of == second.item_id
        assert result.stats["url_exact"] == 1

    def test_same_url_first_wins_when_equal_length(self) -> None:
        first = make_item(title="A", link="https://example.com/x", source="a", full_text="same")
        second = make_item(title="B", link="https://example.com/x", source="b", full_text="same")
        result = _dedupe([first, second])
        assert result.survivors[0].item_id == first.item_id


class TestLayer2Title:
    def test_near_identical_titles_merged(self) -> None:
        first = make_item(
            title="OpenAI 发布 GPT-5 模型",
            link="https://a.com/1",
            source="a",
            full_text="body",
        )
        second = make_item(
            title="OpenAI 发布 GPT-5 模型！",
            link="https://b.com/2",
            source="b",
            full_text="a longer body " * 10,
        )
        result = _dedupe([first, second])
        assert len(result.survivors) == 1
        assert result.duplicates[first.item_id].reason == DUP_TITLE
        assert result.survivors[0].item_id == second.item_id

    def test_distinct_versions_not_merged_by_title(self) -> None:
        first = make_item(title="OpenAI 发布 GPT-5 模型", link="https://a.com/1", source="a")
        second = make_item(title="OpenAI 发布 GPT-5.1 模型", link="https://b.com/2", source="b")
        result = _dedupe([first, second])
        assert len(result.survivors) == 2


class TestLayer3Semantic:
    def test_syndicated_copy_merged(self) -> None:
        summary = (
            "OpenAI 今日正式发布 GPT-5 大模型，支持多模态输入与 100 万 tokens 上下文，"
            "API 价格同步公布，开发者可通过平台调用。"
        )
        first = make_item(
            title="OpenAI 发布 GPT-5 大模型",
            summary=summary,
            link="https://a.com/1",
            source="a",
        )
        second = make_item(
            title="OpenAI GPT-5 大模型正式发布",
            summary=summary,
            link="https://b.com/2",
            source="b",
        )
        result = _dedupe([first, second])
        assert len(result.survivors) == 1
        duplicate = next(iter(result.duplicates.values()))
        assert duplicate.reason == DUP_SEMANTIC
        assert result.stats["semantic"] == 1

    def test_different_events_stay_separate(self) -> None:
        first = make_item(
            title="OpenAI 发布 GPT-5 模型",
            summary="OpenAI 推出新模型。",
            source="a",
        )
        second = make_item(
            title="英伟达财报超预期",
            summary="英伟达公布季度财报。",
            source="b",
        )
        result = _dedupe([first, second])
        assert len(result.survivors) == 2


class TestRetentionPolicy:
    def test_longer_full_text_wins(self) -> None:
        first = make_item(title="Same", link="https://a.com/1", source="a", full_text="x" * 5)
        second = make_item(title="Same", link="https://b.com/2", source="b", full_text="x" * 50)
        result = _dedupe([first, second])
        assert result.survivors[0].item_id == second.item_id

    def test_official_wins_on_equal_length(self) -> None:
        first = make_item(
            title="Same",
            link="https://a.com/1",
            source="media",
            category="news",
            full_text="x" * 10,
        )
        second = make_item(
            title="Same",
            link="https://b.com/2",
            source="official",
            category="official",
            full_text="x" * 10,
        )
        result = _dedupe([first, second])
        assert result.survivors[0].item_id == second.item_id

    def test_earlier_published_wins_on_equal_length(self) -> None:
        first = make_item(
            title="Same",
            link="https://a.com/1",
            source="a",
            full_text="x" * 10,
            published_at="2026-11-09T09:00:00+00:00",
        )
        second = make_item(
            title="Same",
            link="https://b.com/2",
            source="b",
            full_text="x" * 10,
            published_at="2026-11-09T08:00:00+00:00",
        )
        result = _dedupe([first, second])
        assert result.survivors[0].item_id == second.item_id


class TestLayer4CrossDay:
    def _history(self) -> list:
        return [
            make_event(
                "OpenAI 发布 GPT-5 模型",
                summary="OpenAI 发布 GPT-5 大模型。",
                labels=make_labels(
                    actions=("release",),
                    numbers=({"type": "version", "value": "gpt-5"},),
                ),
                event_id="h1",
                continuity_id="c1",
            )
        ]

    def _ledger(self) -> dict:
        return {
            "version": 1,
            "entries": [
                {
                    "event_id": "h1",
                    "title": "OpenAI 发布 GPT-5 模型",
                    "continuity_id": "c1",
                    "links": [],
                }
            ],
        }

    def test_covered_repeat_is_dropped(self) -> None:
        item = make_item(
            title="OpenAI 发布 GPT-5 模型",
            summary="OpenAI 发布 GPT-5 大模型。",
            source="repost",
            link="https://repost.com/gpt5",
        )
        result = dedupe_items(
            [item],
            labeled=_labeled([item]),
            history_events=self._history(),
            ledger=self._ledger(),
            config=ocean_config(),
        )
        assert result.survivors == []
        duplicate = result.duplicates[item.item_id]
        assert duplicate.reason == DUP_CROSS_DAY
        assert duplicate.dup_of == "h1"
        assert result.stats["cross_day_covered"] == 1

    def test_uncovered_repeat_is_kept_and_marked_reappeared(self) -> None:
        item = make_item(
            title="OpenAI 发布 GPT-5 模型",
            summary="OpenAI 发布 GPT-5 大模型。",
            source="repost",
            link="https://repost.com/gpt5",
        )
        result = dedupe_items(
            [item],
            labeled=_labeled([item]),
            history_events=self._history(),
            ledger={"version": 1, "entries": []},
            config=ocean_config(),
        )
        assert len(result.survivors) == 1
        assert item.item_id in result.reappeared

    def test_new_progress_survives_covered_history(self) -> None:
        item = make_item(
            title="OpenAI 发布 GPT-5.1 模型，性能提升 40%",
            summary="OpenAI 发布 GPT-5.1，性能提升 40%。",
            source="openai_blog",
            category="official",
            link="https://openai.com/gpt51",
        )
        result = dedupe_items(
            [item],
            labeled=_labeled([item]),
            history_events=self._history(),
            ledger=self._ledger(),
            config=ocean_config(),
        )
        assert len(result.survivors) == 1
        assert item.item_id not in result.reappeared

    def test_no_history_means_no_layer4(self) -> None:
        item = make_item(title="OpenAI 发布 GPT-5 模型", source="a")
        result = _dedupe([item])
        assert len(result.survivors) == 1
        assert result.stats["kept"] == 1
