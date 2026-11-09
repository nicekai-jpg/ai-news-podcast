"""Stage 2 连续性：历史匹配、新进展、sequence/continuity、已覆盖过滤。"""

from __future__ import annotations

from ai_news_podcast.business.ocean.continuity import (
    assign_continuity,
    detect_new_progress,
    filter_history_window,
)
from tests.test_ocean_support import make_event, make_labels, ocean_config


def _assign(events, *, history=None, ledger=None, date_str="2026-11-10"):
    return assign_continuity(
        events,
        history_events=history or [],
        ledger=ledger or {"version": 1, "entries": []},
        date_str=date_str,
        config=ocean_config(),
    )


def _gpt5_history(sequence: int = 1, continuity_id: str = "c1", **kwargs):
    return make_event(
        "OpenAI 发布 GPT-5 模型",
        summary="OpenAI 发布 GPT-5 大模型。",
        labels=make_labels(
            actions=("release",),
            numbers=({"type": "version", "value": "gpt-5"},),
        ),
        event_id="h1",
        sequence=sequence,
        continuity_id=continuity_id,
        first_seen_at="2026-11-09T08:00:00+00:00",
        last_update_at="2026-11-09T08:00:00+00:00",
        **kwargs,
    )


class TestNewEvent:
    def test_new_event_gets_sequence_one(self) -> None:
        event = make_event(
            "谷歌发布 Gemini 3 模型",
            labels=make_labels(subjects=(("google", 1.0),), actions=("release",)),
        )
        outcome = _assign([event])
        assert len(outcome.events) == 1
        assert event.sequence == 1
        assert event.is_update is False
        assert event.continuity_id.startswith("c_")
        assert outcome.new_events == 1
        assert len(outcome.lines) == 1
        assert outcome.lines[0]["sequence"] == 1

    def test_continuity_id_is_deterministic(self) -> None:
        first = make_event(
            "谷歌发布 Gemini 3 模型", labels=make_labels(subjects=(("google", 1.0),))
        )
        second = make_event(
            "谷歌发布 Gemini 3 模型", labels=make_labels(subjects=(("google", 1.0),))
        )
        id_first = _assign([first]).events[0].continuity_id
        id_second = _assign([second]).events[0].continuity_id
        assert id_first == id_second


class TestContinuityUpdate:
    def test_new_version_increments_sequence(self) -> None:
        history = _gpt5_history()
        event = make_event(
            "OpenAI 发布 GPT-5.1 模型，性能提升 40%",
            summary="OpenAI 发布 GPT-5.1，性能提升 40%。",
            labels=make_labels(
                actions=("release",),
                numbers=(
                    {"type": "version", "value": "gpt-5.1"},
                    {"type": "percent", "value": "40%"},
                ),
            ),
            last_update_at="2026-11-10T08:00:00+00:00",
            first_seen_at="2026-11-10T08:00:00+00:00",
        )
        outcome = _assign([event], history=[history])
        assert len(outcome.events) == 1
        assert event.sequence == 2
        assert event.is_update is True
        assert event.continuity_id == "c1"
        assert outcome.updates == 1
        assert event.labels.lifecycle == "evolving"
        line = outcome.lines[0]
        assert line["sequence"] == 2
        assert "h1" in line["history_event_ids"]

    def test_repeat_without_progress_keeps_sequence(self) -> None:
        history = _gpt5_history()
        event = make_event(
            "OpenAI 发布 GPT-5 模型",
            summary="OpenAI 发布 GPT-5 大模型。",
            labels=make_labels(actions=("release",)),
            first_seen_at="2026-11-10T08:00:00+00:00",
            last_update_at="2026-11-10T08:00:00+00:00",
        )
        outcome = _assign([event], history=[history])
        assert len(outcome.events) == 1
        assert event.sequence == 1
        assert event.is_update is False
        assert event.reappeared is True


class TestCoveredFilter:
    def _ledger(self) -> dict:
        return {
            "version": 1,
            "entries": [
                {
                    "event_id": "h1",
                    "title": "OpenAI 发布 GPT-5 模型",
                    "continuity_id": "c1",
                    "links": ["https://example.com/gpt5"],
                }
            ],
        }

    def test_covered_repeat_is_filtered(self) -> None:
        history = _gpt5_history()
        event = make_event(
            "OpenAI 发布 GPT-5 模型",
            summary="OpenAI 发布 GPT-5 大模型。",
            labels=make_labels(actions=("release",)),
            first_seen_at="2026-11-10T08:00:00+00:00",
            last_update_at="2026-11-10T08:00:00+00:00",
        )
        outcome = _assign([event], history=[history], ledger=self._ledger())
        assert outcome.events == []
        assert outcome.dropped and outcome.dropped[0].event_id == event.event_id
        assert outcome.drop_reasons[event.event_id] == "covered_before"
        assert event.covered_before is True

    def test_uncovered_repeat_is_kept(self) -> None:
        history = _gpt5_history()
        event = make_event(
            "OpenAI 发布 GPT-5 模型",
            summary="OpenAI 发布 GPT-5 大模型。",
            labels=make_labels(actions=("release",)),
            first_seen_at="2026-11-10T08:00:00+00:00",
            last_update_at="2026-11-10T08:00:00+00:00",
        )
        outcome = _assign([event], history=[history])
        assert len(outcome.events) == 1
        assert event.covered_before is False
        assert event.reappeared is True


class TestMatchingGuards:
    def test_same_subject_different_event_not_matched(self) -> None:
        history = make_event(
            "OpenAI 完成 10 亿美元融资",
            summary="OpenAI 宣布完成新一轮融资。",
            labels=make_labels(actions=("funding",)),
            event_id="h_funding",
            continuity_id="c_fin",
        )
        event = make_event(
            "OpenAI 发布 GPT-5 模型",
            summary="OpenAI 发布新模型。",
            labels=make_labels(actions=("release",)),
            first_seen_at="2026-11-10T08:00:00+00:00",
            last_update_at="2026-11-10T08:00:00+00:00",
        )
        outcome = _assign([event], history=[history])
        assert event.sequence == 1
        assert event.continuity_id != "c_fin"
        assert outcome.matched_events == {}

    def test_old_history_outside_window_ignored(self) -> None:
        history = _gpt5_history()
        history.last_update_at = "2026-08-01T08:00:00+00:00"
        history.first_seen_at = "2026-08-01T08:00:00+00:00"
        event = make_event(
            "OpenAI 发布 GPT-5 模型",
            summary="OpenAI 发布 GPT-5 大模型。",
            labels=make_labels(actions=("release",)),
        )
        outcome = _assign([event], history=[history])
        assert event.sequence == 1
        assert outcome.matched_events == {}


class TestHistoryLines:
    def test_line_collects_history_members(self) -> None:
        first = _gpt5_history(sequence=1, continuity_id="c1")
        second = make_event(
            "OpenAI 发布 GPT-5.1 模型",
            summary="OpenAI 发布 GPT-5.1。",
            labels=make_labels(
                actions=("release",),
                numbers=({"type": "version", "value": "gpt-5.1"},),
            ),
            event_id="h2",
            sequence=2,
            continuity_id="c1",
            first_seen_at="2026-11-09T08:00:00+00:00",
            last_update_at="2026-11-10T08:00:00+00:00",
        )
        current = make_event(
            "OpenAI 发布 GPT-5.2 模型，性能提升 50%",
            summary="OpenAI 发布 GPT-5.2。",
            labels=make_labels(
                actions=("release",),
                numbers=(
                    {"type": "version", "value": "gpt-5.2"},
                    {"type": "percent", "value": "50%"},
                ),
            ),
            first_seen_at="2026-11-11T08:00:00+00:00",
            last_update_at="2026-11-11T08:00:00+00:00",
        )
        outcome = _assign([current], history=[first, second], date_str="2026-11-11")
        assert len(outcome.events) == 1
        assert current.sequence == 3
        line = outcome.lines[0]
        assert line["sequence"] == 3
        assert {"h1", "h2", current.event_id} <= set(line["event_ids"])
        assert line["first_seen_at"].startswith("2026-11-09")


class TestDetectNewProgress:
    def test_new_version(self) -> None:
        result = detect_new_progress(
            "OpenAI 发布 GPT-5.1 模型",
            ["OpenAI 发布 GPT-5 模型"],
            config=ocean_config(),
        )
        assert result.is_new is True
        assert any("new_version" in reason for reason in result.reasons)

    def test_new_number(self) -> None:
        result = detect_new_progress(
            "融资 20 亿美元",
            ["融资 10 亿美元"],
            config=ocean_config(),
        )
        assert result.is_new is True
        assert any("new_number" in reason for reason in result.reasons)

    def test_no_progress(self) -> None:
        result = detect_new_progress(
            "OpenAI 发布 GPT-5 模型",
            ["OpenAI 发布 GPT-5 模型"],
            current_actions=("release",),
            previous_actions=("release",),
            config=ocean_config(),
        )
        assert result.is_new is False

    def test_new_action(self) -> None:
        result = detect_new_progress(
            "OpenAI 开源 GPT-5 模型",
            ["OpenAI 发布 GPT-5 模型"],
            current_actions=("open_source", "release"),
            previous_actions=("release",),
            config=ocean_config(),
        )
        assert result.is_new is True
        assert any("new_action" in reason for reason in result.reasons)


class TestFilterHistoryWindow:
    def test_filters_old_and_future(self) -> None:
        recent = make_event("近期事件", first_seen_at="2026-11-09T08:00:00+00:00")
        recent.last_update_at = "2026-11-09T08:00:00+00:00"
        old = make_event("古老事件", first_seen_at="2026-08-01T08:00:00+00:00")
        old.last_update_at = "2026-08-01T08:00:00+00:00"
        kept = filter_history_window([recent, old], "2026-11-10", 30)
        assert [event.representative_title for event in kept] == ["近期事件"]
