"""Tests for business/editorial/review.py（S5 日报点评 / S7 播客点评，假 LLM）。"""

from __future__ import annotations

import json
from typing import Any

from ai_news_podcast.business.contracts import Commentary, Event, Fact
from ai_news_podcast.business.editorial.config import load_config
from ai_news_podcast.business.editorial.review import (
    build_podcast_review,
    build_report_review,
    parse_podcast_review,
    parse_report_review,
)

CONFIG = "config/editorial_v3.yaml"


def _event() -> Event:
    return Event(event_id="e1", representative_title="事件一", summary="摘要", item_ids=["i1"])


def _row() -> Commentary:
    return Commentary(
        event_id="e1",
        headline="标题",
        facts=[Fact(text="事实一", evidence_ids=["i1"], single_source=False)],
    )


def _llm(payload: dict[str, Any] | str) -> Any:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return lambda _prompt: text


class TestReportReview:
    def test_builds_review_items(self) -> None:
        payload = {
            "overall": "整体准确",
            "items": [
                {
                    "event_id": "e1",
                    "title": "标题",
                    "verdict": "准确",
                    "comment": "点评内容",
                    "corrections": ["补充来源"],
                }
            ],
            "risks": ["单一来源"],
        }
        review = build_report_review(
            [_row()], [_event()], _llm(payload), item_evidence={"i1": "正文"}, config_path=CONFIG
        )
        assert review.overall == "整体准确"
        assert review.items[0].verdict == "准确"
        assert review.items[0].corrections == ["补充来源"]
        assert review.risks == ["单一来源"]

    def test_invalid_json_returns_empty(self) -> None:
        review = build_report_review([_row()], [_event()], _llm("不是 JSON"), config_path=CONFIG)
        assert review.items == [] and review.overall == ""

    def test_llm_error_returns_empty(self) -> None:
        def boom(_prompt: str) -> str:
            raise RuntimeError("LLM 故障")

        review = build_report_review([_row()], [_event()], boom, config_path=CONFIG)
        assert review.items == []

    def test_empty_rows_skips_llm(self) -> None:
        review = build_report_review([], [_event()], _llm({}), config_path=CONFIG)
        assert review.items == []


class TestPodcastReview:
    def test_builds_show_notes_and_chapters(self) -> None:
        payload = {
            "overall": "不错",
            "show_notes": "本期看点",
            "chapters": [{"title": "开场", "intro": "导语", "highlights": ["听点"]}],
            "items": [{"event_id": "e1", "title": "标题", "verdict": "补充", "comment": "点评"}],
            "fact_check_notes": ["核对数字"],
        }
        review = build_podcast_review(
            "[CHAPTER] 开场\n[Host A] 你好", [_row()], [_event()], _llm(payload), config_path=CONFIG
        )
        assert review.show_notes == "本期看点"
        assert review.chapters[0]["title"] == "开场"
        assert review.items[0].verdict == "补充"
        assert review.fact_check_notes == ["核对数字"]

    def test_empty_script_skips_llm(self) -> None:
        review = build_podcast_review(" ", [_row()], [_event()], _llm({}), config_path=CONFIG)
        assert review.chapters == []


class TestParsers:
    def test_parse_report_review_tolerates_missing_fields(self) -> None:
        review = parse_report_review({"overall": "ok"}, load_config(CONFIG))
        assert review.items == [] and review.risks == []

    def test_parse_podcast_review_skips_bad_chapters(self) -> None:
        review = parse_podcast_review(
            {"chapters": [{"title": "一"}, "bad", {"intro": "二"}]}, load_config(CONFIG)
        )
        assert len(review.chapters) == 2
        assert review.chapters[1]["title"] == ""
