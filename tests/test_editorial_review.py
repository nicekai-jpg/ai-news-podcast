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
        # fact_check_notes 不再采信 LLM 自评，改为程序核对
        assert review.fact_check_notes == ["未提供事实底稿，跳过数字核对"]

    def test_fact_notes_verified_against_evidence(self) -> None:
        payload = {"chapters": [{"title": "开场", "intro": "导语"}]}
        script = "[CHAPTER] 开场\n[Host A] 模型参数 100B。"
        review = build_podcast_review(
            script,
            [_row()],
            [_event()],
            _llm(payload),
            config_path=CONFIG,
            evidence_text="该模型参数 100B。",
        )
        assert review.fact_check_notes == ["文稿内 1 个数字均可在事实底稿核对"]

    def test_fact_notes_flag_missing_number(self) -> None:
        payload = {"chapters": [{"title": "开场", "intro": "导语"}]}
        script = "[CHAPTER] 开场\n[Host A] 模型参数 999B。"
        review = build_podcast_review(
            script,
            [_row()],
            [_event()],
            _llm(payload),
            config_path=CONFIG,
            evidence_text="该模型参数 100B。",
        )
        assert any("999" in note for note in review.fact_check_notes)

    def test_fact_notes_flag_chapter_mismatch(self) -> None:
        payload = {"chapters": [{"title": "开场", "intro": "导语"}]}
        script = "[CHAPTER] 开场\n[Host A] 你好\n[CHAPTER] 收尾\n[Host B] 再见"
        review = build_podcast_review(
            script, [_row()], [_event()], _llm(payload), config_path=CONFIG
        )
        assert any("章节数" in note for note in review.fact_check_notes)

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


class TestChapterAlignment:
    """章节标题必须与文稿 [CHAPTER] 严格一致：页面导航直接来自它。"""

    SCRIPT = "[CHAPTER] 今日开场\n[Host A] 你好\n[CHAPTER] 数据底座\n[Host B] 继续"

    def _payload(self, titles: list[str]) -> dict:
        return {
            "show_notes": "本期看点" * 20,
            "chapters": [{"title": t, "intro": "导语", "highlights": ["听点"]} for t in titles],
            "items": [],
        }

    def test_drifted_titles_are_corrected(self) -> None:
        review = build_podcast_review(
            self.SCRIPT,
            [_row()],
            [_event()],
            _llm(self._payload(["开场白", "数据底座换挡"])),
            config_path=CONFIG,
        )
        assert [c["title"] for c in review.chapters] == ["今日开场", "数据底座"]
        assert any("章节标题已按文稿校正" in note for note in review.fact_check_notes)
        # LLM 写的 intro 保留
        assert review.chapters[0]["intro"] == "导语"

    def test_matching_titles_keep_no_extra_note(self) -> None:
        review = build_podcast_review(
            self.SCRIPT,
            [_row()],
            [_event()],
            _llm(self._payload(["今日开场", "数据底座"])),
            config_path=CONFIG,
        )
        assert [c["title"] for c in review.chapters] == ["今日开场", "数据底座"]
        assert not any("章节" in note for note in review.fact_check_notes)

    def test_chapter_count_mismatch_rebuilds_from_script(self) -> None:
        review = build_podcast_review(
            self.SCRIPT,
            [_row()],
            [_event()],
            _llm(self._payload(["只有一章"])),
            config_path=CONFIG,
        )
        assert [c["title"] for c in review.chapters] == ["今日开场", "数据底座"]
        assert any("章节数与文稿不一致" in note for note in review.fact_check_notes)

    def test_missing_chapters_generated_from_script(self) -> None:
        review = build_podcast_review(
            self.SCRIPT,
            [_row()],
            [_event()],
            _llm({"show_notes": "看点" * 30, "items": []}),
            config_path=CONFIG,
        )
        assert [c["title"] for c in review.chapters] == ["今日开场", "数据底座"]
        assert any("生成骨架" in note for note in review.fact_check_notes)


def test_review_quality_flags_chapter_mismatch() -> None:
    from ai_news_podcast.business.editorial.review_quality import review_quality

    review = {
        "overall": "整体评价" * 20,
        "items": [{"event_id": "e1", "comment": "有依据"}],
        "show_notes": "本期看点" * 30,
        "chapters": [
            {"title": "开场白", "intro": "导语"},
            {"title": "数据底座换挡", "intro": "导语"},
            {"title": "收尾", "intro": "导语"},
        ],
    }
    aligned = review_quality(
        review, kind="podcast", script_chapters=["今日开场", "数据底座", "明日看点"]
    )
    assert aligned["checks"]["chapter_alignment"] is False
    assert any("不一致" in issue for issue in aligned["issues"])

    ok = review_quality(review, kind="podcast")
    assert "chapter_alignment" not in ok["checks"]
