"""Tests for business/editorial/commentary.py（Stage 4 编辑点评，假 LLM，不联网）。"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from ai_news_podcast.business.contracts import Event, Labels, ShortlistEntry
from ai_news_podcast.business.editorial import (
    EditorialValidationError,
    build_commentaries,
    extract_json_object,
    parse_commentary,
)
from ai_news_podcast.business.editorial.config import config_from_mapping, load_config

_EVENT_SUMMARY = "该公司今日发布 2.1 版本，融资 10 亿美元。"
_EVENT_TITLE = "某公司发布 2.1 版本"


def make_event(
    event_id: str = "evt-1",
    *,
    title: str = _EVENT_TITLE,
    summary: str = _EVENT_SUMMARY,
    item_ids: list[str] | None = None,
    independent_sources: int = 2,
) -> Event:
    return Event(
        event_id=event_id,
        representative_title=title,
        summary=summary,
        item_ids=["item-1", "item-2"] if item_ids is None else item_ids,
        independent_sources=independent_sources,
        labels=Labels(primary_action="发布", topics=["技术"], subjects=[{"name": "某公司"}]),
    )


def make_entry(rank: int = 1, event_id: str = "evt-1") -> ShortlistEntry:
    return ShortlistEntry(rank=rank, event_id=event_id, score=4.0, reasons=["影响面广"])


def make_payload(
    *,
    event_id: str = "evt-1",
    headline: str = _EVENT_TITLE,
    fact_text: str = _EVENT_SUMMARY,
    evidence_ids: list[str] | None = None,
    counterpoint: str = "",
    why_matters: str = "版本迭代会影响开发者工具链。",
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "headline": headline,
        "one_liner": "一句话摘要",
        "facts": [
            {
                "text": fact_text,
                "evidence_ids": ["item-1", "item-2"] if evidence_ids is None else evidence_ids,
                "single_source": False,
            }
        ],
        "why_matters": why_matters,
        "background": "背景脉络",
        "impact": "影响分析",
        "counterpoint": counterpoint,
        "uncertainty": "",
        "sources": [{"label": "官方博客", "link": "https://example.com/blog", "official": True}],
    }


def as_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


class FakeLLM:
    """按顺序返回预置响应，并记录收到的 prompt。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("FakeLLM 没有更多预置响应")
        return self._responses.pop(0)


class TestBuildCommentaries:
    def test_returns_valid_commentaries(self) -> None:
        events = [
            make_event("evt-1"),
            make_event(
                "evt-2",
                title="另一家公司开源 7B 模型",
                summary="另一家公司开源 7B 模型，参数 70 亿。",
                item_ids=["item-3"],
            ),
        ]
        entries = [make_entry(1, "evt-1"), make_entry(2, "evt-2")]
        second_payload = make_payload(
            event_id="evt-2",
            headline="另一家公司开源 7B 模型",
            fact_text="另一家公司开源 7B 模型，参数 70 亿。",
            evidence_ids=["item-3"],
        )
        llm = FakeLLM([as_json(make_payload()), as_json(second_payload)])

        result = build_commentaries(entries, events, llm)

        assert [item.event_id for item in result] == ["evt-1", "evt-2"]
        assert result[0].headline == _EVENT_TITLE
        assert result[0].facts[0].evidence_ids == ["item-1", "item-2"]
        assert result[1].facts[0].evidence_ids == ["item-3"]
        assert len(llm.prompts) == 2
        assert "客观性铁律" in llm.prompts[0]
        assert "evidence_ids" in llm.prompts[0]
        assert "暂无公开反方信息" in llm.prompts[0]

    def test_respects_report_top_n(self) -> None:
        events = [make_event(f"evt-{i}") for i in range(6)]
        entries = [make_entry(i + 1, f"evt-{i}") for i in range(6)]
        llm = FakeLLM([as_json(make_payload()) for _ in range(6)])

        result = build_commentaries(entries, events, llm)

        assert len(result) == 5
        assert len(llm.prompts) == 5

    def test_dedupes_same_event(self) -> None:
        events = [make_event("evt-1")]
        entries = [make_entry(1, "evt-1"), make_entry(2, "evt-1")]
        llm = FakeLLM([as_json(make_payload())])

        result = build_commentaries(entries, events, llm)

        assert len(result) == 1
        assert len(llm.prompts) == 1

    def test_skips_missing_event(self) -> None:
        entries = [make_entry(1, "evt-missing"), make_entry(2, "evt-1")]
        llm = FakeLLM([as_json(make_payload())])

        result = build_commentaries(entries, [make_event("evt-1")], llm)

        assert [item.event_id for item in result] == ["evt-1"]
        assert len(llm.prompts) == 1

    def test_llm_exception_skips_entry(self, caplog: pytest.LogCaptureFixture) -> None:
        def boom(_: str) -> str:
            raise RuntimeError("模拟 LLM 故障")

        with caplog.at_level(logging.ERROR):
            result = build_commentaries([make_entry()], [make_event()], boom)

        assert result == []
        assert any("LLM 调用失败" in record.message for record in caplog.records)

    def test_invalid_json_skips_entry(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = build_commentaries([make_entry()], [make_event()], FakeLLM(["不是 JSON"]))

        assert result == []
        assert any("未通过校验" in record.message for record in caplog.records)

    def test_fenced_json_accepted(self) -> None:
        text = "说明如下：\n```json\n" + as_json(make_payload()) + "\n```\n以上。"
        result = build_commentaries([make_entry()], [make_event()], FakeLLM([text]))

        assert len(result) == 1

    def test_item_evidence_used_for_numbers(self) -> None:
        event = make_event("evt-1", item_ids=["item-1"], summary="摘要不含数字。")
        evidence = {"item-1": "官方披露本次融资 1.5 亿美元，估值未公开。"}
        payload = make_payload(
            headline="某公司融资 1.5 亿美元",
            fact_text="官方披露本次融资 1.5 亿美元。",
            evidence_ids=["item-1"],
        )
        llm = FakeLLM([as_json(payload)])

        result = build_commentaries([make_entry()], [event], llm, item_evidence=evidence)

        assert len(result) == 1
        assert "1.5 亿美元" in llm.prompts[0]

    def test_item_evidence_rejects_missing_item(self) -> None:
        event = make_event("evt-1", item_ids=["item-1"], summary="摘要不含数字。")
        evidence = {"other-item": "正文"}
        payload = make_payload(evidence_ids=["item-1"])

        result = build_commentaries(
            [make_entry()], [event], FakeLLM([as_json(payload)]), item_evidence=evidence
        )

        assert result == []


class TestFactObjectivity:
    def test_fact_without_evidence_rejected(self, caplog: pytest.LogCaptureFixture) -> None:
        payload = make_payload(evidence_ids=[])
        with caplog.at_level(logging.WARNING):
            result = build_commentaries([make_entry()], [make_event()], FakeLLM([as_json(payload)]))

        assert result == []
        assert any("evidence_ids" in record.message for record in caplog.records)

    def test_fact_with_unknown_evidence_rejected(self) -> None:
        payload = make_payload(evidence_ids=["item-nope"])

        result = build_commentaries([make_entry()], [make_event()], FakeLLM([as_json(payload)]))

        assert result == []

    def test_fact_number_not_in_evidence_rejected(self, caplog: pytest.LogCaptureFixture) -> None:
        payload = make_payload(fact_text="该公司今日融资 99 亿美元。")
        with caplog.at_level(logging.WARNING):
            result = build_commentaries([make_entry()], [make_event()], FakeLLM([as_json(payload)]))

        assert result == []
        assert any("数字" in record.message for record in caplog.records)

    def test_percentage_number_accepted(self) -> None:
        event = make_event(summary="该公司营收同比增长 30%。")
        payload = make_payload(fact_text="该公司营收同比增长 30%。")

        result = build_commentaries([make_entry()], [event], FakeLLM([as_json(payload)]))

        assert len(result) == 1

    def test_banned_word_in_fact_rejected(self) -> None:
        event = make_event(summary="该公司发布 2.1 版本，这是一次颠覆式更新。")
        payload = make_payload(fact_text="该公司发布 2.1 版本，这是一次颠覆式更新。")

        result = build_commentaries([make_entry()], [event], FakeLLM([as_json(payload)]))

        assert result == []

    def test_banned_word_in_analysis_sanitized(self) -> None:
        payload = make_payload(why_matters="这是一次重磅发布。")

        result = build_commentaries([make_entry()], [make_event()], FakeLLM([as_json(payload)]))

        assert len(result) == 1
        assert "重磅" not in result[0].why_matters

    def test_single_source_flag_auto_set(self) -> None:
        event = make_event(independent_sources=1)
        result = build_commentaries([make_entry()], [event], FakeLLM([as_json(make_payload())]))

        assert result[0].facts[0].single_source is True

    def test_counterpoint_fallback(self) -> None:
        result = build_commentaries(
            [make_entry()], [make_event()], FakeLLM([as_json(make_payload(counterpoint=""))])
        )

        assert result[0].counterpoint == "暂无公开反方信息"

    def test_headline_falls_back_to_event_title(self) -> None:
        payload = make_payload()
        payload.pop("headline")

        result = build_commentaries([make_entry()], [make_event()], FakeLLM([as_json(payload)]))

        assert result[0].headline == _EVENT_TITLE

    def test_wrong_event_id_corrected(self) -> None:
        result = build_commentaries(
            [make_entry()], [make_event()], FakeLLM([as_json(make_payload(event_id="evt-x"))])
        )

        assert result[0].event_id == "evt-1"


class TestParseCommentary:
    def test_requires_facts(self) -> None:
        cfg = config_from_mapping({})
        payload = make_payload()
        payload["facts"] = []

        with pytest.raises(EditorialValidationError):
            parse_commentary(payload, make_event(), config=cfg)

    def test_requires_facts_array(self) -> None:
        cfg = config_from_mapping({})
        payload = make_payload()
        payload.pop("facts")

        with pytest.raises(EditorialValidationError):
            parse_commentary(payload, make_event(), config=cfg)

    def test_drops_only_invalid_fact(self) -> None:
        cfg = config_from_mapping({})
        payload = make_payload()
        payload["facts"].append(
            {"text": "没有出处的传闻", "evidence_ids": [], "single_source": False}
        )

        commentary = parse_commentary(payload, make_event(), config=cfg)

        assert len(commentary.facts) == 1
        assert commentary.facts[0].evidence_ids


class TestJsonExtraction:
    def test_plain(self) -> None:
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_code_fence_and_chatter(self) -> None:
        text = '前言\n```json\n{"a": {"b": 2}}\n```\n后记'

        assert extract_json_object(text)["a"]["b"] == 2

    def test_leading_array_wrapped_object(self) -> None:
        text = '候选：{"event_id": "e", "facts": []} 结束'

        assert extract_json_object(text)["event_id"] == "e"

    def test_invalid_raises(self) -> None:
        with pytest.raises(EditorialValidationError):
            extract_json_object("完全不是 JSON")


def test_default_config_loaded() -> None:
    cfg = load_config("config/editorial_v3.yaml")

    assert cfg.report_top_n == 5
