"""Tests for business/editorial/script.py（Stage 5 文稿，假 LLM，不联网）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ai_news_podcast.business.contracts import (
    Commentary,
    Event,
    Fact,
    Labels,
    ShortlistEntry,
)
from ai_news_podcast.business.editorial import build_script, normalize_script, quality
from ai_news_podcast.business.editorial.config import load_config

_BASE = "这是用于测试的播客对白内容，讨论事实与观点以及行业影响。"


def utterance(speaker: str, chars: int) -> str:
    text = (_BASE * (chars // len(_BASE) + 1))[:chars]
    return f"[Host {speaker}] {text}"


def make_script(total: int = 3300, chapters: int = 4) -> str:
    """生成一份字数/章节/主持均衡都达标的标准文稿。"""
    titles = ["开场", "主推一", "主推二", "收尾"][:chapters]
    per_chapter = total // chapters
    lines: list[str] = []
    for index in range(chapters):
        lines.append(f"[CHAPTER] {titles[index]}")
        remaining = per_chapter
        speaker = "A" if index % 2 == 0 else "B"
        while remaining > 0:
            chunk = min(120, remaining)
            lines.append(utterance(speaker, chunk))
            remaining -= chunk
            speaker = "B" if speaker == "A" else "A"
    return "\n".join(lines) + "\n"


def make_entry(rank: int = 1, event_id: str = "evt-1") -> ShortlistEntry:
    return ShortlistEntry(rank=rank, event_id=event_id, score=4.0)


def make_event(event_id: str = "evt-1") -> Event:
    return Event(
        event_id=event_id,
        representative_title="某公司发布 2.1 版本",
        summary="该公司今日发布 2.1 版本，融资 10 亿美元。",
        item_ids=["item-1"],
        independent_sources=2,
        labels=Labels(primary_action="发布", topics=["技术"]),
    )


def make_commentary(event_id: str = "evt-1", headline: str = "某公司发布 2.1 版本") -> Commentary:
    return Commentary(
        event_id=event_id,
        headline=headline,
        one_liner="一句话摘要",
        facts=[Fact(text="该公司今日发布 2.1 版本，融资 10 亿美元。", evidence_ids=["item-1"])],
        why_matters="版本迭代会影响工具链。",
        counterpoint="暂无公开反方信息",
    )


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("FakeLLM 没有更多预置响应")
        return self._responses.pop(0)


def write_config(tmp_path: Path, **overrides: Any) -> str:
    config: dict[str, Any] = {
        "report_top_n": 5,
        "target_minutes": 1,
        "min_chars": 100,
        "max_chars": 200,
        "chapters": 2,
        "min_chapters": 1,
        "max_chapters": 2,
        "max_rewrites": 1,
    }
    config.update(overrides)
    path = tmp_path / "editorial_test.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return str(path)


class TestBuildScript:
    def test_returns_quality_ok_script(self) -> None:
        script = make_script(3300, 4)
        llm = FakeLLM([script])

        result = build_script([make_entry()], [make_event()], [make_commentary()], llm)

        report = quality(result)
        assert report["ok"], report["issues"]
        assert report["chars"] == 3300
        assert report["chapters"] == 4
        assert len(llm.prompts) == 1

    def test_prompt_carries_facts_and_targets(self) -> None:
        llm = FakeLLM([make_script(3300, 4)])

        build_script([make_entry()], [make_event()], [make_commentary()], llm)

        prompt = llm.prompts[0]
        assert "某公司发布 2.1 版本" in prompt
        assert "item-1" in prompt
        assert "3000" in prompt and "3600" in prompt
        assert "[CHAPTER]" in prompt
        assert "林夏" in prompt and "陈默" in prompt
        assert "禁止引入点评之外" in prompt

    def test_rewrites_when_too_short(self) -> None:
        valid = make_script(3300, 4)
        llm = FakeLLM([make_script(600, 2), valid])

        result = build_script([make_entry()], [make_event()], [make_commentary()], llm)

        assert len(llm.prompts) == 2
        assert "未通过" in llm.prompts[1]
        assert quality(result)["ok"]

    def test_rewrites_when_too_long(self) -> None:
        valid = make_script(3300, 4)
        llm = FakeLLM([make_script(4000, 4), valid])

        result = build_script([make_entry()], [make_event()], [make_commentary()], llm)

        assert len(llm.prompts) == 2
        assert quality(result)["ok"]

    def test_keeps_better_version_when_rewrite_worse(self) -> None:
        first = make_script(3300, 1)
        second = make_script(600, 2)
        llm = FakeLLM([first, second])

        result = build_script([make_entry()], [make_event()], [make_commentary()], llm)

        assert len(llm.prompts) == 2
        assert quality(result)["chars"] == 3300

    def test_trims_when_rewrite_still_too_long(self, tmp_path: Path) -> None:
        llm = FakeLLM([make_script(400, 2), make_script(500, 2)])

        result = build_script(
            [make_entry()],
            [make_event()],
            [make_commentary()],
            llm,
            config_path=write_config(tmp_path),
        )

        assert quality(result, config_path=write_config(tmp_path))["chars"] <= 200

    def test_short_script_returned_with_issue(self, tmp_path: Path) -> None:
        config_path = write_config(tmp_path)
        llm = FakeLLM([make_script(50, 1), make_script(50, 1)])

        result = build_script(
            [make_entry()], [make_event()], [make_commentary()], llm, config_path=config_path
        )

        report = quality(result, config_path=config_path)
        assert not report["ok"]
        assert report["chars"] == 50
        assert len(llm.prompts) == 2

    def test_llm_error_propagates(self) -> None:
        def boom(_: str) -> str:
            raise RuntimeError("模拟 LLM 故障")

        with pytest.raises(RuntimeError):
            build_script([make_entry()], [make_event()], [make_commentary()], boom)

    def test_commentaries_ordered_by_shortlist_rank(self) -> None:
        first = make_commentary("evt-1", headline="头号事件")
        second = make_commentary("evt-2", headline="二号事件")
        llm = FakeLLM([make_script(3300, 4)])
        entries = [make_entry(1, "evt-1"), make_entry(2, "evt-2")]

        build_script(entries, [make_event("evt-1"), make_event("evt-2")], [second, first], llm)

        prompt = llm.prompts[0]
        assert prompt.index("头号事件") < prompt.index("二号事件")

    def test_accepts_item_evidence(self) -> None:
        llm = FakeLLM([make_script(3300, 4)])

        result = build_script(
            [make_entry()],
            [make_event()],
            [make_commentary()],
            llm,
            item_evidence={"item-1": "补充正文"},
        )

        assert quality(result)["ok"]


class TestNormalizeScript:
    def test_named_hosts_and_bold(self) -> None:
        raw = (
            "林夏：你好，欢迎收听。\n"
            "陈默：大家好。\n"
            "[CHAPTER] 一、今日头条\n"
            "**林夏**：先看第一条。\n"
            "[Host B] 好的。\n"
        )

        result = normalize_script(raw)

        assert "[Host A] 你好，欢迎收听。" in result
        assert "[Host B] 大家好。" in result
        assert "[CHAPTER] 今日头条" in result
        assert "林夏：" not in result

    def test_inline_script_split(self) -> None:
        raw = "[Host A] 第一句 [Host B] 第二句 [CHAPTER] 主题 [Host A] 第三句"

        result = normalize_script(raw)

        assert result.splitlines()[0] == "[Host A] 第一句"
        assert result.splitlines()[1] == "[Host B] 第二句"
        assert result.splitlines()[2] == "[CHAPTER] 主题"

    def test_continuation_merged_into_previous_speaker(self) -> None:
        raw = "[Host A] 第一句，\n继续第二句。\n[Host B] 好的。"

        result = normalize_script(raw)

        assert "[Host A] 第一句，继续第二句。" in result

    def test_drops_markdown_and_thinking_lines(self) -> None:
        raw = "# 播客文稿\n```\n（注：以下是正文）\n[Host A] 正文开始。\n```\n"

        result = normalize_script(raw)

        assert "#" not in result
        assert "```" not in result
        assert "注：" not in result
        assert result.strip() == "[Host A] 正文开始。"

    def test_empty_output(self) -> None:
        assert normalize_script("   \n\n") == ""

    def test_chapter_trailing_punctuation_cleaned(self) -> None:
        result = normalize_script("[CHAPTER] 收尾。")

        assert result == "[CHAPTER] 收尾\n"


def test_repo_config_defaults() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "editorial_v3.yaml"
    cfg = load_config(config_path)

    assert (cfg.min_chars, cfg.max_chars, cfg.chapters) == (3000, 3600, 4)
