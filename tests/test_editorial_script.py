"""Tests for business/editorial/script.py（S6 文稿，假 LLM，不联网）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ai_news_podcast.business.contracts import Event, Labels, ShortlistEntry
from ai_news_podcast.business.editorial import build_script, normalize_script, quality
from ai_news_podcast.business.editorial.config import load_config

_BASE = "这是用于测试的播客对白内容，讨论事实与观点以及行业影响。"
EVIDENCE = {"item-1": "该公司今日发布 2.1 版本，融资 10 亿美元。"}


def utterance(speaker: str, chars: int) -> str:
    text = (_BASE * (chars // len(_BASE) + 1))[:chars]
    return f"[Host {speaker}] {text}"


def _config_baseline() -> tuple[int, int]:
    """取当前配置的「合格文稿」基准（字数、章节），避免测试写死旧配置值。

    2026-11-26 配置放宽（3000–3600 字/4 章 → 4000–4800 字/5 章）后，
    测试里写死的 3300/4 不再达标，会触发重写并耗尽 FakeLLM 响应。
    """
    from ai_news_podcast.business.editorial.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "editorial_v3.yaml")
    return (cfg.min_chars + cfg.max_chars) // 2, cfg.chapters


_OK_CHARS, _OK_CHAPTERS = _config_baseline()


def make_script(total: int | None = None, chapters: int | None = None) -> str:
    """生成一份字数/章节/主持均衡都达标的标准文稿（默认取配置中位值）。"""
    total = _OK_CHARS if total is None else total
    chapters = _OK_CHAPTERS if chapters is None else chapters
    titles = ["开场", "主推一", "主推二", "主推三", "主推四", "收尾"]
    titles = [*titles[: max(0, chapters - 1)], "收尾"] if chapters > 1 else titles[:1]
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
        script = make_script()
        llm = FakeLLM([script])

        result = build_script([make_entry()], [make_event()], EVIDENCE, llm)

        report = quality(result)
        assert report["ok"], report["issues"]
        assert report["chars"] == _OK_CHARS
        assert report["chapters"] == _OK_CHAPTERS
        assert len(llm.prompts) == 1

    def test_prompt_carries_facts_and_targets(self) -> None:
        llm = FakeLLM([make_script()])

        build_script([make_entry()], [make_event()], EVIDENCE, llm)

        prompt = llm.prompts[0]
        assert "某公司发布 2.1 版本" in prompt
        assert "item-1" in prompt
        from ai_news_podcast.business.editorial.config import load_config

        _cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "editorial_v3.yaml")
        assert str(_cfg.min_chars) in prompt and str(_cfg.max_chars) in prompt
        assert "[CHAPTER]" in prompt
        assert "林夏" in prompt and "陈默" in prompt
        assert "事实底稿" in prompt

    def test_rewrites_when_too_short(self) -> None:
        valid = make_script()
        llm = FakeLLM([make_script(600, 2), valid])

        result = build_script([make_entry()], [make_event()], EVIDENCE, llm)

        assert len(llm.prompts) == 2
        assert "未通过" in llm.prompts[1]
        assert quality(result)["ok"]

    def test_rewrites_when_too_long(self) -> None:
        valid = make_script()
        llm = FakeLLM([make_script(_OK_CHARS + 1200, _OK_CHAPTERS), valid])

        result = build_script([make_entry()], [make_event()], EVIDENCE, llm)

        assert len(llm.prompts) == 2
        assert quality(result)["ok"]

    def test_keeps_better_version_when_rewrite_worse(self) -> None:
        first = make_script(3300, 1)
        second = make_script(600, 2)
        llm = FakeLLM([first, second])

        result = build_script([make_entry()], [make_event()], EVIDENCE, llm)

        assert len(llm.prompts) == 2
        assert quality(result)["chars"] == 3300

    def test_trims_when_rewrite_still_too_long(self, tmp_path: Path) -> None:
        llm = FakeLLM([make_script(400, 2), make_script(500, 2)])

        result = build_script(
            [make_entry()],
            [make_event()],
            EVIDENCE,
            llm,
            config_path=write_config(tmp_path),
        )

        assert quality(result, config_path=write_config(tmp_path))["chars"] <= 200

    def test_short_script_returned_with_issue(self, tmp_path: Path) -> None:
        config_path = write_config(tmp_path)
        llm = FakeLLM([make_script(50, 1), make_script(50, 1)])

        result = build_script(
            [make_entry()], [make_event()], EVIDENCE, llm, config_path=config_path
        )

        report = quality(result, config_path=config_path)
        assert not report["ok"]
        assert report["chars"] == 50
        assert len(llm.prompts) == 2

    def test_llm_error_propagates(self) -> None:
        def boom(_: str) -> str:
            raise RuntimeError("模拟 LLM 故障")

        with pytest.raises(RuntimeError):
            build_script([make_entry()], [make_event()], EVIDENCE, boom)

    def test_events_ordered_by_shortlist_rank(self) -> None:
        llm = FakeLLM([make_script()])
        entries = [make_entry(1, "evt-1"), make_entry(2, "evt-2")]
        event_1 = make_event("evt-1")
        event_1.representative_title = "头号事件"
        event_2 = make_event("evt-2")
        event_2.representative_title = "二号事件"

        build_script(entries, [event_2, event_1], EVIDENCE, llm)

        prompt = llm.prompts[0]
        assert prompt.index("头号事件") < prompt.index("二号事件")

    def test_prompt_includes_evidence_text(self) -> None:
        llm = FakeLLM([make_script()])

        result = build_script([make_entry()], [make_event()], EVIDENCE, llm)

        assert quality(result)["ok"]
        assert "融资 10 亿美元" in llm.prompts[0]


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

    # 2026-11-26 放宽：15 分钟/3000–3600 字/4 章 → 20 分钟/3600–4400 字/5 章
    # （shortlist 8→15、report_top_n 5→8 之后，原时长装不下）
    # 字数按「模型实际生成语速」~182 字/分钟定，不是 TTS 的 220–260 字/分钟；
    # 按 TTS 语速定 4000 下限会让模型频繁达不到而触发重写。
    assert (cfg.min_chars, cfg.max_chars, cfg.chapters) == (3600, 4400, 5)
    assert (cfg.min_chapters, cfg.max_chapters) == (4, 6)
    assert cfg.target_minutes == 20


class TestScriptFactCheck:
    def test_rewrites_when_number_not_in_evidence(self) -> None:
        bad = make_script().replace(_BASE, "融资 99 亿美元。", 1)
        good = make_script()
        llm = FakeLLM([bad, good])

        result = build_script([make_entry()], [make_event()], EVIDENCE, llm)

        assert len(llm.prompts) == 2
        assert "99" in llm.prompts[1]  # 重写 prompt 里列出了找不到的数字
        assert "99" not in result

    def test_accepts_numbers_present_in_evidence(self) -> None:
        script = make_script()
        script = script.replace(_BASE, "发布 2.1 版本", 1)
        script = script.replace(_BASE, "融资 10 亿美元", 1)
        llm = FakeLLM([script])

        build_script([make_entry()], [make_event()], EVIDENCE, llm)

        assert len(llm.prompts) == 1  # 数字都能在底稿找到，不触发重写


class TestRadarSection:
    RADAR = (
        "以下是「项目雷达」栏目的结构化素材(与新闻无关,单独成栏)。\n"
        "- [主推] vercel-labs/agent-browser(⭐48463,较昨日 +957,语言:Rust,"
        "许可证:Apache-2.0):Browser automation CLI for AI agents"
    )

    def test_prompt_includes_radar_when_provided(self) -> None:
        llm = FakeLLM([make_script()])

        build_script([make_entry()], [make_event()], EVIDENCE, llm, radar_text=self.RADAR)

        prompt = llm.prompts[0]
        assert "项目雷达" in prompt
        assert "vercel-labs/agent-browser" in prompt

    def test_prompt_omits_radar_when_empty(self) -> None:
        llm = FakeLLM([make_script()])

        build_script([make_entry()], [make_event()], EVIDENCE, llm, radar_text="")

        assert "项目雷达" not in llm.prompts[0]

    def test_radar_numbers_pass_fact_check(self) -> None:
        script = make_script().replace(_BASE, "项目目前⭐48463，较昨日 +957。", 1)
        llm = FakeLLM([script])

        build_script([make_entry()], [make_event()], EVIDENCE, llm, radar_text=self.RADAR)

        assert len(llm.prompts) == 1  # 星数来自雷达素材，不应触发重写
