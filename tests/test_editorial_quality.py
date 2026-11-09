"""Tests for business/editorial/quality.py 与配置加载（纯函数，不联网）。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ai_news_podcast.business.contracts import Commentary, Fact
from ai_news_podcast.business.editorial import check_script, commentary_quality
from ai_news_podcast.business.editorial.config import config_from_mapping, load_config
from ai_news_podcast.business.editorial.quality import (
    chapter_titles,
    count_chapters,
    count_spoken_chars,
    find_banned_words,
    host_char_counts,
    host_char_ratio,
    unmarked_lines,
)

_SMALL = {
    "min_chars": 5,
    "max_chars": 100,
    "chapters": 1,
    "min_chapters": 1,
    "max_chapters": 1,
    "min_host_char_ratio": 0.25,
    "max_unmarked_ratio": 0.2,
}


def small_config():
    return config_from_mapping(dict(_SMALL))


_VALID = "[CHAPTER] 开场\n[Host A] 你好世界\n[Host B] 大家好呀\n"


class TestScriptMetrics:
    def test_count_spoken_chars_ignores_markers(self) -> None:
        assert count_spoken_chars(_VALID) == 4 + 4

    def test_host_char_counts(self) -> None:
        assert host_char_counts(_VALID) == (4, 4)

    def test_host_char_ratio(self) -> None:
        assert host_char_ratio(_VALID) == pytest.approx(0.5)

    def test_ratio_without_hosts(self) -> None:
        assert host_char_ratio("[CHAPTER] 开场\n随意一句话\n") == 0.0

    def test_count_chapters_and_titles(self) -> None:
        script = "[CHAPTER] 开场\n[Host A] 你好\n[CHAPTER] 收尾\n[Host B] 再见"

        assert count_chapters(script) == 2
        assert chapter_titles(script) == ["开场", "收尾"]

    def test_unmarked_lines(self) -> None:
        script = "[CHAPTER] 开场\n[Host A] 台词\n（此处应有音乐）\n## 标题\n"

        assert unmarked_lines(script) == ["（此处应有音乐）", "## 标题"]

    def test_find_banned_words(self) -> None:
        assert find_banned_words("这是一次重磅且颠覆的更新", ["重磅", "颠覆", "炸裂"]) == [
            "重磅",
            "颠覆",
        ]


class TestCheckScript:
    def test_valid_script_passes(self) -> None:
        report = check_script(_VALID, small_config())

        assert report["ok"], report["issues"]
        assert report["chars"] == 8
        assert report["chapters"] == 1
        assert report["unmarked_ratio"] == 0.0
        assert report["banned_words"] == []

    def test_flags_chars_and_chapters(self) -> None:
        config = config_from_mapping({**_SMALL, "min_chars": 100, "max_chars": 200})

        report = check_script("[CHAPTER] 开场\n[Host A] 短\n", config)

        assert not report["ok"]
        assert any("正文字数" in issue for issue in report["issues"])
        assert any("主持人不均衡" in issue or "章节数" in issue for issue in report["issues"])

    def test_flags_host_imbalance(self) -> None:
        script = "[CHAPTER] 开场\n[Host A] " + ("长" * 90) + "\n[Host B] 短\n"

        report = check_script(script, small_config())

        assert not report["ok"]
        assert any("主持人不均衡" in issue for issue in report["issues"])

    def test_flags_banned_words(self) -> None:
        script = "[CHAPTER] 开场\n[Host A] 这是一次重磅发布\n[Host B] 确实颠覆\n"

        report = check_script(script, small_config())

        assert not report["ok"]
        assert set(report["banned_words"]) == {"重磅", "颠覆"}
        assert any("禁用词" in issue for issue in report["issues"])

    def test_flags_unmarked_lines(self) -> None:
        script = "\n".join(["# 播客文稿"] * 5) + "\n[CHAPTER] 开场\n[Host A] 你好\n"

        report = check_script(script, small_config())

        assert not report["ok"]
        assert any("未标记行" in issue for issue in report["issues"])

    def test_no_host_lines_flagged(self) -> None:
        report = check_script("[CHAPTER] 开场\n", small_config())

        assert not report["ok"]
        assert any("没有任何" in issue for issue in report["issues"])


class TestCommentaryQuality:
    def test_reports_missing_evidence_and_banned(self) -> None:
        commentary = Commentary(
            event_id="evt-1",
            headline="标题",
            why_matters="这是一次重磅发布",
            facts=[Fact(text="没有出处的事实")],
        )

        report = commentary_quality(commentary, small_config())

        assert not report["ok"]
        assert report["facts_without_evidence"] == ["没有出处的事实"]
        assert "重磅" in report["banned_words"]
        assert any("counterpoint" in warning for warning in report["warnings"])

    def test_single_source_ratio_and_fact_chars(self) -> None:
        config = config_from_mapping({**_SMALL, "min_fact_chars": 1, "max_fact_chars": 10})
        commentary = Commentary(
            event_id="evt-1",
            headline="标题",
            counterpoint="暂无公开反方信息",
            sources=[{"label": "官方", "link": "https://example.com", "official": True}],
            facts=[
                Fact(text="短事实", evidence_ids=["item-1"], single_source=True),
                Fact(text="另一条短事实", evidence_ids=["item-2"]),
            ],
        )

        report = commentary_quality(commentary, config)

        assert report["ok"]
        assert report["single_source_facts"] == 1
        assert report["single_source_ratio"] == pytest.approx(0.5)
        assert report["fact_chars_in_range"] is True

    def test_fact_chars_out_of_range_warns(self) -> None:
        config = config_from_mapping({**_SMALL, "min_fact_chars": 50, "max_fact_chars": 100})
        commentary = Commentary(
            event_id="evt-1",
            headline="标题",
            facts=[Fact(text="太短", evidence_ids=["item-1"])],
        )

        report = commentary_quality(commentary, config)

        assert report["fact_chars_in_range"] is False
        assert any("fact 字数" in warning for warning in report["warnings"])


class TestConfigLoading:
    def test_repo_config(self) -> None:
        path = Path(__file__).resolve().parents[1] / "config" / "editorial_v3.yaml"
        cfg = load_config(path)

        assert cfg.report_top_n == 5
        assert (cfg.target_minutes, cfg.min_chars, cfg.max_chars) == (15, 3000, 3600)
        assert (cfg.chapters, cfg.min_chapters, cfg.max_chapters) == (4, 3, 4)
        assert cfg.host_a == "林夏" and cfg.host_b == "陈默"
        assert "颠覆" in cfg.banned_words
        assert (cfg.fact_min_chars, cfg.fact_max_chars) == (150, 300)

    def test_section_overrides_flat(self) -> None:
        cfg = config_from_mapping(
            {
                "report_top_n": 5,
                "stage4": {"report_top_n": 3},
                "stage5": {"min_chars": 100, "max_chars": 200},
                "commentary": {"min_fact_chars": 10, "max_fact_chars": 20},
            }
        )

        assert cfg.report_top_n == 3
        assert (cfg.min_chars, cfg.max_chars) == (100, 200)
        assert (cfg.fact_min_chars, cfg.fact_max_chars) == (10, 20)

    def test_hosts_mapping(self) -> None:
        cfg = config_from_mapping({"hosts": {"host_a": "甲", "host_b": "乙"}})

        assert (cfg.host_a, cfg.host_b) == ("甲", "乙")

    def test_missing_file_falls_back(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            cfg = load_config(tmp_path / "nope.yaml")

        assert cfg.report_top_n == 5
        assert any("不存在" in record.message for record in caplog.records)

    def test_invalid_char_range_raises(self) -> None:
        with pytest.raises(ValueError, match="min_chars"):
            config_from_mapping({"min_chars": 500, "max_chars": 100})

    def test_invalid_chapter_range_raises(self) -> None:
        with pytest.raises(ValueError, match="章节"):
            config_from_mapping({"chapters": 5, "min_chapters": 3, "max_chapters": 4})

    def test_invalid_host_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="min_host_char_ratio"):
            config_from_mapping({"min_host_char_ratio": 0.8})
