"""Tests for ai_news_podcast.business.pipeline.podcast_writer_service pure functions."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from ai_news_podcast.business.pipeline import podcast_writer_service
from ai_news_podcast.business.pipeline.podcast_writer_service import (
    _cn_date,
    _replace_banned_words,
    check_banned_words,
)
from ai_news_podcast.data.text_utils_dao import clean_tts_text


class TestCheckBannedWords:
    def test_finds_matches(self) -> None:
        text = "这个产品真的炸裂，简直是王炸"
        found = check_banned_words(text)
        assert "炸裂" in found
        assert "王炸" in found

    def test_empty_when_clean(self) -> None:
        assert check_banned_words("这是一个正常的句子") == []

    def test_custom_banned_list(self) -> None:
        assert check_banned_words("hello world", banned=["world"]) == ["world"]


class TestReplaceBannedWords:
    def test_replaces_known_mappings(self) -> None:
        text = "这个结果炸裂，堪称王炸"
        result = _replace_banned_words(text)
        assert "炸裂" not in result
        assert "王炸" not in result
        assert "非常" in result
        assert "王牌" in result

    def test_removes_unmapped_words(self) -> None:
        text = "废话不多说，众所周知"
        result = _replace_banned_words(text)
        assert "废话不多说" not in result
        assert "众所周知" not in result


class TestSanitizeForTts:
    def test_escapes_literal_newlines(self) -> None:
        assert clean_tts_text("line1\\nline2") == "line1\nline2"

    def test_removes_tags(self) -> None:
        assert clean_tts_text("[FACT] hello [INFERENCE] world") == "hello world"

    def test_removes_html(self) -> None:
        assert clean_tts_text("<p>paragraph</p>") == "paragraph"

    def test_compresses_punctuation(self) -> None:
        assert clean_tts_text("你好，，，世界") == "你好，世界"
        assert clean_tts_text("你好。。。世界") == "你好。世界"

    def test_empty_string(self) -> None:
        assert clean_tts_text("") == ""


class TestCnDate:
    def test_format(self) -> None:
        dt = datetime(2024, 5, 20)
        assert _cn_date(dt) == "2024年5月20日"


class TestGeneratePodcastRadar:
    def test_generate_podcast_injects_radar_into_editor_prompt(self) -> None:
        """雷达素材必须进入 Editor prompt，writer prompt 收到 has_radar 标记。"""
        brief = {
            "stories": [
                {
                    "representative_title": "大新闻",
                    "role": "main",
                    "context": {"factual_summary": ["要点"], "sources_ranked": []},
                }
            ],
            "radar": {
                "projects": [
                    {
                        "repo": "owner/hot",
                        "url": "u",
                        "stars": 1500,
                        "delta_stars": 1000,
                        "language": "Python",
                        "license": "MIT",
                        "description": "hot harness",
                        "readme_excerpt": "pip install hot",
                    }
                ],
                "meta": {"pick_repo": "owner/hot"},
            },
        }
        captured: list[str] = []

        def fake_llm(prompt, cfg):
            captured.append(prompt)
            if len(captured) == 1:
                return (
                    "# 今日播报大纲\n\n## 金句\nx\n\n## 头条 1\n- **标题**: a\n- **摘要**: b\n\n"
                    "## 头条 2\n- **标题**: c\n- **摘要**: d\n\n## 项目雷达\n- **主推 [owner/hot]**"
                )
            return "[Host A] 我们聊聊刚过去的 radar-repo。\n[Host B] 好的，这个项目值得说说。"

        with patch.object(podcast_writer_service, "_call_llm", side_effect=fake_llm):
            podcast_writer_service.generate_podcast(brief, episode_date=datetime(2026, 9, 11))

        assert "owner/hot" in captured[0]  # editor prompt 含雷达素材
        assert "[主推]" in captured[0]  # 主推段(delta/许可证/语言/摘录格式)已渲染
        assert "项目雷达栏目规范" in captured[1]  # writer prompt 含雷达规则


def _brief() -> dict:
    return {
        "stories": [
            {
                "representative_title": "大新闻",
                "role": "main",
                "context": {"factual_summary": ["要点"], "sources_ranked": []},
            }
        ]
    }


_EDITOR_OUTLINE = "# 今日播报大纲\n\n## 金句\nx\n\n## 头条 1\n- **标题**: a\n- **摘要**: b\n"


class TestHostBalanceGate:
    """2026-11-05 事故回归：模型把写稿计划包在 [Host A] 里输出，Host B 只有 4 句。"""

    def test_imbalanced_writer_output_falls_back(self) -> None:
        calls: list[str] = []

        def fake_llm(prompt, cfg):
            calls.append(prompt)
            if len(calls) == 1:
                return _EDITOR_OUTLINE
            return "\n".join(["[Host A] 我先讲一遍今天的新闻。"] * 20 + ["[Host B] 嗯。"])

        with patch.object(podcast_writer_service, "_call_llm", side_effect=fake_llm):
            script, warnings = podcast_writer_service.generate_podcast(
                _brief(), episode_date=datetime(2026, 11, 5)
            )

        assert any("失衡" in warning for warning in warnings)
        # 降级后的模板脚本必须双主持均衡
        assert script.count("[Host A]") > 0
        assert script.count("[Host B]") > 0

    def test_retry_with_hard_constraint_recovers(self) -> None:
        calls: list[str] = []

        def fake_llm(prompt, cfg):
            calls.append(prompt)
            if len(calls) == 1:
                return _EDITOR_OUTLINE
            if len(calls) == 2:
                return "\n".join(["[Host A] 我先讲一遍。"] * 20 + ["[Host B] 嗯。"])
            return "[Host A] 今天聊三条新闻。\n[Host B] 好，先看第一条。"

        with patch.object(podcast_writer_service, "_call_llm", side_effect=fake_llm):
            script, warnings = podcast_writer_service.generate_podcast(
                _brief(), episode_date=datetime(2026, 11, 5)
            )

        assert "输出格式硬约束" in calls[2]
        assert not any("失衡" in warning for warning in warnings)
        assert "今天聊三条新闻" in script

    def test_thinking_leak_inside_host_tags_is_detected(self) -> None:
        leaked = (
            "[Host A] Let's plan dialogue structure:\n"
            "[Host A] Need length. Let's estimate Chinese chars.\n"
            "[Host B] 好。"
        )
        assert podcast_writer_service.contains_thinking_process(leaked) is True
