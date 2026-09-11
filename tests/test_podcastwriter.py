"""Tests for ai_news_podcast.pipeline.podcastwriter pure functions."""

from __future__ import annotations

from datetime import datetime

from ai_news_podcast.pipeline.podcastwriter import (
    _cn_date,
    _replace_banned_words,
    check_banned_words,
)
from ai_news_podcast.text_utils import clean_tts_text


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
        from unittest.mock import patch

        from ai_news_podcast.pipeline import podcastwriter

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
                ]
            },
            "meta": {"pick_repo": "owner/hot"},
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

        with patch.object(podcastwriter, "_call_llm", side_effect=fake_llm):
            podcastwriter.generate_podcast(brief, episode_date=datetime(2026, 9, 11))

        assert "owner/hot" in captured[0]  # editor prompt 含雷达素材
        assert "项目雷达栏目规范" in captured[1]  # writer prompt 含雷达规则
