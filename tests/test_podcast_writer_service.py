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
            return "\n".join(
                [
                    "[Host A] 今天聊三条新闻，先看第一条的具体内容和背景。",
                    "[Host B] 好，这条的行业影响我也关注到了。",
                ]
                * 26
            )

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


class TestScriptQualityGate:
    """2026-11-05 之后新增的质量闸门：短稿/超长稿/85:15 敷衍稿/中文计划稿都要能挡住。"""

    @staticmethod
    def _balanced_script(pairs: int = 30) -> str:
        return "\n".join(
            [
                "[Host A] 今天这条新闻值得展开讲讲，先看它的背景和影响。",
                "[Host B] 同意，我把刚才那条的关键信息补充一下。",
            ]
            * pairs
        )

    def test_balanced_script_passes(self) -> None:
        from ai_news_podcast.business.pipeline.podcast_writer_service import _script_looks_valid

        assert _script_looks_valid(self._balanced_script()) is True

    def test_short_script_is_rejected(self) -> None:
        from ai_news_podcast.business.pipeline.podcast_writer_service import _script_looks_valid

        assert _script_looks_valid("[Host A] 你好。\n[Host B] 嗨。") is False

    def test_oversized_script_is_rejected(self) -> None:
        from ai_news_podcast.business.pipeline.podcast_writer_service import _script_looks_valid

        huge = "\n".join(
            [
                "[Host A] " + "很长的一段台词。" * 20,
                "[Host B] " + "另外一段很长的台词。" * 20,
            ]
            * 20
        )
        assert _script_looks_valid(huge) is False

    def test_chinese_planning_text_is_detected(self) -> None:
        from ai_news_podcast.business.pipeline.podcast_writer_service import _script_looks_valid

        planning = (
            "[Host A] 我先梳理第1段素材，注意口径和数字。\n"
            "[Host B] 要求：这一段要展开，控制在四十字以上。\n"
        ) * 10
        assert _script_looks_valid(planning) is False


class TestFallbackQuality:
    def test_fallback_has_chapters_and_no_numbered_phrasing(self) -> None:
        from ai_news_podcast.business.pipeline.podcast_writer_service import _build_fallback

        brief = {
            "stories": [
                {
                    "role": "main",
                    "representative_title": "某模型发布",
                    "context": {"factual_summary": ["English summary only here."]},
                },
                {
                    "role": "supporting",
                    "representative_title": "另一条新闻",
                    "context": {"factual_summary": ["另一条中文摘要。"]},
                },
            ],
            "radar": {"meta": {"pick_repo": "owner/repo"}, "projects": []},
        }
        script = _build_fallback(brief, datetime(2026, 11, 5), "今日AI看点")
        assert "[CHAPTER] 开场" in script
        assert "[CHAPTER] 收尾" in script
        assert "第一个大头条" not in script
        # 只有英文摘要时不要念英文原文
        assert "English summary only here." not in script
        assert "owner/repo" in script
        assert script.count("[Host A]") > 0 and script.count("[Host B]") > 0


class TestSalvageDialogue:
    """2026-11-06 事故回归：19k 字符的输出里混着写稿计划和成稿，不能整篇丢掉。"""

    @staticmethod
    def _dialogue_block(pairs: int = 30) -> str:
        return "\n".join(
            [
                "[CHAPTER] 开场",
                "[Host A] 听众朋友们好，欢迎收听今天的节目。",
                "[Host B] 大家好，我们来看几条重要的消息。",
            ]
            + [
                f"[Host A] 这是第{index}条的背景和影响分析，信息量不小。\n[Host B] 我把关键数字和判断补充一下。"
                for index in range(pairs)
            ]
        )

    def test_picks_longest_contiguous_run(self) -> None:
        from ai_news_podcast.business.pipeline.podcast_writer_service import (
            _salvage_dialogue,
            _script_looks_valid,
        )

        raw = (
            "Need no 好 start. Fine.\n"
            "[CHAPTER] 草稿\n"
            "[Host A] 这是草稿，不要用。\n"
            "\nLet's count rough.\n"
            "Chapter1 10 lines ~800?\n"
            "Total ~2770.\n"
            "We need to polish.\n" + self._dialogue_block()
        )
        salvaged = _salvage_dialogue(raw)
        assert "草稿" not in salvaged
        assert "We need" not in salvaged
        assert _script_looks_valid(salvaged) is True

    def test_empty_when_no_dialogue(self) -> None:
        from ai_news_podcast.business.pipeline.podcast_writer_service import _salvage_dialogue

        assert _salvage_dialogue("We need to plan.\nLet's count.") == ""


class TestFallbackLength:
    def test_fallback_is_listenable_length(self) -> None:
        from ai_news_podcast.business.pipeline.podcast_writer_service import _build_fallback

        stories = [
            {
                "role": "main" if index == 0 else "supporting",
                "representative_title": f"新闻标题{index}",
                "context": {
                    "factual_summary": [
                        f"这是第{index}条新闻的中文摘要，说明了事件背景、关键数据与对行业的实际影响，值得持续跟进。"
                    ]
                },
            }
            for index in range(5)
        ]
        brief = {"stories": stories, "radar": {"meta": {"pick_repo": "owner/repo"}}}
        script = _build_fallback(brief, datetime(2026, 11, 6), "今日AI看点")
        chars = len("".join(script.split()))
        assert chars >= 600
        assert "第一个大头条" not in script
        assert script.count("[CHAPTER]") >= 3
        assert script.count("[Host A]") >= 6 and script.count("[Host B]") >= 6
