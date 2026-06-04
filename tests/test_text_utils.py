"""Tests for ai_news_podcast.text_utils — shared TTS text cleaning."""

from __future__ import annotations

import pytest

from ai_news_podcast.text_utils import (
    RE_FACT_TAG,
    RE_MOOD_TAG,
    RE_NON_HOST_BRACKET,
    clean_tts_text,
)

# ---------------------------------------------------------------------------
# Basic edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self) -> None:
        assert clean_tts_text("") == ""

    def test_whitespace_only(self) -> None:
        assert clean_tts_text("   \n\n  ") == ""

    def test_plain_text_unchanged(self) -> None:
        text = "今天天气不错，适合出门散步。"
        assert clean_tts_text(text) == text


# ---------------------------------------------------------------------------
# Tag removal
# ---------------------------------------------------------------------------

class TestFactTagRemoval:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("[FACT] 这是事实", "这是事实"),
            ("[INFERENCE] 这是推论", "这是推论"),
            ("[OPINION] 这是观点", "这是观点"),
            ("[FACT]第一段[INFERENCE]第二段", "第一段第二段"),
            ("[FACT]  带空格", "带空格"),
        ],
    )
    def test_fact_tags_removed(self, input_text: str, expected: str) -> None:
        assert clean_tts_text(input_text) == expected


class TestMoodTagRemoval:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("[mood:happy] 今天很开心", "今天很开心"),
            ("[mood:sad] 好难过", "好难过"),
            ("[mood:excited] 太棒了", "太棒了"),
            ("[mood:neutral] 平淡", "平淡"),
            ("[mood:very-happy] 开心", "开心"),
        ],
    )
    def test_mood_tags_removed(self, input_text: str, expected: str) -> None:
        assert clean_tts_text(input_text) == expected


class TestEmojiParenRemoval:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("真的吗（doge）", "真的吗"),
            ("不会吧(狗头)", "不会吧"),
            ("哈哈（笑）", "哈哈"),
            ("手动滑稽", "手动滑稽"),  # not in parentheses — kept
            ("（手动狗头）", ""),
            ("(bushi)", ""),
            ("(划掉)", ""),
        ],
    )
    def test_emoji_parens_removed(self, input_text: str, expected: str) -> None:
        result = clean_tts_text(input_text)
        assert result == expected


class TestFancyQuotesRemoval:
    def test_fancy_quotes_removed(self) -> None:
        assert clean_tts_text("「你好」『世界』【测试】") == "你好世界测试"


class TestEmptyParenRemoval:
    def test_empty_parens_removed(self) -> None:
        assert clean_tts_text("前缀（）后缀") == "前缀后缀"

    def test_empty_parens_with_spaces(self) -> None:
        assert clean_tts_text("前缀( )后缀") == "前缀后缀"


# ---------------------------------------------------------------------------
# Host tag preservation
# ---------------------------------------------------------------------------

class TestHostTagPreservation:
    def test_host_a_preserved(self) -> None:
        text = "[Host A] 大家好"
        assert clean_tts_text(text) == text

    def test_host_b_preserved(self) -> None:
        text = "[Host B] 欢迎收听"
        assert clean_tts_text(text) == text

    def test_non_host_brackets_removed(self) -> None:
        assert clean_tts_text("[旁白] 这段话") == "这段话"

    def test_mixed_host_and_other_brackets(self) -> None:
        text = "[Host A] 你好 [注释] [Host B] 再见"
        assert clean_tts_text(text) == "[Host A] 你好 [Host B] 再见"


# ---------------------------------------------------------------------------
# Escaped newline handling
# ---------------------------------------------------------------------------

class TestEscapedNewlines:
    def test_literal_backslash_n_replaced(self) -> None:
        assert clean_tts_text("第一行\\n第二行") == "第一行\n第二行"

    def test_multiple_escaped_newlines(self) -> None:
        assert clean_tts_text("A\\n\\nB") == "A\n\nB"


# ---------------------------------------------------------------------------
# SSML handling
# ---------------------------------------------------------------------------

class TestSSMLHandling:
    def test_ssml_preserved_by_default(self) -> None:
        text = '<speak><voice name="zh-CN-XiaoxiaoNeural">你好</voice></speak>'
        result = clean_tts_text(text)
        assert "<speak" in result
        assert "<voice" in result

    def test_ssml_stripped_when_preserve_false(self) -> None:
        text = '<speak><voice name="zh-CN-XiaoxiaoNeural">你好</voice></speak>'
        result = clean_tts_text(text, preserve_ssml=False)
        assert "<speak" not in result
        assert "<voice" not in result
        assert "你好" in result

    def test_non_ssml_html_stripped_by_default(self) -> None:
        text = "<b>粗体</b>和<i>斜体</i>"
        result = clean_tts_text(text)
        assert "<b>" not in result
        assert "粗体" in result

    def test_voice_tag_triggers_ssml_detection(self) -> None:
        text = '<voice name="zh-CN-YunxiNeural">你好</voice>'
        result = clean_tts_text(text)
        assert "<voice" in result


# ---------------------------------------------------------------------------
# Punctuation compression
# ---------------------------------------------------------------------------

class TestPunctuationCompression:
    def test_repeated_commas(self) -> None:
        assert clean_tts_text("你好，，，世界") == "你好，世界"

    def test_repeated_english_commas(self) -> None:
        assert clean_tts_text("hello,,,world") == "hello，world"

    def test_repeated_periods(self) -> None:
        assert clean_tts_text("结束。。。") == "结束。"

    def test_repeated_english_periods(self) -> None:
        assert clean_tts_text("end...") == "end。"


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------

class TestWhitespaceNormalisation:
    def test_multiple_spaces_collapsed(self) -> None:
        assert clean_tts_text("你好   世界") == "你好 世界"

    def test_tabs_collapsed(self) -> None:
        assert clean_tts_text("你好\t\t世界") == "你好 世界"

    def test_triple_newlines_compressed(self) -> None:
        assert clean_tts_text("A\n\n\nB") == "A\n\nB"

    def test_leading_trailing_whitespace_stripped(self) -> None:
        assert clean_tts_text("  你好  ") == "你好"

    def test_line_leading_trailing_stripped(self) -> None:
        assert clean_tts_text("  行1  \n  行2  ") == "行1\n行2"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_clean_tts_text_idempotent(self) -> None:
        text = "[FACT] 测试（doge）[mood:happy]  你好，，，世界\\n\\n\\n结束"
        first = clean_tts_text(text)
        second = clean_tts_text(first)
        assert first == second

    def test_ssml_idempotent(self) -> None:
        text = '<speak><voice name="zh-CN-XiaoxiaoNeural">[FACT] 测试</voice></speak>'
        first = clean_tts_text(text)
        second = clean_tts_text(first)
        assert first == second


# ---------------------------------------------------------------------------
# Regex constant exports
# ---------------------------------------------------------------------------

class TestRegexConstants:
    def test_re_fact_tag_matches(self) -> None:
        assert RE_FACT_TAG.search("[FACT]")
        assert RE_FACT_TAG.search("[INFERENCE]")
        assert RE_FACT_TAG.search("[OPINION]")

    def test_re_mood_tag_matches(self) -> None:
        assert RE_MOOD_TAG.search("[mood:happy]")
        assert RE_MOOD_TAG.search("[mood:sad]")

    def test_re_non_host_bracket_preserves_host(self) -> None:
        m = RE_NON_HOST_BRACKET.search("[Host A]")
        assert m is None
        m = RE_NON_HOST_BRACKET.search("[Host B]")
        assert m is None
        m = RE_NON_HOST_BRACKET.search("[other]")
        assert m is not None
