"""Tests for ai_news_podcast.pipeline.scriptwriter pure functions."""

from __future__ import annotations

from datetime import datetime

from ai_news_podcast.pipeline.scriptwriter import (
    _cn_date,
    _replace_banned_words,
    _sanitize_for_tts,
    check_banned_words,
)


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
        assert _sanitize_for_tts("line1\\nline2") == "line1\nline2"

    def test_removes_tags(self) -> None:
        assert _sanitize_for_tts("[FACT] hello [INFERENCE] world") == "hello world"

    def test_removes_html(self) -> None:
        assert _sanitize_for_tts("<p>paragraph</p>") == "paragraph"

    def test_compresses_punctuation(self) -> None:
        assert _sanitize_for_tts("你好，，，世界") == "你好，世界"
        assert _sanitize_for_tts("你好。。。世界") == "你好。世界"

    def test_empty_string(self) -> None:
        assert _sanitize_for_tts("") == ""


class TestCnDate:
    def test_format(self) -> None:
        dt = datetime(2024, 5, 20)
        assert _cn_date(dt) == "2024年5月20日"
