"""Tests for tts_parser — splitting and parsing script dialogue."""

from __future__ import annotations

from ai_news_podcast.business.pipeline.tts_parser_service import (
    has_renderable_text,
    split_text_into_sentences,
)


class TestSplitTextIntoSentences:
    def test_basic_splitting(self) -> None:
        text = "今天天气很好，我们一起去公园吧。大家都觉得这个主意不错。"
        sentences = split_text_into_sentences(text, max_chars=15)
        assert len(sentences) == 3
        assert sentences[0] == "今天天气很好，"
        assert sentences[1] == "我们一起去公园吧。"
        assert sentences[2] == "大家都觉得这个主意不错。"


class TestHasRenderableText:
    def test_letters_digits_cjk_are_renderable(self) -> None:
        assert has_renderable_text("你好世界")
        assert has_renderable_text("hello")
        assert has_renderable_text("2026")

    def test_pure_symbols_are_not_renderable(self) -> None:
        assert not has_renderable_text("")
        assert not has_renderable_text("   ")
        assert not has_renderable_text("……")
        assert not has_renderable_text("——！？")
        assert not has_renderable_text("🎉🎉")

    def test_mixed_symbol_and_text_is_renderable(self) -> None:
        assert has_renderable_text("……好的")


class TestSplitTextIntoSentencesDropsSymbolOnly:
    def test_symbol_only_segment_dropped(self) -> None:
        assert split_text_into_sentences("……") == []
        assert split_text_into_sentences("🎉🎉🎉") == []
        assert split_text_into_sentences("   ") == []

    def test_symbol_surrounded_by_text_kept(self) -> None:
        assert split_text_into_sentences("你好……", max_chars=80) == ["你好……"]
