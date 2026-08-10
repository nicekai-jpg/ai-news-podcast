"""Tests for tts_parser — splitting and parsing script dialogue."""

from __future__ import annotations

from ai_news_podcast.pipeline.tts_parser import (
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
