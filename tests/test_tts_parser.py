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

    def test_repaired_xml_tags_laughter(self) -> None:
        # A long text that gets split right in the middle of <laughing>...</laughing>
        text = "这是一个非常有趣的科技新闻，<laughing>大模型真的在日常工作中非常省电！</laughing>我觉得很神奇。"
        # split by punctuation:
        # parts:
        # 1. "这是一个非常有趣的科技新闻，"
        # 2. "<laughing>大模型真的在日常工作中非常省电！"
        # 3. "</laughing>我觉得很神奇。"
        # set max_chars=40, so part 1 + part 2 exceeds 40, they will split
        sentences = split_text_into_sentences(text, max_chars=30)

        # Check that open and close tags are paired correctly in each sentence
        for s in sentences:
            assert s.count("<laughing>") == s.count("</laughing>")
            assert s.count("<strong>") == s.count("</strong>")

    def test_repaired_xml_tags_strong(self) -> None:
        text = "重点内容是，<strong>这非常重要！</strong>我们必须要记住。"
        sentences = split_text_into_sentences(text, max_chars=20)
        for s in sentences:
            assert s.count("<strong>") == s.count("</strong>")
