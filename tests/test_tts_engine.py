"""Tests for ai_news_podcast.pipeline.tts_engine pure functions."""

from __future__ import annotations

from ai_news_podcast.pipeline.tts_engine import (
    DialogueChunk,
    _clean_tts_text,
    parse_dialogue_chunks,
)


class TestCleanTtsText:
    def test_removes_fact_tags(self) -> None:
        text = "[FACT] This is true. [INFERENCE] Probably. [OPINION] I think so."
        assert _clean_tts_text(text) == "This is true. Probably. I think so."

    def test_removes_mood_tags(self) -> None:
        text = "[mood:excited] Hello world"
        assert _clean_tts_text(text) == "Hello world"

    def test_removes_square_brackets_except_host(self) -> None:
        text = "[note] Hello [Host A] there"
        assert "[note]" not in _clean_tts_text(text)
        # Host tag itself is handled by parser, not cleaner
        assert "Hello" in _clean_tts_text(text)

    def test_removes_parenthetical_annotations(self) -> None:
        assert _clean_tts_text("Hello (doge) world") == "Hello world"
        # Original text has no spaces around the parenthetical, so removal joins words.
        assert _clean_tts_text("Hello（狗头）world") == "Helloworld"
        assert _clean_tts_text("Hello(bushi)") == "Hello"

    def test_normalizes_whitespace(self) -> None:
        text = "Line 1\n\n\n\nLine 2"
        assert _clean_tts_text(text) == "Line 1\n\nLine 2"

    def test_removes_fancy_quotes(self) -> None:
        text = "「 quoted 」 and 『 another 』"
        assert _clean_tts_text(text) == "quoted and another"


class TestParseDialogueChunks:
    def test_simple_host_switch(self) -> None:
        text = "[Host A] Hello\n[Host B] Hi there"
        chunks = parse_dialogue_chunks(text)
        assert len(chunks) == 2
        assert chunks[0] == DialogueChunk(host="A", text="Hello")
        assert chunks[1] == DialogueChunk(host="B", text="Hi there")

    def test_implicit_leading_text_goes_to_default_a(self) -> None:
        text = "Welcome everyone. [Host B] Thanks"
        chunks = parse_dialogue_chunks(text)
        assert chunks[0] == DialogueChunk(host="A", text="Welcome everyone.")

    def test_trailing_text_after_last_marker(self) -> None:
        text = "[Host A] Hello\nGoodbye"
        chunks = parse_dialogue_chunks(text)
        # Everything after [Host A] until EOF belongs to Host A.
        assert chunks[-1] == DialogueChunk(host="A", text="Hello\nGoodbye")

    def test_empty_result_for_empty_string(self) -> None:
        assert parse_dialogue_chunks("") == []

    def test_cleans_each_chunk(self) -> None:
        text = "[Host A] Hello (doge) world"
        chunks = parse_dialogue_chunks(text)
        assert chunks[0].text == "Hello world"
