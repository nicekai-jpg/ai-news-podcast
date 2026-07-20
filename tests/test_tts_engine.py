"""Tests for ai_news_podcast.pipeline.tts_engine pure functions."""

from __future__ import annotations

from ai_news_podcast.pipeline.tts_engine import parse_dialogue_chunks
from ai_news_podcast.pipeline.tts_types import DialogueChunk
from ai_news_podcast.text_utils import clean_tts_text


class TestCleanTtsText:
    def test_removes_fact_tags(self) -> None:
        text = "[FACT] This is true. [INFERENCE] Probably. [OPINION] I think so."
        assert clean_tts_text(text) == "This is true. Probably. I think so."

    def test_removes_mood_tags(self) -> None:
        text = "[mood:excited] Hello world"
        assert clean_tts_text(text) == "Hello world"

    def test_removes_square_brackets_except_host(self) -> None:
        text = "[note] Hello [Host A] there"
        assert "[note]" not in clean_tts_text(text)
        # Host tag itself is handled by parser, not cleaner
        assert "Hello" in clean_tts_text(text)

    def test_removes_parenthetical_annotations(self) -> None:
        assert clean_tts_text("Hello (doge) world") == "Hello world"
        # Original text has no spaces around the parenthetical, so removal joins words.
        assert clean_tts_text("Hello（狗头）world") == "Helloworld"
        assert clean_tts_text("Hello(bushi)") == "Hello"

    def test_normalizes_whitespace(self) -> None:
        text = "Line 1\n\n\n\nLine 2"
        assert clean_tts_text(text) == "Line 1\n\nLine 2"

    def test_removes_fancy_quotes(self) -> None:
        text = "「 quoted 」 and 『 another 』"
        assert clean_tts_text(text) == "quoted and another"

    def test_strips_html_tags(self) -> None:
        """HTML tags should be stripped (SSML support removed)."""
        text = "<p>Hello <b>world</b></p>"
        assert clean_tts_text(text) == "Hello world"


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
        assert chunks[-1].host == "A"
        assert "Hello" in chunks[-1].text and "Goodbye" in chunks[-1].text

    def test_empty_result_for_empty_string(self) -> None:
        assert parse_dialogue_chunks("") == []

    def test_cleans_each_chunk(self) -> None:
        text = "[Host A] Hello (doge) world"
        chunks = parse_dialogue_chunks(text)
        assert chunks[0].text == "Hello world"

    def test_case_insensitive_host_markers(self) -> None:
        """Host markers should be case-insensitive."""
        text = "[host a] Hello\n[HOST B] Hi"
        chunks = parse_dialogue_chunks(text)
        assert len(chunks) == 2
        assert chunks[0] == DialogueChunk(host="A", text="Hello")
        assert chunks[1] == DialogueChunk(host="B", text="Hi")

    def test_multiple_chunks_same_host(self) -> None:
        """Multiple consecutive chunks for the same host."""
        text = "[Host A] First\n[Host A] Second\n[Host B] Third"
        chunks = parse_dialogue_chunks(text)
        assert len(chunks) == 3
        assert chunks[0] == DialogueChunk(host="A", text="First")
        assert chunks[1] == DialogueChunk(host="A", text="Second")
        assert chunks[2] == DialogueChunk(host="B", text="Third")


class TestAnnotateTextInBatches:
    def test_batching_and_lossless_verification(self, monkeypatch) -> None:
        from ai_news_podcast.pipeline.tts_engine import _annotate_text_in_batches

        # Create a 4-turn script, batch size 2
        script = (
            "[Host A] 大家好，欢迎收听AI先锋。\n"
            "[Host B] 确实是个好消息，咱们详细说说。\n"
            "[Host A] 第一条新闻是关于大模型部署升级。\n"
            "[Host B] 没错，性能提升了数倍之多。"
        )

        def mock_call_llm(prompt: str, cfg: dict) -> str:
            # If batch 1, return normal annotated
            if "大家好，欢迎收听AI先锋" in prompt:
                return (
                    "[Host A] 大家好，<laughter>欢迎收听AI先锋。</laughter>\n"
                    "[Host B] 确实是个好消息，<breath>咱们详细说说。"
                )
            # If batch 2, simulate LLM truncation/dropping a turn or shrinking text
            return "[Host A] 第一条新闻"

        monkeypatch.setattr("ai_news_podcast.pipeline.llm_client.call_llm", mock_call_llm)

        res = _annotate_text_in_batches(script, "AI 每日先锋", {}, batch_size=2)
        # Batch 1 should be annotated
        assert "<laughter>" in res or "欢迎收听AI先锋" in res
        # Batch 2 should fallback to exact original text because of truncation verification
        assert "[Host A] 第一条新闻是关于大模型部署升级。" in res
        assert "[Host B] 没错，性能提升了数倍之多。" in res
