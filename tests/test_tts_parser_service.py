"""Tests for tts_parser — splitting and parsing script dialogue."""

from __future__ import annotations

from ai_news_podcast.business.pipeline.podcast_writer_service import _normalize_host_tags
from ai_news_podcast.business.pipeline.tts_parser_service import (
    has_renderable_text,
    parse_dialogue_chunks,
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


class TestParseDialogueChunksChapters:
    """v2 章节真标题：`[CHAPTER] 标题` 行归属其后的 Host 段且不进 TTS 文本。"""

    def test_chapter_before_host_marker_goes_to_next_chunk(self) -> None:
        text = "[Host A] 开场白。\n[CHAPTER] 今日头条\n[Host B] 回应一下。"
        chunks = parse_dialogue_chunks(text)
        assert [(c.host, c.chapter, c.text) for c in chunks] == [
            ("A", None, "开场白。"),
            ("B", "今日头条", "回应一下。"),
        ]

    def test_chapter_lines_never_leak_into_tts_text(self) -> None:
        text = "[Host A] 开场白。\n[CHAPTER] 今日头条\n[Host A] 正文内容。"
        chunks = parse_dialogue_chunks(text)
        assert all("[CHAPTER]" not in chunk.text for chunk in chunks)
        assert [chunk.chapter for chunk in chunks] == [None, "今日头条"]

    def test_no_chapter_markers_yields_none(self) -> None:
        chunks = parse_dialogue_chunks("[Host A] 你好\n[Host B] 嗯")
        assert [chunk.chapter for chunk in chunks] == [None, None]

    def test_consecutive_chapter_markers_take_last(self) -> None:
        text = "[Host A] 开场白。\n[CHAPTER] 旧标题\n[CHAPTER] 新标题\n[Host B] 正文内容。"
        chunks = parse_dialogue_chunks(text)
        assert chunks[-1].chapter == "新标题"
        assert len(chunks) == 2
        assert all("[CHAPTER]" not in chunk.text for chunk in chunks)

    def test_duplicate_same_chapter_recorded_once(self) -> None:
        text = "[Host A] 开场白。\n[CHAPTER] 今日头条\n[CHAPTER] 今日头条\n[Host B] 正文内容。"
        chunks = parse_dialogue_chunks(text)
        assert [chunk.chapter for chunk in chunks] == [None, "今日头条"]
        assert chunks[-1].text == "正文内容。"

    def test_trailing_chapter_without_host_is_ignored(self) -> None:
        text = "[Host A] 正文内容。\n[CHAPTER] 尾部标题"
        chunks = parse_dialogue_chunks(text)
        assert len(chunks) == 1
        assert chunks[0].chapter is None
        assert "[CHAPTER]" not in chunks[0].text

    def test_existing_behavior_without_chapters_unchanged(self) -> None:
        chunks = parse_dialogue_chunks("[Host A] First\n[Host A] Second\n[Host B] Third")
        assert [(c.host, c.text, c.chapter) for c in chunks] == [
            ("A", "First", None),
            ("A", "Second", None),
            ("B", "Third", None),
        ]

    def test_writer_normalized_sample_script_yields_chapters(self) -> None:
        """格式样例级链路：writer 归一化 → parser，章节正确归属且标记不被念出。"""
        script = _normalize_host_tags(
            "[Host A] 大家好。\n"
            "[Host B] 我是周航。\n"
            "[CHAPTER] 今日头条\n"
            "[Host A] 第一个话题。\n"
            "[CHAPTER] 快讯速览\n"
            "[Host B] 最后一条快讯。"
        )
        assert script.count("[CHAPTER]") == 2
        assert "[Host A] 今日头条" not in script
        assert "[Host B] 快讯速览" not in script
        chunks = parse_dialogue_chunks(script)
        assert [chunk.chapter for chunk in chunks] == [None, None, "今日头条", "快讯速览"]
        assert all("[CHAPTER]" not in chunk.text for chunk in chunks)


def test_non_renderable_chunk_is_dropped() -> None:
    """纯符号块（如 /）会被 CosyVoice tokenize 成空列表，必须丢弃而不是生成 0.1s 空段。"""
    from ai_news_podcast.business.pipeline.tts_parser_service import parse_dialogue_chunks

    chunks = parse_dialogue_chunks("[Host A] /\n[Host B] 正常的一句话台词。")
    assert [chunk.text for chunk in chunks] == ["正常的一句话台词。"]


def test_long_text_without_punctuation_is_hard_cut() -> None:
    from ai_news_podcast.business.pipeline.tts_parser_service import split_text_into_sentences

    parts = split_text_into_sentences("啊" * 200, max_chars=80)
    assert [len(part) for part in parts] == [80, 80, 40]
