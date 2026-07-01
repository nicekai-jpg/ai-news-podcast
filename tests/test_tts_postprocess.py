"""Tests for shared TTS post-processing."""

from __future__ import annotations

import sys
from unittest.mock import patch

from ai_news_podcast.pipeline.tts_postprocess import assemble_dialogue_audio
from ai_news_podcast.pipeline.tts_types import DialogueChunk


class FakeSeg:
    def __init__(self, ms: int = 1000):
        self._ms = ms

    def __len__(self) -> int:
        return self._ms

    def __add__(self, other: object) -> FakeSeg:
        if isinstance(other, FakeSeg):
            return FakeSeg(self._ms + other._ms)
        return self

    @classmethod
    def empty(cls) -> FakeSeg:
        return cls(0)

    @classmethod
    def silent(cls, *, duration: int) -> FakeSeg:
        return cls(duration)


def test_assemble_dialogue_audio_returns_timestamps() -> None:
    chunks = [DialogueChunk(host="A", text="a"), DialogueChunk(host="B", text="b")]
    segments = [FakeSeg(1000), FakeSeg(2000)]
    fake_pydub = type("M", (), {"AudioSegment": FakeSeg})()
    with patch.dict(sys.modules, {"pydub": fake_pydub}):
        combined, timestamps = assemble_dialogue_audio(
            chunks,
            segments,
            chunk_silence_base=300,
            vocal_pad_ms=0,
            silence_min=100,
            silence_max=100,
            silence_jitter=0,
        )
    assert len(combined) == 3100
    assert timestamps == [(0.0, 1.0), (1.1, 2.0)]
