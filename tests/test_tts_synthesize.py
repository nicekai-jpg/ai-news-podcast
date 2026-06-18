"""Tests for tts_engine synthesize flow with mocked audio backends."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_news_podcast.pipeline.tts_engine import synthesize
from ai_news_podcast.pipeline.tts_types import DialogueChunk


class FakeAudioSegment:
    """Minimal stand-in for pydub.AudioSegment."""

    def __init__(self, duration: int = 1000):
        self._duration = duration

    def __len__(self) -> int:
        return self._duration

    def __add__(self, other: Any) -> "FakeAudioSegment":
        if isinstance(other, FakeAudioSegment):
            return FakeAudioSegment(self._duration + other._duration)
        if isinstance(other, int):
            return self
        return NotImplemented

    def __sub__(self, val: int) -> "FakeAudioSegment":
        return self

    def __mul__(self, n: int) -> "FakeAudioSegment":
        return FakeAudioSegment(self._duration * n)

    def __getitem__(self, idx: slice) -> "FakeAudioSegment":
        if isinstance(idx, slice):
            stop = idx.stop if idx.stop is not None else self._duration
            return FakeAudioSegment(min(stop, self._duration))
        return self

    def fade_in(self, _ms: int) -> "FakeAudioSegment":
        return self

    def fade_out(self, _ms: int) -> "FakeAudioSegment":
        return self

    def overlay(self, _other: Any) -> "FakeAudioSegment":
        return self

    def export(self, _path: str, format: str | None = None) -> None:
        pass

    @classmethod
    def empty(cls) -> "FakeAudioSegment":
        return cls(0)

    @classmethod
    def silent(cls, *, duration: int) -> "FakeAudioSegment":
        return cls(duration)

    @classmethod
    def from_file(cls, _path: str) -> "FakeAudioSegment":
        return cls(1000)


def _make_pydub_module():
    module = MagicMock()
    module.AudioSegment = FakeAudioSegment
    return module


class FakeCommunicate:
    """Mock for edge_tts.Communicate."""

    def __init__(self, text: str, voice: str, rate: str = "+0%", volume: str = "+0%"):
        self.text = text
        self._should_fail = False

    def set_fail(self) -> None:
        self._should_fail = True

    async def save(self, path: str) -> None:
        if self._should_fail:
            raise RuntimeError("TTS failed")
        Path(path).write_text("fake mp3 data", encoding="utf-8")


@pytest.fixture(autouse=True)
def mock_loudnorm():
    """Prevent real ffmpeg calls during TTS tests."""

    async def _fake_loudnorm(input_path, output_path, **kwargs):
        Path(output_path).write_text("fake normalized mp3", encoding="utf-8")

    with patch("ai_news_podcast.pipeline.tts_postprocess.run_loudnorm", side_effect=_fake_loudnorm):
        yield




@pytest.mark.asyncio
async def test_synthesize_entrypoint_archived(tmp_path: Path) -> None:
    """synthesize() should raise ValueError when edge-tts backend is requested."""
    with pytest.raises(ValueError, match="Edge-TTS backend has been archived"):
        await synthesize(
            "[Host A] Hello\n[Host B] World",
            backend="edge-tts",
            output_path=tmp_path / "out.mp3",
        )


@pytest.mark.asyncio
async def test_synthesize_empty_dialogue_raises(tmp_path: Path) -> None:
    """synthesize() should raise when no dialogue chunks are parsed."""
    with pytest.raises(ValueError, match="empty after dialogue parsing"):
        await synthesize("", backend="cosyvoice2", output_path=tmp_path / "out.mp3")


@pytest.mark.asyncio
async def test_synthesize_cosyvoice2_backend(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "out.mp3"
    called: dict = {}

    async def fake_synth(chunks, output_path, **kwargs):
        called["chunks"] = chunks
        called["output_path"] = output_path
        Path(output_path).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(
        "ai_news_podcast.pipeline.tts_engine.synthesize_cosyvoice2",
        fake_synth,
    )
    await synthesize(
        "[Host A] 你好\n[Host B] 欢迎",
        backend="cosyvoice2",
        output_path=output,
        cfg={"tts": {}},
    )
    assert output.exists()
    assert len(called["chunks"]) == 2


@pytest.mark.asyncio
async def test_synthesize_unsupported_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported TTS backend"):
        await synthesize(
            "[Host A] Hello",
            backend="unknown-tts",
            output_path=tmp_path / "out.mp3",
        )
