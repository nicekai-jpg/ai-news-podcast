"""Tests for tts_engine synthesize flow with mocked audio backends."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_news_podcast.pipeline.tts_engine import (
    DialogueChunk,
    synthesize,
    synthesize_edge_tts,
)


class FakeAudioSegment:
    """Minimal stand-in for pydub.AudioSegment."""

    def __init__(self, duration: int = 1000):
        self._duration = duration

    def __len__(self) -> int:
        return self._duration

    def __add__(self, other: Any) -> "FakeAudioSegment":
        if isinstance(other, FakeAudioSegment):
            return FakeAudioSegment(self._duration + other._duration)
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

    async def _fake_loudnorm(input_path, output_path):
        Path(output_path).write_text("fake normalized mp3", encoding="utf-8")

    with patch("ai_news_podcast.pipeline.tts_engine._run_loudnorm", side_effect=_fake_loudnorm):
        yield


@pytest.mark.asyncio
async def test_synthesize_edge_tts_success(tmp_path: Path) -> None:
    chunks = [
        DialogueChunk(host="A", text="Hello"),
        DialogueChunk(host="B", text="World"),
    ]
    output = tmp_path / "out.mp3"

    with patch.dict(sys.modules, {"pydub": _make_pydub_module()}):
        with patch("edge_tts.Communicate", FakeCommunicate):
            await synthesize_edge_tts(
                chunks,
                voices=("voice-a", "voice-b"),
                output_path=output,
            )
    assert output.exists()


@pytest.mark.asyncio
async def test_synthesize_edge_tts_retries_on_failure(tmp_path: Path) -> None:
    """edge-tts retries up to 3 times per chunk."""
    call_count = 0

    class CountingCommunicate(FakeCommunicate):
        async def save(self, path: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("fail")
            Path(path).write_text("ok", encoding="utf-8")

    chunks = [DialogueChunk(host="A", text="One")]
    output = tmp_path / "out.mp3"

    with patch.dict(sys.modules, {"pydub": _make_pydub_module()}):
        with patch("edge_tts.Communicate", CountingCommunicate):
            await synthesize_edge_tts(
                chunks,
                voices=("voice-a",),
                output_path=output,
            )
    assert call_count == 3
    assert output.exists()


@pytest.mark.asyncio
async def test_synthesize_edge_tts_raises_after_3_failures(tmp_path: Path) -> None:
    class AlwaysFail(FakeCommunicate):
        async def save(self, path: str) -> None:
            raise RuntimeError("always fails")

    chunks = [DialogueChunk(host="A", text="One")]
    output = tmp_path / "out.mp3"

    with patch.dict(sys.modules, {"pydub": _make_pydub_module()}):
        with patch("edge_tts.Communicate", AlwaysFail):
            with pytest.raises(RuntimeError):
                await synthesize_edge_tts(
                    chunks,
                    voices=("voice-a",),
                    output_path=output,
                )


@pytest.mark.asyncio
async def test_synthesize_edge_tts_with_bgm(tmp_path: Path) -> None:
    bgm = tmp_path / "bgm.wav"
    bgm.write_text("bgm data", encoding="utf-8")
    output = tmp_path / "out.mp3"

    chunks = [DialogueChunk(host="A", text="Hello")]

    with patch.dict(sys.modules, {"pydub": _make_pydub_module()}):
        with patch("edge_tts.Communicate", FakeCommunicate):
            await synthesize_edge_tts(
                chunks,
                voices=("voice-a",),
                output_path=output,
                bgm_path=str(bgm),
            )
    assert output.exists()


@pytest.mark.asyncio
async def test_synthesize_entrypoint(tmp_path: Path) -> None:
    """synthesize() should delegate to synthesize_edge_tts for edge-tts backend."""
    output = tmp_path / "out.mp3"

    with patch.dict(sys.modules, {"pydub": _make_pydub_module()}):
        with patch("edge_tts.Communicate", FakeCommunicate):
            await synthesize(
                "[Host A] Hello\n[Host B] World",
                backend="edge-tts",
                voices=("a", "b"),
                output_path=output,
            )
    assert output.exists()


@pytest.mark.asyncio
async def test_synthesize_empty_dialogue_raises(tmp_path: Path) -> None:
    """synthesize() should raise when no dialogue chunks are parsed."""
    with pytest.raises(ValueError, match="empty after dialogue parsing"):
        await synthesize("", backend="edge-tts", output_path=tmp_path / "out.mp3")


@pytest.mark.asyncio
async def test_synthesize_unsupported_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported TTS backend"):
        await synthesize(
            "[Host A] Hello",
            backend="cosyvoice",
            output_path=tmp_path / "out.mp3",
        )
