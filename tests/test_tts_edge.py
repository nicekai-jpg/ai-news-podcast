"""Edge-case and unit tests for tts_engine utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_news_podcast.pipeline.tts_engine import (
    _chunk_silence_ms,
    _to_audio_segment_from_cosy_output,
)


class TestChunkSilenceMs:
    def test_clamps_to_range(self) -> None:
        # random.randint makes exact value unpredictable, but we can check bounds
        result = _chunk_silence_ms(300)
        assert 400 <= result <= 800

        result = _chunk_silence_ms(1000)
        assert 400 <= result <= 800

    def test_variance_within_200ms(self) -> None:
        # With base=500, low=400, high=600
        results = {_chunk_silence_ms(500) for _ in range(50)}
        assert all(400 <= r <= 600 for r in results)


class DummyAudioSegment:
    """Stand-in for pydub.AudioSegment so isinstance() works in tests."""

    @classmethod
    def from_file(cls, path_or_fp, format=None):
        return cls()


class TestToAudioSegmentFromCosyOutput:
    def _make_pydub_module(self):
        module = MagicMock()
        module.AudioSegment = DummyAudioSegment
        return module

    def test_bytes_input(self) -> None:
        with patch.dict("sys.modules", {"pydub": self._make_pydub_module()}):
            result = _to_audio_segment_from_cosy_output(b"wavdata")
        assert isinstance(result, DummyAudioSegment)

    def test_audio_segment_input_returned_directly(self) -> None:
        seg = DummyAudioSegment()
        with patch.dict("sys.modules", {"pydub": self._make_pydub_module()}):
            result = _to_audio_segment_from_cosy_output(seg)
        assert result is seg

    def test_dict_with_audio_key(self) -> None:
        with patch.dict("sys.modules", {"pydub": self._make_pydub_module()}):
            result = _to_audio_segment_from_cosy_output({"audio": b"data"})
        assert isinstance(result, DummyAudioSegment)

    def test_unsupported_type_raises(self) -> None:
        with (
            patch.dict("sys.modules", {"pydub": self._make_pydub_module()}),
            pytest.raises(TypeError, match="Unsupported CosyVoice output type"),
        ):
            _to_audio_segment_from_cosy_output(12345)
