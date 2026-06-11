"""Edge-case and unit tests for TTS postprocess utilities."""

from __future__ import annotations

from ai_news_podcast.pipeline.tts_postprocess import chunk_silence_ms


class TestChunkSilenceMs:
    def test_clamps_to_range(self) -> None:
        result = chunk_silence_ms(300)
        assert 400 <= result <= 800

        result = chunk_silence_ms(1000)
        assert 400 <= result <= 800

    def test_variance_within_200ms(self) -> None:
        results = {chunk_silence_ms(500) for _ in range(50)}
        assert all(400 <= r <= 600 for r in results)
