"""Tests for tts_engine _run_loudnorm."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ai_news_podcast.pipeline.tts_engine import _run_loudnorm


class TestRunLoudnorm:
    @pytest.mark.asyncio
    async def test_success(self, tmp_path: Path) -> None:
        input_path = tmp_path / "in.mp3"
        output_path = tmp_path / "out.mp3"
        input_path.write_text("fake mp3", encoding="utf-8")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await _run_loudnorm(input_path, output_path)

        assert mock_exec.called

    @pytest.mark.asyncio
    async def test_failure_raises(self, tmp_path: Path) -> None:
        input_path = tmp_path / "in.mp3"
        output_path = tmp_path / "out.mp3"
        input_path.write_text("fake mp3", encoding="utf-8")

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"ffmpeg error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="ffmpeg loudnorm failed"):
                await _run_loudnorm(input_path, output_path)

    @pytest.mark.asyncio
    async def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        input_path = tmp_path / "in.mp3"
        output_path = tmp_path / "nested" / "dir" / "out.mp3"
        input_path.write_text("fake mp3", encoding="utf-8")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await _run_loudnorm(input_path, output_path)

        assert output_path.parent.exists()
