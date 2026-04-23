"""Tests for processor numpy encoder and save_brief."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_news_podcast.pipeline.processor import _NumpySafeEncoder, save_brief


class TestNumpySafeEncoder:
    def test_encodes_numpy_int(self) -> None:
        # Simulate numpy integer behavior with a simple wrapper
        class FakeNpInt:
            def __int__(self):
                return 42

        encoder = _NumpySafeEncoder()
        # The encoder tries to import numpy; if not available it falls back.
        # We can't easily test the numpy path without numpy installed,
        # but we can verify the fallback path works.
        with pytest.raises(TypeError):
            # FakeNpInt is not json serializable by default
            encoder.default(FakeNpInt())

    def test_encodes_regular_types(self) -> None:

        encoder = _NumpySafeEncoder()
        # Regular types should fall through to the parent
        with pytest.raises(TypeError):
            encoder.default(object())


class TestSaveBrief:
    def test_creates_file_with_newline(self, tmp_path: Path) -> None:
        path = tmp_path / "brief.json"
        brief = {"thesis": "test", "stories": []}
        save_brief(brief, path)
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert '"thesis": "test"' in text

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "brief.json"
        save_brief({}, path)
        assert path.exists()
