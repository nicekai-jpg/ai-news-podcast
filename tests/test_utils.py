"""Tests for ai_news_podcast.utils."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ai_news_podcast.utils import load_sources, read_json, read_yaml, write_json, write_text


class TestReadYaml:
    def test_reads_simple_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("key: value\nlist: [1, 2]\n", encoding="utf-8")
        data = read_yaml(path)
        assert data == {"key": "value", "list": [1, 2]}

    def test_raises_on_non_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid YAML object"):
            read_yaml(path)


class TestReadJson:
    def test_reads_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        path.write_text('[{"id": 1}]', encoding="utf-8")
        assert read_json(path) == [{"id": 1}]

    def test_returns_empty_list_when_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.json"
        assert read_json(path) == []


class TestWriteJson:
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "file.json"
        write_json(path, {"ok": True})
        assert path.exists()
        assert read_json(path) == {"ok": True}

    def test_ends_with_newline(self, tmp_path: Path) -> None:
        path = tmp_path / "file.json"
        write_json(path, [1])
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")


class TestWriteText:
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "c.txt"
        write_text(path, "hello")
        assert path.read_text(encoding="utf-8") == "hello"


class TestLoadSources:
    def test_loads_valid_sources(self, tmp_path: Path) -> None:
        path = tmp_path / "sources.yaml"
        path.write_text(
            yaml.safe_dump({"sources": [{"name": "A", "url": "https://a.com"}]}),
            encoding="utf-8",
        )
        sources = load_sources(path)
        assert sources == [{"name": "A", "url": "https://a.com"}]

    def test_raises_when_missing_sources_key(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("other: []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a 'sources' list"):
            load_sources(path)

    def test_skips_non_dict_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "mixed.yaml"
        path.write_text(
            yaml.safe_dump({"sources": [{"name": "A"}, "not-a-dict", 123]}),
            encoding="utf-8",
        )
        sources = load_sources(path)
        assert sources == [{"name": "A"}]
