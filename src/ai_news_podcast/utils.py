"""Shared utility functions for I/O and config loading."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dict."""
    yaml = importlib.import_module("yaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML object at {path}")
    return data


def read_json(path: Path) -> Any:
    """Read a JSON file; return empty list if missing."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    """Write data to a JSON file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    """Write text to a file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_sources(sources_path: Path) -> list[dict[str, Any]]:
    """Load and validate a sources.yaml file."""
    data = read_yaml(sources_path)
    sources = data.get("sources")
    if not isinstance(sources, list):
        raise ValueError("sources.yaml must contain a 'sources' list")
    out: list[dict[str, Any]] = []
    for src in sources:
        if isinstance(src, dict):
            out.append(src)
    return out
