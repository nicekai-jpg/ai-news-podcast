"""站点静态资源质量门：JS 语法、无外链、无占位残留、播放引擎契约。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_STATIC_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_news_podcast"
    / "presentation"
    / "site_builder"
    / "static"
)
_NODE = shutil.which("node")
_ASSETS = ("app.js", "player.js", "style.css")


def _read(name: str) -> str:
    return (_STATIC_DIR / name).read_text(encoding="utf-8")


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
@pytest.mark.parametrize("filename", ["app.js", "player.js"])
def test_javascript_syntax(filename: str) -> None:
    result = subprocess.run(
        [str(_NODE), "--check", str(_STATIC_DIR / filename)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_assets_have_no_external_urls() -> None:
    for name in _ASSETS:
        text = _read(name)
        assert "http://" not in text, name
        assert "https://" not in text, name
        assert "cdn." not in text, name


def test_assets_have_no_placeholder_leftovers() -> None:
    for name in _ASSETS:
        assert "placeholder: replaced" not in _read(name), name


def test_player_engine_contract() -> None:
    assert "PodcastEngine" in _read("app.js")
    assert "class PodcastEngine" in _read("player.js")
