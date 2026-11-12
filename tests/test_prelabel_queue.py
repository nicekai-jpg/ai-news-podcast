"""Tests for scripts/prelabel_queue.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tests.test_ocean_support import ocean_config, ocean_lexicon


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prelabel_queue.py"
    spec = importlib.util.spec_from_file_location("prelabel_queue", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prelabels_have_rule_fields() -> None:
    module = _load_module()
    rows = [{"title": "OpenAI 发布新模型", "summary": "支持 Agent"}]
    enriched = module.build_prelabels(rows, config=ocean_config(), lexicon=ocean_lexicon())[0]
    assert enriched["prelabel_ai_relevance"] in {"core", "adjacent", "unknown"}
    assert "prelabel_topics" in enriched
    assert enriched.get("label_source", True)
