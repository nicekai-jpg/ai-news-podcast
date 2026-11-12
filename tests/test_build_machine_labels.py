"""Tests for scripts/build_machine_labels.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_machine_labels.py"
    spec = importlib.util.spec_from_file_location("build_machine_labels", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_maps_relevance_to_worth() -> None:
    module = _load_module()
    core = module.to_eval_label({"title": "t", "prelabel_ai_relevance": "core"})
    none = module.to_eval_label({"title": "t", "prelabel_ai_relevance": "none"})
    assert core["worth_publishing"] is True
    assert none["worth_publishing"] is False
    assert core["reviewed"] is False


def teststratified_takes_all_groups() -> None:
    module = _load_module()
    labels = [
        {"ai_relevance": "core"},
        {"ai_relevance": "none"},
        {"ai_relevance": "core"},
    ]
    picked = module.stratified(labels, 2)
    assert {row["ai_relevance"] for row in picked} == {"core", "none"}
