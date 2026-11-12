"""Tests for scripts/ops_snapshot.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ops_snapshot.py"
    spec = importlib.util.spec_from_file_location("ops_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_snapshot_counts_and_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    _write(tmp_path / "ocean" / "2026-11-12" / "events.json", [{"labels": {}}])
    _write(tmp_path / "selection" / "2026-11-12" / "shortlist.json", [{"event_id": "e1"}])
    _write(
        tmp_path / "editorial" / "2026-11-12" / "report_review.json",
        {"overall": "评价" * 30, "items": [{"comment": "ok"}]},
    )
    snapshot = module.build_snapshot(tmp_path, "2026-11-12")
    assert snapshot["events"] == 1
    assert snapshot["shortlist"] == 1
    assert "运维快照" in module.render(snapshot)
