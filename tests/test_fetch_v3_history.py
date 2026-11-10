"""Tests for scripts/fetch_v3_history.py pure helpers."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_v3_history.py"
_spec = importlib.util.spec_from_file_location("fetch_v3_history", _SCRIPT)
assert _spec and _spec.loader
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)


def _run(run_id: int, created: str, status: str = "completed", conclusion: str = "success") -> dict:
    return {
        "databaseId": run_id,
        "createdAt": created,
        "status": status,
        "conclusion": conclusion,
        "event": "schedule",
    }


class TestSelectRuns:
    def test_filters_old_and_failed(self) -> None:
        now = datetime(2026, 11, 10, tzinfo=UTC)
        runs = [
            _run(1, "2026-11-10T00:00:00Z"),
            _run(2, "2026-10-01T00:00:00Z"),  # too old
            _run(3, "2026-11-09T00:00:00Z", conclusion="failure"),
            _run(4, "2026-11-08T00:00:00Z", status="in_progress"),
        ]
        picked = fetch.select_runs(runs, days=14, exclude_run_id="", now=now)
        assert [r["databaseId"] for r in picked] == [1]

    def test_excludes_current_run(self) -> None:
        now = datetime(2026, 11, 10, tzinfo=UTC)
        runs = [_run(1, "2026-11-10T00:00:00Z"), _run(2, "2026-11-09T00:00:00Z")]
        picked = fetch.select_runs(runs, days=14, exclude_run_id="1", now=now)
        assert [r["databaseId"] for r in picked] == [2]


class TestArtifactDates:
    def test_parses_release_artifact_names(self) -> None:
        arts = [
            {"name": "v3-output-2026-11-10"},
            {"name": "v3-output-2026-11-09"},
            {"name": "other-artifact"},
            {"name": "v3-output-bad"},
        ]
        assert fetch.artifact_dates(arts) == {
            "v3-output-2026-11-10": "2026-11-10",
            "v3-output-2026-11-09": "2026-11-09",
        }
