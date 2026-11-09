"""Tests for coverage ledger (cross-day dedup memory)."""

from __future__ import annotations

from pathlib import Path

from ai_news_podcast.business.coverage.ledger import (
    CoverageEntry,
    append_entry,
    is_covered,
    load_ledger,
    save_ledger,
)


class TestLedger:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        ledger = load_ledger(tmp_path / "nope.json")
        assert ledger == {"version": 1, "entries": []}

    def test_append_and_roundtrip(self, tmp_path: Path) -> None:
        ledger = load_ledger(tmp_path / "ledger.json")
        entry = CoverageEntry(
            event_id="e1",
            title="DeepSeek 发布 V4.1",
            links=["https://a.com/x?utm_source=t"],
            first_covered="2026-11-01",
            last_covered="2026-11-01",
        )
        append_entry(ledger, entry)
        path = save_ledger(tmp_path / "ledger.json", ledger)
        reloaded = load_ledger(path)
        assert len(reloaded["entries"]) == 1
        assert reloaded["entries"][0]["event_id"] == "e1"

    def test_append_idempotent_and_updates_sequence(self) -> None:
        ledger = {"version": 1, "entries": []}
        append_entry(ledger, CoverageEntry(event_id="e1", last_covered="2026-11-01", sequence=1))
        append_entry(ledger, CoverageEntry(event_id="e1", last_covered="2026-11-03", sequence=2))
        assert len(ledger["entries"]) == 1
        assert ledger["entries"][0]["last_covered"] == "2026-11-03"
        assert ledger["entries"][0]["sequence"] == 2

    def test_is_covered_by_event_id_or_link(self) -> None:
        ledger = {
            "version": 1,
            "entries": [
                {"event_id": "e1", "links": ["https://a.com/x"]},
            ],
        }
        assert is_covered(ledger, event_id="e1")
        assert is_covered(ledger, links=["https://a.com/x?utm_source=1"])
        assert not is_covered(ledger, links=["https://b.com/y"])
