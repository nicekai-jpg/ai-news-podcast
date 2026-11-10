"""Tests for the v3 pipeline orchestrator (stages mocked, no network)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

from ai_news_podcast.business.contracts import (
    Commentary,
    EditorialReview,
    Event,
    Fact,
    Item,
    PodcastReview,
    ShortlistEntry,
)
from ai_news_podcast.business.pipeline.v3_runner_service import (
    load_item_evidence,
    make_llm_judge,
    run_v3_pipeline,
)


def _fake_modules(monkeypatch: Any, tmp_path: Path) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def fake_collect(config_path: Any, date_str: str, output_path: Any, **kwargs: Any) -> Path:
        calls["collect"] = {"config_path": str(config_path), "date_str": date_str}
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-raw")
        return path

    monkeypatch.setattr("ai_news_podcast.business.collect.runner.collect_and_write", fake_collect)

    event = Event(
        event_id="e1",
        representative_title="DeepSeek 发布新模型",
        item_ids=["i1"],
        source_count=3,
        independent_sources=2,
        sequence=1,
        continuity_id="c1",
        evidence_score=4.0,
    )

    def fake_build_ocean(
        raw_path: Any, *, date_str: str, history_dir: Any, ledger_path: Any, config_path: Any
    ) -> Any:
        calls["ocean"] = {"raw_path": str(raw_path), "history_dir": str(history_dir)}
        ocean_dir = Path(history_dir) / date_str
        ocean_dir.mkdir(parents=True, exist_ok=True)
        (ocean_dir / "events.json").write_text(json.dumps([event.to_dict()]), encoding="utf-8")
        items = [
            Item(
                item_id="i1",
                channel="c1_rss",
                source="s",
                source_label="S",
                title="t",
                link="https://example.com/1",
                full_text="DeepSeek 发布新模型，参数 100B。",
            )
        ]
        (ocean_dir / "items.json").write_text(
            json.dumps([item.to_dict() for item in items]), encoding="utf-8"
        )
        return types.SimpleNamespace(events=[event], items=items, continuity=[])

    ocean_mod = types.ModuleType("ai_news_podcast.business.ocean.build")
    ocean_mod.build_ocean = fake_build_ocean  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_news_podcast.business.ocean.build", ocean_mod)

    shortlist = [
        ShortlistEntry(
            rank=1, event_id="e1", score=4.5, reasons=["impact"], evidence_item_ids=["i1"]
        )
    ]

    def fake_recommend(events: Any, **kwargs: Any) -> list[ShortlistEntry]:
        calls["recommend"] = {"events": len(list(events))}
        return shortlist

    monkeypatch.setattr("ai_news_podcast.business.recommend.service.recommend", fake_recommend)

    def fake_commentaries(
        shortlist_arg: Any, events_arg: Any, llm_call: Any, **kwargs: Any
    ) -> list[Commentary]:
        calls["commentaries"] = {
            "llm": callable(llm_call),
            "item_evidence": bool(kwargs.get("item_evidence")),
        }
        return [
            Commentary(event_id="e1", headline="h", facts=[Fact(text="f", evidence_ids=["i1"])])
        ]

    monkeypatch.setattr(
        "ai_news_podcast.business.editorial.commentary.build_commentaries", fake_commentaries
    )

    def fake_report_review(*args: Any, **kwargs: Any) -> EditorialReview:
        calls["report_review"] = {"rows": len(list(args[0])) if args else 0}
        return EditorialReview(overall="ok", items=[])

    def fake_podcast_review(*args: Any, **kwargs: Any) -> PodcastReview:
        calls["podcast_review"] = {"script": bool(args[0]) if args else False}
        return PodcastReview(overall="ok", show_notes="notes")

    monkeypatch.setattr(
        "ai_news_podcast.business.editorial.review.build_report_review", fake_report_review
    )
    monkeypatch.setattr(
        "ai_news_podcast.business.editorial.review.build_podcast_review", fake_podcast_review
    )

    def fake_script(*args: Any, **kwargs: Any) -> str:
        calls["script"] = {"item_evidence": bool(args[2]) if len(args) > 2 else False}
        return "[CHAPTER] 开场\n[Host A] 你好\n[Host B] 你好"

    monkeypatch.setattr("ai_news_podcast.business.editorial.script.build_script", fake_script)
    return calls


def test_run_v3_pipeline_flow(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _fake_modules(monkeypatch, tmp_path)
    data_dir = tmp_path / "data"
    result = run_v3_pipeline("2026-11-09", data_dir=data_dir, llm_cfg={"model": "fake"})

    assert result.raw_path.exists()
    assert result.events_path.exists()
    assert result.shortlist_path.exists()
    assert result.commentary_path.exists()
    assert result.report_path is not None and result.report_path.exists()
    assert result.script_path is not None and result.script_path.exists()
    assert calls["recommend"]["events"] == 1
    assert calls["commentaries"]["item_evidence"] is True
    assert calls["script"]["item_evidence"] is True
    assert calls["report_review"]["rows"] == 1
    assert calls["podcast_review"]["script"] is True
    assert result.report_review_path is not None and result.report_review_path.exists()
    assert result.podcast_review_path is not None and result.podcast_review_path.exists()

    ledger = json.loads((data_dir / "coverage" / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["entries"][0]["event_id"] == "e1"
    assert ledger["entries"][0]["links"] == ["https://example.com/1"]


def test_load_item_evidence(tmp_path: Path) -> None:
    ocean_dir = tmp_path / "ocean" / "2026-11-09"
    ocean_dir.mkdir(parents=True)
    (ocean_dir / "items.json").write_text(
        json.dumps([{"item_id": "i1", "full_text": "body", "summary": "s"}]), encoding="utf-8"
    )
    assert load_item_evidence(ocean_dir) == {"i1": "body"}

    (ocean_dir / "items.json").unlink()
    assert load_item_evidence(ocean_dir) == {}


def test_make_llm_judge_parses_json() -> None:
    judge = make_llm_judge(lambda prompt: '```json\n{"e1": {"impact": 5, "reason": "重要"}}\n```')
    result = judge([Event(event_id="e1", representative_title="t")])
    assert result["e1"]["impact"] == 5


def test_make_llm_judge_bad_output_returns_empty() -> None:
    judge = make_llm_judge(lambda prompt: "not json at all")
    assert judge([Event(event_id="e1", representative_title="t")]) == {}
