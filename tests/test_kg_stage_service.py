"""Stage 2.5 知识图谱：逐事件抽图 + 落盘。"""

from __future__ import annotations

import json

from ai_news_podcast.business.contracts import Event
from ai_news_podcast.business.pipeline.kg_stage_service import (
    build_kg_artifact,
    write_kg_artifact,
)


def _event(event_id: str, title: str, item_ids: list[str]) -> Event:
    return Event(
        event_id=event_id,
        representative_title=title,
        item_ids=item_ids,
        source_count=len(item_ids),
        independent_sources=len(item_ids),
    )


def test_builds_artifact_for_events() -> None:
    events = [
        _event("e1", "阿里巴巴发布通义千问", ["i1"]),
        _event("e2", "某个没有全文的事件", ["i2"]),
    ]
    items = [
        {"item_id": "i1", "full_text": "阿里巴巴发布了通义千问，性能超过GPT-5。"},
        {"item_id": "i2", "full_text": ""},
    ]
    payload = build_kg_artifact(events, items, date_str="2026-11-16")
    assert payload["stats"]["events"] == 2
    assert payload["events"][0]["edges"]
    assert payload["events"][1]["edges"] == []
    assert payload["date"] == "2026-11-16"


def test_min_confidence_can_filter_everything() -> None:
    events = [_event("e1", "阿里巴巴发布通义千问", ["i1"])]
    items = [{"item_id": "i1", "full_text": "阿里巴巴发布了通义千问。"}]
    payload = build_kg_artifact(events, items, date_str="2026-11-16", min_confidence=0.95)
    assert payload["stats"]["edges"] == 0


def test_write_artifact(tmp_path) -> None:
    payload = {"date": "2026-11-16", "events": [], "stats": {}}
    path = write_kg_artifact(tmp_path / "kg" / "kg.json", payload)
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["date"] == "2026-11-16"


def test_item_texts_falls_back_to_summary() -> None:
    events = [_event("e1", "阿里巴巴发布通义千问", ["i1"])]
    items = [{"item_id": "i1", "full_text": "", "summary": "阿里巴巴发布了通义千问。"}]
    payload = build_kg_artifact(events, items, date_str="2026-11-16")
    assert payload["events"][0]["edges"]
