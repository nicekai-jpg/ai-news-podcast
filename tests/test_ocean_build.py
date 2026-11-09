"""Stage 2 build_ocean 端到端：artifact → 去重/聚类/连续性 → ocean 快照文件。"""

from __future__ import annotations

import json
from pathlib import Path

from ai_news_podcast.business.ocean.build import build_ocean
from tests.test_ocean_support import (
    OCEAN_CONFIG_PATH,
    make_item,
    utc_iso,
    write_raw_artifact,
)


def _item(
    title: str,
    *,
    source: str,
    summary: str,
    full_text: str,
    category: str = "news",
    link: str | None = None,
    date: str = "2026-11-09",
):
    return make_item(
        title=title,
        source=source,
        category=category,
        summary=summary,
        full_text=full_text,
        link=link,
        published_at=utc_iso(date),
    )


def _day1_items() -> list:
    return [
        _item(
            "OpenAI 发布 GPT-5 模型",
            source="openai_blog",
            category="official",
            summary="OpenAI 推出 GPT-5 大模型，性能提升。",
            full_text="OpenAI 今日发布 GPT-5 大模型，支持多模态与更长上下文。",
            link="https://openai.com/gpt5",
        ),
        _item(
            "OpenAI GPT-5 正式发布，能力大幅提升",
            source="techcrunch",
            summary="OpenAI 发布 GPT-5，支持更长上下文。",
            full_text="OpenAI 发布 GPT-5，支持 100 万 tokens 上下文，性能提升 30%。",
            link="https://techcrunch.com/gpt5",
        ),
        _item(
            "OpenAI 发布 GPT-5 模型",
            source="repost",
            summary="转载：OpenAI 推出 GPT-5 大模型。",
            full_text="OpenAI 今日发布 GPT-5 大模型，转载自官方博客。",
            link="https://openai.com/gpt5?utm_source=repost",
        ),
        _item(
            "英伟达财报超预期",
            source="reuters",
            summary="英伟达公布季度财报，营收增长。",
            full_text="英伟达公布财报，营收增长 50%。",
            link="https://reuters.com/nvda",
        ),
    ]


def _day2_items() -> list:
    return [
        _item(
            "OpenAI 发布 GPT-5.1 模型，性能提升 40%",
            source="openai_blog",
            category="official",
            summary="OpenAI 推出 GPT-5.1，性能提升 40%。",
            full_text="OpenAI 发布 GPT-5.1 模型，性能提升 40%，上下文扩展到 200 万 tokens。",
            link="https://openai.com/gpt51",
            date="2026-11-10",
        ),
        _item(
            "谷歌发布 Gemini 3 模型",
            source="google_blog",
            category="official",
            summary="谷歌发布 Gemini 3。",
            full_text="谷歌发布 Gemini 3 模型，全面上线。",
            link="https://blog.google/gemini3",
            date="2026-11-10",
        ),
    ]


class TestBuildOcean:
    def test_two_day_snapshot_and_continuity(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        day1_path = write_raw_artifact(
            raw_dir / "raw-2026-11-09.json.gz", _day1_items(), "2026-11-09"
        )
        day2_path = write_raw_artifact(
            raw_dir / "raw-2026-11-10.json.gz", _day2_items(), "2026-11-10"
        )

        snapshot1 = build_ocean(
            day1_path,
            date_str="2026-11-09",
            config_path=OCEAN_CONFIG_PATH,
        )
        ocean_root = tmp_path / "data" / "ocean"
        assert (ocean_root / "2026-11-09" / "events.json").exists()
        assert (ocean_root / "2026-11-09" / "items.json").exists()
        assert (ocean_root / "2026-11-09" / "continuity.json").exists()
        assert (ocean_root / "2026-11-09" / "manifest.json").exists()

        assert snapshot1.events
        gpt_event = next(
            event for event in snapshot1.events if "GPT-5" in event.representative_title
        )
        assert gpt_event.sequence == 1
        assert gpt_event.continuity_id.startswith("c_")
        assert gpt_event.source_count >= 2
        assert gpt_event.labels.evidence_tier in {"strong", "medium"}
        assert gpt_event.labels.confidence["fact_consistency"] >= 0
        assert gpt_event.item_ids

        rows = snapshot1.items
        assert len(rows) == len(_day1_items())
        duplicate_rows = [row for row in rows if row["dup_reason"]]
        assert duplicate_rows and duplicate_rows[0]["dup_reason"] == "url_exact"
        assert duplicate_rows[0]["dup_of"]
        for row in rows:
            assert "full_text" in row
            assert "event_id" in row
            assert "continuity_id" in row

        snapshot2 = build_ocean(
            day2_path,
            date_str="2026-11-10",
            config_path=OCEAN_CONFIG_PATH,
        )
        update_event = next(
            event
            for event in snapshot2.events
            if event.representative_title.startswith("OpenAI 发布 GPT-5.1")
        )
        assert update_event.sequence == 2
        assert update_event.is_update is True
        assert update_event.continuity_id == gpt_event.continuity_id
        assert update_event.labels.lifecycle in {"evolving", "long_tail"}
        assert snapshot2.continuity
        line = next(
            line
            for line in snapshot2.continuity
            if line["continuity_id"] == gpt_event.continuity_id
        )
        assert line["sequence"] == 2
        assert gpt_event.event_id in line["event_ids"]

        manifest = json.loads(
            (ocean_root / "2026-11-10" / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["date"] == "2026-11-10"
        assert manifest["stats"]["input_items"] == 2
        assert manifest["stats"]["continuity_updates"] == 1
        assert "parameters" in manifest

    def test_build_is_idempotent(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        path = write_raw_artifact(raw_dir / "raw-2026-11-09.json.gz", _day1_items(), "2026-11-09")
        first = build_ocean(path, date_str="2026-11-09", config_path=OCEAN_CONFIG_PATH)
        second = build_ocean(path, date_str="2026-11-09", config_path=OCEAN_CONFIG_PATH)
        first_events = [event.to_dict() for event in first.events]
        second_events = [event.to_dict() for event in second.events]
        assert first_events == second_events
        assert all(event.reappeared is False for event in second.events)

    def test_ledger_dict_covers_repeat_item(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        day1_path = write_raw_artifact(
            raw_dir / "raw-2026-11-09.json.gz", _day1_items(), "2026-11-09"
        )
        day1 = build_ocean(day1_path, date_str="2026-11-09", config_path=OCEAN_CONFIG_PATH)
        gpt_event = next(event for event in day1.events if "GPT-5" in event.representative_title)
        ledger = {
            "version": 1,
            "entries": [
                {
                    "event_id": gpt_event.event_id,
                    "title": gpt_event.representative_title,
                    "continuity_id": gpt_event.continuity_id,
                    "links": ["https://openai.com/gpt5"],
                }
            ],
        }
        repeat_item = _item(
            "OpenAI 发布 GPT-5 模型",
            source="repost2",
            summary="旧闻重发。",
            full_text="OpenAI 今日发布 GPT-5 大模型，旧闻重发。",
            link="https://repost.example.com/gpt5",
            date="2026-11-10",
        )
        day2_path = write_raw_artifact(
            raw_dir / "raw-2026-11-10.json.gz", [repeat_item], "2026-11-10"
        )
        day2 = build_ocean(
            day2_path,
            date_str="2026-11-10",
            ledger_path=ledger,
            config_path=OCEAN_CONFIG_PATH,
        )
        assert day2.events == []
        dropped_row = next(row for row in day2.items if row["item_id"] == repeat_item.item_id)
        assert dropped_row["dup_reason"] in {"cross_day_covered", "covered_before"}
        assert dropped_row["event_id"] == ""

    def test_empty_artifact_writes_empty_snapshot(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        path = write_raw_artifact(raw_dir / "raw-2026-11-09.json.gz", [], "2026-11-09")
        snapshot = build_ocean(path, date_str="2026-11-09", config_path=OCEAN_CONFIG_PATH)
        assert snapshot.events == []
        assert snapshot.items == []
        output_dir = tmp_path / "data" / "ocean" / "2026-11-09"
        assert json.loads((output_dir / "events.json").read_text(encoding="utf-8")) == []
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["stats"]["input_items"] == 0

    def test_missing_artifact_raises(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            build_ocean(tmp_path / "missing.json.gz", date_str="2026-11-09")
