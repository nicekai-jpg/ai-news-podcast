"""Tests for business/contracts (shared schemas + artifact IO)."""

from __future__ import annotations

import gzip
from pathlib import Path

from ai_news_podcast.business.contracts import (
    Commentary,
    Event,
    Fact,
    Item,
    Labels,
    ShortlistEntry,
    canonical_link,
    make_event_id,
    make_item_id,
)
from ai_news_podcast.business.contracts.artifact import read_artifact, write_artifact


class TestCanonicalLink:
    def test_strips_tracking_and_fragment(self) -> None:
        url = "https://WWW.Example.com/a/b/?utm_source=x&id=3#frag"
        assert canonical_link(url) == "https://example.com/a/b?id=3"

    def test_trailing_slash(self) -> None:
        assert canonical_link("https://example.com/path/") == "https://example.com/path"

    def test_empty(self) -> None:
        assert canonical_link("") == ""


class TestIds:
    def test_item_id_stable(self) -> None:
        a = make_item_id("https://example.com/x?utm_source=t", "标题")
        b = make_item_id("https://example.com/x", "标题")
        assert a == b

    def test_event_id_order_independent(self) -> None:
        assert make_event_id(["b", "a"]) == make_event_id(["a", "b"])


class TestSerialization:
    def test_item_roundtrip(self) -> None:
        item = Item(item_id="1", channel="c1_rss", source="s", source_label="S", title="t")
        assert Item.from_dict(item.to_dict()) == item

    def test_event_roundtrip_with_labels(self) -> None:
        event = Event(
            event_id="e",
            representative_title="t",
            labels=Labels(subjects=[{"id": "openai", "name": "OpenAI"}], topics=["技术"]),
        )
        restored = Event.from_dict(event.to_dict())
        assert restored.labels.subjects[0]["id"] == "openai"
        assert restored.labels.topics == ["技术"]

    def test_shortlist_and_commentary(self) -> None:
        entry = ShortlistEntry(rank=1, event_id="e", score=4.2, reasons=["r"])
        assert entry.to_dict()["rank"] == 1
        commentary = Commentary(
            event_id="e", headline="h", facts=[Fact(text="f", evidence_ids=["i"])]
        )
        data = commentary.to_dict()
        assert data["facts"][0]["evidence_ids"] == ["i"]


class TestArtifact:
    def test_write_and_read(self, tmp_path: Path) -> None:
        items = [
            Item(item_id="1", channel="c1_rss", source="s", source_label="S", title="t"),
            Item(item_id="2", channel="c5_wechat2rss", source="w", source_label="W", title="t2"),
        ]
        path = write_artifact(tmp_path / "raw-2026-11-09.json.gz", items, date_str="2026-11-09")
        with gzip.open(path, "rb") as fh:
            assert fh.read()
        manifest, loaded = read_artifact(path)
        assert manifest["item_count"] == 2
        assert [i.item_id for i in loaded] == ["1", "2"]
        assert loaded[1].channel == "c5_wechat2rss"
