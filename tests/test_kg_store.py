"""事实层存储：去重键、归并、落盘、幂等。"""

from __future__ import annotations

import json

from ai_news_podcast.business.kg.store import (
    fact_key,
    load_day,
    merge_cards,
    write_c_samples,
    write_day,
)
from ai_news_podcast.business.kg.types import DayStats, FactCard


def _card(fact_id: str, source: str, grade: str = "B") -> FactCard:
    return FactCard(
        fact_id=fact_id,
        date="2026-11-16",
        subject={"id": "lenovo", "name": "联想", "kind": "company"},
        role="main",
        relation="release",
        relation_raw="发布",
        slots={"object": {"text": "新模型", "entity_id": "", "kind": "product"}},
        polarity="positive",
        modality="actual",
        grade=grade,
        grade_reasons=("dep_confirmed",),
        evidence={"sentence": "联想发布新模型", "item_id": source, "url": "", "published_at": ""},
        source={"id": "qbitai", "name": "量子位", "category": "news"},
        continuity_id="c1",
        is_new=True,
        sources=(source,),
        created_at="2026-11-16T00:00:00Z",
    )


def test_fact_key_stable() -> None:
    a = fact_key("2026-11-16", "lenovo", "联想", "release", "新模型")
    b = fact_key("2026-11-16", "lenovo", "联想", "release", "新模型")
    c = fact_key("2026-11-16", "lenovo", "联想", "release", "旧模型")
    assert a == b
    assert a != c


def test_merge_same_fact_keeps_sources() -> None:
    merged = merge_cards([_card("f1", "a"), _card("f1", "b", grade="A")])
    assert len(merged) == 1
    assert set(merged[0].sources) == {"a", "b"}
    assert merged[0].grade == "A"


def test_write_day_and_load(tmp_path) -> None:
    path = write_day(
        tmp_path,
        "2026-11-16",
        [_card("f1", "a")],
        rules_version="abc",
        stats=DayStats(cards_b=1),
        partial=False,
        elapsed_ms=1,
    )
    assert path.exists()
    rows = load_day(tmp_path, "2026-11-16")
    assert len(rows) == 1
    manifest = json.loads((tmp_path / "manifest-2026-11-16.json").read_text(encoding="utf-8"))
    assert manifest["rules_version"] == "abc"


def test_write_day_idempotent(tmp_path) -> None:
    write_day(
        tmp_path,
        "2026-11-16",
        [_card("f1", "a")],
        rules_version="abc",
        stats=DayStats(cards_b=1),
        partial=False,
        elapsed_ms=1,
    )
    first = (tmp_path / "facts-2026-11-16.jsonl").read_bytes()
    write_day(
        tmp_path,
        "2026-11-16",
        [_card("f1", "a")],
        rules_version="abc",
        stats=DayStats(cards_b=1),
        partial=False,
        elapsed_ms=1,
    )
    assert (tmp_path / "facts-2026-11-16.jsonl").read_bytes() == first


def test_c_samples_written(tmp_path) -> None:
    path = write_c_samples(tmp_path, "2026-11-16", [{"clause": "x"}])
    assert path is not None and path.exists()


class TestGradeMerge:
    def _card(self, grade: str, sources: tuple[str, ...]):
        from ai_news_podcast.business.kg.types import FactCard

        return FactCard(
            fact_id="f1",
            date="2026-11-16",
            subject={"id": "google", "name": "谷歌", "kind": "company"},
            role="main",
            relation="release",
            relation_raw="发布",
            slots={"object": {"text": "Gemini 4 Hydrogen"}},
            polarity="positive",
            modality="actual",
            grade=grade,
            grade_reasons=(),
            evidence={"sentence": "s", "item_id": "i1"},
            source={"id": "x", "name": "x", "category": "news"},
            continuity_id="c1",
            is_new=True,
            sources=sources,
            created_at="2026-11-16T00:00:00+00:00",
        )

    def test_a_beats_a2(self) -> None:
        from ai_news_podcast.business.kg.store import merge_cards

        merged = merge_cards([self._card("A2", ("a",)), self._card("A", ("b",))])
        assert merged[0].grade == "A"
        assert merged[0].sources == ("a", "b")

    def test_a2_beats_b(self) -> None:
        from ai_news_podcast.business.kg.store import merge_cards

        merged = merge_cards([self._card("B", ("a",)), self._card("A2", ("b",))])
        assert merged[0].grade == "A2"
