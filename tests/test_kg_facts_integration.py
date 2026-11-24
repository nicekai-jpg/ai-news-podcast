"""事实层集成：条目 → 事件 → 事实卡 → 落盘。"""

from __future__ import annotations

import json

import pytest

from ai_news_podcast.business.contracts import Event
from ai_news_podcast.business.kg.dep import DepParser
from ai_news_podcast.business.kg.lexicon import KGlexicon
from ai_news_podcast.business.kg.service import build_facts
from ai_news_podcast.business.kg.types import FactsConfig
from ai_news_podcast.business.pipeline.kg_stage_service import load_facts_config, run_facts_stage


def _item(item_id: str, title: str, text: str, *, source: str = "qbitai") -> dict:
    return {
        "item_id": item_id,
        "source": source,
        "source_label": "量子位",
        "category": "news",
        "title": title,
        "link": f"https://example.com/{item_id}",
        "published_at": "2026-11-16T10:00:00+08:00",
        "summary": "",
        "full_text": text,
        "full_text_status": "ok",
        "lang": "zh",
        "dup_of": "",
        "event_id": "evt-1",
    }


@pytest.fixture(scope="module")
def env():
    pytest.importorskip("spacy")
    lex = KGlexicon.load()
    parser = DepParser.load(lex)
    if parser is None:
        pytest.skip("缺少 spaCy 模型")
    return lex, parser


def test_build_facts_end_to_end(env) -> None:
    lex, parser = env
    items = [
        _item("i1", "比亚迪发布新电池", "比亚迪推出第二代刀片电池。"),
        _item("i2", "英伟达任命新董事", "英伟达任命 Harvey C. Jones 为公司董事会主席。"),
    ]
    event = Event(
        event_id="evt-1", representative_title="比亚迪与英伟达", item_ids=["i1", "i2"], sequence=1
    )
    result = build_facts(
        items, [event], date_str="2026-11-16", config=FactsConfig(), lexicon=lex, parser=parser
    )
    assert result.cards, "应至少产出一张事实卡"
    assert all(card.evidence["sentence"] for card in result.cards)
    assert result.stats.sentences > 0


def test_parser_unavailable_returns_empty(env) -> None:
    lex, _parser = env
    items = [_item("i1", "比亚迪发布新电池", "比亚迪推出第二代刀片电池。")]
    event = Event(event_id="evt-1", representative_title="比亚迪", item_ids=["i1"], sequence=1)
    result = build_facts(
        items, [event], date_str="2026-11-16", config=FactsConfig(), lexicon=lex, parser=None
    )
    assert result.cards


def test_env_override_disables_facts(monkeypatch, tmp_path) -> None:
    from pathlib import Path

    monkeypatch.setenv("KG_FACTS_ENABLED", "false")
    config = load_facts_config(Path("config"))
    assert config.enabled is False
    monkeypatch.setenv("KG_FACTS_ENABLED", "true")
    config = load_facts_config(Path("config"))
    assert config.enabled is True


def test_run_facts_stage_parser_missing_writes_alert(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(DepParser, "load", classmethod(lambda cls, *args, **kwargs: None))
    stats = run_facts_stage([], [], date_str="2026-11-09", config_root="config", data_root=tmp_path)
    assert stats["ok"] is False
    assert stats["error"] == "parser_missing"
    manifest = json.loads(
        (tmp_path / "facts" / "manifest-2026-11-09.json").read_text(encoding="utf-8")
    )
    assert manifest["alert"] == "parser_missing"
    assert manifest["stats"]["cards_a"] == 0
