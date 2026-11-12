"""Tests for ocean/semantic.py — LLM 语义标注（枚举/证据/置信度）。"""

from __future__ import annotations

import json

from ai_news_podcast.business.contracts import Item, Labels
from ai_news_podcast.business.ocean.semantic import (
    SemanticLabeler,
    build_prompt,
    parse_response,
)
from ai_news_podcast.business.ocean.taxonomy import default_taxonomy
from tests.test_ocean_support import make_item


def _item(item_id: str = "i1", title: str = "OpenAI 发布新模型") -> Item:
    return make_item(item_id=item_id, title=title, summary="支持 Agent")


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "item_id": "i1",
        "ai_centrality": "central",
        "primary_domain": "ai_core",
        "industry": "ai",
        "evidence": {"ai_relevance": "OpenAI", "primary_domain": "模型"},
        "confidence": 0.9,
    }
    row.update(overrides)
    return row


def _labeler(payload: object) -> SemanticLabeler:
    return SemanticLabeler(
        lambda _prompt: json.dumps(payload, ensure_ascii=False),
        default_taxonomy(),
        min_confidence=0.6,
    )


def test_build_prompt_lists_enums() -> None:
    prompt = build_prompt([_item()], default_taxonomy())
    assert "ai_centrality" in prompt
    assert "primary_domain" in prompt
    assert "OpenAI 发布新模型" in prompt


def test_parse_response_fenced_json() -> None:
    assert parse_response('```json\n[{"item_id": "i1"}]\n```')[0]["item_id"] == "i1"


def test_suggest_and_apply_sets_hybrid() -> None:
    labeler = _labeler([_row()])
    suggestions = labeler.suggest([_item()])
    labels = Labels()
    updated = labeler.apply(labels, suggestions["i1"])
    assert updated.ai_relevance == "core"
    assert updated.primary_domain == "ai_core"
    assert updated.industry == "ai"
    assert updated.label_method["ai_relevance"] == "hybrid"
    assert updated.label_confidence["ai_relevance"] == 0.9


def test_invalid_enum_or_evidence_is_dropped() -> None:
    bad_enum = _labeler([_row(ai_centrality="banana")]).suggest([_item()])
    assert bad_enum == {}
    no_evidence = _labeler([_row(evidence={"ai_relevance": "不存在的句子"})]).suggest([_item()])
    assert "i1" in no_evidence
    assert no_evidence["i1"].evidence == {}


def test_low_confidence_keeps_rule_labels() -> None:
    labeler = _labeler([_row(confidence=0.2)])
    updated = labeler.apply(Labels(), labeler.suggest([_item()])["i1"])
    assert updated.ai_relevance == "unknown"
