"""知识图谱：服务层（接地、去重、置信度、节点 id）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_news_podcast.business.kg.service import build_kg

# 词表用绝对路径：别的测试可能 monkeypatch.chdir()，相对路径会失效。
NLP_DIR = Path(__file__).resolve().parents[1] / "config" / "nlp"


@pytest.fixture(scope="module")
def lexicon():
    from ai_news_podcast.business.kg.lexicon import KGlexicon

    return KGlexicon.load(NLP_DIR)


def test_nodes_are_grounded_in_text(lexicon) -> None:
    text = "阿里巴巴发布了通义千问，Qwen3.6在评测中超过GPT-5。"
    doc = build_kg(text, "阿里发布新模型", lexicon=lexicon)
    for node in doc.nodes:
        assert node.name in text or node.name in "阿里发布新模型"


def test_edges_use_canonical_relation(lexicon) -> None:
    doc = build_kg("阿里巴巴发布了通义千问。", "发布", lexicon=lexicon)
    assert doc.edges
    assert all(edge.relation for edge in doc.edges)
    assert all(edge.relation_raw for edge in doc.edges)


def test_open_node_id_is_stable(lexicon) -> None:
    doc = build_kg("中专生拍出爆款电影。", "中专生拍出爆款电影", lexicon=lexicon)
    open_nodes = [node for node in doc.nodes if node.node_id.startswith("open:")]
    for node in open_nodes:
        again = build_kg("中专生拍出爆款电影。", "中专生拍出爆款电影", lexicon=lexicon)
        assert node.node_id in {item.node_id for item in again.nodes}


def test_default_confidence_without_nlp(lexicon) -> None:
    doc = build_kg("阿里巴巴发布了通义千问。", "发布", lexicon=lexicon)
    assert doc.edges
    assert all(edge.confidence == pytest.approx(0.55) for edge in doc.edges)


def test_min_confidence_filters(lexicon) -> None:
    strict = build_kg("阿里巴巴发布了通义千问。", "发布", lexicon=lexicon, min_confidence=0.9)
    assert strict.edges == []


def test_dependency_confirmation_when_available(lexicon) -> None:
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("zh_core_web_sm")
    except OSError:
        pytest.skip("缺少 zh_core_web_sm")
    doc = build_kg(
        "阿里巴巴发布了通义千问。", "发布", lexicon=lexicon, nlp=nlp, min_confidence=0.55
    )
    assert doc.edges
    assert any(edge.confidence >= 0.85 for edge in doc.edges)


def test_retarget_clause_points_to_body_sentence() -> None:
    """标题派生的证据句，若正文能同时支撑两端，应改指到正文。"""
    from ai_news_podcast.business.kg.service import _retarget_clause
    from ai_news_podcast.business.kg.types import FactCandidate, SlotValue

    candidate = FactCandidate(
        clause="OpenAI 和 Meta 联手",
        subject=SlotValue("subject", "蚂蚁集团", "ant", "company"),
        relation="partnership",
        relation_raw="合作",
        slots={"object": SlotValue("object", "中金公司", "cicc", "company")},
    )
    body = "双方签署协议。中金公司与蚂蚁集团达成战略合作，共同探索 AI 赋能财富管理。"
    _retarget_clause(candidate, body)
    assert candidate.clause == "中金公司与蚂蚁集团达成战略合作，共同探索 AI 赋能财富管理"


def test_retarget_clause_keeps_original_when_body_lacks_support() -> None:
    from ai_news_podcast.business.kg.service import _retarget_clause
    from ai_news_podcast.business.kg.types import FactCandidate, SlotValue

    candidate = FactCandidate(
        clause="OpenAI 和 Meta 联手",
        subject=SlotValue("subject", "OpenAI", "openai", "company"),
        relation="partnership",
        relation_raw="合作",
        slots={"object": SlotValue("object", "Meta", "meta", "company")},
    )
    body = "Sierra 联合 Meta 发布 Poppy。OpenAI 推出 ChatGPT Wallet。"
    _retarget_clause(candidate, body)
    assert candidate.clause == "OpenAI 和 Meta 联手"


def test_retarget_rejects_distant_mentions() -> None:
    """同一句里两个实体隔很远且不是合作关系 → 不改指（否则会制造假事实）。"""
    from ai_news_podcast.business.kg.service import _retarget_clause
    from ai_news_podcast.business.kg.types import FactCandidate, SlotValue

    candidate = FactCandidate(
        clause="OpenAI 和 Meta 联手",
        subject=SlotValue("subject", "OpenAI", "openai", "company"),
        relation="partnership",
        relation_raw="合作",
        slots={"object": SlotValue("object", "Meta", "meta", "company")},
    )
    body = (
        "首批合作伙伴 OpenAI 与 Visa 均在其中。"
        "在消费端市场，Meta 的 Muse 和 OpenAI 的 dots 是竞争对手。"
    )
    _retarget_clause(candidate, body)
    assert candidate.clause == "OpenAI 和 Meta 联手"
