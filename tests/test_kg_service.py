"""知识图谱：服务层（接地、去重、置信度、节点 id）。"""

from __future__ import annotations

import pytest

from ai_news_podcast.business.kg.service import build_kg


@pytest.fixture(scope="module")
def lexicon():
    from ai_news_podcast.business.kg.lexicon import KGlexicon

    return KGlexicon.load()


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
