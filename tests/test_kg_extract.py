"""知识图谱：候选三元组生成与关系归一。"""

from __future__ import annotations

from ai_news_podcast.business.kg.extract import extract_candidates, split_sentences
from ai_news_podcast.business.kg.lexicon import KGlexicon
from ai_news_podcast.business.kg.relations import canonical_relation


def test_split_sentences_keeps_meaningful_pieces() -> None:
    sentences = split_sentences("阿里巴巴发布新模型。它很快！短")
    assert sentences == ["阿里巴巴发布新模型"]


def test_extracts_release_triple() -> None:
    lexicon = KGlexicon.load()
    candidates = extract_candidates("阿里巴巴发布了通义千问。", "阿里巴巴发布新模型", lexicon)
    assert candidates, "应该抽到至少一条候选"
    first = candidates[0]
    assert first.head_node_id == "alibaba"
    assert first.action_id == "release"
    assert "通义" in first.tail_text or "qwen" in first.tail_text.lower()


def test_extracts_personnel_triple() -> None:
    lexicon = KGlexicon.load()
    candidates = extract_candidates("黄仁勋辞去公司CEO职务。", "英伟达换帅", lexicon)
    assert any(item.head_node_id == "jensen_huang" for item in candidates)


def test_canonical_relation_mapping() -> None:
    assert canonical_relation("funding") == "funding"
    assert canonical_relation("ipo") == "funding"
    assert canonical_relation("investigation") == "regulation"
    assert canonical_relation("lawsuit") == "legal"
    assert canonical_relation("unknown_action") == "other"
