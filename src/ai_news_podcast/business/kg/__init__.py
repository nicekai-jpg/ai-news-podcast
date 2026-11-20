"""知识图谱抽取（零 LLM 调用）。

入口：`build_kg(text, title, nlp=None) -> KGDocument`。

链路：词典实体/动作 → 未登录实体补充 → 动作锚点候选 → 依存确认 → 关系归一。
"""

from ai_news_podcast.business.kg.entities import OpenEntity, detect_open_entities
from ai_news_podcast.business.kg.extract import Candidate, extract_candidates, split_sentences
from ai_news_podcast.business.kg.lexicon import KGlexicon, MatchedAction, MatchedEntity
from ai_news_podcast.business.kg.relations import CANONICAL_LABELS, canonical_label
from ai_news_podcast.business.kg.service import build_kg
from ai_news_podcast.business.kg.types import KGDocument, KGEdge, KGNode

__all__ = [
    "CANONICAL_LABELS",
    "Candidate",
    "KGDocument",
    "KGEdge",
    "KGNode",
    "KGlexicon",
    "MatchedAction",
    "MatchedEntity",
    "OpenEntity",
    "build_kg",
    "canonical_label",
    "detect_open_entities",
    "extract_candidates",
    "split_sentences",
]
