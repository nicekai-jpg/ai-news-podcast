"""知识图谱抽取的数据结构：节点、边、文档。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

KINDS = ("company", "person", "org", "product", "tech", "number", "date", "event", "other")


@dataclass(frozen=True)
class KGNode:
    """图谱节点。node_id 优先用词表 id，未登录实体用 `open:<hash>`。"""

    node_id: str
    name: str
    kind: str = "other"
    source: str = "lexicon"  # lexicon / open
    mentions: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "mentions": self.mentions,
        }


@dataclass(frozen=True)
class KGEdge:
    """图谱边：头节点、关系、尾节点 + 出处句 + 置信度。"""

    head_id: str
    head_name: str
    relation: str
    relation_raw: str
    tail_id: str
    tail_name: str
    confidence: float = 0.0
    sentence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_id": self.head_id,
            "head_name": self.head_name,
            "relation": self.relation,
            "relation_raw": self.relation_raw,
            "tail_id": self.tail_id,
            "tail_name": self.tail_name,
            "confidence": round(float(self.confidence), 3),
            "sentence": self.sentence,
        }

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.head_id, self.relation, self.tail_id)


@dataclass
class KGDocument:
    """一篇文章的抽取结果。"""

    title: str
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "stats": self.stats,
        }


__all__ = ["KINDS", "KGDocument", "KGEdge", "KGNode"]


# ---------------------------------------------------------------------------
# V2 事实层结构（详细设计 v0.6）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LexiconSpan:
    """词表实体在原文里的位置与归一信息。"""

    start: int
    end: int
    name: str
    entity_id: str = ""
    kind: str = "other"
    source: str = "lexicon"


@dataclass(frozen=True)
class DepToken:
    """依存树上的一个词。"""

    i: int
    text: str
    idx: int
    dep: str
    head: int
    pos: str
    ent_type: str = ""


@dataclass
class ParseResult:
    """一个子句的解析结果。"""

    text: str
    tokens: list[DepToken] = field(default_factory=list)
    spans: list[LexiconSpan] = field(default_factory=list)
    root_index: int = -1
    ents: list[tuple[str, int, int, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Predicate:
    """一个动作及其论元（token 下标）。"""

    verb_index: int
    verb_text: str
    subjects: tuple[int, ...] = ()
    objects: tuple[int, ...] = ()
    attrs: tuple[int, ...] = ()
    preps: tuple[tuple[str, int], ...] = ()
    reported: bool = False


@dataclass(frozen=True)
class SlotValue:
    """填进事实卡的一个槽位值。"""

    name: str
    text: str
    entity_id: str = ""
    kind: str = "other"
    value: float | None = None
    unit: str = ""


@dataclass
class FactCandidate:
    """判档前的候选事实。"""

    clause: str
    subject: SlotValue
    relation: str
    relation_raw: str
    slots: dict[str, SlotValue] = field(default_factory=dict)
    polarity: str = "positive"
    modality: str = "actual"
    reasons: list[str] = field(default_factory=list)
    item_id: str = ""
    published_at: str = ""
    secondary: bool = False
    event_time: str = ""


@dataclass
class FactCard:
    """落盘结构。"""

    fact_id: str
    date: str
    subject: dict[str, str]
    role: str
    relation: str
    relation_raw: str
    slots: dict[str, dict[str, object]]
    polarity: str
    modality: str
    grade: str
    grade_reasons: tuple[str, ...]
    evidence: dict[str, str]
    source: dict[str, str]
    continuity_id: str
    is_new: bool
    sources: tuple[str, ...]
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "date": self.date,
            "subject": self.subject,
            "role": self.role,
            "relation": self.relation,
            "relation_raw": self.relation_raw,
            "slots": self.slots,
            "polarity": self.polarity,
            "modality": self.modality,
            "grade": self.grade,
            "grade_reasons": list(self.grade_reasons),
            "evidence": self.evidence,
            "source": self.source,
            "continuity_id": self.continuity_id,
            "is_new": self.is_new,
            "sources": list(self.sources),
            "created_at": self.created_at,
        }


@dataclass
class DayStats:
    """一天的统计（写 manifest/日志）。"""

    events: int = 0
    events_with_cards: int = 0
    skipped_dup_items: int = 0
    en_items: int = 0
    skipped_long_sentences: int = 0
    split_long_sentences: int = 0
    truncated_items: int = 0
    sentences: int = 0
    clauses: int = 0
    predicates: int = 0
    candidates: int = 0
    cards_a: int = 0
    cards_a2: int = 0
    cards_b: int = 0
    cards_new: int = 0
    errors: int = 0
    inherited_subjects: int = 0
    title_only_cards: int = 0
    c_samples: int = 0
    relation_counts: dict[str, int] = field(default_factory=dict)
    unmatched_subjects: list[tuple[str, int]] = field(default_factory=list)
    unmatched_verbs: list[tuple[str, int]] = field(default_factory=list)
    unmatched_objects: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "events": self.events,
            "events_with_cards": self.events_with_cards,
            "skipped_dup_items": self.skipped_dup_items,
            "en_items": self.en_items,
            "skipped_long_sentences": self.skipped_long_sentences,
            "split_long_sentences": self.split_long_sentences,
            "truncated_items": self.truncated_items,
            "sentences": self.sentences,
            "clauses": self.clauses,
            "predicates": self.predicates,
            "candidates": self.candidates,
            "cards_a": self.cards_a,
            "cards_a2": self.cards_a2,
            "cards_b": self.cards_b,
            "cards_new": self.cards_new,
            "errors": self.errors,
            "inherited_subjects": self.inherited_subjects,
            "title_only_cards": self.title_only_cards,
            "c_samples": self.c_samples,
            "relation_counts": dict(self.relation_counts),
            "unmatched_subjects": [list(x) for x in self.unmatched_subjects],
            "unmatched_verbs": [list(x) for x in self.unmatched_verbs],
            "unmatched_objects": [list(x) for x in self.unmatched_objects],
        }


@dataclass(frozen=True)
class FactsConfig:
    """Stage 2.5 事实层配置（6 个旋钮 + 固定值）。"""

    enabled: bool = True
    max_sentences_guard: int = 500
    write_b: bool = True
    n_process: int = 2
    max_seconds: int = 600
    debug_kg: bool = False
    tokenizer: str = "spacy"  # spacy | custom
    parser: str = "spacy"  # spacy | ltp
    # 固定值（写死，不开放配置）
    max_sentence_chars: int = 600
    max_clause_chars: int = 120
    object_max_chars: int = 30
    c_sample_limit: int = 100
    langs: tuple[str, ...] = ("zh", "en")


@dataclass
class FactsBuildResult:
    """build_facts 的返回：卡 + 统计 + C 样本。"""

    cards: list[FactCard] = field(default_factory=list)
    stats: DayStats = field(default_factory=DayStats)
    c_samples: list[dict[str, object]] = field(default_factory=list)


__all__ += [
    "DayStats",
    "DepToken",
    "FactCandidate",
    "FactCard",
    "FactsBuildResult",
    "FactsConfig",
    "LexiconSpan",
    "ParseResult",
    "Predicate",
    "SlotValue",
]
