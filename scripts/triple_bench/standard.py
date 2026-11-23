"""评测标准：关系词表、文本归一化、统一分句、数据结构。

所有参评方案必须走这里的 `split_sentences()`，避免"切句不同"混进模型差异。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

RELATIONS: tuple[str, ...] = (
    "release",
    "open_source",
    "update",
    "pricing",
    "supply",
    "deploy",
    "support",
    "funding",
    "acquisition",
    "partnership",
    "benchmark",
    "personnel",
    "layoff",
    "regulation",
    "legal",
    "incident",
    "research",
    "event",
    "info",
    "award",
    "other",
)

RELATION_ZH: dict[str, str] = {
    "release": "发布",
    "open_source": "开源",
    "update": "更新",
    "pricing": "价格",
    "supply": "量产交付",
    "deploy": "部署接入",
    "support": "支持",
    "funding": "融资上市",
    "acquisition": "收购",
    "partnership": "合作",
    "benchmark": "评测对比",
    "personnel": "人事",
    "layoff": "裁员",
    "regulation": "监管",
    "legal": "诉讼",
    "incident": "事故风险",
    "research": "研究",
    "event": "活动",
    "info": "信息披露",
    "award": "获奖认证",
    "other": "其他",
}

# 各模型/工具常见的别名写法，全部收口到 21 类。
RELATION_ALIASES: dict[str, str] = {
    "发布": "release",
    "推出": "release",
    "开源": "open_source",
    "更新": "update",
    "价格": "pricing",
    "定价": "pricing",
    "量产交付": "supply",
    "供货": "supply",
    "部署接入": "deploy",
    "部署": "deploy",
    "支持": "support",
    "兼容": "support",
    "融资上市": "funding",
    "融资": "funding",
    "上市": "funding",
    "收购": "acquisition",
    "并购": "acquisition",
    "合作": "partnership",
    "联合": "partnership",
    "评测对比": "benchmark",
    "评测": "benchmark",
    "人事": "personnel",
    "裁员": "layoff",
    "监管": "regulation",
    "处罚": "regulation",
    "诉讼": "legal",
    "事故风险": "incident",
    "事故": "incident",
    "研究": "research",
    "活动": "event",
    "信息披露": "info",
    "获奖认证": "award",
    "获奖": "award",
    "其他": "other",
    "open source": "open_source",
    "deployment": "deploy",
    "provide": "support",
    "cooperate": "partnership",
    "cooperation": "partnership",
    "lawsuit": "legal",
    "accident": "incident",
    "recall": "incident",
    "disclosed": "info",
    "rumour": "info",
    "rumor": "info",
    "prediction": "info",
}

_TRAILING = " \t\r\n，。；、：:,.!?！？;\"'“”‘’《》()（）[]【】"
_QUOTES = "\"'“”‘’《》()（）[]【】"


def normalize_relation(relation: str) -> str:
    """关系 id 归一；未知值返回 other（调用方应记录告警）。"""
    value = unicodedata.normalize("NFKC", str(relation or "")).strip().lower()
    value = re.sub(r"[\s_\-]+", " ", value)
    if value in RELATIONS:
        return value
    if value in RELATION_ALIASES:
        return RELATION_ALIASES[value]
    compact = value.replace(" ", "")
    if compact in RELATIONS:
        return compact
    if compact in RELATION_ALIASES:
        return RELATION_ALIASES[compact]
    return "other"


def is_known_relation(relation: str) -> bool:
    return normalize_relation(relation) != "other"


def normalize_text(text: str) -> str:
    """端点归一化：全半角、大小写、空白、首尾标点。"""
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.strip(_TRAILING)
    value = value.strip(_QUOTES)
    value = re.sub(r"[\s\u3000·・]+", "", value)
    return value.lower()


def sentence_split(text: str, *, min_chars: int = 8) -> list[str]:
    """统一分句。所有 adapter 都必须用这份结果，保证输入一致。"""
    raw = re.split(r"(?<=[。！？!?；;\n])", str(text or ""))
    out: list[str] = []
    for part in raw:
        value = part.strip()
        if len(value) >= min_chars:
            out.append(value)
    return out


@dataclass(slots=True)
class Doc:
    item_id: str
    title: str
    text: str
    lang: str
    published_at: str = ""
    event_id: str = ""

    def sentences(self) -> list[str]:
        body = f"{self.title}。{self.text}" if self.title else self.text
        return sentence_split(body)

    def to_json(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "text": self.text,
            "lang": self.lang,
            "published_at": self.published_at,
            "event_id": self.event_id,
        }

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> Doc:
        return cls(
            item_id=str(row.get("item_id") or ""),
            title=str(row.get("title") or ""),
            text=str(row.get("text") or row.get("full_text") or ""),
            lang=str(row.get("lang") or ""),
            published_at=str(row.get("published_at") or ""),
            event_id=str(row.get("event_id") or ""),
        )


@dataclass(slots=True)
class Triple:
    subject: str
    relation: str
    object: str
    confidence: float = 0.0
    evidence: str = ""
    object_kind: str = "entity"  # entity | value
    meta: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str, str]:
        return (
            normalize_relation(self.relation),
            normalize_text(self.subject),
            normalize_text(self.object),
        )

    def to_json(self) -> dict[str, Any]:
        out = {
            "subject": self.subject,
            "relation": normalize_relation(self.relation),
            "object": self.object,
            "confidence": round(float(self.confidence), 4),
        }
        if self.evidence:
            out["evidence"] = self.evidence
        if self.object_kind != "entity":
            out["object_kind"] = self.object_kind
        if self.meta:
            out["meta"] = self.meta
        return out


@dataclass(slots=True)
class GoldItem:
    item_id: str
    title: str
    lang: str
    split: str
    status: str
    triples: list[Triple]

    def to_json(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "lang": self.lang,
            "split": self.split,
            "status": self.status,
            "triples": [t.to_json() for t in self.triples],
        }

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> GoldItem:
        triples = [
            Triple(
                subject=str(t.get("subject") or ""),
                relation=normalize_relation(str(t.get("relation") or "")),
                object=str(t.get("object") or ""),
                confidence=float(t.get("confidence") or 0.0),
                evidence=str(t.get("evidence") or ""),
                object_kind=str(t.get("object_kind") or "entity"),
                meta=dict(t.get("meta") or {}),
            )
            for t in row.get("triples") or []
        ]
        return cls(
            item_id=str(row.get("item_id") or ""),
            title=str(row.get("title") or ""),
            lang=str(row.get("lang") or ""),
            split=str(row.get("split") or "test"),
            status=str(row.get("status") or "draft"),
            triples=triples,
        )
