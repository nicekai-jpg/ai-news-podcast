"""把 Stage 2 的 `config/nlp` 词典包装成知识图谱用的匹配器。

只做四件事：加载词典、按类别找别名、给出统一节点 id、给事实层做 span 匹配。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_news_podcast.business.kg.types import LexiconSpan
from ai_news_podcast.business.ocean.labels import (
    KIND_ACTION,
    KIND_ENTITY,
    KIND_OBJECT,
    Lexicon,
    load_lexicon,
    match_aliases,
)

DEFAULT_NLP_DIR = "config/nlp"

_ENTITY_KINDS = {"company": "company", "lab": "org", "person": "person"}
_LATIN_NAME = re.compile(r"[A-Z][A-Za-z.]*(?:\s+[A-Z][A-Za-z.]*)+")


@dataclass(frozen=True)
class MatchedEntity:
    """词典命中的一个实体/对象。"""

    node_id: str
    name: str
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class MatchedAction:
    """词典命中的一个动作。"""

    action_id: str
    name: str
    start: int
    end: int
    surface: str


class KGlexicon:
    """知识图谱侧的词典门面。"""

    def __init__(self, lexicon: Lexicon, nlp_dir: str = DEFAULT_NLP_DIR) -> None:
        self.lexicon = lexicon
        self.nlp_dir = nlp_dir
        self._action_map_cache: dict[str, str] | None = None

    @classmethod
    def load(cls, nlp_dir: str | Path = DEFAULT_NLP_DIR) -> KGlexicon:
        return cls(load_lexicon(nlp_dir), str(nlp_dir))

    def entities(self, text_lower: str) -> list[MatchedEntity]:
        """主体（公司 / 实验室 / 人物）。"""
        rows = match_aliases(text_lower, self.lexicon.index, KIND_ENTITY)
        return [
            MatchedEntity(
                node_id=str(row.term.id),
                name=str(row.term.name),
                kind=_ENTITY_KINDS.get(str(row.term.type), "org"),
                start=row.start,
                end=row.end,
            )
            for row in rows
        ]

    def objects(self, text_lower: str) -> list[MatchedEntity]:
        """产品 / 模型（只作为尾节点候选）。"""
        rows = match_aliases(text_lower, self.lexicon.index, KIND_OBJECT)
        return [
            MatchedEntity(
                node_id=str(row.term.id),
                name=str(row.term.name),
                kind="product",
                start=row.start,
                end=row.end,
            )
            for row in rows
        ]

    def actions(self, text_lower: str) -> list[MatchedAction]:
        """动作锚点。"""
        rows = match_aliases(text_lower, self.lexicon.index, KIND_ACTION)
        return [
            MatchedAction(
                action_id=str(row.term.id),
                name=str(row.term.name),
                start=row.start,
                end=row.end,
                surface=text_lower[row.start : row.end],
            )
            for row in rows
        ]

    @property
    def action_map(self) -> dict[str, str]:
        """别名（小写）→ 动作 id，供谓词动词查表。"""
        if self._action_map_cache is None:
            mapping: dict[str, str] = {}
            for action in self.lexicon.actions:
                for alias in (action.name, *action.aliases):
                    text = str(alias or "").strip().lower()
                    if text:
                        mapping.setdefault(text, str(action.id))
            self._action_map_cache = mapping
        return self._action_map_cache

    def action_id(self, surface: str) -> str:
        """按动词/名词表面词查动作 id；没有返回空串。"""
        return self.action_map.get(str(surface or "").strip().lower(), "")

    def find_spans(self, text: str) -> list[LexiconSpan]:
        """词表实体/对象 + 拉丁人名，长词优先、去重叠。"""
        lowered = text.lower()
        rows: list[LexiconSpan] = [
            LexiconSpan(
                item.start,
                item.end,
                text[item.start : item.end] or item.name,
                item.node_id,
                item.kind,
                "lexicon",
            )
            for item in self.entities(lowered)
        ]
        rows.extend(
            LexiconSpan(
                item.start,
                item.end,
                text[item.start : item.end] or item.name,
                item.node_id,
                "product",
                "lexicon",
            )
            for item in self.objects(lowered)
        )
        for match in _LATIN_NAME.finditer(text):
            value = match.group(0).strip()
            if len(value) >= 3:
                rows.append(
                    LexiconSpan(
                        match.start(),
                        match.start() + len(value),
                        value,
                        "",
                        "person",
                        "latin",
                    )
                )
        rows.sort(key=lambda row: (row.start, -(row.end - row.start)))
        kept: list[LexiconSpan] = []
        last_end = -1
        for row in rows:
            if row.start < last_end:
                continue
            kept.append(row)
            last_end = row.end
        return kept


def merge_candidates(
    matched: list[MatchedEntity],
    opened: list[Any],
    *,
    source: str = "lexicon",
) -> list[dict[str, Any]]:
    """把词典实体和未登录实体合成候选：词典优先，重叠时丢弃未登录的那个。"""
    lex_rows: list[dict[str, Any]] = [
        {
            "text": item.name,
            "start": item.start,
            "end": item.end,
            "node_id": item.node_id,
            "kind": item.kind,
            "source": source,
        }
        for item in matched
    ]
    open_rows: list[dict[str, Any]] = [
        {
            "text": item.text,
            "start": item.start,
            "end": item.end,
            "node_id": "",
            "kind": item.kind,
            "source": "open",
        }
        for item in opened
    ]
    lex_rows.sort(key=lambda row: (row["start"], -(row["end"] - row["start"])))
    open_rows = [
        row
        for row in open_rows
        if not any(row["start"] < item["end"] and item["start"] < row["end"] for item in lex_rows)
    ]
    rows = sorted(
        [*lex_rows, *open_rows], key=lambda row: (row["start"], -(row["end"] - row["start"]))
    )
    kept: list[dict[str, Any]] = []
    last_end = -1
    for row in rows:
        if row["start"] < last_end:
            continue
        kept.append(row)
        last_end = row["end"]
    return kept


__all__ = [
    "DEFAULT_NLP_DIR",
    "KGlexicon",
    "LexiconSpan",
    "MatchedAction",
    "MatchedEntity",
    "merge_candidates",
]
