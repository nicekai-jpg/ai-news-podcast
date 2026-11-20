"""候选三元组生成：动作锚点 + 最近主体/对象。

这一步只负责「召回」，不保证主谓宾是对的；置信度靠 service 里的依存确认。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ai_news_podcast.business.kg.entities import detect_open_entities
from ai_news_podcast.business.kg.lexicon import KGlexicon, merge_candidates

_SENT_SPLIT = re.compile(r"[。！？!?；;\n]+")
_WS = re.compile(r"\s+")
_LEADING = re.compile(r"^(了|的|为|在|与|和|是|对|把|被|将|于)\s*")
_TRAILING = re.compile(r"(的|了|着|等)$")
_NOUN_POS = ("n", "nr", "ns", "nt", "nz", "nl", "ng", "eng", "j")
_MAX_SPAN = 24


@dataclass
class Candidate:
    """一条候选边（未经依存确认）。"""

    head_text: str
    head_node_id: str
    head_kind: str
    head_source: str
    action_id: str
    action_name: str
    action_surface: str
    tail_text: str
    tail_node_id: str
    tail_kind: str
    tail_source: str
    sentence: str
    confidence: float = 0.55


def split_sentences(text: str) -> list[str]:
    """按中文标点切句，过滤过短片段。"""
    sentences: list[str] = []
    for raw in _SENT_SPLIT.split(text or ""):
        sentence = _WS.sub(" ", raw).strip()
        if len(sentence) >= 8:
            sentences.append(sentence)
    return sentences


def _nearest_before(rows: list[dict[str, Any]], position: int, *, max_gap: int = 40):
    candidates = [
        row for row in rows if row["end"] <= position and position - row["end"] <= max_gap
    ]
    return max(candidates, key=lambda row: row["end"]) if candidates else None


def _nearest_after(rows: list[dict[str, Any]], position: int, *, max_gap: int = 30):
    candidates = [
        row for row in rows if row["start"] >= position and row["start"] - position <= max_gap
    ]
    return min(candidates, key=lambda row: row["start"]) if candidates else None


def _span_after(sentence: str, start: int, *, limit: int = _MAX_SPAN) -> str:
    """动作后面截一段名词短语；不像名词就返回空串。"""
    import jieba.posseg as pseg

    tail = sentence[start : start + limit]
    tail = re.split(r"[，,、。；;：:（(“”\"）)]", tail)[0]
    tail = _TRAILING.sub("", _LEADING.sub("", tail.strip())).strip()
    if len(tail) < 2:
        return ""
    words = [word for word in pseg.cut(tail) if word.word.strip()]
    if not words or not any(word.flag.startswith(_NOUN_POS) for word in words):
        return ""
    if all(word.flag.startswith(("v", "d", "r", "c", "u")) for word in words):
        return ""
    return tail


_HEAD_KINDS = {"company", "org", "person"}


def _valid_head(head: dict[str, Any]) -> bool:
    """主语必须是机构/人物，且长度合理；产品、泛词不能当主语。"""
    text = str(head.get("text") or "")
    return bool(text) and len(text) <= 20 and head.get("kind") in _HEAD_KINDS


_BAD_TAIL_CHARS = set("，。、；：（）()《》“”\"'—…!?！？/\\|")
_BAD_TAIL_POS = ("v", "d", "r", "c", "u", "p", "q")


def _valid_tail(tail: str) -> bool:
    """尾巴必须是短名词短语：不含标点、不超过 14 字、不含动词。"""
    import jieba.posseg as pseg

    text = tail.strip()
    if not 2 <= len(text) <= 14:
        return False
    if any(char in _BAD_TAIL_CHARS for char in text):
        return False
    words = [word for word in pseg.cut(text) if word.word.strip()]
    return not any(word.flag.startswith(_BAD_TAIL_POS) for word in words)


def extract_candidates(
    text: str, title: str, lexicon: KGlexicon, *, strict: bool = False
) -> list[Candidate]:
    """从正文里抽候选三元组，按 (头, 动作, 尾) 去重。"""
    candidates: list[Candidate] = []
    seen: set[tuple[str, str, str]] = set()
    for sentence in split_sentences(text):
        lower = sentence.lower()
        actions = lexicon.actions(lower)
        if not actions:
            continue
        entities = lexicon.entities(lower)
        objects = lexicon.objects(lower)
        opened = detect_open_entities(sentence, title)
        heads = merge_candidates(entities, [] if strict else opened)
        tails = merge_candidates([*objects, *entities], opened)
        tails.sort(key=lambda row: row["start"])
        for action in actions:
            head = _nearest_before(heads, action.start)
            if head is None or not _valid_head(head):
                continue
            tail = _nearest_after(tails, action.end)
            if tail is not None:
                tail_text = tail["text"]
                tail_meta: dict[str, Any] = tail
            else:
                tail_text = _span_after(sentence, action.end)
                if len(tail_text) < 2:
                    continue
                tail_meta = {"text": tail_text, "node_id": "", "kind": "other", "source": "span"}
            if head["text"] == tail_text or not 2 <= len(tail_text) <= 24:
                continue
            if strict and not _valid_tail(tail_text):
                continue
            key = (head["text"], action.action_id, tail_text)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                Candidate(
                    head_text=head["text"],
                    head_node_id=head["node_id"],
                    head_kind=head["kind"],
                    head_source=head["source"],
                    action_id=action.action_id,
                    action_name=action.name,
                    action_surface=action.surface,
                    tail_text=tail_text,
                    tail_node_id=tail_meta.get("node_id", ""),
                    tail_kind=tail_meta.get("kind", "other"),
                    tail_source=tail_meta.get("source", "open"),
                    sentence=sentence,
                )
            )
    return candidates


__all__ = ["Candidate", "extract_candidates", "split_sentences"]
