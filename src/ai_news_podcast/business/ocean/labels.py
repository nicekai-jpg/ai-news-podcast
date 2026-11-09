"""Stage 2 规则标注：从 `config/nlp/*.yaml` 词典抽取事件属性。

只产出"事实/属性"，不产出分数/排名（v5 审查结论）：
- subjects（含 weight/confidence）、object_type（多标签）、actions（多标签 + primary）；
- tech_stack、regions、topics（多标签 + primary）；
- lifecycle（仅粗排/点评用，由 sequence/来源数推导）；
- evidence_tier（仅粗排门槛，来自 evidence 模块，不在本模块计算）；
- 新增 6 个标签：novelty_numbers / narrative_angle / audience_persona /
  sensitivity / quote_availability / momentum_growth（momentum 由 build 计算）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ai_news_podcast.business.contracts import Item, Labels
from ai_news_podcast.business.ocean.config import cfg_get
from ai_news_podcast.business.ocean.textutils import (
    clean_text,
    extract_novelty_numbers,
    has_quoted_speech,
    keywords,
)

log = logging.getLogger(__name__)

KIND_ENTITY = "entity"
KIND_OBJECT = "object"
KIND_ACTION = "action"
KIND_TECH = "tech"
KIND_REGION = "region"
KIND_TOPIC = "topic"
KIND_NARRATIVE = "narrative"
KIND_SENSITIVITY = "sensitivity"

_ALL_KINDS = (
    KIND_ENTITY,
    KIND_OBJECT,
    KIND_ACTION,
    KIND_TECH,
    KIND_REGION,
    KIND_TOPIC,
    KIND_NARRATIVE,
    KIND_SENSITIVITY,
)


# ---------------------------------------------------------------------------
# 词典结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityDef:
    id: str
    name: str
    aliases: tuple[str, ...]
    type: str = "company"
    weight: float = 0.5
    ambiguous: bool = False
    headquarters: str = ""


@dataclass(frozen=True)
class ObjectDef:
    id: str
    name: str
    aliases: tuple[str, ...]
    type: str = "model"
    family: str = ""


@dataclass(frozen=True)
class ActionDef:
    id: str
    name: str
    aliases: tuple[str, ...]
    priority: int = 0


@dataclass(frozen=True)
class TermDef:
    id: str
    name: str
    aliases: tuple[str, ...]
    priority: int = 0


@dataclass(frozen=True)
class PersonaDef:
    id: str
    name: str
    keywords: tuple[str, ...] = ()
    tech_stack: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    object_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class _AliasMatch:
    term: Any
    start: int
    end: int


@dataclass(frozen=True)
class _AliasIndex:
    ascii_patterns: dict[str, Any] = field(default_factory=dict)
    ascii_terms: dict[str, dict[str, Any]] = field(default_factory=dict)
    cjk_aliases: dict[str, dict[str, tuple[tuple[str, Any], ...]]] = field(default_factory=dict)
    context_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Lexicon:
    entities: tuple[EntityDef, ...]
    objects: tuple[ObjectDef, ...]
    actions: tuple[ActionDef, ...]
    tech: tuple[TermDef, ...]
    regions: tuple[TermDef, ...]
    topics: tuple[TermDef, ...]
    narrative: tuple[TermDef, ...]
    personas: tuple[PersonaDef, ...]
    sensitivity: tuple[TermDef, ...]
    type_keywords: dict[str, tuple[str, ...]]
    index: _AliasIndex


@dataclass
class LabeledItem:
    """条目级标注结果 + 聚类所需实体键/关键词。"""

    labels: Labels
    entity_keys: frozenset[str] = frozenset()
    keywords: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# 词典加载
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        log.warning("NLP dictionary missing: %s", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not value:
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _aliases(row: dict[str, Any], name: str) -> tuple[str, ...]:
    values = [name, *_string_list(row.get("aliases"))]
    seen: dict[str, None] = {}
    for value in values:
        if value:
            seen.setdefault(value, None)
    return tuple(seen)


def _load_entities(nlp_dir: Path) -> tuple[EntityDef, ...]:
    data = _read_yaml(nlp_dir / "entities.yaml")
    result: list[EntityDef] = []
    for group, default_type in (("companies", "company"), ("labs", "lab"), ("people", "person")):
        for row in data.get(group) or []:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            name = str(row.get("name") or row["id"])
            result.append(
                EntityDef(
                    id=str(row["id"]),
                    name=name,
                    aliases=_aliases(row, name),
                    type=str(row.get("type") or default_type),
                    weight=float(row.get("weight", 0.5)),
                    ambiguous=bool(row.get("ambiguous", False)),
                    headquarters=str(row.get("headquarters") or ""),
                )
            )
    return tuple(result)


def _load_objects(nlp_dir: Path) -> tuple[ObjectDef, ...]:
    data = _read_yaml(nlp_dir / "objects.yaml")
    result: list[ObjectDef] = []
    for row in data.get("objects") or []:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        name = str(row.get("name") or row["id"])
        result.append(
            ObjectDef(
                id=str(row["id"]),
                name=name,
                aliases=_aliases(row, name),
                type=str(row.get("type") or "model"),
                family=str(row.get("family") or ""),
            )
        )
    return tuple(result)


def _load_actions(nlp_dir: Path) -> tuple[ActionDef, ...]:
    data = _read_yaml(nlp_dir / "actions.yaml")
    result: list[ActionDef] = []
    for row in data.get("actions") or []:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        name = str(row.get("name") or row["id"])
        result.append(
            ActionDef(
                id=str(row["id"]),
                name=name,
                aliases=_aliases(row, name),
                priority=int(row.get("priority", 0)),
            )
        )
    return tuple(result)


def _load_terms(nlp_dir: Path, filename: str, key: str) -> tuple[TermDef, ...]:
    data = _read_yaml(nlp_dir / filename)
    result: list[TermDef] = []
    for row in data.get(key) or []:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        name = str(row.get("name") or row["id"])
        result.append(
            TermDef(
                id=str(row["id"]),
                name=name,
                aliases=_aliases(row, name),
                priority=int(row.get("priority", 0)),
            )
        )
    return tuple(result)


def _load_personas(nlp_dir: Path) -> tuple[PersonaDef, ...]:
    data = _read_yaml(nlp_dir / "persona.yaml")
    result: list[PersonaDef] = []
    for row in data.get("personas") or []:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        result.append(
            PersonaDef(
                id=str(row["id"]),
                name=str(row.get("name") or row["id"]),
                keywords=tuple(_string_list(row.get("keywords"))),
                tech_stack=tuple(_string_list(row.get("tech_stack"))),
                actions=tuple(_string_list(row.get("actions"))),
                object_types=tuple(_string_list(row.get("object_types"))),
            )
        )
    return tuple(result)


def _load_type_keywords(nlp_dir: Path) -> dict[str, tuple[str, ...]]:
    data = _read_yaml(nlp_dir / "objects.yaml")
    result: dict[str, tuple[str, ...]] = {}
    for key, values in (data.get("type_keywords") or {}).items():
        result[str(key)] = tuple(_string_list(values))
    return result


def _alias_groups(
    terms: list[tuple[str, Any]],
) -> tuple[list[tuple[str, Any]], dict[str, list[tuple[str, Any]]], dict[str, Any]]:
    """拆分 ASCII / 非 ASCII 别名（ASCII 用边界正则，CJK 用首字索引）。"""
    ascii_pairs = sorted(
        {(alias.lower(), term) for alias, term in terms if alias.isascii()},
        key=lambda pair: (-len(pair[0]), pair[0]),
    )
    cjk_map: dict[str, list[tuple[str, Any]]] = {}
    cjk_pairs = sorted(
        {(alias.lower(), term) for alias, term in terms if not alias.isascii()},
        key=lambda pair: (-len(pair[0]), pair[0]),
    )
    for alias, term in cjk_pairs:
        cjk_map.setdefault(alias[0], []).append((alias, term))
    return ascii_pairs, cjk_map, dict(ascii_pairs)


def _build_alias_index(lexicon_parts: dict[str, list[tuple[str, Any]]]) -> _AliasIndex:
    ascii_patterns: dict[str, Any] = {}
    ascii_terms: dict[str, dict[str, Any]] = {}
    cjk_aliases: dict[str, dict[str, tuple[tuple[str, Any], ...]]] = {}
    import re

    for kind in _ALL_KINDS:
        ascii_pairs, cjk_map, ascii_map = _alias_groups(lexicon_parts.get(kind, []))
        if ascii_pairs:
            alternation = "|".join(re.escape(alias) for alias, _ in ascii_pairs)
            ascii_patterns[kind] = re.compile(
                rf"(?<![a-z0-9])(?:{alternation})(?![a-z0-9])", re.IGNORECASE
            )
            ascii_terms[kind] = ascii_map
        if cjk_map:
            cjk_aliases[kind] = {key: tuple(value) for key, value in cjk_map.items()}
    context_terms: tuple[str, ...] = tuple(
        sorted(
            {
                alias.lower()
                for kind in (KIND_TECH, KIND_OBJECT, KIND_ACTION)
                for alias, _ in lexicon_parts.get(kind, [])
                if len(alias) >= 2
            },
            key=len,
            reverse=True,
        )
    )
    return _AliasIndex(
        ascii_patterns=ascii_patterns,
        ascii_terms=ascii_terms,
        cjk_aliases=cjk_aliases,
        context_terms=context_terms,
    )


def _build_lexicon(nlp_dir: Path) -> Lexicon:
    entities = _load_entities(nlp_dir)
    objects = _load_objects(nlp_dir)
    actions = _load_actions(nlp_dir)
    tech = _load_terms(nlp_dir, "tech.yaml", "tech_stack")
    regions = _load_terms(nlp_dir, "regions.yaml", "regions")
    topics = _load_terms(nlp_dir, "topics.yaml", "topics")
    narrative = _load_terms(nlp_dir, "narrative.yaml", "narrative_angles")
    sensitivity = _load_terms(nlp_dir, "sensitivity.yaml", "sensitivity")
    personas = _load_personas(nlp_dir)
    type_keywords = _load_type_keywords(nlp_dir)

    parts: dict[str, list[tuple[str, Any]]] = {kind: [] for kind in _ALL_KINDS}
    for entity in entities:
        parts[KIND_ENTITY].extend((alias, entity) for alias in entity.aliases)
    for obj in objects:
        parts[KIND_OBJECT].extend((alias, obj) for alias in obj.aliases)
    for action in actions:
        parts[KIND_ACTION].extend((alias, action) for alias in action.aliases)
    for kind, terms in (
        (KIND_TECH, tech),
        (KIND_REGION, regions),
        (KIND_TOPIC, topics),
        (KIND_NARRATIVE, narrative),
        (KIND_SENSITIVITY, sensitivity),
    ):
        for term in terms:
            parts[kind].extend((alias, term) for alias in term.aliases)
    return Lexicon(
        entities=entities,
        objects=objects,
        actions=actions,
        tech=tech,
        regions=regions,
        topics=topics,
        narrative=narrative,
        personas=personas,
        sensitivity=sensitivity,
        type_keywords=type_keywords,
        index=_build_alias_index(parts),
    )


@lru_cache(maxsize=16)
def _load_lexicon_cached(nlp_dir: str) -> Lexicon:
    return _build_lexicon(Path(nlp_dir))


def load_lexicon(nlp_dir: str | Path) -> Lexicon:
    """加载（并缓存）`config/nlp` 词典。"""
    return _load_lexicon_cached(str(Path(nlp_dir)))


# ---------------------------------------------------------------------------
# 多模匹配
# ---------------------------------------------------------------------------


def _match_aliases(text_lower: str, index: _AliasIndex, kind: str) -> list[_AliasMatch]:
    raw: list[_AliasMatch] = []
    pattern = index.ascii_patterns.get(kind)
    if pattern is not None:
        terms = index.ascii_terms.get(kind, {})
        for match in pattern.finditer(text_lower):
            term = terms.get(match.group(0).lower())
            if term is not None:
                raw.append(_AliasMatch(term=term, start=match.start(), end=match.end()))
    for first_char, entries in index.cjk_aliases.get(kind, {}).items():
        position = text_lower.find(first_char)
        while position != -1:
            for alias, term in entries:
                if text_lower.startswith(alias, position):
                    raw.append(_AliasMatch(term=term, start=position, end=position + len(alias)))
                    break
            position = text_lower.find(first_char, position + 1)
    return _resolve_overlaps(raw)


def _resolve_overlaps(matches: list[_AliasMatch]) -> list[_AliasMatch]:
    selected: list[_AliasMatch] = []
    last_end = -1
    for match in sorted(matches, key=lambda item: (item.start, -(item.end - item.start))):
        if match.start < last_end:
            continue
        selected.append(match)
        last_end = match.end
    return selected


def _contains_keyword(text_lower: str, keyword: str) -> bool:
    if not keyword:
        return False
    lowered = keyword.lower()
    if lowered.isascii():
        import re

        return re.search(rf"(?<![a-z0-9]){re.escape(lowered)}(?![a-z0-9])", text_lower) is not None
    return lowered in text_lower


def _has_tech_context(text_lower: str, match: _AliasMatch, index: _AliasIndex, window: int) -> bool:
    start = max(0, match.start - window)
    end = min(len(text_lower), match.end + window)
    snippet = text_lower[start:end]
    return any(term and term in snippet for term in index.context_terms)


def _dedupe_by_id(matches: list[_AliasMatch]) -> list[_AliasMatch]:
    seen: set[str] = set()
    result: list[_AliasMatch] = []
    for match in matches:
        term_id = str(getattr(match.term, "id", ""))
        if term_id and term_id not in seen:
            seen.add(term_id)
            result.append(match)
    return result


# ---------------------------------------------------------------------------
# 各维度抽取
# ---------------------------------------------------------------------------


def _extract_entities(
    text_lower: str, lexicon: Lexicon, config: dict[str, Any]
) -> list[tuple[EntityDef, float]]:
    window = int(cfg_get(config, "labels.context_window_chars", 40))
    results: list[tuple[EntityDef, float]] = []
    for match in _match_aliases(text_lower, lexicon.index, KIND_ENTITY):
        entity = match.term
        if entity.ambiguous and not _has_tech_context(text_lower, match, lexicon.index, window):
            continue
        confidence = 0.6 if entity.ambiguous else 1.0
        results.append((entity, confidence))
    # 同一实体多次命中取最高置信度
    merged: dict[str, tuple[EntityDef, float]] = {}
    for entity, confidence in results:
        existing = merged.get(entity.id)
        if existing is None or confidence > existing[1]:
            merged[entity.id] = (entity, confidence)
    return list(merged.values())


def _extract_object_keys(text_lower: str, lexicon: Lexicon) -> tuple[list[str], set[str]]:
    types: list[str] = []
    keys: set[str] = set()
    for match in _match_aliases(text_lower, lexicon.index, KIND_OBJECT):
        obj = match.term
        if obj.type and obj.type not in types:
            types.append(obj.type)
        keys.add(obj.id)
        if obj.family:
            keys.add(obj.family)
    for type_name, type_words in lexicon.type_keywords.items():
        if type_name not in types and any(
            _contains_keyword(text_lower, word) for word in type_words
        ):
            types.append(type_name)
    return types, keys


def _extract_actions(text_lower: str, lexicon: Lexicon) -> tuple[list[str], str]:
    actions: list[ActionDef] = [
        match.term for match in _match_aliases(text_lower, lexicon.index, KIND_ACTION)
    ]
    unique: dict[str, ActionDef] = {}
    for action in actions:
        unique.setdefault(action.id, action)
    ordered = sorted(unique.values(), key=lambda item: (-item.priority, item.id))
    return [action.id for action in ordered], (ordered[0].id if ordered else "")


def _extract_term_ids(text_lower: str, lexicon: Lexicon, kind: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for match in _match_aliases(text_lower, lexicon.index, kind):
        term = match.term
        if term.id not in seen:
            seen.add(term.id)
            result.append(term.id)
    return result


def _priority_order(ids: list[str], term_defs: tuple[TermDef, ...]) -> list[str]:
    priorities = {term.id: term.priority for term in term_defs}
    return sorted(ids, key=lambda term_id: (-priorities.get(term_id, 0), term_id))


def _extract_personas(
    text_lower: str,
    lexicon: Lexicon,
    tech_ids: list[str],
    action_ids: list[str],
    object_types: list[str],
) -> list[str]:
    result: list[str] = []
    for persona in lexicon.personas:
        triggered = (
            any(_contains_keyword(text_lower, word) for word in persona.keywords)
            or bool(set(persona.tech_stack) & set(tech_ids))
            or bool(set(persona.actions) & set(action_ids))
            or bool(set(persona.object_types) & set(object_types))
        )
        if triggered:
            result.append(persona.id)
    return result


def _extract_regions(text_lower: str, lexicon: Lexicon, subjects: list[EntityDef]) -> list[str]:
    explicit = _extract_term_ids(text_lower, lexicon, KIND_REGION)
    result = list(explicit)
    for subject in subjects:
        headquarters = subject.headquarters
        if headquarters and headquarters not in result:
            result.append(headquarters)
    return result


def _quote_available(text_lower: str, config: dict[str, Any]) -> bool:
    if has_quoted_speech(text_lower):
        return True
    markers = [str(marker).lower() for marker in cfg_get(config, "labels.quote_markers", [])]
    return any(marker and marker in text_lower for marker in markers)


def _build_subjects(
    entities: list[tuple[EntityDef, float]],
) -> tuple[list[dict[str, Any]], dict[str, EntityDef]]:
    definitions: dict[str, EntityDef] = {entity.id: entity for entity, _ in entities}
    subjects: list[dict[str, Any]] = [
        {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "weight": round(float(entity.weight), 3),
            "confidence": round(confidence, 3),
        }
        for entity, confidence in entities
    ]
    subjects.sort(key=lambda row: (-float(row["weight"]), str(row["id"])))
    return subjects, definitions


# ---------------------------------------------------------------------------
# 标注入口
# ---------------------------------------------------------------------------


def _item_text(item: Item, config: dict[str, Any]) -> str:
    max_chars = int(cfg_get(config, "preprocess.max_full_text_chars", 4000))
    return clean_text(f"{item.title}。{item.summary}。{item.full_text[:max_chars]}")


def label_text(
    title: str,
    summary: str = "",
    full_text: str = "",
    *,
    lexicon: Lexicon,
    config: dict[str, Any],
) -> LabeledItem:
    """对一段文本做规则标注（无 Item 依赖，便于复用/测试）。"""
    max_chars = int(cfg_get(config, "preprocess.max_full_text_chars", 4000))
    text = clean_text(f"{title}。{summary}。{full_text[:max_chars]}")
    text_lower = text.lower()

    entity_pairs = _extract_entities(text_lower, lexicon, config)
    subjects, entity_defs = _build_subjects(entity_pairs)
    object_types, object_keys = _extract_object_keys(text_lower, lexicon)
    action_ids, primary_action = _extract_actions(text_lower, lexicon)
    tech_ids = _extract_term_ids(text_lower, lexicon, KIND_TECH)
    topic_ids = _priority_order(_extract_term_ids(text_lower, lexicon, KIND_TOPIC), lexicon.topics)
    narrative_ids = _extract_term_ids(text_lower, lexicon, KIND_NARRATIVE)
    sensitivity_ids = _priority_order(
        _extract_term_ids(text_lower, lexicon, KIND_SENSITIVITY), lexicon.sensitivity
    )
    persona_ids = _extract_personas(text_lower, lexicon, tech_ids, action_ids, object_types)
    regions = _extract_regions(text_lower, lexicon, list(entity_defs.values()))
    novelty_numbers = extract_novelty_numbers(text)
    max_numbers = int(cfg_get(config, "labels.novelty.max_values", 12))
    novelty_numbers = novelty_numbers[:max_numbers]
    quote_available = _quote_available(text_lower, config)

    confidence = {
        str(key): float(value)
        for key, value in (cfg_get(config, "labels.confidence", {}) or {}).items()
    }
    if any(entity.ambiguous for entity, _ in entity_pairs):
        confidence["subjects"] = min(confidence.get("subjects", 0.95), 0.6)

    labels = Labels(
        subjects=subjects,
        object_type=object_types,
        actions=action_ids,
        primary_action=primary_action,
        tech_stack=tech_ids,
        regions=regions,
        lifecycle="unknown",
        topics=topic_ids,
        primary_topic=topic_ids[0] if topic_ids else "",
        evidence_tier="unknown",
        novelty_numbers=novelty_numbers,
        narrative_angle=narrative_ids,
        audience_persona=persona_ids,
        sensitivity=sensitivity_ids,
        quote_availability=quote_available,
        confidence=confidence,
        method="rule",
    )
    entity_keys = frozenset({entity.id for entity, _ in entity_pairs} | object_keys)
    top_k = int(cfg_get(config, "preprocess.keyword_top_k", 15))
    return LabeledItem(
        labels=labels,
        entity_keys=entity_keys,
        keywords=tuple(keywords(f"{title} {summary}", top_k=top_k)),
    )


def label_item(item: Item, lexicon: Lexicon, config: dict[str, Any]) -> LabeledItem:
    """对单条 Item 做规则标注。"""
    return label_text(
        item.title,
        item.summary,
        item.full_text,
        lexicon=lexicon,
        config=config,
    )


# ---------------------------------------------------------------------------
# 事件级标签合并 / 生命周期
# ---------------------------------------------------------------------------


def _merge_ordered(values_per_item: list[list[str]]) -> list[str]:
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for position, values in enumerate(values_per_item):
        for value in values:
            counts[value] = counts.get(value, 0) + 1
            first_seen.setdefault(value, position * 1000 + len(first_seen))
    return sorted(counts, key=lambda value: (-counts[value], first_seen[value]))


def _merge_subjects(labels_list: list[Labels]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for labels in labels_list:
        for subject in labels.subjects:
            subject_id = str(subject.get("id", ""))
            if not subject_id:
                continue
            counts[subject_id] = counts.get(subject_id, 0) + 1
            current = merged.get(subject_id)
            if current is None or float(subject.get("weight", 0)) > float(current.get("weight", 0)):
                merged[subject_id] = dict(subject)
    result = list(merged.values())
    result.sort(key=lambda row: (-float(row.get("weight", 0)), -counts[str(row["id"])], row["id"]))
    return result


def _merge_novelty(labels_list: list[Labels], max_values: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for labels in labels_list:
        for number in labels.novelty_numbers:
            key = (str(number.get("type", "")), str(number.get("value", "")))
            if key not in seen:
                seen.add(key)
                merged.append(dict(number))
                if len(merged) >= max_values:
                    return merged
    return merged


def _merge_primary(labels_list: list[Labels], field_name: str, fallback: list[str]) -> str:
    ordered = _merge_ordered(
        [
            [str(getattr(labels, field_name))]
            for labels in labels_list
            if getattr(labels, field_name)
        ]
    )
    if ordered:
        return ordered[0]
    return fallback[0] if fallback else ""


def _merged_confidence(labels_list: list[Labels], config: dict[str, Any]) -> dict[str, float]:
    confidence = {
        str(key): float(value)
        for key, value in (cfg_get(config, "labels.confidence", {}) or {}).items()
    }
    if labels_list:
        confidence.update(
            {str(key): float(value) for key, value in labels_list[0].confidence.items()}
        )
    return confidence


def merge_labels(
    labeled_items: list[LabeledItem],
    *,
    evidence_tier: str = "unknown",
    config: dict[str, Any] | None = None,
) -> Labels:
    """把事件内多条 item 标签合并为事件级标签（并集 + 主标签投票）。"""
    cfg = config or {}
    labels_list = [item.labels for item in labeled_items]
    if not labels_list:
        return Labels(evidence_tier=evidence_tier, confidence={}, method="rule")
    actions = _merge_ordered([labels.actions for labels in labels_list])
    topics = _merge_ordered([labels.topics for labels in labels_list])
    max_values = int(cfg_get(cfg, "labels.novelty.max_values", 12))
    confidence = _merged_confidence(labels_list, cfg)
    confidence["evidence_tier"] = float(cfg_get(cfg, "labels.confidence.evidence_tier", 0.9))
    return Labels(
        subjects=_merge_subjects(labels_list),
        object_type=_merge_ordered([labels.object_type for labels in labels_list]),
        actions=actions,
        primary_action=_merge_primary(labels_list, "primary_action", actions),
        tech_stack=_merge_ordered([labels.tech_stack for labels in labels_list]),
        regions=_merge_ordered([labels.regions for labels in labels_list]),
        lifecycle="unknown",
        topics=topics,
        primary_topic=_merge_primary(labels_list, "primary_topic", topics),
        evidence_tier=evidence_tier,
        novelty_numbers=_merge_novelty(labels_list, max_values),
        narrative_angle=_merge_ordered([labels.narrative_angle for labels in labels_list]),
        audience_persona=_merge_ordered([labels.audience_persona for labels in labels_list]),
        sensitivity=_merge_ordered([labels.sensitivity for labels in labels_list]),
        quote_availability=any(labels.quote_availability for labels in labels_list),
        confidence=confidence,
        method="rule",
    )


def infer_lifecycle(
    *,
    sequence: int,
    source_count: int,
    is_update: bool,
    age_days: float,
    gap_days: float,
    config: dict[str, Any] | None = None,
) -> str:
    """生命周期（v4 §1.4）：只供粗排/点评，不进精排（避免与 momentum 循环）。"""
    cfg = config or {}
    burst_sources = int(cfg_get(cfg, "labels.lifecycle.burst_sources", 3))
    long_tail_days = float(cfg_get(cfg, "labels.lifecycle.long_tail_days", 7))
    fading_days = float(cfg_get(cfg, "labels.lifecycle.fading_days", 3))
    if sequence <= 1 and source_count >= burst_sources:
        return "burst"
    if is_update and sequence >= 2:
        if age_days >= long_tail_days and source_count <= 2:
            return "long_tail"
        return "evolving"
    if gap_days >= fading_days:
        return "fading"
    if age_days >= long_tail_days:
        return "long_tail"
    return "unknown"


def match_subjects(left: Labels, right: Labels) -> tuple[float, int]:
    """两个标签集合的主体 Jaccard 与共享主体数。"""
    left_ids = {str(subject.get("id", "")) for subject in left.subjects if subject.get("id")}
    right_ids = {str(subject.get("id", "")) for subject in right.subjects if subject.get("id")}
    if not left_ids and not right_ids:
        return 0.0, 0
    shared = left_ids & right_ids
    union = left_ids | right_ids
    return (len(shared) / len(union) if union else 0.0), len(shared)
