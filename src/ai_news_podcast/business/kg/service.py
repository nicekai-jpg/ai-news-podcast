"""事实层编排：按事件抽候选 → 判档 → 事实卡（S5-S7 调度）。"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter
from pathlib import Path

from ai_news_podcast.business.contracts import Event, utc_now_iso
from ai_news_podcast.business.kg import grade as grade_mod
from ai_news_podcast.business.kg import store
from ai_news_podcast.business.kg.clause import split_clauses
from ai_news_podcast.business.kg.dep import DepParser, extract_predicates
from ai_news_podcast.business.kg.extract import (
    Candidate,
    extract_candidates,
)
from ai_news_podcast.business.kg.extract import split_sentences as split_news_sentences
from ai_news_podcast.business.kg.fill import (
    FillContext,
    build_candidates,
    build_nominal_candidate,
    classify_relation,
    load_rules,
    reporting_verbs,
    resolve_mention,
)
from ai_news_podcast.business.kg.lexicon import KGlexicon
from ai_news_podcast.business.kg.noise import (
    detect_language,
    item_text,
    pick_article_subject,
    select_news_sentences,
    split_sentences,
)
from ai_news_podcast.business.kg.relations import canonical_relation
from ai_news_podcast.business.kg.schema import load_schemas
from ai_news_podcast.business.kg.types import (
    DayStats,
    FactCandidate,
    FactCard,
    FactsBuildResult,
    FactsConfig,
    KGDocument,
    KGEdge,
    KGNode,
    SlotValue,
)

logger = logging.getLogger(__name__)

MAX_SOURCES_PER_EVENT = 8
_SUBJECT_DEPS = {"nsubj", "nsubjpass", "nsubj:xsubj"}
_OBJECT_DEPS = {"dobj", "obj", "attr"}


def _serialize_slot(slot: SlotValue) -> dict[str, object]:
    payload: dict[str, object] = {
        "text": slot.text,
        "entity_id": slot.entity_id,
        "kind": slot.kind,
    }
    if slot.value is not None:
        payload["value"] = slot.value
    if slot.unit:
        payload["unit"] = slot.unit
    return payload


def _object_key(candidate: FactCandidate) -> str:
    if candidate.relation == "personnel":
        # 人事事实的去重依据是「谁 + 哪家公司」，不包含职务：
        # 「辞去总裁/CEO/董事」是同一件事被分句后重复抽到的，职务写进 role 槽即可。
        person = candidate.slots.get("person")
        if person is not None and person.text:
            return person.text
    for name in ("object", "person", "role"):
        slot = candidate.slots.get(name)
        if slot is not None and slot.text:
            return slot.text
    return ""


def _peer_key(candidate: FactCandidate) -> str:
    if candidate.relation == "personnel":
        # 人事事实按「人」去重；公司归属不稳定（有的分句带定语、有的不带），
        # 放进键里会把同一件事拆成两张卡。
        return ""
    for name in ("peer", "company", "investor"):
        slot = candidate.slots.get(name)
        if slot is not None and slot.text:
            return slot.text
    return ""


_RETARGET_MAX_GAP = 40


def _retarget_clause(candidate: FactCandidate, text: str) -> None:
    """证据句取自标题时，把它改指到正文里能同时支撑两端的句子。

    标题派生的句子常常是概括或标题党（"OpenAI 和 Meta 联手"），
    但同一事实在正文里可能有明确表述（"中金公司与蚂蚁集团签署战略合作协议"）。
    能找到就改指，找不到则保留原句，交给 `grade()` 的 `evidence_from_title` 降档。
    """
    clause = candidate.clause.strip()
    if not clause or clause in text:
        return
    anchors = [candidate.subject.text.strip()]
    if candidate.relation == "personnel":
        role = candidate.slots.get("role")
        anchors.append(str(role.text or "").strip() if role is not None else "")
    else:
        obj = candidate.slots.get("object")
        anchors.append(str(obj.text or "").strip() if obj is not None else "")
    anchors = [anchor for anchor in anchors if anchor]
    if len(anchors) < 2:
        return
    for sentence in split_news_sentences(text):
        positions = [sentence.find(anchor) for anchor in anchors]
        if any(position < 0 for position in positions):
            continue
        # 两端必须挨得够近：同一句里隔了几十个字的实体谈的往往不是同一件事
        # （如「OpenAI 与 Visa、Mastercard 均在其中……Meta 的 Muse 与 OpenAI 的 dots 是竞争对手」）。
        if max(positions) - min(positions) > _RETARGET_MAX_GAP:
            continue
        window = sentence[min(positions) : max(positions) + max(len(a) for a in anchors) + 20]
        if not any(cue and cue in window for cue in (candidate.relation_raw,)):
            continue
        candidate.clause = sentence.strip()
        return


def _make_card(
    candidate: FactCandidate,
    grade: str,
    reasons: list[str],
    *,
    date: str,
    item: dict,
    evidence_scope: str,
    continuity_id: str,
    is_new: bool,
) -> FactCard:
    object_key = _object_key(candidate)
    fact_id = store.fact_key(
        date,
        candidate.subject.entity_id,
        candidate.subject.text,
        candidate.relation,
        object_key,
        _peer_key(candidate),
    )
    evidence = {
        "sentence": candidate.clause,
        "item_id": str(item.get("item_id") or ""),
        "url": str(item.get("link") or ""),
        "published_at": str(item.get("published_at") or candidate.published_at or ""),
        "scope": evidence_scope,
    }
    source = {
        "id": str(item.get("source") or ""),
        "name": str(item.get("source_label") or ""),
        "category": str(item.get("category") or ""),
    }
    return FactCard(
        fact_id=fact_id,
        date=date,
        subject={
            "id": candidate.subject.entity_id,
            "name": candidate.subject.text,
            "kind": candidate.subject.kind,
        },
        role="secondary" if candidate.secondary else "main",
        relation=candidate.relation,
        relation_raw=candidate.relation_raw,
        slots={name: _serialize_slot(value) for name, value in candidate.slots.items()},
        polarity=candidate.polarity,
        modality=candidate.modality,
        grade=grade,
        grade_reasons=tuple(reasons),
        evidence=evidence,
        source=source,
        continuity_id=continuity_id,
        is_new=is_new,
        sources=(str(item.get("item_id") or ""),),
        created_at=utc_now_iso(),
    )


def _evidence_scope(item: dict) -> str:
    if len(str(item.get("full_text") or "")) >= 20:
        return "body"
    if len(str(item.get("summary") or "")) >= 20:
        return "summary"
    return "title"


class _ItemJob:
    """一个待处理条目。"""

    def __init__(self, item: dict, event: Event | None) -> None:
        self.item = item
        self.event = event
        self.continuity_id = str(event.continuity_id or "") if event is not None else ""
        sequence = int(event.sequence or 1) if event is not None else 1
        self.is_new = sequence == 1


def _collect_jobs(items: list[dict], events: list[Event], stats: DayStats) -> list[_ItemJob]:
    """事件级优先；无事件的条目兜底；重复条目跳过。"""
    by_id = {str(row.get("item_id") or ""): row for row in items if row.get("item_id")}
    jobs: list[_ItemJob] = []
    used: set[str] = set()
    stats.events = len(events)
    for event in events:
        usable: list[dict] = []
        for item_id in event.item_ids:
            item = by_id.get(str(item_id))
            if item is None:
                continue
            if str(item.get("dup_of") or ""):
                stats.skipped_dup_items += 1
                continue
            text = item_text(item)
            if not text and len(str(item.get("title") or "")) < 6:
                continue
            usable.append(item)
        for item in usable[:MAX_SOURCES_PER_EVENT]:
            used.add(str(item.get("item_id") or ""))
            jobs.append(_ItemJob(item, event))
    for row in items:
        item_id = str(row.get("item_id") or "")
        if not item_id or item_id in used:
            continue
        if str(row.get("dup_of") or ""):
            stats.skipped_dup_items += 1
            continue
        text = item_text(row)
        if not text and len(str(row.get("title") or "")) < 6:
            continue
        jobs.append(_ItemJob(row, None))
    return jobs


def build_facts(  # noqa: C901, PLR0912, PLR0915
    items: list[dict],
    events: list[Event],
    *,
    date_str: str,
    config: FactsConfig,
    lexicon: KGlexicon | None = None,
    parser: DepParser | None = None,
    schema_path: str | Path | None = None,
    rules_path: str | Path | None = None,
) -> FactsBuildResult:
    """纯计算：不写盘、不联网、不调 LLM。"""
    lex = lexicon or KGlexicon.load()
    schemas = load_schemas(schema_path) if schema_path is not None else load_schemas()
    rules = load_rules(rules_path) if rules_path is not None else load_rules()
    reporting = reporting_verbs(rules_path) if rules_path is not None else reporting_verbs()
    if parser is None:
        parser = DepParser.load(lex)
    stats = DayStats()
    result = FactsBuildResult(stats=stats)
    if parser is None and config.enabled:
        stats.errors += 1
        return result

    jobs = _collect_jobs(items, events, stats)
    started = time.monotonic()
    subject_counter: Counter[str] = Counter()
    verb_counter: Counter[str] = Counter()
    object_counter: Counter[str] = Counter()
    c_samples: list[dict[str, object]] = []

    for job in jobs:
        if time.monotonic() - started > config.max_seconds:
            result.stats.errors += 0
            break
        row = job.item
        lang = detect_language(row)
        if lang not in config.langs:
            continue
        if lang == "en":
            stats.en_items += 1
        title = str(row.get("title") or "")
        text = item_text(row)
        scope = _evidence_scope(row)
        article_subject = pick_article_subject(title, lex)
        if article_subject is None and text:
            for lead in split_sentences(text)[:3]:
                article_subject = pick_article_subject(lead, lex)
                if article_subject is not None:
                    break
        sentences = select_news_sentences(text, config, title=title if len(title) >= 6 else "")
        if not sentences:
            continue
        last_subject: SlotValue | None = None
        item_cards = 0
        for sentence in sentences:
            last_subject = None
            lowered_sentence = sentence.lower()
            sentence_modality = "actual"
            if any(word and word in lowered_sentence for word in rules.get("rumor", ())):
                sentence_modality = "rumor"
            elif any(word and word in lowered_sentence for word in rules.get("forecast", ())):
                sentence_modality = "forecast"
            clauses = split_clauses(
                sentence,
                max_chars=config.max_clause_chars,
            )
            stats.sentences += 1
            stats.clauses += len(clauses)
            for clause in clauses:
                try:
                    parse = parser.analyze(clause, lang, title=title)
                except Exception:
                    stats.errors += 1
                    continue
                predicates = extract_predicates(parse, reporting)
                stats.predicates += len(predicates)
                ctx = FillContext(
                    lexicon=lex,
                    schemas=schemas,
                    item_id=str(row.get("item_id") or ""),
                    published_at=str(row.get("published_at") or ""),
                    article_subject=article_subject,
                    prev_subject=last_subject,
                    object_max_chars=config.object_max_chars,
                    lang=lang,
                    sentence_modality=sentence_modality,
                )
                clause_cards = 0
                for pred in predicates:
                    for index in pred.subjects:
                        value = resolve_mention(parse, index)
                        if value is not None:
                            last_subject = value
                            break
                    match = classify_relation(parse, pred, lex)
                    if match is None:
                        verb_counter[pred.verb_text] += 1
                        continue
                    candidates = build_candidates(parse, pred, match, ctx)
                    for candidate in candidates:
                        stats.candidates += 1
                        candidate.item_id = ctx.item_id
                        _retarget_clause(candidate, text)
                        grade, reasons = grade_mod.grade(
                            candidate,
                            article_subject_text=(
                                article_subject.text if article_subject is not None else ""
                            ),
                            object_max_chars=config.object_max_chars,
                            evidence_scope=scope,
                            source_text=text,
                            title=title,
                        )
                        if grade == "C":
                            if len(c_samples) < config.c_sample_limit:
                                c_samples.append(
                                    {
                                        "item_id": ctx.item_id,
                                        "clause": candidate.clause,
                                        "subject": candidate.subject.text,
                                        "relation": candidate.relation,
                                        "reasons": reasons,
                                    }
                                )
                            continue
                        card = _make_card(
                            candidate,
                            grade,
                            reasons,
                            date=date_str,
                            item=row,
                            evidence_scope=scope,
                            continuity_id=str(job.continuity_id or row.get("continuity_id") or ""),
                            is_new=bool(job.is_new),
                        )
                        result.cards.append(card)
                        stats.relation_counts[candidate.relation] = (
                            stats.relation_counts.get(candidate.relation, 0) + 1
                        )
                        item_cards += 1
                        clause_cards += 1
                        last_subject = candidate.subject
                        if not candidate.subject.entity_id:
                            subject_counter[candidate.subject.text] += 1
                        object_slot = candidate.slots.get("object")
                        if object_slot is not None and not object_slot.entity_id:
                            object_counter[object_slot.text] += 1
                if clause_cards:
                    continue
                nominal = build_nominal_candidate(parse, ctx)
                if nominal is None:
                    continue
                stats.candidates += 1
                nominal.item_id = ctx.item_id
                _retarget_clause(nominal, text)
                grade, reasons = grade_mod.grade(
                    nominal,
                    article_subject_text=(
                        article_subject.text if article_subject is not None else ""
                    ),
                    object_max_chars=config.object_max_chars,
                    evidence_scope=scope,
                    source_text=text,
                    title=title,
                )
                if grade == "C":
                    continue
                result.cards.append(
                    _make_card(
                        nominal,
                        grade,
                        reasons,
                        date=date_str,
                        item=row,
                        evidence_scope=scope,
                        continuity_id=str(job.continuity_id or row.get("continuity_id") or ""),
                        is_new=bool(job.is_new),
                    )
                )
                stats.relation_counts[nominal.relation] = (
                    stats.relation_counts.get(nominal.relation, 0) + 1
                )
                item_cards += 1
        if item_cards:
            stats.events_with_cards += 1

    result.cards = store.merge_cards(result.cards)
    stats.cards_a = sum(1 for card in result.cards if card.grade == "A")
    stats.cards_a2 = sum(1 for card in result.cards if card.grade == "A2")
    stats.cards_b = len(result.cards) - stats.cards_a - stats.cards_a2
    stats.cards_new = sum(1 for card in result.cards if card.is_new)
    stats.unmatched_subjects = subject_counter.most_common(20)
    stats.unmatched_verbs = verb_counter.most_common(20)
    stats.unmatched_objects = object_counter.most_common(20)
    stats.c_samples = len(c_samples)
    result.c_samples = c_samples
    return result


__all__ = ["MAX_SOURCES_PER_EVENT", "build_facts"]


# ---------------------------------------------------------------------------
# V1 接口（build_kg）：旧 Stage 2.5 / 调试产物 / 旧测试继续用
# ---------------------------------------------------------------------------


def _node_id(text: str, node_id: str, source: str) -> str:
    if node_id:
        return node_id
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]
    return f"open:{digest}"


def _add_node(nodes: dict[str, KGNode], node_id: str, name: str, kind: str, source: str) -> None:
    existing = nodes.get(node_id)
    if existing is None:
        nodes[node_id] = KGNode(node_id=node_id, name=name, kind=kind, source=source)
    else:
        nodes[node_id] = KGNode(
            node_id=node_id,
            name=existing.name,
            kind=existing.kind,
            source=existing.source,
            mentions=existing.mentions + 1,
        )


def _confirm_with_dependency(candidates: list[Candidate], nlp: object) -> None:
    """用依存树核对主语/宾语角色：双确认 0.9，单确认 0.65，都没 0.4。"""
    cache: dict[str, object] = {}
    for index, candidate in enumerate(candidates):
        sentence = candidate.sentence
        doc = cache.get(sentence)
        if doc is None:
            try:
                doc = nlp(sentence)  # type: ignore[operator]
            except Exception:
                continue
            cache[sentence] = doc
        action_tokens = [t for t in doc if t.text == candidate.action_surface]
        if not action_tokens:
            candidate.confidence = 0.4
            continue
        action = action_tokens[0]
        head_ok = any(
            t.text == candidate.head_text and t.dep_ in _SUBJECT_DEPS and t.head.i == action.i
            for t in doc
        )
        tail_ok = any(
            (t.text == candidate.tail_text or candidate.tail_text.startswith(t.text))
            and t.dep_ in _OBJECT_DEPS
            and t.head.i == action.i
            for t in doc
        )
        if head_ok and tail_ok:
            confidence = 0.9
        elif head_ok or tail_ok:
            confidence = 0.65
        else:
            confidence = 0.4
        candidates[index] = Candidate(
            head_text=candidate.head_text,
            head_node_id=candidate.head_node_id,
            head_kind=candidate.head_kind,
            head_source=candidate.head_source,
            action_id=candidate.action_id,
            action_name=candidate.action_name,
            action_surface=candidate.action_surface,
            tail_text=candidate.tail_text,
            tail_node_id=candidate.tail_node_id,
            tail_kind=candidate.tail_kind,
            tail_source=candidate.tail_source,
            sentence=candidate.sentence,
            confidence=confidence,
        )


def build_kg(
    text: str,
    title: str = "",
    *,
    lexicon: KGlexicon | None = None,
    nlp: object | None = None,
    min_confidence: float = 0.4,
    strict: bool = False,
) -> KGDocument:
    """把一篇文章抽成知识图谱文档（V1 接口）。"""
    lex = lexicon or KGlexicon.load()
    body = f"{title}。{text}" if title else text
    candidates = extract_candidates(body, title, lex, strict=strict)
    if nlp is not None:
        _confirm_with_dependency(candidates, nlp)
    kept = [item for item in candidates if item.confidence >= min_confidence]
    nodes: dict[str, KGNode] = {}
    edges: list[KGEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for item in kept:
        head_id = _node_id(item.head_text, item.head_node_id, item.head_source)
        tail_id = _node_id(item.tail_text, item.tail_node_id, item.tail_source)
        _add_node(nodes, head_id, item.head_text, item.head_kind, item.head_source)
        _add_node(nodes, tail_id, item.tail_text, item.tail_kind, item.tail_source)
        edge = KGEdge(
            head_id=head_id,
            head_name=item.head_text,
            relation=canonical_relation(item.action_id),
            relation_raw=item.action_name,
            tail_id=tail_id,
            tail_name=item.tail_text,
            confidence=item.confidence,
            sentence=item.sentence[:160],
        )
        if edge.key in seen:
            continue
        seen.add(edge.key)
        edges.append(edge)
    stats = {
        "sentences": len(split_news_sentences(body)),
        "candidates": len(candidates),
        "kept": len(kept),
        "high_confidence": sum(1 for item in kept if item.confidence >= 0.85),
        "nodes": len(nodes),
        "edges": len(edges),
    }
    return KGDocument(title=title, nodes=list(nodes.values()), edges=edges, stats=stats)


__all__ += ["build_kg"]
