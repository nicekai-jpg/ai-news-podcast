"""Stage 2 四层去重。

| 层 | 规则 | 输出 |
|---|---|---|
| L1 URL | 规范化链接完全相同 | 后者并入前者，reason=url_exact |
| L2 标题 | rapidfuzz ratio ≥ title_threshold（版本号不同时保护） | reason=title_near |
| L3 语义 | TF-IDF char n-gram cosine ≥ semantic_threshold（保留正文最长/官方/更早） | reason=semantic |
| L4 跨日 | 与最近 30 天 events + coverage ledger 比对 | 已覆盖且无新进展 → 过滤；未覆盖 → reappeared |

每条 item 记录 `dup_of`（并入的 item_id 或历史 event_id）与 `dup_reason`。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ai_news_podcast.business.contracts import Event, Item, canonical_link
from ai_news_podcast.business.coverage.ledger import CoverageEntry, is_covered
from ai_news_podcast.business.ocean.config import cfg_get
from ai_news_podcast.business.ocean.continuity import detect_new_progress
from ai_news_podcast.business.ocean.labels import LabeledItem
from ai_news_podcast.business.ocean.similarity import SimilarityIndex
from ai_news_podcast.business.ocean.textutils import (
    clean_text,
    extract_version_strings,
    keywords,
    normalize_title,
    parse_time,
    title_similarity,
)

log = logging.getLogger(__name__)

DUP_URL = "url_exact"
DUP_TITLE = "title_near"
DUP_SEMANTIC = "semantic"
DUP_CROSS_DAY = "cross_day_covered"

_REASON_PRIORITY = {DUP_URL: 3, DUP_TITLE: 2, DUP_SEMANTIC: 1}


@dataclass(frozen=True)
class DupInfo:
    dup_of: str
    reason: str
    score: float = 0.0
    detail: str = ""


@dataclass
class DedupResult:
    survivors: list[Item] = field(default_factory=list)
    duplicates: dict[str, DupInfo] = field(default_factory=dict)
    reappeared: set[str] = field(default_factory=set)
    stats: dict[str, int] = field(default_factory=dict)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.members: dict[int, list[int]] = {index: [index] for index in range(size)}

    def find(self, index: int) -> int:
        root = index
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[index] != root:
            self.parent[index], index = root, self.parent[index]
        return root

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        self.parent[right_root] = left_root
        self.members.setdefault(left_root, []).extend(self.members.pop(right_root, []))
        return left_root

    def representative(self, root: int) -> int:
        return min(self.members.get(root, [root]))


def _clean_item_text(item: Item, config: dict[str, Any], *, include_full_text: bool = True) -> str:
    max_chars = int(cfg_get(config, "preprocess.max_full_text_chars", 4000))
    if include_full_text:
        raw = f"{item.title} {item.summary} {item.full_text[:max_chars]}"
    else:
        raw = f"{item.title} {item.summary}"
    return clean_text(raw) or item.title or " "


def _official(item: Item, config: dict[str, Any]) -> bool:
    categories = {
        str(value).strip().lower()
        for value in cfg_get(config, "evidence.official_categories", ["official"])
    }
    return (item.category or "").strip().lower() in categories


def _retention_key(item: Item, index: int, config: dict[str, Any]) -> tuple[float, int, float, int]:
    """保留规则：正文更长 > 官方源 > 更早发布时间 > 输入顺序。"""
    published = parse_time(item.published_at)
    stamp = published.timestamp() if published else float("inf")
    return (float(len(item.full_text or "")), int(_official(item, config)), -stamp, -index)


def _versions_distinct(left: str, right: str) -> bool:
    left_versions = extract_version_strings(left)
    right_versions = extract_version_strings(right)
    if not left_versions or not right_versions:
        return False
    return not (left_versions <= right_versions or right_versions <= left_versions)


def _blocking_tokens(title: str) -> list[str]:
    tokens = keywords(title, top_k=8)
    normalized = normalize_title(title)
    bigrams = [
        normalized[position : position + 2] for position in range(0, min(len(normalized), 16), 2)
    ]
    return list(dict.fromkeys([*tokens, *[gram for gram in bigrams if len(gram) == 2]]))


def _mark_reason(
    uf: _UnionFind,
    root: int,
    reason: str,
    score: float,
    reasons: dict[int, str],
    scores: dict[int, float],
) -> None:
    for member in uf.members.get(root, [root]):
        current = reasons.get(member)
        if current is None or _REASON_PRIORITY[reason] > _REASON_PRIORITY[current]:
            reasons[member] = reason
            scores[member] = score


def _group_by_url(items: list[Item], uf: _UnionFind) -> None:
    first_by_url: dict[str, int] = {}
    for index, item in enumerate(items):
        link = item.normalized_link or canonical_link(item.link)
        if not link:
            continue
        first = first_by_url.get(link)
        if first is None:
            first_by_url[link] = index
        else:
            uf.union(first, index)


def _merge_by_title(
    items: list[Item],
    uf: _UnionFind,
    config: dict[str, Any],
    reasons: dict[int, str],
    scores: dict[int, float],
) -> None:
    threshold = float(cfg_get(config, "dedupe.title_threshold", 0.90))
    min_chars = int(cfg_get(config, "dedupe.min_title_chars", 6))
    max_candidates = int(cfg_get(config, "dedupe.max_title_candidates", 300))
    token_index: dict[str, list[int]] = {}
    for position, item in enumerate(items):
        candidates: list[int] = []
        seen_candidates: set[int] = set()
        for token in _blocking_tokens(item.title):
            for candidate in token_index.get(token, []):
                if candidate not in seen_candidates:
                    seen_candidates.add(candidate)
                    candidates.append(candidate)
        best: tuple[float, int] | None = None
        if len(item.title) >= min_chars:
            for candidate in candidates[:max_candidates]:
                if uf.find(candidate) == uf.find(position):
                    continue
                similarity = title_similarity(item.title, items[candidate].title)
                if similarity < threshold:
                    continue
                if _versions_distinct(item.title, items[candidate].title):
                    continue
                if best is None or similarity > best[0]:
                    best = (similarity, candidate)
        if best is not None:
            root = uf.union(uf.find(best[1]), position)
            _mark_reason(uf, root, DUP_TITLE, best[0], reasons, scores)
        for token in _blocking_tokens(item.title):
            token_index.setdefault(token, []).append(position)


def _merge_by_semantic(
    items: list[Item],
    roots: list[int],
    uf: _UnionFind,
    config: dict[str, Any],
    reasons: dict[int, str],
    scores: dict[int, float],
) -> None:
    threshold = float(cfg_get(config, "dedupe.semantic_threshold", 0.30))
    docs = [
        _clean_item_text(items[uf.representative(root)], config, include_full_text=False)
        for root in roots
    ]
    index = SimilarityIndex(docs)
    if not index.available:
        return
    matrix = index.matrix_similarity()
    if matrix is None:
        return
    for position, root in enumerate(roots):
        anchor_root = uf.find(root)
        if anchor_root != root:
            continue
        for right_position in range(position + 1, len(roots)):
            right_root = uf.find(roots[right_position])
            if right_root == anchor_root:
                continue
            similarity = float(matrix[position][right_position])
            if similarity < threshold:
                continue
            if _versions_distinct(docs[position], docs[right_position]):
                continue
            merged = uf.union(anchor_root, right_root)
            anchor_root = merged
            _mark_reason(uf, merged, DUP_SEMANTIC, similarity, reasons, scores)


def _coverage_entries(ledger: dict[str, Any]) -> list[CoverageEntry]:
    return [
        CoverageEntry.from_dict(raw) for raw in ledger.get("entries") or [] if isinstance(raw, dict)
    ]


def _ledger_covers_item(
    item: Item,
    matched_event: Event,
    entries: list[CoverageEntry],
    ledger: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    link = canonical_link(item.link)
    links = [link] if link else []
    if is_covered(ledger, links=links, event_id=matched_event.event_id):
        return True
    if matched_event.covered_before:
        return True
    title_threshold = float(cfg_get(config, "continuity.covered_title_threshold", 0.85))
    for entry in entries:
        if matched_event.continuity_id and entry.continuity_id == matched_event.continuity_id:
            return True
        entry_links = {canonical_link(entry_link) for entry_link in entry.links if entry_link}
        if link and link in entry_links:
            return True
        if (
            entry.title
            and title_similarity(entry.title, matched_event.representative_title) >= title_threshold
        ):
            return True
    return False


def _match_history(
    item: Item,
    index: SimilarityIndex,
    history_events: list[Event],
    config: dict[str, Any],
) -> tuple[Event | None, float]:
    title_threshold = float(cfg_get(config, "dedupe.cross_day_title_threshold", 0.88))
    semantic_threshold = float(cfg_get(config, "dedupe.cross_day_semantic_threshold", 0.40))
    text = _clean_item_text(item, config, include_full_text=False)
    similarities = index.similarities_to(text) if index.available else []
    best: tuple[Event | None, float] = (None, 0.0)
    for position, history in enumerate(history_events):
        title_sim = title_similarity(item.title, history.representative_title)
        semantic_sim = float(similarities[position]) if position < len(similarities) else 0.0
        if title_sim < title_threshold and semantic_sim < semantic_threshold:
            continue
        score = max(title_sim, semantic_sim)
        if best[0] is None or score > best[1]:
            best = (history, score)
    return best


def _cross_day_dedupe(
    survivors: list[Item],
    labeled: Mapping[str, LabeledItem],
    history: list[Event],
    ledger: dict[str, Any],
    config: dict[str, Any],
    duplicates: dict[str, DupInfo],
) -> tuple[list[Item], set[str], dict[str, int]]:
    """L4：与历史 events + coverage ledger 比对。"""
    entries = _coverage_entries(ledger)
    history_docs = [
        clean_text(f"{event.representative_title} {event.summary}") or event.representative_title
        for event in history
    ]
    index = SimilarityIndex(history_docs)
    reappeared: set[str] = set()
    kept: list[Item] = []
    dropped = 0
    for item in survivors:
        matched, _score = _match_history(item, index, history, config)
        if matched is None:
            kept.append(item)
            continue
        item_label = labeled.get(item.item_id)
        progress = detect_new_progress(
            _clean_item_text(item, config),
            [f"{matched.representative_title} {matched.summary}"],
            current_actions=item_label.labels.actions if item_label else (),
            previous_actions=matched.labels.actions,
            current_numbers=item_label.labels.novelty_numbers if item_label else None,
            previous_numbers=matched.labels.novelty_numbers,
            config=config,
        )
        covered = _ledger_covers_item(item, matched, entries, ledger, config)
        if covered and not progress.is_new:
            duplicates[item.item_id] = DupInfo(
                dup_of=matched.event_id,
                reason=DUP_CROSS_DAY,
                score=1.0,
                detail=f"already covered event '{matched.representative_title}'",
            )
            dropped += 1
            continue
        if not covered:
            reappeared.add(item.item_id)
        kept.append(item)
    return (
        kept,
        reappeared,
        {
            "cross_day_covered": dropped,
            "reappeared": len(reappeared),
        },
    )


def dedupe_items(
    items: list[Item],
    *,
    labeled: Mapping[str, LabeledItem] | None = None,
    history_events: Iterable[Event | dict[str, Any]] | None = None,
    ledger: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> DedupResult:
    """执行 L1–L4 去重，返回存活条目、重复映射与 reappeared 标记。"""
    cfg = config or {}
    ledger = ledger or {"version": 1, "entries": []}
    labeled = labeled or {}
    history = [
        event if isinstance(event, Event) else Event.from_dict(event)
        for event in (history_events or [])
    ]
    if not items:
        return DedupResult(stats={"input": 0, "kept": 0})

    size = len(items)
    uf = _UnionFind(size)
    reasons: dict[int, str] = {}
    scores: dict[int, float] = {}

    _group_by_url(items, uf)
    for index in range(size):
        root = uf.find(index)
        if len(uf.members.get(root, [])) > 1:
            _mark_reason(uf, root, DUP_URL, 1.0, reasons, scores)
    _merge_by_title(items, uf, cfg, reasons, scores)
    roots = sorted({uf.find(index) for index in range(size)}, key=uf.representative)
    _merge_by_semantic(items, roots, uf, cfg, reasons, scores)

    final_roots = sorted({uf.find(index) for index in range(size)}, key=uf.representative)
    duplicates: dict[str, DupInfo] = {}
    survivors: list[Item] = []
    for root in final_roots:
        members = sorted(uf.members.get(root, [root]))
        best_index = max(members, key=lambda index: _retention_key(items[index], index, cfg))
        survivors.append(items[best_index])
        for index in members:
            if index == best_index:
                continue
            duplicates[items[index].item_id] = DupInfo(
                dup_of=items[best_index].item_id,
                reason=reasons.get(index, DUP_SEMANTIC),
                score=round(scores.get(index, 0.0), 4),
                detail=f"merged into '{items[best_index].title}'",
            )

    stats = {
        "input": size,
        "url_exact": sum(1 for info in duplicates.values() if info.reason == DUP_URL),
        "title_near": sum(1 for info in duplicates.values() if info.reason == DUP_TITLE),
        "semantic": sum(1 for info in duplicates.values() if info.reason == DUP_SEMANTIC),
    }

    reappeared: set[str] = set()
    if history and bool(cfg_get(cfg, "dedupe.cross_day_new_progress", True)):
        survivors, reappeared, cross_day_stats = _cross_day_dedupe(
            survivors, labeled, history, ledger, cfg, duplicates
        )
        stats.update(cross_day_stats)
    stats["kept"] = len(survivors)
    return DedupResult(
        survivors=survivors,
        duplicates=duplicates,
        reappeared=reappeared,
        stats=stats,
    )
