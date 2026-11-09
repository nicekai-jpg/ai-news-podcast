"""Stage 2 事件聚类：候选对生成 + 相似度 + 连通分量/HDBSCAN。

先按“共享实体/关键词 + ±48h 时间窗”生成候选对（倒排索引，避免 O(N²)），
再用 TF-IDF 余弦 + 实体 Jaccard + 标题相似度判定是否同一事件；
默认用连通分量（确定性、可回溯），可选 sklearn HDBSCAN。
代表标题取“含实体+动作的最短标题”，fallback 到最短标题。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ai_news_podcast.business.contracts import Item
from ai_news_podcast.business.ocean.config import cfg_get
from ai_news_podcast.business.ocean.labels import LabeledItem
from ai_news_podcast.business.ocean.similarity import SimilarityIndex
from ai_news_podcast.business.ocean.textutils import (
    clean_text,
    extract_version_strings,
    hours_between,
    normalize_title,
    parse_time,
    title_similarity,
)

log = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    item_ids: list[str]
    representative_item_id: str
    representative_title: str
    size: int = 0
    score: float = 0.0

    def __post_init__(self) -> None:
        self.size = len(self.item_ids)


@dataclass
class ClusterOutcome:
    clusters: list[ClusterResult] = field(default_factory=list)
    candidate_pairs: int = 0


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        root = index
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[index] != root:
            self.parent[index], index = root, self.parent[index]
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _item_doc(item: Item) -> str:
    return clean_text(f"{item.title} {item.summary}") or item.title or " "


def _versions_distinct(left: str, right: str) -> bool:
    left_versions = extract_version_strings(left)
    right_versions = extract_version_strings(right)
    if not left_versions or not right_versions:
        return False
    return not (left_versions <= right_versions or right_versions <= left_versions)


def _candidate_positions(
    items: list[Item],
    labeled: Mapping[str, LabeledItem],
    config: dict[str, Any],
) -> list[set[int]]:
    """倒排索引生成候选（共享实体或共享关键词）。"""
    max_candidates = int(cfg_get(config, "cluster.max_candidates_per_item", 200))
    entity_index: dict[str, list[int]] = {}
    keyword_index: dict[str, list[int]] = {}
    entity_keys: list[frozenset[str]] = []
    item_keywords: list[tuple[str, ...]] = []
    for position, item in enumerate(items):
        label = labeled.get(item.item_id)
        entities = label.entity_keys if label else frozenset()
        tokens = label.keywords if label else ()
        entity_keys.append(entities)
        item_keywords.append(tokens)
        for entity in entities:
            entity_index.setdefault(entity, []).append(position)
        for keyword in tokens:
            keyword_index.setdefault(keyword, []).append(position)
    candidates: list[set[int]] = []
    for position in range(len(items)):
        pool: set[int] = set()
        for entity in entity_keys[position]:
            pool.update(entity_index.get(entity, []))
        for keyword in item_keywords[position]:
            pool.update(keyword_index.get(keyword, []))
        pool.discard(position)
        if len(pool) > max_candidates:
            pool = set(sorted(pool)[:max_candidates])
        candidates.append(pool)
    return candidates


def _pair_accepted(
    position: int,
    other: int,
    items: list[Item],
    labeled: Mapping[str, LabeledItem],
    similarity: float,
    config: dict[str, Any],
) -> bool:
    semantic_threshold = float(cfg_get(config, "cluster.semantic_threshold", 0.55))
    entity_threshold = float(cfg_get(config, "cluster.entity_jaccard_threshold", 0.50))
    title_threshold = float(cfg_get(config, "cluster.title_threshold", 0.85))
    min_shared_keywords = int(cfg_get(config, "cluster.min_shared_keywords", 2))
    reject_versions = bool(cfg_get(config, "cluster.reject_distinct_versions", True))

    if reject_versions and _versions_distinct(items[position].title, items[other].title):
        return False
    title_sim = title_similarity(items[position].title, items[other].title)
    if title_sim >= title_threshold:
        return True
    if similarity < semantic_threshold:
        return False
    left = labeled.get(items[position].item_id)
    right = labeled.get(items[other].item_id)
    left_entities = left.entity_keys if left else frozenset()
    right_entities = right.entity_keys if right else frozenset()
    union = left_entities | right_entities
    entity_jaccard = len(left_entities & right_entities) / len(union) if union else 0.0
    shared_keywords = len(set(left.keywords if left else ()) & set(right.keywords if right else ()))
    return entity_jaccard >= entity_threshold or shared_keywords >= min_shared_keywords


def _within_time_window(left: Item, right: Item, window_hours: float) -> bool:
    left_time = parse_time(left.published_at)
    right_time = parse_time(right.published_at)
    if left_time is None or right_time is None:
        return True
    return hours_between(left_time, right_time) <= window_hours


def _component_groups(
    items: list[Item],
    labeled: Mapping[str, LabeledItem],
    index: SimilarityIndex,
    config: dict[str, Any],
    candidates: list[set[int]],
) -> tuple[list[list[int]], int]:
    window_hours = float(cfg_get(config, "cluster.time_window_hours", 48))
    max_pairs = int(cfg_get(config, "cluster.max_candidate_pairs", 50000))
    matrix = index.matrix_similarity()
    uf = _UnionFind(len(items))
    pair_count = 0
    for position in range(len(items)):
        for other in sorted(candidates[position]):
            if other <= position:
                continue
            if not _within_time_window(items[position], items[other], window_hours):
                continue
            similarity = 0.0
            if matrix is not None:
                similarity = float(matrix[position][other])
            if not _pair_accepted(position, other, items, labeled, similarity, config):
                continue
            pair_count += 1
            if pair_count > max_pairs:
                log.warning("Cluster candidate pairs exceeded cap (%s); stop adding.", max_pairs)
                break
            uf.union(position, other)
        if pair_count > max_pairs:
            break
    groups: dict[int, list[int]] = {}
    for position in range(len(items)):
        groups.setdefault(uf.find(position), []).append(position)
    return list(groups.values()), pair_count


def _hdbscan_groups(
    items: list[Item],
    labeled: Mapping[str, LabeledItem],
    index: SimilarityIndex,
    config: dict[str, Any],
    candidates: list[set[int]],
) -> tuple[list[list[int]], int] | None:
    """可选 HDBSCAN 聚类；失败返回 None 走连通分量。"""
    matrix = index.matrix_similarity()
    if matrix is None:
        return None
    try:
        import numpy as np
        from sklearn.cluster import HDBSCAN

        distances = np.clip(1.0 - matrix, 0.0, 1.0)
        min_cluster_size = int(cfg_get(config, "cluster.hdbscan_min_cluster_size", 2))
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric="precomputed",
            copy=False,
        ).fit_predict(distances)
    except (ImportError, ValueError, TypeError) as exc:
        log.warning("HDBSCAN clustering unavailable (%s); using connected components.", exc)
        return None
    groups: dict[int, list[int]] = {}
    for position, label in enumerate(labels):
        groups.setdefault(int(label), []).append(position)
    allowed: set[tuple[int, int]] = set()
    for position, others in enumerate(candidates):
        for other in others:
            if other > position and _pair_accepted(
                position, other, items, labeled, float(matrix[position][other]), config
            ):
                allowed.add((position, other))
    # 用候选边把 HDBSCAN 簇再切开，保证“版本不同不合并”
    result: list[list[int]] = []
    for members in groups.values():
        uf = _UnionFind(len(items))
        member_set = set(members)
        for left, right in allowed:
            if left in member_set and right in member_set:
                uf.union(left, right)
        split: dict[int, list[int]] = {}
        for member in members:
            split.setdefault(uf.find(member), []).append(member)
        result.extend(split.values())
    return result, len(allowed)


def _pick_representative(
    members: list[int],
    items: list[Item],
    labeled: Mapping[str, LabeledItem],
) -> int:
    def quality(position: int) -> tuple[int, int, int, int]:
        label = labeled.get(items[position].item_id)
        labels = label.labels if label else None
        has_entity = int(bool(labels and (labels.subjects or labels.object_type)))
        has_action = int(bool(labels and labels.primary_action))
        return (
            int(has_entity and has_action),
            has_entity,
            -len(normalize_title(items[position].title)),
            -position,
        )

    return max(members, key=quality)


def cluster_items(
    items: list[Item],
    *,
    labeled: Mapping[str, LabeledItem],
    config: dict[str, Any] | None = None,
) -> ClusterOutcome:
    """把今日条目聚成事件，返回稳定排序的聚类结果。"""
    cfg = config or {}
    if not items:
        return ClusterOutcome()
    if len(items) == 1:
        only = items[0]
        return ClusterOutcome(
            clusters=[
                ClusterResult(
                    item_ids=[only.item_id],
                    representative_item_id=only.item_id,
                    representative_title=only.title,
                )
            ]
        )
    index = SimilarityIndex([_item_doc(item) for item in items])
    candidates = _candidate_positions(items, labeled, cfg)
    method = str(cfg_get(cfg, "cluster.method", "components"))
    grouped: tuple[list[list[int]], int] | None = None
    if method == "hdbscan":
        grouped = _hdbscan_groups(items, labeled, index, cfg, candidates)
    if grouped is None:
        grouped = _component_groups(items, labeled, index, cfg, candidates)
    clusters: list[ClusterResult] = []
    for members in grouped[0]:
        representative = _pick_representative(members, items, labeled)
        chosen = items[representative]
        clusters.append(
            ClusterResult(
                item_ids=sorted(items[position].item_id for position in members),
                representative_item_id=chosen.item_id,
                representative_title=chosen.title,
            )
        )
    clusters.sort(key=lambda cluster: (-cluster.size, cluster.representative_title))
    return ClusterOutcome(clusters=clusters, candidate_pairs=grouped[1])
