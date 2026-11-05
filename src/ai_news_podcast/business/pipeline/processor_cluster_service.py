"""Stage 2 — 聚类模块。

TF-IDF char n-gram + DBSCAN(cosine) 聚类。
"""

from __future__ import annotations

import logging

from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from ai_news_podcast.business.pipeline.fetcher_service import RawItem
from ai_news_podcast.business.pipeline.processor_types_service import Cluster

logger = logging.getLogger(__name__)


def _merge_similar_singletons(
    clusters: list[Cluster], *, threshold: float = 80.0, max_size: int = 6
) -> list[Cluster]:
    """DBSCAN 在长文本上几乎全判噪声（192 条 → 192 簇），多源合并完全失效。

    对「单条成簇」的条目按标题相似度做一次贪心合并，让同一事件的不同报道能并进同一簇，
    source_richness/impact_scope 才有意义；标题不相似的保持单源，不强行凑。
    """
    merged = list(clusters)
    changed = True
    while changed:
        changed = False
        for i, candidate in enumerate(merged):
            if len(candidate.items) > 1:
                continue
            for j, target in enumerate(merged):
                if i == j or len(target.items) >= max_size:
                    continue
                score = fuzz.token_set_ratio(
                    candidate.representative.title, target.representative.title
                )
                if score < threshold:
                    continue
                target.items.extend(candidate.items)
                target.representative = max(
                    target.items, key=lambda item: len(item.full_text_snippet)
                )
                merged.pop(i)
                changed = True
                break
            if changed:
                break
    return merged


def cluster_stories(
    items: list[RawItem],
    *,
    ngram_range: tuple[int, int] = (2, 4),
    eps: float = 0.35,
    min_samples: int = 2,
    merge_title_threshold: float = 80.0,
) -> list[Cluster]:
    """TF-IDF char n-gram + DBSCAN(cosine) 聚类。"""
    if len(items) < 2:
        return [Cluster(cluster_id=0, items=items, representative=items[0])] if items else []

    texts = [f"{it.title} {it.summary}" for it in items]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=10000,
    )
    tfidf = vectorizer.fit_transform(texts)
    dist = cosine_distances(tfidf)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dist)

    clusters_map: dict[int, list[int]] = {}
    noise_id = max(labels) + 1 if len(labels) > 0 else 0
    for idx, label in enumerate(labels):
        if label == -1:
            # 噪声点各成一簇
            clusters_map[noise_id] = [idx]
            noise_id += 1
        else:
            clusters_map.setdefault(label, []).append(idx)

    clusters: list[Cluster] = []
    for cid, indices in sorted(clusters_map.items()):
        cluster_items = [items[i] for i in indices]
        # 选全文最长的作为代表
        representative = max(cluster_items, key=lambda x: len(x.full_text_snippet))
        clusters.append(Cluster(cluster_id=cid, items=cluster_items, representative=representative))

    if merge_title_threshold > 0:
        before = len(clusters)
        clusters = _merge_similar_singletons(clusters, threshold=merge_title_threshold)
        if len(clusters) != before:
            logger.info("Merged similar singleton clusters: %d → %d", before, len(clusters))

    logger.info("Clustered %d items → %d clusters", len(items), len(clusters))
    return clusters
