"""Stage 2 — 聚类模块。

TF-IDF char n-gram + DBSCAN(cosine) 聚类。
"""

from __future__ import annotations

import logging

from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from ai_news_podcast.pipeline.fetcher import RawItem
from ai_news_podcast.pipeline.processor_types import Cluster

logger = logging.getLogger(__name__)


def cluster_stories(
    items: list[RawItem],
    *,
    ngram_range: tuple[int, int] = (2, 4),
    eps: float = 0.35,
    min_samples: int = 2,
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

    logger.info("Clustered %d items → %d clusters", len(items), len(clusters))
    return clusters
