"""聚类合并单测：DBSCAN 全噪声时，相似标题的孤立条目要能合并成多源簇。"""

from __future__ import annotations

from ai_news_podcast.business.pipeline.fetcher_service import RawItem
from ai_news_podcast.business.pipeline.processor_cluster_service import cluster_stories


def _item(title: str, source: str) -> RawItem:
    return RawItem(
        id=title,
        title=title,
        link=f"https://example.com/{title}",
        normalized_link=f"https://example.com/{title}",
        source_name=source,
        source_category="news",
        published_at="2026-11-05T00:00:00+00:00",
        summary="s",
        full_text_snippet="x" * 100,
        category="other",
        language="zh",
    )


def test_similar_singletons_are_merged() -> None:
    items = [
        _item("Google 发布 Gemini 4 Carbon 前沿模型", "DeepMind Blog"),
        _item("Google 发布 Gemini 4 Carbon 前沿模型！", "TechCrunch"),
        _item("完全不同的新闻：某公司裁员", "Other"),
    ]
    clusters = cluster_stories(items, merge_title_threshold=80)
    sizes = sorted(len(cluster.items) for cluster in clusters)
    assert sizes == [1, 2]
    merged = next(cluster for cluster in clusters if len(cluster.items) == 2)
    assert {item.source_name for item in merged.items} == {"DeepMind Blog", "TechCrunch"}
