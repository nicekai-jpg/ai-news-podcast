"""Stage 2 — 信息加工模块

职责：RawItem 列表 → 三层去重 → 聚类 → context_block → 五维打分 →
      角色分配 → Thesis 提炼 → 输出 episode_brief dict。
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_news_podcast.pipeline.fetcher import RawItem
from ai_news_podcast.pipeline.processor_cluster import cluster_stories
from ai_news_podcast.pipeline.processor_context import _build_context
from ai_news_podcast.pipeline.processor_dedup import (  # noqa: F401
    _dedup_url,
    _extract_keywords,
    dedup_pipeline,
)
from ai_news_podcast.pipeline.processor_score import _assign_role, _score_cluster
from ai_news_podcast.pipeline.processor_thesis import _extract_thesis, set_thesis_templates

# Re-export types and functions for backward compatibility with existing tests
from ai_news_podcast.pipeline.processor_types import (  # noqa: F401
    Cluster,
    ContextBlock,
    ScoredStory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 主入口 — 加工管线
# ---------------------------------------------------------------------------


def process(
    raw_items: list[RawItem],
    *,
    processing_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    完整加工管线。返回 episode_brief dict。

    Parameters
    ----------
    raw_items : 来自 fetcher 的原始条目
    processing_cfg : config.yaml 中的 processing 段
    """
    cfg = processing_cfg or {}
    dedup_cfg = cfg.get("dedup", {})
    cluster_cfg = cfg.get("clustering", {})
    scoring_cfg = cfg.get("scoring", {})
    authority_cfg = cfg.get("source_authority", {})

    # Load thesis templates from config, keeping defaults as fallback
    set_thesis_templates(cfg.get("thesis_templates"))

    # 1) 三层去重
    deduped = dedup_pipeline(
        raw_items,
        rapidfuzz_threshold=int(dedup_cfg.get("rapidfuzz_threshold", 92)),
        jieba_overlap=float(dedup_cfg.get("jieba_keyword_overlap", 0.35)),
        title_sim=int(dedup_cfg.get("title_sim_threshold", 85)),
        window_hours=int(dedup_cfg.get("dedup_window_hours", 48)),
    )

    if not deduped:
        logger.warning("No items after dedup")
        return {
            "stories": [],
            "thesis": "",
            "metadata": {"total_raw": len(raw_items), "total_deduped": 0},
        }

    # 2) 聚类
    ngram = tuple(cluster_cfg.get("ngram_range", [2, 4]))
    clusters = cluster_stories(
        deduped,
        ngram_range=ngram,  # type: ignore[arg-type]
        eps=float(cluster_cfg.get("eps", 0.35)),
        min_samples=int(cluster_cfg.get("min_samples", 2)),
    )

    # 3) 为每个 cluster 构建 context + 打分 + 角色分配
    role_thresholds = scoring_cfg.get(
        "role_thresholds",
        {
            "main": [12, 15],
            "supporting": [8, 11],
            "quick": [5, 7],
            "skip_below": 5,
        },
    )

    scored_stories: list[ScoredStory] = []
    for cluster in clusters:
        context = _build_context(cluster, source_authority=authority_cfg)
        scores = _score_cluster(cluster)
        total = sum(scores.values())
        role, emoji = _assign_role(total, role_thresholds)

        scored_stories.append(
            ScoredStory(
                cluster_id=cluster.cluster_id,
                representative_title=cluster.representative.title,
                items=[item.to_dict() for item in cluster.items],
                context=context,
                scores=scores,
                total_score=total,
                role=role,
                role_emoji=emoji,
            )
        )

    # 按总分降序排列
    scored_stories.sort(key=lambda x: x.total_score, reverse=True)

    # 4) Thesis 提炼
    thesis = _extract_thesis(scored_stories)

    # 5) 组装 episode_brief
    brief: dict[str, Any] = {
        "thesis": thesis,
        "stories": [],
        "metadata": {
            "total_raw": len(raw_items),
            "total_deduped": len(deduped),
            "total_clusters": len(clusters),
            "generated_at": datetime.now(tz=UTC).isoformat(),
        },
    }

    for ss in scored_stories:
        brief["stories"].append(
            {
                "cluster_id": ss.cluster_id,
                "representative_title": ss.representative_title,
                "role": ss.role,
                "role_emoji": ss.role_emoji,
                "total_score": ss.total_score,
                "scores": ss.scores,
                "context": {
                    "factual_summary": ss.context.factual_summary,
                    "historical_background": ss.context.historical_background,
                    "sources_ranked": ss.context.sources_ranked,
                },
                "items": ss.items,
            }
        )

    return brief


class _NumpySafeEncoder(json.JSONEncoder):
    """Handle numpy int/float types that stdlib json cannot serialize."""

    def default(self, o: Any) -> Any:
        try:
            import numpy as np

            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        return super().default(o)


def save_brief(brief: dict[str, Any], path: Path) -> None:
    """保存 episode_brief.json。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(brief, f, ensure_ascii=False, indent=2, cls=_NumpySafeEncoder)
        f.write("\n")
    logger.info("Saved episode_brief to %s", path)
