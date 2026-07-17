"""Stage 2 — Context block 构建模块。

为每个 cluster 构建事实摘要、历史背景和来源排序。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ai_news_podcast.pipeline.processor_types import AUTHORITY_ORDER, Cluster, ContextBlock

logger = logging.getLogger(__name__)


def _build_context(
    cluster: Cluster,
    *,
    story_memory: dict[str, Any] | None = None,
    source_authority: dict[str, int] | None = None,
) -> ContextBlock:
    """构建 context_block。"""
    auth_map = source_authority or AUTHORITY_ORDER
    rep = cluster.representative

    # 3 句事实摘要 — 取前 3 个 item 的 summary 首句
    summaries: list[str] = []
    for item in cluster.items[:3]:
        text = item.summary or item.full_text_snippet
        if text:
            # Use a regex that doesn't split on decimal points (e.g. 5.6)
            sentences = [
                s.strip() for s in re.split(r"(?<!\d)\.(?!\d)|[。！!？?\n]", text) if s.strip()
            ]
            if sentences:
                summaries.append(sentences[0])
    if not summaries:
        summaries = [rep.title]

    # 历史背景 — 简单占位（后续可从 story_memory 拉取）
    background = ""
    if story_memory:
        # 搜索 story_memory 中相关条目
        for mem_key, mem_val in story_memory.items():
            if any(kw in rep.title for kw in mem_key.split()):
                background = str(mem_val)
                break

    # 来源排序
    sources_ranked: list[dict[str, Any]] = []
    for item in cluster.items:
        auth = auth_map.get(item.source_category, 5)
        sources_ranked.append(
            {
                "name": item.source_name,
                "authority": auth,
                "link": item.link,
            }
        )
    sources_ranked.sort(key=lambda x: x["authority"])

    return ContextBlock(
        factual_summary=summaries,
        historical_background=background,
        sources_ranked=sources_ranked,
    )
