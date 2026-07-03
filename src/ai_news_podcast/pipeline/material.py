"""新闻素材准备模块

提供统一的素材文本构建功能，支持多种选择策略（多样性重排 / 纯分数排序）。
"""

from __future__ import annotations

import logging
from typing import Any

from ai_news_podcast.pipeline.strategies import StrategyRegistry

logger = logging.getLogger(__name__)


def build_material_text(
    brief: dict[str, Any],
    *,
    max_stories: int = 8,
    strategy: str = "score_diversity",
) -> str:
    """把 episode_brief 中最重要的新闻素材整理为结构化文本。

    Parameters
    ----------
    brief: episode_brief dict，包含 stories 列表。
    max_stories: 最多选取的新闻条数。
    strategy: 选择策略。
        - "score_diversity": 按分数降序 + 实体多样性惩罚（MMR-like），
          防止同一家公司霸榜。适合播客脚本生成。
        - "pure_score": 纯按分数排序取 top N。适合日报生成。

    Returns
    -------
    str: 结构化素材文本，可直接注入 LLM prompt。
    """
    stories = brief.get("stories", [])

    strategy_obj = StrategyRegistry.get(strategy)
    selected = strategy_obj.select(stories, max_stories)

    sections: list[str] = []
    for i, story in enumerate(selected, 1):
        role = story.get("role", "quick")
        role_label = {"main": "重要", "supporting": "次要", "quick": "简讯"}.get(role, "简讯")
        title = story.get("representative_title", "无标题")
        context = story.get("context", {})
        summaries = context.get("factual_summary", [])
        background = context.get("historical_background", "")
        sources = context.get("sources_ranked", [])

        part = f"【素材{i}】（{role_label}）\n标题：{title}\n"

        if summaries:
            part += "摘要：\n"
            for sm in summaries:
                part += f"  - {sm}\n"

        if background:
            part += f"背景：{background}\n"

        if sources:
            src_names = "、".join(src["name"] for src in sources[:3])
            part += f"综合来源：{src_names}\n"

        sections.append(part)

    return "\n".join(sections)
