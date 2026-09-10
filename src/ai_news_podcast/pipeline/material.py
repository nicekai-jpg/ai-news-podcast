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


def build_radar_text(radar: dict[str, Any] | None) -> str:
    """把项目雷达结果格式化为固定栏目素材文本(空雷达返回空串)。

    每日只主推一个项目;其余候选仅作对比背景。数字与摘录全部由代码注入,LLM 不得增删。
    """
    if not radar or not radar.get("projects"):
        return ""
    meta = radar.get("meta", {})
    pick_repo = meta.get("pick_repo")
    lines = ["以下是「项目雷达」栏目的结构化素材(与新闻无关,单独成栏)。"]
    for p in radar.get("projects", []):
        if p.get("repo") != pick_repo:
            continue
        delta = p.get("delta_stars")
        delta_str = f"较昨日 +{delta}" if isinstance(delta, int) else "首日无对比数据"
        lines.append(
            f"- [主推] {p.get('repo')}(⭐{p.get('stars')},{delta_str},"
            f"语言:{p.get('language') or '未知'},许可证:{p.get('license') or '无'}):"
            f"{str(p.get('description') or '').strip()}"
        )
        excerpt = str(p.get("readme_excerpt") or "").strip()
        if excerpt:
            lines.append(f"  README 上手摘录(原文,安装命令必须逐字引用):\n  {excerpt}")
        lines.append(f"  链接:{p.get('url')}")
    others = [
        f"{p.get('repo')} ⭐{p.get('stars')}"
        for p in radar.get("projects", [])
        if p.get("repo") != pick_repo
    ]
    if others:
        lines.append("  其余候选(仅供对比参考,播客中不要展开):" + ";".join(others))
    lines.append(
        "使用规则:仓库名、数字、安装命令必须原样引用,禁止编造或修改;"
        "只讲主推项目,不要展开其余候选;禁止把项目与新闻混在同一栏目。"
    )
    return "\n".join(lines)
