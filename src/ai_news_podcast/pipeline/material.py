"""新闻素材准备模块

提供统一的素材文本构建功能，支持多种选择策略（多样性重排 / 纯分数排序）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认实体列表（当 config.yaml 中未配置 entities.companies 时使用）
# ---------------------------------------------------------------------------

_DEFAULT_COMPANIES = [
    "谷歌",
    "google",
    "openai",
    "微软",
    "microsoft",
    "英伟达",
    "nvidia",
    "苹果",
    "apple",
    "meta",
    "anthropic",
    "claude",
    "字节",
    "腾讯",
    "百度",
    "阿里",
    "华为",
    "奥迪",
    "audi",
    "特斯拉",
    "tesla",
]


def _load_companies_from_config() -> list[str]:
    """从 config.yaml 读取 entities.companies，读取失败时返回默认值。"""
    try:
        config_path = (
            Path(__file__).resolve().parent.parent.parent.parent / "config" / "config.yaml"
        )
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        companies = cfg.get("entities", {}).get("companies", [])
        return companies if companies else _DEFAULT_COMPANIES
    except (OSError, yaml.YAMLError, KeyError):
        return _DEFAULT_COMPANIES


def _get_story_entities(story: dict[str, Any]) -> set[str]:
    """提取新闻故事中的公司/品牌实体词，用于多样性去重。"""
    title = str(story.get("representative_title", "")).lower()
    entities = set()
    COMPANIES = _load_companies_from_config()
    for c in COMPANIES:
        if c in title:
            # 标准化实体名
            norm = c
            if c in ("google", "谷歌"):
                norm = "谷歌"
            elif c in ("microsoft", "微软"):
                norm = "微软"
            elif c in ("nvidia", "英伟达"):
                norm = "英伟达"
            elif c in ("apple", "苹果"):
                norm = "苹果"
            elif c in ("audi", "奥迪"):
                norm = "奥迪"
            elif c in ("tesla", "特斯拉"):
                norm = "特斯拉"
            entities.add(norm)
    return entities


def _select_with_diversity(
    stories: list[dict[str, Any]],
    max_stories: int,
) -> list[dict[str, Any]]:
    """动态实体多样性重排（MMR-like）：防止同一家公司霸榜。

    按分数降序挑选，但如果候选新闻包含已选中实体的关键词，
    则对其施加分数惩罚（每冲突一次扣 3 分）。
    """
    active = [s for s in stories if isinstance(s, dict) and s.get("role") != "skip"]
    selected: list[dict[str, Any]] = []
    entity_counts: dict[str, int] = {}
    candidates = [dict(s) for s in active]

    while len(selected) < max_stories and candidates:
        for c in candidates:
            orig_score = c.get("total_score", 0)
            c_entities = _get_story_entities(c)
            penalty = sum(3 * entity_counts.get(ent, 0) for ent in c_entities)
            c["_temp_score"] = orig_score - penalty

        candidates.sort(key=lambda x: x.get("_temp_score", 0), reverse=True)
        best = candidates.pop(0)
        selected.append(best)

        for ent in _get_story_entities(best):
            entity_counts[ent] = entity_counts.get(ent, 0) + 1

    return selected


def _select_pure_score(
    stories: list[dict[str, Any]],
    max_stories: int,
) -> list[dict[str, Any]]:
    """纯按分数排序，取 top N。"""
    active = [s for s in stories if isinstance(s, dict) and s.get("role") != "skip"]
    active.sort(key=lambda s: s.get("total_score", 0), reverse=True)
    return active[:max_stories]


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

    if strategy == "score_diversity":
        selected = _select_with_diversity(stories, max_stories)
    elif strategy == "pure_score":
        selected = _select_pure_score(stories, max_stories)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

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
