"""Stage 2 — 五维打分与角色分配模块。

五维评分：impact_scope, novelty, explainability, listener_relevance, source_richness。
角色分配：main / supporting / quick / skip。
"""

from __future__ import annotations

import logging
from typing import Any

from ai_news_podcast.pipeline.processor_types import AUTHORITY_ORDER, Cluster

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 五维打分 (PLAN §2.4)
# ---------------------------------------------------------------------------


def _is_model_release_or_update(text: str) -> bool:
    """判断是否为大模型或核心AI产品的发布/开源/版本升级新闻。"""
    import re

    text = text.lower()
    model_kws = (
        "kimi",
        "qwen",
        "千问",
        "通义",
        "deepseek",
        "claude",
        "gpt",
        "gemini",
        "llama",
        "glm",
        "智谱",
        "minimax",
        "hunyuan",
        "混元",
        "doubao",
        "豆包",
        "gemma",
        "smollm",
        "mistral",
        "o1",
        "o3",
        "grok",
        "百川",
        "零一万物",
        "大模型",
        "大语言模型",
        "llm",
        "vlm",
        "基础模型",
        "foundation model",
    )
    release_kws = (
        "发布",
        "首发",
        "推出",
        "上线",
        "开源",
        "问世",
        "亮相",
        "新一代",
        "新模型",
        "launch",
        "release",
        "unveil",
        "announce",
        "open weight",
        "open-source",
        "available",
    )
    has_model = any(k in text for k in model_kws)
    has_release = any(k in text for k in release_kws) or bool(
        re.search(r"\b(?:v|k|r)?\d+(?:\.\d+)+\b", text)
    )
    return has_model and has_release


def _score_cluster(cluster: Cluster) -> dict[str, int]:
    """
    五维评分，每维 1-3 分，总分 5-15。
    - impact_scope: 多源报道 → 3，2源 → 2，单源 → 1
    - novelty: 含新产品/论文关键词 → 3
    - explainability: 全文丰富度
    - listener_relevance: 中文内容 + AI 强相关 → 3
    - source_richness: 来源权威性
    """
    items = cluster.items
    rep = cluster.representative
    text = f"{rep.title} {rep.summary} {rep.full_text_snippet}".lower()

    # 判定是否为大模型或核心产品发布/升级新闻，大幅增加打分权重
    is_model_rel = _is_model_release_or_update(text)

    # impact_scope
    unique_sources = len({it.source_name for it in items})
    impact = 3 if unique_sources >= 3 else (2 if unique_sources >= 2 else 1)
    if is_model_rel:
        impact = max(impact, 2)  # 大模型发布默认具备行业广泛影响，保底2分

    # novelty
    novel_kws = (
        "首次",
        "突破",
        "创新",
        "first",
        "novel",
        "breakthrough",
        "state-of-the-art",
        "新发布",
        "新模型",
        "launch",
        "release",
        "开源",
    )
    novelty = (
        3
        if is_model_rel or any(k in text for k in novel_kws)
        else (2 if any(k in text for k in ("更新", "update", "升级")) else 1)
    )

    # explainability
    fulltext_len = len(rep.full_text_snippet)
    explain = 3 if fulltext_len >= 1200 else (2 if fulltext_len >= 600 else 1)
    if is_model_rel:
        explain = max(explain, 2)  # 大模型发布规格硬核，保底2分

    # listener_relevance
    zh_count = sum(1 for it in items if it.language == "zh")
    relevance_kws = ("大模型", "llm", "chatgpt", "claude", "gpt", "agent", "智能体")
    has_relevance = any(k in text for k in relevance_kws)
    relevance = (
        3
        if (is_model_rel or (zh_count > 0 and has_relevance))
        else (2 if has_relevance or zh_count > 0 else 1)
    )

    # source_richness
    auth_scores = [AUTHORITY_ORDER.get(it.source_category, 5) for it in items]
    min_auth = min(auth_scores) if auth_scores else 5
    richness = 3 if min_auth <= 1 else (2 if min_auth <= 2 else 1)
    if is_model_rel:
        richness = max(richness, 3)  # 大模型重磅发布赋予高权威权重，保底3分

    return {
        "impact_scope": impact,
        "novelty": novelty,
        "explainability": explain,
        "listener_relevance": relevance,
        "source_richness": richness,
    }


# ---------------------------------------------------------------------------
# 角色分配 (PLAN §2.5)
# ---------------------------------------------------------------------------


def _assign_role(total: int, thresholds: dict[str, Any]) -> tuple[str, str]:
    """
    12-15 → 🔴main  |  8-11 → 🟡supporting  |  5-7 → 🟢quick  |  <5 → ⚪skip
    """
    main_range = thresholds.get("main", [12, 15])
    supporting_range = thresholds.get("supporting", [8, 11])
    quick_range = thresholds.get("quick", [5, 7])

    if total >= main_range[0]:
        return "main", "🔴"
    if total >= supporting_range[0]:
        return "supporting", "🟡"
    if total >= quick_range[0]:
        return "quick", "🟢"
    return "skip", "⚪"
