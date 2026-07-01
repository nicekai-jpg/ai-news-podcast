"""Stage 2 共享数据结构与常量。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# 输出数据结构
# ---------------------------------------------------------------------------


@dataclass
class ContextBlock:
    factual_summary: list[str]  # 3 句事实摘要
    historical_background: str
    sources_ranked: list[dict[str, Any]]  # [{name, authority, link}]


@dataclass
class Cluster:
    cluster_id: int
    items: list[Any]  # RawItem instances
    representative: Any  # RawItem — 标题最长 / 全文最丰富的代表


@dataclass
class ScoredStory:
    """加工后的故事，用于输出 episode_brief。"""

    cluster_id: int
    representative_title: str
    items: list[dict[str, Any]]  # 原始 RawItem dicts
    context: ContextBlock
    scores: dict[str, int]  # 五维得分
    total_score: int
    role: str  # main / supporting / quick / skip
    role_emoji: str


# ---------------------------------------------------------------------------
# 来源权威性排序（数值越小越权威）
# ---------------------------------------------------------------------------

AUTHORITY_ORDER: dict[str, int] = {
    "official": 1,
    "research": 2,
    "open_source": 2,
    "news": 3,
    "product": 3,
    "analysis": 3,
    "community": 3,
    "tools": 4,
    "events": 4,
    "other": 5,
}
