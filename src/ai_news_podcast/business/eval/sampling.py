"""评测集抽样：按 AI 相关性分组做分层抽样，避免样本被单一类型占满。"""

from __future__ import annotations

import hashlib
from typing import Any

from ai_news_podcast.business.recommend.base import matched_keywords
from ai_news_podcast.business.recommend.config import DEFAULT_AI_TEXT_KEYWORDS

ADJACENT_KEYWORDS: tuple[str, ...] = (
    "芯片",
    "半导体",
    "算力",
    "gpu",
    "英伟达",
    "nvidia",
    "机器人",
    "具身",
    "自动驾驶",
    "智能驾驶",
    "数据中心",
    "光刻",
)

NONE_KEYWORDS: tuple[str, ...] = (
    "汽车",
    "车型",
    "车企",
    "造车",
    "轿车",
    "suv",
    "销量",
    "停车",
    "公交",
    "车祸",
    "事故",
    "股价",
    "a股",
    "指数",
    "涨停",
    "跌停",
    "基金",
    "融资买入",
    "短剧",
    "好物",
    "优惠",
    "折扣",
    "耳机",
    "笔记本",
    "双十一",
)

GROUPS = ("core", "adjacent", "none", "unknown")

_INFRA_TERMS = {
    "芯片",
    "半导体",
    "gpu",
    "算力",
    "英伟达",
    "nvidia",
    "机器人",
    "具身",
    "自动驾驶",
    "智能驾驶",
}
_CORE_HINTS = tuple(term for term in DEFAULT_AI_TEXT_KEYWORDS if term not in _INFRA_TERMS)


def classify_suggestion(title: str, summary: str = "") -> str:
    """给标注模板一个“建议分组”；人工只需确认/修改，不是最终金标准。"""
    text = f"{title} {summary}"
    if matched_keywords(text, _CORE_HINTS):
        return "core"
    if matched_keywords(text, ADJACENT_KEYWORDS):
        return "adjacent"
    if matched_keywords(text, NONE_KEYWORDS):
        return "none"
    return "unknown"


def _stable_order(row: dict[str, Any]) -> str:
    key = f"{row.get('date', '')}|{row.get('title', '')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def stratified_sample(
    rows: list[dict[str, Any]],
    quotas: dict[str, int],
    *,
    total_limit: int | None = None,
) -> list[dict[str, Any]]:
    """按分组配额抽样；某组不足时用其他组补足到 total_limit。

    每个分组内部按标题哈希排序，保证可复现；输入行需含 `group` 与 `title`。
    """
    buckets: dict[str, list[dict[str, Any]]] = {group: [] for group in GROUPS}
    for row in rows:
        group = str(row.get("group") or "unknown")
        buckets.setdefault(group, []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=_stable_order)

    picked: list[dict[str, Any]] = []
    used: set[str] = set()
    for group in GROUPS:
        for row in buckets.get(group, [])[: quotas.get(group, 0)]:
            picked.append(row)
            used.add(_stable_order(row))

    limit = total_limit if total_limit is not None else sum(quotas.values())
    if len(picked) < limit:
        remaining: list[dict[str, Any]] = []
        for group in GROUPS:
            remaining.extend(
                row for row in buckets.get(group, []) if _stable_order(row) not in used
            )
        remaining.sort(key=_stable_order)
        picked.extend(remaining[: limit - len(picked)])

    return picked


def split_by_day(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按 date 字段分组，保持原始顺序；用于把大队列拆成按天标注文件。"""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        date = str(row.get("date") or "unknown")
        groups.setdefault(date, []).append(row)
    return dict(sorted(groups.items()))
