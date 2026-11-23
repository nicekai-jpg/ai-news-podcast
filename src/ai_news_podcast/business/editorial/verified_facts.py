"""把 Stage 2.5 的 A 档事实卡转成「可核验事实」文本，供 editorial 交叉核对。

设计原则：
1. 只取 A 档——B 档每天 3000+ 条，且精度不足，注入只会引入噪声；
2. 按事件匹配——事实必须能对到当前事件的条目，不能把别的事件的事实塞进来；
3. 只做只读提示——prompt 里明确写「不是新的事实来源」，facts 仍须带 evidence_ids。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

RELATION_ZH: dict[str, str] = {
    "release": "发布",
    "open_source": "开源",
    "update": "更新",
    "pricing": "价格",
    "supply": "量产交付",
    "deploy": "部署接入",
    "support": "支持",
    "funding": "融资上市",
    "acquisition": "收购",
    "partnership": "合作",
    "benchmark": "评测对比",
    "personnel": "人事",
    "layoff": "裁员",
    "regulation": "监管",
    "legal": "诉讼",
    "incident": "事故风险",
    "research": "研究",
    "event": "活动",
    "info": "信息披露",
    "award": "获奖认证",
    "other": "其他",
}

MAX_FACTS_PER_EVENT = 6


def load_a_cards(
    facts_dir: str | Path,
    date: str,
    *,
    grades: Sequence[str] = ("A",),
) -> list[dict[str, Any]]:
    """读取指定日期、指定档位的事实卡；文件缺失时返回空列表。

    默认只取 A（严格档）。把 ``grades`` 写成 ``("A", "A2")`` 可以连
    「主语继承」的中间档一起取——A2 抽样精度约 50%，只适合当素材补充。
    """
    path = Path(facts_dir) / f"facts-{date}.jsonl"
    if not path.is_file():
        return []
    cards: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("grade") in grades:
            cards.append(row)
    return cards


def _object_text(card: dict[str, Any]) -> str:
    slots = card.get("slots") or {}
    for name in ("object", "person", "company", "role"):
        slot = slots.get(name)
        if isinstance(slot, dict) and slot.get("text"):
            return str(slot["text"])
    return ""


def render_fact(card: dict[str, Any]) -> str:
    """一条事实渲染成「主体 —关系→ 宾语」；人事额外带上角色与公司。"""
    subject = str((card.get("subject") or {}).get("name") or "")
    relation = RELATION_ZH.get(str(card.get("relation") or ""), str(card.get("relation") or ""))
    slots = card.get("slots") or {}
    if card.get("relation") == "personnel":
        person = str((slots.get("person") or {}).get("text") or subject)
        role = str((slots.get("role") or {}).get("text") or "")
        company = str((slots.get("company") or {}).get("text") or "")
        parts = [part for part in (person, role, company) if part]
        return f"{' · '.join(parts)}（人事）"
    obj = _object_text(card)
    return f"{subject} —{relation}→ {obj}" if obj else f"{subject} —{relation}→（未知宾语）"


def facts_by_item(cards: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """按 item_id 归档；同一 item 内去重并限制条数。"""
    out: dict[str, list[str]] = {}
    for card in cards:
        item_id = str((card.get("evidence") or {}).get("item_id") or "")
        if not item_id:
            continue
        rendered = render_fact(card)
        bucket = out.setdefault(item_id, [])
        if rendered in bucket:
            continue
        if len(bucket) < MAX_FACTS_PER_EVENT:
            bucket.append(rendered)
    return out


def verified_facts_for_event(
    item_ids: Sequence[str],
    by_item: dict[str, list[str]],
    *,
    limit: int = MAX_FACTS_PER_EVENT,
) -> list[str]:
    """给一个事件挑出它自己条目上的 A 档事实。"""
    out: list[str] = []
    for item_id in item_ids:
        for rendered in by_item.get(str(item_id), []):
            if rendered not in out:
                out.append(rendered)
            if len(out) >= limit:
                return out
    return out


__all__ = [
    "MAX_FACTS_PER_EVENT",
    "RELATION_ZH",
    "facts_by_item",
    "load_a_cards",
    "render_fact",
    "verified_facts_for_event",
]
