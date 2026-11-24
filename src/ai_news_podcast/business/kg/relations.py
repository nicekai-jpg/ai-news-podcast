"""关系归一：把 config/nlp/actions.yaml 的 48 个细动作收口成 21 个规范关系。

细动作保留在 `relation_raw` 里做审计；下游按 `relation` 做聚合和查询。
"""

from __future__ import annotations

CANONICAL_LABELS: dict[str, str] = {
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

# 细动作 id → 规范关系 id
_ACTION_TO_CANONICAL: dict[str, str] = {
    "release": "release",
    "open_source": "open_source",
    "open_access": "open_source",
    "demo": "release",
    "product_update": "update",
    "migrate": "update",
    "pricing": "pricing",
    "supply": "supply",
    "build": "event",
    "deployment": "deploy",
    "support": "support",
    "provide": "support",
    "funding": "funding",
    "ipo": "funding",
    "acquisition": "acquisition",
    "partnership": "partnership",
    "sponsor": "partnership",
    "cooperate_std": "partnership",
    "benchmark": "benchmark",
    "outperform": "benchmark",
    "ranking": "benchmark",
    "record": "benchmark",
    "verify": "benchmark",
    "compare": "benchmark",
    "personnel": "personnel",
    "return": "personnel",
    "hiring": "personnel",
    "layoff": "layoff",
    "regulation": "regulation",
    "investigation": "regulation",
    "ban": "regulation",
    "export_control": "regulation",
    "suspend": "regulation",
    "resume": "regulation",
    "lawsuit": "legal",
    "settle": "legal",
    "accident": "incident",
    "recall": "incident",
    "warn": "incident",
    "research": "research",
    "training": "research",
    "event": "event",
    "found": "event",
    "award": "award",
    "disclosed": "info",
    "rumor": "info",
    "deny": "info",
    "predict": "info",
}


def canonical_relation(action_id: str) -> str:
    """细动作 id 映射成规范关系；未知动作归到 other。"""
    return _ACTION_TO_CANONICAL.get(str(action_id or "").strip().lower(), "other")


def canonical_label(relation: str) -> str:
    """规范关系 id 转中文标签。"""
    return CANONICAL_LABELS.get(str(relation or ""), "其他")


__all__ = ["CANONICAL_LABELS", "canonical_label", "canonical_relation"]
