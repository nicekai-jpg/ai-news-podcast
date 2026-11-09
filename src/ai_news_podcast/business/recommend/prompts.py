"""Stage 3 LLM rubric 提示词：六维 0–5 打分 + 一句话理由。

设计约束（对齐 docs/pipeline_redesign_v5.md §3.3/§7.4）：

- 六维：影响力 / 新颖度 / 势头 / 受众相关 / 可讲性 / 证据与风险；
- 每档（0–5）都有明确描述，保证跨天可复现；
- 只喂“事实/属性 + 原始统计量”，**不喂 lifecycle / evidence_tier**，
  避免 LLM 把召回/粗排用过的结论再计一次分；
- 输出必须是纯 JSON：`{event_id: {六维分数..., "reason": "..."}}`。
"""

from __future__ import annotations

import json
from typing import Any

from ai_news_podcast.business.contracts import Event

RUBRIC_DIMENSIONS = (
    "impact",
    "novelty",
    "momentum",
    "audience_fit",
    "narratability",
    "evidence_risk",
)

RUBRIC_TITLES: dict[str, str] = {
    "impact": "影响力",
    "novelty": "新颖度",
    "momentum": "势头",
    "audience_fit": "受众相关",
    "narratability": "可讲性",
    "evidence_risk": "证据与风险",
}

RUBRIC_DEFINITIONS: dict[str, str] = {
    "impact": "主体有多重要、影响范围多大（全球/中国/行业/单一产品）。",
    "novelty": "是否新发布、新版本、新数字、新主体；纯复述旧闻应给低分。",
    "momentum": "来源数增长快不快、是否持续事件的新进展、是否再次出现。",
    "audience_fit": "开发者/从业者/投资人/政策制定者等目标受众是否关心。",
    "narratability": "有没有数字、对比、冲突、反转、首次/纪录，能否讲成故事。",
    "evidence_risk": "来源质量、是否官方/多源、是否有传闻/敏感/标题党风险。",
}

# 每维 0–5 档的锚点描述，顺序即分值 0..5。
RUBRIC_LEVELS: dict[str, tuple[str, ...]] = {
    "impact": (
        "0 分：无主体或与 AI 行业无关。",
        "1 分：小型团队/地方事件，影响限于局部。",
        "2 分：细分领域玩家，行业内少量关注。",
        "3 分：知名公司/实验室，行业内有明确影响。",
        "4 分：头部厂商重要动作，跨行业关注。",
        "5 分：全球级影响，可能改变竞争格局或行业规则。",
    ),
    "novelty": (
        "0 分：旧闻复述，无任何新信息。",
        "1 分：仅措辞变化或例行更新。",
        "2 分：有新数字/新版本，但增量有限。",
        "3 分：明确的新发布/新版本/新主体。",
        "4 分：多项新信息叠加，明显领先于既有报道。",
        "5 分：行业首次/首创，信息增量显著。",
    ),
    "momentum": (
        "0 分：无新增来源，事件已停止发酵。",
        "1 分：仅单一来源，几乎没有讨论。",
        "2 分：少量来源跟进，热度平稳。",
        "3 分：多源持续报道，处于上升期。",
        "4 分：来源快速增长，明显热点。",
        "5 分：当日爆发式传播，多源集中报道。",
    ),
    "audience_fit": (
        "0 分：与目标受众完全无关。",
        "1 分：仅极少数人关心。",
        "2 分：小众但明确的受众。",
        "3 分：开发者/从业者普遍关心。",
        "4 分：多类受众（开发者/投资人/消费者）都关心。",
        "5 分：几乎所有 AI 从业者都需要知道。",
    ),
    "narratability": (
        "0 分：只有一句公告，无法展开。",
        "1 分：信息单薄，缺少可讲细节。",
        "2 分：有基本事实，但缺少数字/对比。",
        "3 分：有数字或对比，可讲成一段。",
        "4 分：有冲突/反转/首次等叙事抓手。",
        "5 分：强故事性，天然适合播客展开。",
    ),
    "evidence_risk": (
        "0 分：单一来源且明显传闻/标题党，风险极高。",
        "1 分：证据弱、来源存疑，风险较高。",
        "2 分：单一来源，无官方确认。",
        "3 分：多源可交叉验证，基本可信。",
        "4 分：多源 + 官方/权威来源，风险低。",
        "5 分：官方一手信息或多源高度一致，几乎无风险。",
    ),
}

SYSTEM_PROMPT = (
    "你是资深 AI 新闻主编，负责为每日播客挑选选题。"
    "请严格按给定 rubric 给每个事件打 0–5 分，只输出 JSON，不要输出任何解释性文字或 Markdown 代码块。"
)

_JSON_SCHEMA_HINT = {
    "<event_id>": {
        "impact": "0-5 整数",
        "novelty": "0-5 整数",
        "momentum": "0-5 整数",
        "audience_fit": "0-5 整数",
        "narratability": "0-5 整数",
        "evidence_risk": "0-5 整数",
        "reason": "一句话中文理由",
    }
}


def build_rubric_prompt(events: list[Event]) -> str:
    """构造 LLM rubric 打分 prompt。"""
    sections = [
        "请对以下 AI 新闻事件按六维 rubric 打分（0–5，整数）。",
        "",
        _rubric_text(),
        "",
        "打分规则：",
        "1. 只依据下面给出的事实/属性与原始统计量，不要臆测未提供的信息；",
        "2. 六个维度都要给分，分数必须是 0–5 的整数；",
        "3. 每个事件给一句不超过 40 字的中文理由；",
        "4. 只输出 JSON 对象，key 必须严格使用事件的 event_id；",
        "5. 输出格式示例（值是示例，请替换为真实评分）：",
        json.dumps(_JSON_SCHEMA_HINT, ensure_ascii=False, indent=2),
        "",
        "待评分事件：",
        "",
    ]
    for index, event in enumerate(events or [], 1):
        sections.append(_format_event(event, index))
        sections.append("")
    return "\n".join(sections).strip()


def _rubric_text() -> str:
    blocks = ["六维 rubric："]
    for dim in RUBRIC_DIMENSIONS:
        blocks.append(f"\n【{RUBRIC_TITLES[dim]}】{RUBRIC_DEFINITIONS[dim]}")
        blocks.extend(f"- {level}" for level in RUBRIC_LEVELS[dim])
    return "\n".join(blocks)


def _format_event(event: Event, index: int) -> str:
    labels = event.labels
    subjects = "、".join(
        f"{_subject_name(subject)}（权重 {_number(subject.get('weight'))}，"
        f"置信度 {_number(subject.get('confidence'), 1.0)}）"
        for subject in labels.subjects
        if isinstance(subject, dict)
    )
    numbers = "；".join(
        f"{item.get('type') or '数字'}={item.get('value')}"
        for item in labels.novelty_numbers
        if isinstance(item, dict)
    )
    actions = "、".join(_unique([labels.primary_action, *labels.actions]))
    lines = [
        f"事件 {index}",
        f"event_id: {event.event_id}",
        f"标题: {event.representative_title}",
        f"摘要: {_truncate(event.summary, 300)}",
        f"主体: {subjects or '无'}",
        f"动作: {actions or '无'}（主动作：{labels.primary_action or '无'}）",
        f"对象类型: {'、'.join(labels.object_type) or '无'}",
        f"主题: {'、'.join(labels.topics) or '无'}",
        f"技术栈: {'、'.join(labels.tech_stack) or '无'}",
        f"地域: {'、'.join(labels.regions) or '无'}",
        f"新数字/新版本: {numbers or '无'}",
        f"叙事角度: {'、'.join(labels.narrative_angle) or '无'}",
        f"受众画像: {'、'.join(labels.audience_persona) or '未标注'}",
        f"敏感标签: {'、'.join(labels.sensitivity) or '无'}",
        f"是否可引用官方原话: {'是' if labels.quote_availability else '否'}",
        f"来源数: {event.source_count}；独立来源数: {event.independent_sources}；"
        f"6h 来源增量: {event.momentum_growth:.1f}",
        f"是否持续事件新进展: {'是' if event.is_update else '否'}；事件序号: {event.sequence}",
    ]
    return "\n".join(lines)


def _subject_name(subject: dict[str, Any]) -> str:
    return str(subject.get("name") or subject.get("id") or "未知主体")


def _number(value: Any, default: float = 0.0) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return f"{default:.2f}"


def _truncate(text: str, limit: int) -> str:
    normalized = (text or "").strip()
    return normalized if len(normalized) <= limit else f"{normalized[:limit]}…"


def _unique(values: list[str]) -> list[str]:
    return [value for value in dict.fromkeys(str(item).strip() for item in values) if value]


__all__ = [
    "RUBRIC_DEFINITIONS",
    "RUBRIC_DIMENSIONS",
    "RUBRIC_LEVELS",
    "RUBRIC_TITLES",
    "SYSTEM_PROMPT",
    "build_rubric_prompt",
]
