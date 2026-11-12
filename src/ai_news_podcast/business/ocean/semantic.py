"""Stage 2 语义标注（LLM，可选）：只输出受控枚举 + 证据 + 置信度。

默认关闭；开启后与规则标签合并（method=hybrid）。LLM 只能从 taxonomy 枚举里选，
证据必须能在标题/摘要里找到，否则丢弃该条建议。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from ai_news_podcast.business.contracts import Item, Labels
from ai_news_podcast.business.ocean.taxonomy import Taxonomy

_CENTRALITY_TO_RELEVANCE = {
    "central": "core",
    "infra": "core",
    "incidental": "adjacent",
    "none": "none",
    "unknown": "unknown",
}
_JSON_BLOCK = re.compile(r"\[.*\]", re.S)


@dataclass(frozen=True)
class SemanticSuggestion:
    item_id: str
    ai_centrality: str
    primary_domain: str
    industry: str
    evidence: dict[str, str]
    confidence: float


def _enum_line(taxonomy: Taxonomy, dimension: str) -> str:
    values = taxonomy.enum_values(dimension)
    return " / ".join(values)


def build_prompt(items: list[Item], taxonomy: Taxonomy) -> str:
    rows = [
        {
            "item_id": item.item_id,
            "title": item.title[:120],
            "summary": (item.summary or "")[:300],
        }
        for item in items
    ]
    return (
        "你是 AI 新闻标注器。只输出 JSON 数组，不要解释。\n"
        f"ai_centrality 只能选：{_enum_line(taxonomy, 'ai_centrality')}\n"
        f"primary_domain 只能选：{_enum_line(taxonomy, 'primary_domain')}\n"
        f"industry 只能选：{' / '.join(taxonomy.industry_ids)}\n"
        "每条：item_id / ai_centrality / primary_domain / industry / "
        "evidence（字段→原文片段，必须出现在标题或摘要）/ confidence（0-1）。\n"
        "事件：\n" + json.dumps(rows, ensure_ascii=False)
    )


def parse_response(text: str) -> list[dict[str, Any]]:
    match = _JSON_BLOCK.search(str(text or ""))
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


class SemanticLabeler:
    """把 LLM 建议校验后合并进规则标签。"""

    def __init__(
        self,
        llm: Callable[[str], str | None],
        taxonomy: Taxonomy,
        *,
        min_confidence: float = 0.6,
        batch_size: int = 20,
    ) -> None:
        self.llm = llm
        self.taxonomy = taxonomy
        self.min_confidence = min_confidence
        self.batch_size = batch_size

    def suggest(self, items: list[Item]) -> dict[str, SemanticSuggestion]:
        results: dict[str, SemanticSuggestion] = {}
        by_id = {item.item_id: item for item in items}
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            raw = self.llm(build_prompt(batch, self.taxonomy))
            for row in parse_response(raw or ""):
                suggestion = self._validate(row, by_id)
                if suggestion is not None:
                    results[suggestion.item_id] = suggestion
        return results

    def _enum_fields(self, row: dict[str, Any]) -> tuple[str, str, str, float] | None:
        centrality = str(row.get("ai_centrality") or "")
        domain = str(row.get("primary_domain") or "")
        industry = str(row.get("industry") or "unknown")
        valid = (
            centrality in self.taxonomy.enum_values("ai_centrality")
            and domain in self.taxonomy.enum_values("primary_domain")
            and industry in self.taxonomy.industry_ids
        )
        try:
            confidence = float(row.get("confidence") or 0.0)
        except (TypeError, ValueError):
            return None
        if not valid or not 0.0 <= confidence <= 1.0:
            return None
        return centrality, domain, industry, confidence

    def _validate(self, row: dict[str, Any], by_id: dict[str, Item]) -> SemanticSuggestion | None:
        item = by_id.get(str(row.get("item_id") or ""))
        fields = self._enum_fields(row) if item is not None else None
        if item is None or fields is None:
            return None
        centrality, domain, industry, confidence = fields
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        haystack = f"{item.title}。{item.summary or ''}"
        clean_evidence = {
            str(key): str(value)
            for key, value in evidence.items()
            if value and str(value) in haystack
        }
        return SemanticSuggestion(
            item_id=item.item_id,
            ai_centrality=centrality,
            primary_domain=domain,
            industry=industry,
            evidence=clean_evidence,
            confidence=confidence,
        )

    def apply(self, labels: Labels, suggestion: SemanticSuggestion) -> Labels:
        if suggestion.confidence < self.min_confidence:
            return labels
        relevance = _CENTRALITY_TO_RELEVANCE.get(suggestion.ai_centrality, "unknown")
        label_evidence = dict(labels.label_evidence)
        label_confidence = dict(labels.label_confidence)
        label_method = dict(labels.label_method)
        for field in ("ai_relevance", "primary_domain", "industry"):
            label_method[field] = "hybrid"
            label_confidence[field] = max(label_confidence.get(field, 0.0), suggestion.confidence)
            if field in suggestion.evidence:
                bucket = list(label_evidence.get(field, []))
                if suggestion.evidence[field] not in bucket:
                    bucket.append(suggestion.evidence[field])
                label_evidence[field] = bucket
        return replace(
            labels,
            ai_relevance=relevance,
            primary_domain=suggestion.primary_domain,
            industry=suggestion.industry,
            label_evidence=label_evidence,
            label_confidence=label_confidence,
            label_method=label_method,
        )


__all__ = [
    "SemanticLabeler",
    "SemanticSuggestion",
    "build_prompt",
    "parse_response",
]
