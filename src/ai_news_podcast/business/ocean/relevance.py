"""Stage 2 AI 相关性分级：core / adjacent / none / unknown + 证据词。

分级标准（写进 docs/recommend_evaluation.md）：
- core：标题/摘要/正文出现 AI 模型、Agent、算法、AI 产品、AI 研究；
- adjacent：AI 基础设施或强相关（芯片、算力、机器人、自动驾驶、数据中心）；
- none：没有任何 AI 信号（汽车、消费、股市、社会新闻）；
- unknown：信息不足，规则不敢定（标题/摘要过短）。

证据词只记录命中的关键词，方便人工审计；不是最终金标准。
"""

from __future__ import annotations

import re

from ai_news_podcast.business.recommend.base import matched_keywords
from ai_news_podcast.business.recommend.config import DEFAULT_AI_TEXT_KEYWORDS

INFRA_TERMS: frozenset[str] = frozenset(
    {
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
        "数据中心",
        "光刻",
    }
)

CORE_HINTS: tuple[str, ...] = tuple(
    term for term in DEFAULT_AI_TEXT_KEYWORDS if term not in INFRA_TERMS
)
ADJACENT_HINTS: tuple[str, ...] = tuple(
    term for term in DEFAULT_AI_TEXT_KEYWORDS if term in INFRA_TERMS
)

_WHITESPACE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text or "")).strip()


def _grade(matches: list[str]) -> tuple[str, list[str]] | None:
    if not matches:
        return None
    core = [term for term in matches if term in CORE_HINTS]
    if core:
        return "core", core[:5]
    return "adjacent", matches[:5]


def classify_text(
    title: str, summary: str = "", full_text: str = "", *, category: str = "news"
) -> tuple[str, list[str], float]:
    """返回 (ai_relevance, evidence, confidence)。

    只有 official/research 来源允许用正文兜底；新闻/社区来源只看标题+摘要，
    避免“汽车新闻正文提了一句自动驾驶”就被判成 AI 相关。
    """
    head = _clean(f"{title} {summary}")
    body = _clean(full_text)[:2000]
    if len(head) < 8 and len(body) < 20:
        return "unknown", [], 0.2

    head_grade = _grade(matched_keywords(head, DEFAULT_AI_TEXT_KEYWORDS))
    if head_grade is not None:
        grade, evidence = head_grade
        return grade, evidence, 0.8 if grade == "core" else 0.7

    if str(category or "").strip().lower() in ("official", "research"):
        body_grade = _grade(matched_keywords(body, DEFAULT_AI_TEXT_KEYWORDS))
        if body_grade is not None:
            grade, evidence = body_grade
            return grade, evidence, 0.55 if grade == "core" else 0.5

    if len(head) < 4:
        return "unknown", [], 0.2
    return "none", [], 0.6


def stronger_grade(current: str, candidate: str) -> str:
    """合并事件内多条 item 的等级：core > adjacent > none > unknown。"""
    order = {"core": 3, "adjacent": 2, "none": 1, "unknown": 0}
    return candidate if order.get(candidate, 0) > order.get(current, 0) else current
