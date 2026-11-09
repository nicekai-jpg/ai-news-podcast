"""推荐阶段的共享基础工具：时间解析、文本匹配、标签权重。

本模块只消费 `business.contracts` 里的 Event/Labels，不依赖 Stage 1–2 的实现，
也不 import `business.pipeline`，方便单测与独立回测。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

from ai_news_podcast.business.contracts import Event, Labels

# 精排规则特征（七个分量）；其中 evidence / risk 对应 LLM rubric 的「证据与风险」一维。
DIMENSIONS = (
    "impact",
    "novelty",
    "momentum",
    "audience_fit",
    "narratability",
    "evidence",
    "risk",
)

# LLM rubric 六维（0–5），与 docs/pipeline_redesign_v5.md §7.4 对齐。
JUDGE_DIMENSIONS = (
    "impact",
    "novelty",
    "momentum",
    "audience_fit",
    "narratability",
    "evidence_risk",
)

# 只允许在召回/粗排使用的标签：它们由原始统计量推导，进入精排会造成循环/重复计数。
COARSE_ONLY_LABELS = ("lifecycle", "evidence_tier")

_WHITESPACE_RE = re.compile(r"\s+")


def clamp(value: float, low: float = 0.0, high: float = 5.0) -> float:
    """把数值夹到 [low, high]。"""
    return max(low, min(high, value))


def to_float(value: Any, default: float = 0.0) -> float:
    """尽量把任意值转成 float，失败返回 default。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_iso(value: str) -> datetime | None:
    """解析 RFC 3339 时间串；空值/非法值返回 None。"""
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def ensure_utc(value: datetime) -> datetime:
    """把无时区 datetime 视为 UTC。"""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def event_time(event: Event) -> datetime | None:
    """事件最近活动时间：优先 last_update_at，其次 first_seen_at。"""
    return parse_iso(event.last_update_at) or parse_iso(event.first_seen_at)


def event_age_hours(event: Event, reference_time: datetime | None = None) -> float | None:
    """事件距参考时间的小时数；时间缺失时返回 None。"""
    seen = event_time(event)
    if seen is None:
        return None
    reference = ensure_utc(reference_time or datetime.now(tz=UTC))
    return max(0.0, (reference - seen).total_seconds() / 3600.0)


def normalize_text(text: str) -> str:
    """文本归一化：全半角统一、大小写折叠、空白压缩。"""
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def contains_keyword(text: str, keyword: str) -> bool:
    """判断文本是否命中关键词。

    ASCII 关键词按单词边界匹配（避免 `job` 命中 `jobs`/`Steve Jobs` 之外的词），
    中文关键词按子串匹配。两侧都会先做归一化。
    """
    normalized_text = normalize_text(text)
    normalized_keyword = normalize_text(keyword)
    if not normalized_keyword:
        return False
    if not normalized_keyword.isascii():
        return normalized_keyword in normalized_text
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def matched_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    """返回文本命中的关键词（保持配置顺序、去重）。"""
    matches: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        normalized = normalize_text(keyword)
        if not normalized or normalized in seen:
            continue
        if contains_keyword(text, normalized):
            matches.append(keyword)
            seen.add(normalized)
    return matches


def title_similarity(left: str, right: str) -> float:
    """两个标题的相似度（0–1），用于重排阶段去重。"""
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def subject_significance(subject: Mapping[str, Any]) -> float:
    """主体显著度 = 人工权重 × 标注置信度（0–1 量纲）。

    v5 §3.3 指出 subjects.weight 被粗排/精排重复使用的风险：这里把「相关性/门槛」
    交给召回（只看是否过线），精排只把该显著度作为 impact 的一次输入。
    """
    weight = clamp(to_float(subject.get("weight"), 0.0), 0.0, 1.0)
    confidence = clamp(to_float(subject.get("confidence"), 1.0), 0.0, 1.0)
    return weight * confidence


def max_subject_significance(labels: Labels) -> float:
    """事件中最重要主体的显著度；无主体返回 0。"""
    values = [
        subject_significance(subject) for subject in labels.subjects if isinstance(subject, Mapping)
    ]
    return max(values, default=0.0)


def label_confidence(labels: Labels, key: str, default: float = 1.0) -> float:
    """读取 Labels.confidence 中某项置信度（0–1）。"""
    raw = labels.confidence.get(key)
    if raw is None:
        return default
    return clamp(to_float(raw, default), 0.0, 1.0)
