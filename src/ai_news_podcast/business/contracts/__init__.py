"""跨阶段共享数据契约（Stage 1–4）。

所有阶段只通过本模块定义的结构交换数据，避免各自造格式。
阶段之间以 JSON artifact 传递（`to_dict()` / `from_dict()` 可序列化）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

CHANNELS = ("c1_rss", "c2_atom", "c3_rsshub", "c4_web", "c5_wechat2rss")
FULL_TEXT_STATUS = ("ok", "failed", "truncated", "none")
LIFECYCLES = ("burst", "evolving", "long_tail", "fading", "unknown")
EVIDENCE_TIERS = ("strong", "medium", "weak", "rumor", "unknown")


def utc_now_iso() -> str:
    """当前 UTC 时间（RFC 3339）。"""
    return datetime.now(tz=UTC).isoformat()


def canonical_link(link: str) -> str:
    """URL 规范化：小写 host、去 www、去 fragment、去跟踪参数、去尾斜杠。"""
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    link = (link or "").strip()
    if not link:
        return ""
    parts = urlsplit(link)
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    drop_keys = ("utm_", "fbclid", "gclid", "spm", "from", "ref")
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not any(k.lower().startswith(d) or k.lower() == d for d in drop_keys)
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme or "https", host, path, urlencode(query), ""))


def make_item_id(link: str, title: str = "", source: str = "") -> str:
    """生成 Stage 1 item_id：优先规范化链接，其次标题+来源。"""
    key = canonical_link(link) or f"{source}|{title}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def make_event_id(item_ids: list[str]) -> str:
    """事件 ID：成员条目 ID 排序后哈希，保证幂等（成员变化时会重新生成）。"""
    joined = "|".join(sorted(item_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass
class Item:
    """Stage 1 采集入海产出的统一条目。"""

    item_id: str
    channel: str
    source: str
    source_label: str
    category: str = "news"  # 来源类别：official/news/research/community
    title: str = ""
    link: str = ""
    normalized_link: str = ""
    published_at: str = ""
    fetched_at: str = ""
    summary: str = ""
    full_text: str = ""
    full_text_status: str = "none"
    lang: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Item:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Labels:
    """Stage 2 多维标注（只放事实/属性，不放分数/排名）。"""

    subjects: list[dict[str, Any]] = field(default_factory=list)  # [{id, name, weight, confidence}]
    object_type: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    primary_action: str = ""
    tech_stack: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    lifecycle: str = "unknown"  # 仅用于粗排/点评，禁止进精排
    topics: list[str] = field(default_factory=list)
    primary_topic: str = ""
    evidence_tier: str = "unknown"  # 仅用于粗排门槛
    novelty_numbers: list[dict[str, Any]] = field(default_factory=list)  # [{type, value}]
    narrative_angle: list[str] = field(default_factory=list)
    audience_persona: list[str] = field(default_factory=list)
    sensitivity: list[str] = field(default_factory=list)
    ai_relevance: str = "unknown"  # core / adjacent / none / unknown
    ai_evidence: list[str] = field(default_factory=list)  # 命中的 AI 关键词
    quote_availability: bool = False
    confidence: dict[str, float] = field(default_factory=dict)
    method: str = "rule"  # rule / llm / hybrid

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Event:
    """Stage 2 新闻潮汐的事件（同一事件多源合并后的结果）。"""

    event_id: str
    representative_title: str
    summary: str = ""
    item_ids: list[str] = field(default_factory=list)
    source_count: int = 0
    independent_sources: int = 0
    first_seen_at: str = ""
    last_update_at: str = ""
    continuity_id: str = ""
    sequence: int = 1
    is_update: bool = False
    labels: Labels = field(default_factory=Labels)
    source_categories: list[str] = field(default_factory=list)  # official/news/research/community
    evidence_score: float = 0.0
    momentum_growth: float = 0.0
    reappeared: bool = False
    covered_before: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        known = set(cls.__dataclass_fields__)
        labels = data.get("labels") or {}
        return cls(
            **{k: v for k, v in data.items() if k in known and k != "labels"},
            labels=Labels(**{k: v for k, v in labels.items() if k in Labels.__dataclass_fields__}),
        )


@dataclass
class ShortlistEntry:
    """Stage 3 选题推荐的候选条目。"""

    rank: int
    event_id: str
    score: float
    rule_score: float = 0.0
    llm_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    evidence_item_ids: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    rejected_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Fact:
    text: str
    evidence_ids: list[str] = field(default_factory=list)
    single_source: bool = False


@dataclass
class Commentary:
    """Stage 4 日报正文（事实与观点分离，02 深度日报/页面使用）。"""

    event_id: str
    headline: str
    one_liner: str = ""
    facts: list[Fact] = field(default_factory=list)
    why_matters: str = ""
    background: str = ""
    impact: str = ""
    counterpoint: str = ""
    uncertainty: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewItem:
    """单条结果的点评（S5 日报点评 / S7 播客点评共用）。"""

    event_id: str = ""
    title: str = ""
    verdict: str = ""  # 准确 / 需修正 / 补充
    comment: str = ""
    corrections: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EditorialReview:
    """S5 日报点评：对已生成的日报做编辑审读。"""

    overall: str = ""
    items: list[ReviewItem] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PodcastReview:
    """S7 播客点评：对已生成的播客文稿做审读 + 01 页面 show notes。"""

    overall: str = ""
    show_notes: str = ""
    chapters: list[dict[str, Any]] = field(default_factory=list)
    items: list[ReviewItem] = field(default_factory=list)
    fact_check_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dumps(data: Any) -> str:
    """稳定 JSON 序列化（中文不转义、字段排序）。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)


def loads(text: str) -> Any:
    return json.loads(text)
