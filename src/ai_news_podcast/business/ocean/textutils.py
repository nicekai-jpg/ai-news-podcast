"""文本预处理与规则抽取的公共工具（去 HTML、规范化、分词、版本/数字抽取）。"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from datetime import UTC, datetime

import jieba
from rapidfuzz import fuzz

log = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

_STOPWORDS = frozenset(
    (
        "的",
        "了",
        "和",
        "与",
        "及",
        "在",
        "是",
        "为",
        "对",
        "将",
        "等",
        "中",
        "也",
        "有",
        "被",
        "从",
        "到",
        "而",
        "或",
        "一个",
        "一种",
        "我们",
        "他们",
        "它们",
        "这",
        "那",
        "其",
        "于",
        "上",
        "下",
        "就",
        "都",
        "会",
        "可能",
        "已经",
        "正在",
        "目前",
        "今天",
        "昨日",
        "表示",
        "认为",
        "指出",
        "称",
        "宣布",
        "消息",
        "新闻",
        "最新",
        "相关",
        "进行",
        "通过",
        "以及",
        "可以",
        "没有",
        "不是",
        "就是",
        "还是",
        "但是",
        "因为",
        "所以",
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "from",
        "as",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "have",
        "has",
        "had",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
    )
)

_VERSION_FAMILIES = (
    "gpt",
    "claude",
    "gemini",
    "gemma",
    "llama",
    "qwen",
    "deepseek",
    "kimi",
    "glm",
    "ernie",
    "grok",
    "mistral",
    "minimax",
    "hunyuan",
    "doubao",
    "seed",
    "phi",
    "sora",
    "veo",
)
_FAMILY_VERSION_RE = re.compile(
    r"\b(" + "|".join(_VERSION_FAMILIES) + r")\s*[-_ ]?\s*(?:[vVmM])?\s*(\d+(?:\.\d+){0,3})",
    re.IGNORECASE,
)
_GENERIC_VERSION_RE = re.compile(r"\bv(\d+(?:\.\d+){1,3})\b", re.IGNORECASE)

_MONEY_PATTERNS = (
    re.compile(r"[$￥¥]\s?\d+(?:\.\d+)?(?:\s?(?:亿|万|千|million|billion|bn|mn|m|b))?", re.I),
    re.compile(
        r"\d+(?:,\d{3})*(?:\.\d+)?\s?(?:亿|万|千)?\s?(?:美元|人民币|元|dollars?|usd|cny|rmb)",
        re.I,
    ),
    re.compile(r"\d+(?:\.\d+)?\s?(?:million|billion|bn|mn)\s?(?:dollars?|usd)?", re.I),
)
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s?%|百分之[零一二三四五六七八九十百千\d]+")
_SCORE_RE = re.compile(
    r"(?:得分|评分|score|准确率|准确度|通过率|胜率|成功率)\s*[:：]?\s*\d+(?:\.\d+)?%?"
    r"|\b\d{1,3}(?:\.\d+)?\s?(?:分|points?|pt)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\d{4}\s?[-/年]\s?\d{1,2}\s?[-/月]\s?\d{1,2}\s?日?"
    r"|\d{1,2}\s?月\s?\d{1,2}\s?日"
)
_COUNT_RE = re.compile(
    r"\d+(?:,\d{3})*(?:\.\d+)?\s?(?:万|亿|千|倍|条|个|篇|台|张|名|位|轮|次|年|天|小时)"
)

_QUOTE_RE = re.compile(r"[“”「」『』\"]\s*[^“”「」『』\"]{2,}\s*[“”「」『』\"]")


def clean_text(text: str) -> str:
    """去 HTML 标签/实体、全角规范化、压缩空白。"""
    if not text:
        return ""
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_title(title: str) -> str:
    """标题规范化：只保留字母/数字/汉字，用于近似比较。"""
    cleaned = clean_text(title).lower()
    return "".join(ch for ch in cleaned if ch.isalnum())


def title_similarity(left: str, right: str) -> float:
    """规范化标题的 rapidfuzz ratio（0–1）。"""
    norm_left = normalize_title(left)
    norm_right = normalize_title(right)
    if not norm_left or not norm_right:
        return 0.0
    return fuzz.ratio(norm_left, norm_right) / 100.0


def tokenize(text: str) -> list[str]:
    """jieba 分词 + 去停用词/纯数字，用于关键词与候选生成。"""
    cleaned = clean_text(text).lower()
    if not cleaned:
        return []
    tokens: list[str] = []
    for raw in jieba.cut(cleaned):
        token = raw.strip()
        if not token or token in _STOPWORDS:
            continue
        if token.isdigit():
            continue
        if len(token) < 2 and not _CJK_RE.search(token):
            continue
        tokens.append(token)
    return tokens


def keywords(text: str, top_k: int = 15) -> list[str]:
    """文本关键词（按出现顺序去重，最多 top_k 个）。"""
    seen: dict[str, None] = {}
    for token in tokenize(text):
        seen.setdefault(token, None)
        if len(seen) >= top_k:
            break
    return list(seen)


def parse_time(value: str) -> datetime | None:
    """解析 ISO / 日期字符串为 UTC datetime；失败返回 None。"""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def hours_between(left: datetime, right: datetime) -> float:
    return abs((left - right).total_seconds()) / 3600.0


def days_between(left: datetime, right: datetime) -> float:
    return abs((left - right).total_seconds()) / 86400.0


def _normalize_version(family: str, version: str) -> str | None:
    version = version.strip("._-")
    if not version:
        return None
    if len(version) == 4 and version.isdigit() and 1900 <= int(version) <= 2100:
        return None
    return f"{family.lower()}-{version}"


def extract_version_strings(text: str) -> set[str]:
    """抽取版本号集合，如 {gpt-5.1, v5.1}；用于新进展与去重保护。"""
    if not text:
        return set()
    cleaned = clean_text(text)
    versions: set[str] = set()
    for match in _FAMILY_VERSION_RE.finditer(cleaned):
        normalized = _normalize_version(match.group(1), match.group(2))
        if normalized:
            versions.add(normalized)
    for match in _GENERIC_VERSION_RE.finditer(cleaned):
        versions.add(f"v{match.group(1)}")
    return versions


def extract_money(text: str) -> list[str]:
    values: list[str] = []
    for pattern in _MONEY_PATTERNS:
        for match in pattern.finditer(text):
            value = _WS_RE.sub("", match.group(0)).lower()
            if value not in values:
                values.append(value)
    return values


def _normalize_date(raw: str) -> str:
    value = _WS_RE.sub("", raw)
    year_match = re.match(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", value)
    if year_match:
        year, month, day = year_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    month_match = re.match(r"(\d{1,2})月(\d{1,2})日", value)
    if month_match:
        month, day = month_match.groups()
        return f"{int(month):02d}-{int(day):02d}"
    return value


def extract_novelty_numbers(text: str) -> list[dict[str, str]]:
    """抽取新颖度数字/版本：[{type, value}]，按 version/money/percent/score/date/count 顺序。"""
    if not text:
        return []
    cleaned = clean_text(text)
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(kind: str, value: str) -> None:
        key = (kind, value)
        if value and key not in seen:
            seen.add(key)
            found.append({"type": kind, "value": value})

    for version in sorted(extract_version_strings(cleaned)):
        _add("version", version)
    for money in extract_money(cleaned):
        _add("money", money)
    for match in _PERCENT_RE.finditer(cleaned):
        _add("percent", _WS_RE.sub("", match.group(0)).lower())
    for match in _SCORE_RE.finditer(cleaned):
        _add("score", _WS_RE.sub("", match.group(0)).lower())
    for match in _DATE_RE.finditer(cleaned):
        _add("date", _normalize_date(match.group(0)))
    for match in _COUNT_RE.finditer(cleaned):
        _add("count", _WS_RE.sub("", match.group(0)).lower())
    return found


def extract_fact_tokens(text: str) -> set[str]:
    """事实一致性用的数字/版本 token（排除日期，避免跨源发布日期差异）。"""
    tokens: set[str] = set()
    for item in extract_novelty_numbers(text):
        if item["type"] == "date":
            continue
        tokens.add(f"{item['type']}:{item['value']}")
    return tokens


def has_quoted_speech(text: str) -> bool:
    """是否出现可引用的引语（引号包裹的 2 字以上内容）。"""
    return bool(_QUOTE_RE.search(clean_text(text)))


def is_official_category(category: str, official_categories: list[str]) -> bool:
    return (category or "").strip().lower() in {
        str(value).strip().lower() for value in official_categories
    }
