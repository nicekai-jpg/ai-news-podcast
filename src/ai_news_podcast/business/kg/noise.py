"""选句、噪声过滤、语言判断、文章主体识别（事实层 S1）。"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import yaml

from ai_news_podcast.business.kg.entities import detect_open_entities
from ai_news_podcast.business.kg.lexicon import KGlexicon
from ai_news_podcast.business.kg.types import FactsConfig, LexiconSpan, SlotValue

logger = logging.getLogger(__name__)

DEFAULT_NOISE_PATH = "config/nlp/noise.yaml"

_SENT_END = re.compile(r"(?<=[。！？!?；;])|\n+")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_MIN_SENTENCE_CHARS = 6
_MIN_ITEM_CHARS = 20

# 噪声兜底：yaml 缺失时仍能工作
_FALLBACK_PATTERNS = ("来源[:：]", "发自", "公众号", "原文链接", "参考链接", "扫码关注", "往期回顾")


@lru_cache(maxsize=4)
def _load_noise_patterns(path: str) -> tuple[re.Pattern[str], ...]:
    file = Path(path)
    raw: list[str] = []
    if file.is_file():
        try:
            data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
            for row in data.get("patterns") or []:
                if isinstance(row, dict) and row.get("pattern"):
                    raw.append(str(row["pattern"]))
                elif isinstance(row, str):
                    raw.append(row)
        except (OSError, yaml.YAMLError):
            logger.warning("噪声配置读取失败，用兜底模式", exc_info=True)
    if not raw:
        raw = list(_FALLBACK_PATTERNS)
    patterns: list[re.Pattern[str]] = []
    for item in raw:
        try:
            patterns.append(re.compile(item))
        except re.error:
            logger.warning("噪声模式非法，跳过：%s", item)
    return tuple(patterns)


def split_sentences(text: str) -> list[str]:
    """按句末标点/换行切句，去空白，丢弃过短句。"""
    value = str(text or "").strip()
    if not value:
        return []
    parts = [x.strip() for x in _SENT_END.split(value)]
    return [x for x in parts if len(x) >= _MIN_SENTENCE_CHARS]


def is_noise_sentence(sentence: str, patterns: tuple[re.Pattern[str], ...] | None = None) -> bool:
    """命中任一噪声模式 → 视为噪声句。"""
    if not sentence:
        return True
    pats = patterns if patterns is not None else _load_noise_patterns(DEFAULT_NOISE_PATH)
    return any(p.search(sentence) for p in pats)


def detect_language(item: dict) -> str:
    """返回 zh / en / 空串（空串表示无法判断）。"""
    lang = str(item.get("lang") or "").strip().lower()
    if lang.startswith("zh"):
        return "zh"
    if lang.startswith("en"):
        return "en"
    text = str(item.get("full_text") or item.get("summary") or item.get("title") or "")
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN.findall(text))
    if cjk + latin == 0:
        return ""
    return "zh" if cjk >= latin else "en"


_ARTICLE_STOP = {
    "FHD",
    "IPS",
    "Hz",
    "GT2",
    "GB",
    "TB",
    "GHz",
    "OLED",
    "LCD",
    "USB",
    "CPU",
    "GPU",
    "NPU",
    "SSD",
    "PC",
    "AR",
    "VR",
    "XR",
}
_MEDIA_WORDS = (
    "快科技",
    "IT之家",
    "新浪科技",
    "腾讯科技",
    "网易科技",
    "搜狐科技",
    "凤凰科技",
    "36氪",
    "钛媒体",
    "虎嗅",
    "机器之心",
    "新智元",
    "量子位",
    "澎湃新闻",
    "第一财经",
    "财联社",
    "界面新闻",
    "cnBeta",
    "TechCrunch",
    "The Verge",
    "Wired",
    "Reuters",
    "Bloomberg",
)
_PRODUCT_WORDS = {
    "显示器",
    "手机",
    "芯片",
    "模型",
    "版本",
    "系统",
    "平台",
    "产品",
    "服务",
    "应用",
    "功能",
    "设备",
    "汽车",
    "机器人",
    "显卡",
    "电池",
    "屏幕",
    "笔记本",
    "耳机",
    "眼镜",
    "相机",
}
_LEAD_SUBJECT = re.compile(
    r"([\u4e00-\u9fa5]{2,6})(?:推出|发布|上线|宣布|收购|并购|开源|融资|上市|完成|获得|签署|任命)"
)


def _unit_phrase(text: str) -> bool:
    if not text.isascii():
        return False
    tokens = [token for token in re.split(r"\s+", text) if token]
    if not tokens:
        return True
    for token in tokens:
        if token.upper() in _ARTICLE_STOP:
            return True
        if token.isupper() and 2 <= len(token) <= 4:
            return True
    return False


def _valid_subject_name(name: str) -> bool:
    text = str(name or "").strip()
    if len(text) < 2:
        return False
    if text[0].isdigit():
        return False
    return any(char.isalpha() or "\u4e00" <= char <= "\u9fa5" for char in text)


def _plausible_article_subject(item) -> bool:
    text = str(item.text or "")
    if not text or text in _MEDIA_WORDS:
        return False
    if text.isascii():
        if text in _ARTICLE_STOP or len(text) < 3 or text.isupper():
            return False
        return item.kind == "company"
    return item.kind in ("company", "person", "org")


def pick_article_subject(title: str, lexicon: KGlexicon) -> SlotValue | None:
    """标题/导语里最像文章主体的实体：词表 → 未登录实体 → 动作前专名。"""
    text = str(title or "")
    spans: list[LexiconSpan] = lexicon.find_spans(text)
    for span in spans:
        if span.source == "latin" and _unit_phrase(span.name):
            continue
        if span.kind in ("company", "person", "org") and _valid_subject_name(span.name):
            return SlotValue("subject", span.name, span.entity_id, span.kind)
    try:
        opened = detect_open_entities(text)
    except Exception:
        opened = []
    for item in sorted(opened, key=lambda row: -(row.end - row.start)):
        if _plausible_article_subject(item) and _valid_subject_name(item.text):
            return SlotValue("subject", item.text, "", item.kind)
    for match in _LEAD_SUBJECT.finditer(text):
        name = match.group(1)
        if name in _PRODUCT_WORDS or name in _MEDIA_WORDS:
            continue
        if not _valid_subject_name(name):
            continue
        return SlotValue("subject", name, "", "company")
    for span in spans:
        if span.source == "latin" and _unit_phrase(span.name):
            continue
        if _valid_subject_name(span.name):
            return SlotValue("subject", span.name, span.entity_id, span.kind)
    return None


def select_news_sentences(
    text: str,
    config: FactsConfig,
    *,
    title: str = "",
    noise_path: str = DEFAULT_NOISE_PATH,
) -> list[str]:
    """标题 + 正文全句；去噪声、长句转入分句、超限截断。"""
    patterns = _load_noise_patterns(noise_path)
    sentences: list[str] = []
    if title and len(title.strip()) >= _MIN_SENTENCE_CHARS:
        sentences.append(title.strip())
    for sentence in split_sentences(text):
        if is_noise_sentence(sentence, patterns):
            continue
        sentences.append(sentence)
    # 相邻重复句去重
    deduped: list[str] = []
    for sentence in sentences:
        if deduped and deduped[-1] == sentence:
            continue
        deduped.append(sentence)
    return deduped[: config.max_sentences_guard]


def item_text(item: dict) -> str:
    """条目可解析文本：全文优先，其次摘要。"""
    text = str(item.get("full_text") or "").strip()
    if len(text) >= _MIN_ITEM_CHARS:
        return text
    return str(item.get("summary") or "").strip()


__all__ = [
    "DEFAULT_NOISE_PATH",
    "detect_language",
    "is_noise_sentence",
    "item_text",
    "pick_article_subject",
    "select_news_sentences",
    "split_sentences",
]
