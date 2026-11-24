"""未登录实体识别：jieba 词性 + 机构后缀 + 英文大写串。

词表永远追不上新公司，这一步负责找出「词表里没有但像实体」的词。
准入从严，避免把「大模型」「智能化」这类泛词当成主体。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PERSON_FLAG = "nr"
_PROPER_FLAGS = ("nr", "nt", "nz")

# 只保留强机构后缀；「科技」「智能」「机器人」太泛，会把「比亚迪智能」切出来
_STRONG_SUFFIX: tuple[tuple[str, str], ...] = (
    ("公司", "company"),
    ("集团", "company"),
    ("科技", "company"),
    ("半导体", "company"),
    ("通信", "company"),
    ("电子", "company"),
    ("软件", "company"),
    ("网络", "company"),
    ("医药", "company"),
    ("药业", "company"),
    ("生物", "company"),
    ("能源", "company"),
    ("控股", "company"),
    ("实业", "company"),
    ("传媒", "company"),
    ("游戏", "company"),
    ("数据", "company"),
    ("信息", "company"),
    ("电气", "company"),
    ("家电", "company"),
    ("汽车", "company"),
    ("银行", "company"),
    ("医院", "company"),
    ("基金会", "company"),
    ("协会", "company"),
    ("联盟", "company"),
    ("实验室", "org"),
    ("研究所", "org"),
    ("研究院", "org"),
    ("大学", "org"),
    ("学院", "org"),
)

_GENERIC = {
    "人工智能",
    "大模型",
    "大语言模型",
    "语言模型",
    "模型",
    "算法",
    "技术",
    "产品",
    "服务",
    "平台",
    "系统",
    "数据",
    "算力",
    "芯片",
    "能力",
    "功能",
    "版本",
    "价格",
    "成本",
    "收入",
    "利润",
    "市值",
    "股价",
    "时间",
    "今天",
    "昨天",
    "今年",
    "明年",
    "上周",
    "本周",
    "目前",
    "未来",
    "国内",
    "海外",
    "全球",
    "中国",
    "美国",
    "市场",
    "行业",
    "用户",
    "客户",
    "员工",
    "公司",
    "团队",
    "研究",
    "实验",
    "论文",
    "智能化",
    # 2026-11-19 实体评测补入：新闻里高频出现但不指代具体实体的泛词
    "材料公司",
    "半导体材料公司",
    "这家公司",
    "该公司",
    "车企",
    "德系",
    "德系车",
    "都市",
    "中国智能",
    "中国团队",
    "官方",
    "开发者",
    "玩家",
    "观众",
    "读者",
    "记者",
    "分析师",
    "研究员",
    "业内人士",
}

_MEDIA = {"量子位", "QbitAI", "凹非寺", "公众号", "作者", "编辑", "来源", "本文", "记者"}

_EN_STOP = {
    "AI",
    "API",
    "APP",
    "App",
    "CEO",
    "CFO",
    "COO",
    "CTO",
    "Demo",
    "Contact",
    "Path",
    "PFS",
    "SUV",
    "Top",
    "New",
    "The",
    "And",
    "For",
    "With",
    "From",
}

_EN_CAPS = re.compile(r"[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3}")
_SUFFIX_PATTERN = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9]{2,12}(?:" + "|".join(s for s, _ in _STRONG_SUFFIX) + r")"
)


@dataclass(frozen=True)
class OpenEntity:
    """未登录实体：原文词、字符区间、粗粒度类型。"""

    text: str
    start: int
    end: int
    kind: str = "other"


def _suffix_kind(text: str) -> str | None:
    for suffix, kind in _STRONG_SUFFIX:
        if text.endswith(suffix):
            return kind
    return None


def _contains_verb(text: str) -> bool:
    import jieba.posseg as pseg

    return any(word.flag.startswith("v") for word in pseg.cut(text))


_PRODUCT_PREFIXES = ("新能源", "智能", "电动", "燃油", "自动驾驶", "无人驾驶")


def _keep(text: str, flag: str, title: str) -> bool:  # noqa: PLR0911
    """准入规则：人名 / 强机构后缀 / 出现在标题里，三者之一。"""
    if len(text) < 2 or text in _GENERIC or text in _MEDIA:
        return False
    if text.endswith("汽车") and text.startswith(_PRODUCT_PREFIXES):
        return False
    if text.isascii() and (len(text) < 3 or text in _EN_STOP):
        return False
    if text.isascii():
        return " " in text or (bool(title) and text in title)
    if flag == _PERSON_FLAG:
        return True
    if _suffix_kind(text) is not None:
        return not _contains_verb(text)
    if flag in ("nt", "nz") and 2 <= len(text) <= 8:
        return not _contains_verb(text)
    return bool(title) and text in title


def detect_open_entities(sentence: str, title: str = "") -> list[OpenEntity]:
    """从一句话里找出未登录实体，重叠时保留更长的那个。"""
    import jieba.posseg as pseg

    found: list[OpenEntity] = []
    cursor = 0
    for word, flag in pseg.cut(sentence):
        position = sentence.find(word, cursor)
        if position < 0:
            continue
        cursor = position + len(word)
        if flag not in _PROPER_FLAGS or not _keep(word, flag, title):
            continue
        kind = "person" if flag == _PERSON_FLAG else (_suffix_kind(word) or "other")
        found.append(OpenEntity(word, position, position + len(word), kind))

    for match in _SUFFIX_PATTERN.finditer(sentence):
        text = match.group(0)
        if not _keep(text, "nt", title):
            continue
        found.append(OpenEntity(text, match.start(), match.end(), _suffix_kind(text) or "other"))

    for match in _EN_CAPS.finditer(sentence):
        text = match.group(0).strip()
        if match.start() == 0 and text not in title:
            continue
        if not _keep(text, "nt", title):
            continue
        found.append(OpenEntity(text, match.start(), match.start() + len(text), "org"))

    return _resolve_overlaps(found)


def _resolve_overlaps(items: list[OpenEntity]) -> list[OpenEntity]:
    ordered = sorted(items, key=lambda item: (item.start, -(item.end - item.start)))
    kept: list[OpenEntity] = []
    last_end = -1
    for item in ordered:
        if item.start < last_end:
            continue
        kept.append(item)
        last_end = item.end
    return kept


__all__ = ["OpenEntity", "detect_open_entities"]
