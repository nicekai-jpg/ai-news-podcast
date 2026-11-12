"""Stage 2 主题域分级：ai_core / ai_infra / other / unknown。

目的不是"标题里有没有 AI 词"，而是判断**这条新闻的主领域**：
- ai_core：模型、Agent、算法、AI 产品、AI 研究；
- ai_infra：芯片/算力/机器人/自动驾驶等 AI 基础设施；
- other：汽车、消费、金融、社会、体育、娱乐等非 AI 主领域；
- unknown：信息不足。

规则顺序（先强后弱）：
1. **标题**命中强 AI 词 → ai_core；
2. 标题命中自动驾驶/智能驾驶等 AI 驾驶词 → ai_infra（即使同时有汽车词）；
3. 标题命中非 AI 类目词 → other；
4. 标题命中基础设施词 → ai_infra；
5. official/research 来源允许正文 AI 词兜底 → ai_core/ai_infra；
6. 否则 unknown。

只用标题判定主领域；摘要/正文只给 official/research 做兜底，
避免汽车新闻因摘要里提了一句“自动驾驶”被归到 AI。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from ai_news_podcast.business.ocean.relevance import CORE_HINTS, INFRA_TERMS
from ai_news_podcast.business.ocean.taxonomy import default_taxonomy
from ai_news_podcast.business.recommend.base import matched_keywords

_CONFIG_PATH = Path(__file__).resolve().parents[4] / "config" / "domain_keywords.yaml"


@lru_cache(maxsize=1)
def load_domain_keywords(path: str | None = None) -> dict[str, tuple[str, ...]]:
    """读取 config/domain_keywords.yaml；缺失时回退内置列表。"""
    config_path = Path(path) if path else _CONFIG_PATH
    raw: dict = {}
    if config_path.is_file():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raw = {}
    return {
        "ai_core": tuple(CORE_HINTS) + tuple(raw.get("ai_core_extra") or ()),
        "ai_infra": tuple(INFRA_TERMS) + tuple(raw.get("ai_infra_extra") or ()),
        "ai_driving": tuple(raw.get("ai_driving") or AI_DRIVING_TERMS),
        "other_category": tuple(raw.get("other_category") or OTHER_CATEGORY_TERMS),
        "auto_brands": tuple(raw.get("auto_brands") or AUTO_BRANDS),
    }


AI_CORE = "ai_core"
AI_INFRA = "ai_infra"
AI_APPLICATION = "ai_application"
OTHER = "other"
UNKNOWN = "unknown"
AI_DOMAINS = (AI_CORE, AI_INFRA, AI_APPLICATION)

# 明确属于 AI 驾驶/座舱的标题，优先于"汽车"类目词。
AI_DRIVING_TERMS: tuple[str, ...] = (
    "自动驾驶",
    "智能驾驶",
    "无人驾驶",
    "智能座舱",
    "robotaxi",
    "fsd",
)

# 非 AI 主领域的高置信标题词；命中即判 other（前提是没命中强 AI 词）。
OTHER_CATEGORY_TERMS: tuple[str, ...] = (
    "汽车",
    "车型",
    "车企",
    "造车",
    "轿车",
    "suv",
    "销量",
    "停产",
    "召回",
    "碰撞",
    "车祸",
    "公交车",
    "停车",
    "股票",
    "股价",
    "a股",
    "港股",
    "美股",
    "指数",
    "涨停",
    "跌停",
    "基金",
    "融资买入",
    "财报",
    "营收",
    "利润",
    "房价",
    "楼市",
    "体育",
    "足球",
    "篮球",
    "奥运",
    "电影",
    "电视剧",
    "明星",
    "综艺",
    "短剧",
    "游戏",
    "手机",
    "耳机",
    "笔记本",
    "好物",
    "优惠",
    "折扣",
    "双十一",
    "玩具",
    "摆件",
    "遥控",
    "天气",
    "事故",
    "社会",
    "政治",
)

# 汽车品牌：标题带品牌不等于 AI（除非同时有 AI 驾驶词）。
AUTO_BRANDS: tuple[str, ...] = (
    "特斯拉",
    "比亚迪",
    "蔚来",
    "小鹏",
    "理想汽车",
    "问界",
    "小米汽车",
    "极氪",
    "长城汽车",
    "吉利",
    "丰田",
    "本田",
    "大众汽车",
    "奔驰",
    "宝马",
    "奥迪",
    "福特",
    "现代汽车",
    "起亚",
    "沃尔沃",
)


def _matches(text: str, terms: tuple[str, ...]) -> list[str]:
    return matched_keywords(text, terms)


def _head(title: str, summary: str) -> str:
    # 主领域只看标题；summary 参数保留是为了接口兼容/未来可选策略
    del summary
    return title.strip()


def _keyword_domain(head: str, terms: dict[str, tuple[str, ...]]) -> tuple[str, list[str]] | None:
    for domain, term_list in (
        (AI_CORE, terms["ai_core"]),
        (AI_INFRA, terms["ai_driving"]),
    ):
        matches = _matches(head, term_list)
        if matches:
            return domain, matches[:5]

    other = _matches(head, terms["other_category"] + terms["auto_brands"])
    if other:
        return OTHER, other[:5]

    infra = _matches(head, terms["ai_infra"])
    if infra:
        return AI_INFRA, infra[:5]
    return None


def classify_domain(
    title: str,
    summary: str = "",
    full_text: str = "",
    *,
    category: str = "news",
    keywords: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str, list[str]]:
    """返回 (primary_domain, evidence)。"""
    head = _head(title, summary)
    terms = keywords or load_domain_keywords()

    keyword_domain = _keyword_domain(head, terms)
    if keyword_domain is not None:
        return keyword_domain

    industry_domain = _industry_domain(title)
    if industry_domain is not None:
        return industry_domain

    if str(category or "").lower() in ("official", "research"):
        return _keyword_domain(full_text[:2000], terms) or (UNKNOWN, [])
    return UNKNOWN, []


def _industry_domain(title: str) -> tuple[str, list[str]] | None:
    industry = default_taxonomy().industry_for_title(title)
    if industry is None:
        return None
    related = str(industry.get("ai_related") or "none")
    domain = {
        "core": AI_CORE,
        "infra": AI_INFRA,
        "application": AI_APPLICATION,
    }.get(related, OTHER)
    return domain, [str(industry.get("name"))]


def stronger_domain(current: str, candidate: str) -> str:
    """事件内多 item 合并：ai_core > ai_infra > unknown > other。"""
    order = {AI_CORE: 3, AI_INFRA: 2, UNKNOWN: 1, OTHER: 0}
    return candidate if order.get(candidate, 0) > order.get(current, 0) else current


__all__ = [
    "AI_APPLICATION",
    "AI_CORE",
    "AI_DOMAINS",
    "AI_INFRA",
    "OTHER",
    "UNKNOWN",
    "classify_domain",
    "stronger_domain",
]
