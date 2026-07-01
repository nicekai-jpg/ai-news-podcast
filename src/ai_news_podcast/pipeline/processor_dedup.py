"""Stage 2 — 去重管线。

三层去重：URL 硬去重 → rapidfuzz 标题软去重 → jieba 关键词重叠。
"""

from __future__ import annotations

import logging
from datetime import datetime

from rapidfuzz import fuzz

from ai_news_podcast.pipeline.fetcher import RawItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 第一层去重：URL 硬去重
# ---------------------------------------------------------------------------


def _dedup_url(items: list[RawItem]) -> list[RawItem]:
    """基于 normalized_link 去重。"""
    seen: set[str] = set()
    out: list[RawItem] = []
    for item in items:
        key = item.normalized_link
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    logger.info("URL dedup: %d → %d", len(items), len(out))
    return out


# ---------------------------------------------------------------------------
# 第二层去重：rapidfuzz 标题软去重
# ---------------------------------------------------------------------------


def _dedup_title_fuzzy(
    items: list[RawItem],
    *,
    threshold: int = 92,
    window_hours: int = 48,
) -> list[RawItem]:
    """token_set_ratio ≥ threshold 且时间差 ≤ window_hours 时视为重复。"""
    out: list[RawItem] = []
    for item in items:
        is_dup = False
        item_dt = datetime.fromisoformat(item.published_at)
        for kept in out:
            kept_dt = datetime.fromisoformat(kept.published_at)
            if abs((item_dt - kept_dt).total_seconds()) > window_hours * 3600:
                continue
            ratio = fuzz.token_set_ratio(item.title, kept.title)
            if ratio >= threshold:
                is_dup = True
                break
        if not is_dup:
            out.append(item)
    logger.info("Fuzzy title dedup: %d → %d", len(items), len(out))
    return out


# ---------------------------------------------------------------------------
# 第三层去重：jieba 关键词重叠
# ---------------------------------------------------------------------------


def _extract_keywords(text: str, topk: int = 10) -> set[str]:
    """提取 jieba 关键词集合。"""
    try:
        import jieba.analyse

        keywords = jieba.analyse.extract_tags(text, topK=topk)
        return set(keywords)
    except (ImportError, RuntimeError):
        words = set(jieba.cut(text))
        return {w for w in words if len(w) >= 2}


def _dedup_keyword_overlap(
    items: list[RawItem],
    *,
    overlap_threshold: float = 0.35,
    title_sim_threshold: int = 85,
) -> list[RawItem]:
    """jieba 关键词重叠 ≥ overlap_threshold 且标题相似度 ≥ title_sim 视为重复。"""
    out: list[RawItem] = []
    kw_cache: list[set[str]] = []
    for item in items:
        text = f"{item.title} {item.summary}"
        kws = _extract_keywords(text)
        is_dup = False
        for i, kept in enumerate(out):
            # 标题相似度门槛
            title_ratio = fuzz.token_set_ratio(item.title, kept.title)
            if title_ratio < title_sim_threshold:
                continue
            # 关键词重叠
            kept_kws = kw_cache[i]
            if not kws or not kept_kws:
                continue
            overlap = len(kws & kept_kws) / min(len(kws), len(kept_kws))
            if overlap >= overlap_threshold:
                is_dup = True
                break
        if not is_dup:
            out.append(item)
            kw_cache.append(kws)
    logger.info("Keyword overlap dedup: %d → %d", len(items), len(out))
    return out


def dedup_pipeline(
    items: list[RawItem],
    *,
    rapidfuzz_threshold: int = 92,
    jieba_overlap: float = 0.35,
    title_sim: int = 85,
    window_hours: int = 48,
) -> list[RawItem]:
    """三层去重管线。"""
    items = _dedup_url(items)
    items = _dedup_title_fuzzy(items, threshold=rapidfuzz_threshold, window_hours=window_hours)
    items = _dedup_keyword_overlap(
        items, overlap_threshold=jieba_overlap, title_sim_threshold=title_sim
    )
    return items
