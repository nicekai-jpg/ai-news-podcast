"""Pipeline runner — 数据基础层。

职责：抓取 → 三层去重 → DBSCAN 聚类 → 五维打分 → 保存 brief JSON。

上层业务（播客、科技日报等）均通过 run_pipeline() 获取当日的 episode_brief，
不允许直接调用 fetch_all() 或 process()。

使用示例::

    brief = await run_pipeline(cfg, sources, date_str="2026-06-02", data_dir=Path("data"))
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_news_podcast.config.models import AppConfig
from ai_news_podcast.events import StageCompleted, StageFailed, StageStarted, event_bus
from ai_news_podcast.pipeline.dedup import (
    get_recent_broadcasted_texts,
    get_recent_broadcasted_urls,
    semantic_dedup,
)
from ai_news_podcast.pipeline.fetcher import fetch_all
from ai_news_podcast.pipeline.processor import process, save_brief
from ai_news_podcast.utils import read_json

log = logging.getLogger(__name__)


def _to_dict(cfg: AppConfig | dict[str, Any]) -> dict[str, Any]:
    """Convert AppConfig back to dict for backward compatibility."""
    if isinstance(cfg, dict):
        return cfg
    # AppConfig is a dataclass — convert to dict recursively
    import dataclasses
    import json

    return json.loads(json.dumps(dataclasses.asdict(cfg), default=str))


async def run_pipeline(  # noqa: PLR0915
    cfg: AppConfig | dict[str, Any],
    sources: list[Any],
    date_str: str,
    data_dir: Path,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """执行数据基础管线，返回当日 episode_brief。

    Parameters
    ----------
    cfg:
        AppConfig 实例或兼容的 dict。
    sources:
        由 load_sources() 读取的信源列表。
    date_str:
        日期字符串，格式 ``YYYY-MM-DD``，用于命名 brief 文件。
    data_dir:
        数据目录（例如 ``<root>/data``），brief JSON 保存于此。
    force_refresh:
        若为 True，则忽略已有 brief 文件，强制重新抓取并处理。

    Returns
    -------
    dict
        episode_brief 字典，包含 ``thesis``、``stories``、``metadata`` 等字段。
    """
    cfg_dict = _to_dict(cfg)
    brief_path = data_dir / "briefs" / f"brief_{date_str}.json"

    # ── 复用已有 brief ──────────────────────────────────────────────────────
    if brief_path.exists() and not force_refresh:
        log.info("Found existing brief at %s — skipping fetch and process", brief_path)
        brief = read_json(brief_path)
        if brief and isinstance(brief, dict) and brief.get("stories") is not None:
            log.info(
                "Loaded brief: %d stories (main=%d, supporting=%d, quick=%d)",
                len(brief.get("stories", [])),
                sum(1 for s in brief.get("stories", []) if s.get("role") == "main"),
                sum(1 for s in brief.get("stories", []) if s.get("role") == "supporting"),
                sum(1 for s in brief.get("stories", []) if s.get("role") == "quick"),
            )
            return brief
        log.warning("Existing brief at %s appears invalid, re-running pipeline", brief_path)

    # ── Stage 1: 抓取 ───────────────────────────────────────────────────────
    fetch_cfg = cfg_dict.get("fetch", {})
    processing_cfg = cfg_dict.get("processing", {})

    timeout_seconds = int(fetch_cfg.get("timeout_seconds", 20))
    connect_timeout = int(fetch_cfg.get("connect_timeout", 5))
    user_agent = str(fetch_cfg.get("user_agent") or "ai-news-podcast/0.1")
    max_items_per_feed = int(fetch_cfg.get("max_items_per_feed", 30))
    max_pages = int(processing_cfg.get("max_pages", 80))

    log.info("Stage 1 [pipeline]: fetching RSS feeds …")
    event_bus.emit(StageStarted(stage="fetch", episode_id=date_str, timestamp=datetime.now(tz=UTC)))
    raw_items = await fetch_all(
        sources,
        timeout_seconds=timeout_seconds,
        connect_timeout=connect_timeout,
        user_agent=user_agent,
        max_items_per_feed=max_items_per_feed,
        max_pages=max_pages,
    )

    if not raw_items:
        log.warning("No items fetched — pipeline aborted")
        event_bus.emit(
            StageFailed(
                stage="fetch",
                episode_id=date_str,
                error="No items fetched",
                timestamp=datetime.now(tz=UTC),
            )
        )
        return {"stories": [], "thesis": "", "metadata": {"total_raw": 0, "error": "no_items"}}

    log.info("Stage 1 [pipeline]: fetched %d raw items", len(raw_items))
    event_bus.emit(
        StageCompleted(
            stage="fetch", episode_id=date_str, duration_ms=0, result={"count": len(raw_items)}
        )
    )

    # ── 跨期历史去重 ──────────────────────────────────────────────────────────
    dedup_details: list[dict[str, Any]] = []
    dedup_cfg = processing_cfg.get("dedup", {})
    recent_episodes_limit = int(dedup_cfg.get("recent_episodes_limit", 14))
    semantic_model = str(dedup_cfg.get("semantic_model", "paraphrase-multilingual-MiniLM-L12-v2"))

    episodes_index = data_dir / "episodes.json"
    recent_urls = get_recent_broadcasted_urls(
        episodes_index, limit=recent_episodes_limit, current_episode_id=date_str
    )
    if recent_urls:
        filtered_items = []
        for item in raw_items:
            if item.normalized_link in recent_urls:
                dedup_details.append(
                    {
                        "title": item.title,
                        "link": item.link,
                        "source": item.source_name,
                        "reason": "cross_episode_url",
                        "detail": "URL matches a recently broadcasted article.",
                    }
                )
            else:
                filtered_items.append(item)

        filtered_count = len(raw_items) - len(filtered_items)
        raw_items = filtered_items
        if filtered_count > 0:
            log.info(
                "Cross-episode dedup: filtered out %d already broadcasted articles, %d items remaining",
                filtered_count,
                len(raw_items),
            )

    # ── 跨期语义相似度去重 ──────────────────────────────────────────────────────
    recent_records = get_recent_broadcasted_texts(
        episodes_index, limit=recent_episodes_limit, current_episode_id=date_str
    )
    if recent_records and raw_items:
        raw_items, semantic_details = semantic_dedup(
            raw_items,
            recent_records,
            dedup_cfg,
            semantic_model=semantic_model,
        )
        dedup_details.extend(semantic_details)

    # ── Stage 2: 处理（三层去重 → DBSCAN 聚类 → 五维打分） ──────────────────
    log.info("Stage 2 [pipeline]: dedup → cluster → score …")
    event_bus.emit(
        StageStarted(stage="process", episode_id=date_str, timestamp=datetime.now(tz=UTC))
    )
    brief = process(raw_items, processing_cfg=processing_cfg)

    # ── 注入去重详细计算信息 ────────────────────────────────────────────────────
    brief.setdefault("metadata", {})["dedup_details"] = dedup_details

    # ── 持久化 brief ─────────────────────────────────────────────────────────
    data_dir.mkdir(parents=True, exist_ok=True)
    save_brief(brief, brief_path)

    log.info(
        "Pipeline complete: %d stories (main=%d, supporting=%d, quick=%d) → %s",
        len(brief.get("stories", [])),
        sum(1 for s in brief.get("stories", []) if s.get("role") == "main"),
        sum(1 for s in brief.get("stories", []) if s.get("role") == "supporting"),
        sum(1 for s in brief.get("stories", []) if s.get("role") == "quick"),
        brief_path,
    )
    event_bus.emit(
        StageCompleted(
            stage="process",
            episode_id=date_str,
            duration_ms=0,
            result={"stories": len(brief.get("stories", []))},
        )
    )

    return brief
