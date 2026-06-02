"""Pipeline runner — 数据基础层。

职责：抓取 → 三层去重 → DBSCAN 聚类 → 五维打分 → 保存 brief JSON。

上层业务（播客、科技日报等）均通过 run_pipeline() 获取当日的 episode_brief，
不允许直接调用 fetch_all() 或 process()。

使用示例::

    brief = await run_pipeline(cfg, sources, date_str="2026-06-02", data_dir=Path("data"))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ai_news_podcast.pipeline.fetcher import fetch_all
from ai_news_podcast.pipeline.processor import process, save_brief
from ai_news_podcast.utils import read_json

log = logging.getLogger(__name__)


async def run_pipeline(
    cfg: dict[str, Any],
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
        完整的 config.yaml 内容（dict）。
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
    brief_path = data_dir / f"brief_{date_str}.json"

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
    fetch_cfg = cfg.get("fetch", {})
    processing_cfg = cfg.get("processing", {})

    timeout_seconds = int(fetch_cfg.get("timeout_seconds", 20))
    connect_timeout = int(fetch_cfg.get("connect_timeout", 5))
    user_agent = str(fetch_cfg.get("user_agent") or "ai-news-podcast/0.1")
    max_items_per_feed = int(fetch_cfg.get("max_items_per_feed", 30))
    max_pages = int(processing_cfg.get("max_pages", 80))

    log.info("Stage 1 [pipeline]: fetching RSS feeds …")
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
        return {"stories": [], "thesis": "", "metadata": {"total_raw": 0, "error": "no_items"}}

    log.info("Stage 1 [pipeline]: fetched %d raw items", len(raw_items))

    # ── Stage 2: 处理（三层去重 → DBSCAN 聚类 → 五维打分） ──────────────────
    log.info("Stage 2 [pipeline]: dedup → cluster → score …")
    brief = process(raw_items, processing_cfg=processing_cfg)

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

    return brief
