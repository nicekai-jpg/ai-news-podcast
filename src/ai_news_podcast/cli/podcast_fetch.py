"""Stage 1 CLI 入口：podcast-fetch。

抓取所有 RSS 源，输出原始数据。
输入: config/sources.yaml
输出: data/raw/{date}.json
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ai_news_podcast.pipeline.fetcher import fetch_all
from ai_news_podcast.utils import load_sources, read_yaml

log = logging.getLogger("podcast_fetch")


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="抓取新闻源 (Stage 1)")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--sources", default="config/sources.yaml")
    ap.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD (默认今天)")
    ap.add_argument("--output", default=None, help="输出文件路径 (默认 data/raw/{date}.json)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    cfg = read_yaml(root / args.config)
    sources = load_sources(root / args.sources)

    date_str = args.date or datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")

    fetch_cfg = cfg.get("fetch", {})
    timeout_seconds = int(fetch_cfg.get("timeout_seconds", 20))
    connect_timeout = int(fetch_cfg.get("connect_timeout", 5))
    user_agent = str(fetch_cfg.get("user_agent") or "ai-news-podcast/0.1")
    max_items_per_feed = int(fetch_cfg.get("max_items_per_feed", 30))
    max_pages = int(cfg.get("processing", {}).get("max_pages", 80))

    log.info("Stage 1: fetching RSS feeds …")
    raw_items = await fetch_all(
        sources,
        timeout_seconds=timeout_seconds,
        connect_timeout=connect_timeout,
        user_agent=user_agent,
        max_items_per_feed=max_items_per_feed,
        max_pages=max_pages,
    )

    if not raw_items:
        log.warning("No items fetched")
        return 1

    log.info("Stage 1: fetched %d raw items", len(raw_items))

    # Save raw items
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else raw_dir / f"{date_str}.json"
    output_path.write_text(
        json.dumps([item.to_dict() for item in raw_items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Raw items saved: %s", output_path)

    return 0


def entrypoint() -> int:
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(entrypoint())
