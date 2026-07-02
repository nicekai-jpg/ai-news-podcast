"""Stage 2 CLI 入口：podcast-process。

处理原始数据：去重 → 聚类 → 打分。
输入: data/raw/{date}.json
输出: data/briefs/brief_{date}.json
"""

import argparse
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

from ai_news_podcast.pipeline.fetcher import RawItem
from ai_news_podcast.pipeline.processor import process
from ai_news_podcast.utils import read_yaml, write_json

log = logging.getLogger("podcast_process")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="处理新闻数据 (Stage 2)")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD (默认今天)")
    ap.add_argument("--input", default=None, help="输入文件路径 (默认 data/raw/{date}.json)")
    ap.add_argument(
        "--output", default=None, help="输出文件路径 (默认 data/briefs/brief_{date}.json)"
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    cfg = read_yaml(root / args.config)

    date_str = args.date or datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")

    # Load raw items
    input_path = Path(args.input) if args.input else root / "data" / "raw" / f"{date_str}.json"
    if not input_path.exists():
        log.error("Raw items not found: %s", input_path)
        return 1

    raw_data = json.loads(input_path.read_text(encoding="utf-8"))
    raw_items = [RawItem.from_dict(item) for item in raw_data]
    log.info("Loaded %d raw items from %s", len(raw_items), input_path)

    # Process
    processing_cfg = cfg.get("processing", {})
    log.info("Stage 2: dedup → cluster → score …")
    brief = process(raw_items, processing_cfg=processing_cfg)

    # Save brief
    briefs_dir = root / "data" / "briefs"
    briefs_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else briefs_dir / f"brief_{date_str}.json"
    write_json(output_path, brief)
    log.info(
        "Brief saved: %s (%d stories: main=%d, supporting=%d, quick=%d)",
        output_path,
        len(brief.get("stories", [])),
        sum(1 for s in brief.get("stories", []) if s.get("role") == "main"),
        sum(1 for s in brief.get("stories", []) if s.get("role") == "supporting"),
        sum(1 for s in brief.get("stories", []) if s.get("role") == "quick"),
    )

    return 0


def entrypoint() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
