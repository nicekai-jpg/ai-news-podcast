"""数据基础管线 CLI 入口：podcast-pipeline。

执行完整数据处理流程：抓取 → 三层去重 → DBSCAN 聚类 → 五维打分。
输出 brief_{date}.json，供 podcast-daily 和 podcast-report 等上层业务复用。

用法::

    # 运行今日管线（如已有 brief 则复用）
    podcast-pipeline

    # 强制刷新，忽略已有缓存
    podcast-pipeline --force-refresh

    # 指定日期（回测用）
    podcast-pipeline --date 2026-06-01
"""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ai_news_podcast.pipeline.runner import run_pipeline
from ai_news_podcast.utils import load_sources, read_yaml

log = logging.getLogger("pipeline_run")


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="运行数据基础管线（抓取 → 去重 → 聚类 → 打分）")
    ap.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    ap.add_argument("--sources", default="config/sources.yaml", help="信源配置路径")
    ap.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD（默认今天）")
    ap.add_argument(
        "--force-refresh",
        action="store_true",
        help="强制重新抓取，忽略已有 brief 缓存",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    cfg = read_yaml(root / args.config)
    sources = load_sources(root / args.sources)

    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")

    data_dir = root / "data"

    brief = await run_pipeline(
        cfg,
        sources,
        date_str=date_str,
        data_dir=data_dir,
        force_refresh=args.force_refresh,
    )

    stories = brief.get("stories", [])
    if not stories:
        log.error("Pipeline produced no stories — check fetch/process configuration")
        return 1

    print("\n" + "=" * 55)
    print(f"✅ 数据基础管线完成 | {date_str}")
    print(f"   故事总数: {len(stories)}")
    print(f"   🔴 主故事: {sum(1 for s in stories if s.get('role') == 'main')}")
    print(f"   🟡 支撑故事: {sum(1 for s in stories if s.get('role') == 'supporting')}")
    print(f"   🟢 快讯: {sum(1 for s in stories if s.get('role') == 'quick')}")
    print(f"   Brief 路径: data/brief_{date_str}.json")
    print("=" * 55 + "\n")

    return 0


def entrypoint() -> int:
    """Synchronous entrypoint for console_scripts."""
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(entrypoint())
