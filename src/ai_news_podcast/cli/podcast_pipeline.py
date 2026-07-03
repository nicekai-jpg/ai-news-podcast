"""Stage 1 CLI entry: podcast-pipeline.

Runs the full data pipeline: fetch → dedup → cluster → score → brief JSON.
"""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_news_podcast.cli.base import AsyncCommand
from ai_news_podcast.config.models import AppConfig
from ai_news_podcast.pipeline.runner import run_pipeline
from ai_news_podcast.utils import load_sources


class PipelineCommand(AsyncCommand):
    """Run data pipeline (Stage 1)."""

    description = "运行数据基础管线（抓取 → 去重 → 聚类 → 打分）"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--sources", default="config/sources.yaml", help="信源配置路径")
        parser.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD（默认今天）")
        parser.add_argument(
            "--force-refresh",
            action="store_true",
            help="强制重新抓取，忽略已有 brief 缓存",
        )

    async def execute_async(self, args: argparse.Namespace, cfg: AppConfig, root: Path) -> int:
        sources = load_sources(root / args.sources)
        date_str = args.date or datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")

        brief = await run_pipeline(
            cfg,
            sources,
            date_str=date_str,
            data_dir=root / "data",
            force_refresh=args.force_refresh,
        )

        stories = brief.get("stories", [])
        if not stories:
            print("Pipeline produced no stories — check fetch/process configuration")
            return 1

        print("\n" + "=" * 55)
        print(f"✅ 数据基础管线完成 | {date_str}")
        print(f"   故事总数: {len(stories)}")
        print(f"   🔴 主故事: {sum(1 for s in stories if s.get('role') == 'main')}")
        print(f"   🟡 支撑故事: {sum(1 for s in stories if s.get('role') == 'supporting')}")
        print(f"   🟢 快讯: {sum(1 for s in stories if s.get('role') == 'quick')}")
        print(f"   Brief 路径: data/briefs/brief_{date_str}.json")
        print("=" * 55 + "\n")

        return 0


def entrypoint() -> int:
    return PipelineCommand().run()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
