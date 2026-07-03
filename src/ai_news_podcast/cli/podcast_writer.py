"""Stage 3 CLI entry: podcast-writer.

根据 brief 生成播客对话脚本。
"""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_news_podcast.cli.base import AsyncCommand
from ai_news_podcast.config.models import AppConfig
from ai_news_podcast.pipeline.podcastwriter import generate_podcast
from ai_news_podcast.text_utils import clean_tts_text
from ai_news_podcast.utils import write_text


class WriterCommand(AsyncCommand):
    """Generate podcast script from brief."""

    description = "生成播客对话脚本 (Stage 3)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD (默认今天)")

    async def execute_async(self, args: argparse.Namespace, cfg: AppConfig, root: Path) -> int:
        from ai_news_podcast.utils import read_yaml

        date_str = args.date or datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
        day = datetime.fromisoformat(date_str).replace(tzinfo=ZoneInfo("Asia/Shanghai"))

        brief_path = root / "data" / "briefs" / f"brief_{date_str}.json"
        if not brief_path.exists():
            print(f"Brief not found: {brief_path}")
            return 1

        brief = read_yaml(brief_path)
        if not brief.get("stories"):
            print("Brief has no stories")
            return 1

        podcast_title = cfg.podcast.title
        script_cfg = cfg.script
        llm_cfg = cfg.llm

        print(f"Stage 3: generating script for {date_str} …")
        podcast_text, warnings = generate_podcast(
            brief,
            episode_date=day,
            podcast_title=podcast_title,
            script_cfg=script_cfg,
            llm_cfg=llm_cfg,
        )
        for w in warnings:
            print(f"Script warning: {w}")

        episodes_dir = root / cfg.build.episodes_dir
        episodes_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = episodes_dir / f"{date_str}.txt"
        clean_transcript = clean_tts_text(podcast_text) + "\n"
        write_text(transcript_path, clean_transcript)
        print(f"Script saved: {transcript_path} ({len(clean_transcript)} chars)")

        return 0


def entrypoint() -> int:
    return WriterCommand().run()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
