"""Stage 3 CLI entry: podcast-writer.

根据 brief 生成播客对话脚本。
"""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_news_podcast.business.pipeline.podcast_writer_service import generate_podcast
from ai_news_podcast.data.config.models_dao import AppConfig
from ai_news_podcast.data.text_utils_dao import clean_tts_text_preserving_chapters
from ai_news_podcast.data.utils_dao import write_text
from ai_news_podcast.presentation.cli.base_controller import AsyncCommand


class WriterCommand(AsyncCommand):
    """Generate podcast script from brief."""

    description = "生成播客对话脚本 (Stage 3)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD (默认今天)")

    async def execute_async(self, args: argparse.Namespace, cfg: AppConfig, root: Path) -> int:
        from ai_news_podcast.data.utils_dao import read_yaml

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

        import dataclasses

        podcast_title = cfg.podcast.title
        script_cfg = cfg.script
        llm_cfg = cfg.llm

        print(f"Stage 3: generating script for {date_str} …")
        podcast_text, warnings = generate_podcast(
            brief,
            episode_date=day,
            podcast_title=podcast_title,
            writer_cfg=dataclasses.asdict(script_cfg),
            llm_cfg=dataclasses.asdict(llm_cfg),
        )
        for w in warnings:
            print(f"Script warning: {w}")

        episodes_dir = root / cfg.build.episodes_dir
        episodes_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = episodes_dir / f"{date_str}.txt"
        # 兜底再清洗一次，但必须保住 [CHAPTER] 章节行（clean_tts_text 会把章节当普通中括号删掉）
        final_script = clean_tts_text_preserving_chapters(podcast_text).rstrip("\n") + "\n"
        write_text(transcript_path, final_script)
        print(f"Script saved: {transcript_path} ({len(final_script)} chars)")

        return 0


def entrypoint() -> int:
    return WriterCommand().run()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
