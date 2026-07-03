"""Stage 4 CLI entry: podcast-tts.

根据播客脚本生成音频。
"""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_news_podcast.cli.base import AsyncCommand
from ai_news_podcast.config.models import AppConfig
from ai_news_podcast.pipeline.tts_engine import synthesize


class TTSCommand(AsyncCommand):
    """Synthesize podcast audio from script."""

    description = "合成播客音频 (Stage 4)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD (默认今天)")
        parser.add_argument("--script", default=None, help="脚本文件路径")
        parser.add_argument("--output", default=None, help="输出 MP3 路径")

    async def execute_async(self, args: argparse.Namespace, cfg: AppConfig, root: Path) -> int:
        date_str = args.date or datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")

        episodes_dir = root / cfg.build.episodes_dir
        script_path = Path(args.script) if args.script else episodes_dir / f"{date_str}.txt"
        output_path = Path(args.output) if args.output else episodes_dir / f"{date_str}.mp3"

        if not script_path.exists():
            print(f"Script not found: {script_path}")
            return 1

        bgm_rel = cfg.tts.bgm_path
        bgm_path = str(root / bgm_rel) if bgm_rel and (root / bgm_rel).exists() else None

        podcast_text = script_path.read_text(encoding="utf-8")

        print(f"Stage 4: synthesizing audio for {date_str} …")
        await synthesize(
            podcast_text,
            backend=cfg.tts.backend,
            output_path=output_path,
            bgm_path=bgm_path,
            transcript_path=script_path,
            cfg=cfg,
            project_root=root,
        )
        print(f"Audio saved: {output_path}")
        return 0


def entrypoint() -> int:
    return TTSCommand().run()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
