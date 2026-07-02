"""Stage 4 CLI 入口：podcast-tts。

根据播客脚本生成音频。
输入: site/episodes/{date}.txt
输出: site/episodes/{date}.mp3
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

from ai_news_podcast.pipeline.tts_engine import synthesize
from ai_news_podcast.utils import read_yaml

log = logging.getLogger("podcast_tts")


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="合成播客音频 (Stage 4)")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD (默认今天)")
    ap.add_argument("--script", default=None, help="脚本文件路径 (默认 site/episodes/{date}.txt)")
    ap.add_argument("--output", default=None, help="输出 MP3 路径 (默认 site/episodes/{date}.mp3)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    cfg = read_yaml(root / args.config)

    date_str = args.date or datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")

    episodes_dir = root / str(cfg.get("build", {}).get("episodes_dir") or "site/episodes")
    script_path = Path(args.script) if args.script else episodes_dir / f"{date_str}.txt"
    output_path = Path(args.output) if args.output else episodes_dir / f"{date_str}.mp3"

    if not script_path.exists():
        log.error("Script not found: %s", script_path)
        return 1

    tts_cfg = cfg.get("tts", {})
    bgm_rel = str(tts_cfg.get("bgm_path") or "assets/bgm_placeholder.wav")
    bgm_path = str(root / bgm_rel) if (root / bgm_rel).exists() else None

    script_text = script_path.read_text(encoding="utf-8")

    log.info("Stage 4: synthesizing audio for %s …", date_str)
    await synthesize(
        script_text,
        backend=str(tts_cfg.get("backend") or "cosyvoice2"),
        output_path=output_path,
        bgm_path=bgm_path,
        transcript_path=script_path,
        cfg=cfg,
        project_root=root,
    )
    log.info("Audio saved: %s", output_path)

    return 0


def entrypoint() -> int:
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(entrypoint())
