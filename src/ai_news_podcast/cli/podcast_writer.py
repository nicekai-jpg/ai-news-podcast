"""Stage 3 CLI 入口：podcast-writer。

根据 brief 生成播客对话脚本。
输入: data/briefs/brief_{date}.json
输出: site/episodes/{date}.txt
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

from ai_news_podcast.pipeline.podcastwriter import generate_podcast
from ai_news_podcast.text_utils import clean_tts_text
from ai_news_podcast.utils import read_yaml, write_text

log = logging.getLogger("podcast_writer")


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="生成播客对话脚本 (Stage 3)")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD (默认今天)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    cfg = read_yaml(root / args.config)

    date_str = args.date or datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    day = datetime.fromisoformat(date_str).replace(tzinfo=ZoneInfo("Asia/Shanghai"))

    brief_path = root / "data" / "briefs" / f"brief_{date_str}.json"
    if not brief_path.exists():
        log.error("Brief not found: %s", brief_path)
        return 1

    brief = read_yaml(brief_path)
    if not brief.get("stories"):
        log.error("Brief has no stories")
        return 1

    podcast_title = str(cfg.get("podcast", {}).get("title") or "AI 每日先锋").strip()
    script_cfg = cfg.get("script", {})
    llm_cfg = cfg.get("llm", {})

    log.info("Stage 3: generating script for %s …", date_str)
    podcast_text, warnings = generate_podcast(
        brief,
        episode_date=day,
        podcast_title=podcast_title,
        script_cfg=script_cfg,
        llm_cfg=llm_cfg,
    )
    for w in warnings:
        log.warning("Script warning: %s", w)

    episodes_dir = root / str(cfg.get("build", {}).get("episodes_dir") or "site/episodes")
    episodes_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = episodes_dir / f"{date_str}.txt"
    clean_transcript = clean_tts_text(podcast_text) + "\n"
    write_text(transcript_path, clean_transcript)
    log.info("Script saved: %s (%d chars)", transcript_path, len(clean_transcript))

    return 0


def entrypoint() -> int:
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(entrypoint())
