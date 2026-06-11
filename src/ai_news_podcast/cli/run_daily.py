"""Daily podcast pipeline: fetch → process → script → TTS → publish."""

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ai_news_podcast.cli.episode_utils import episode_id as _episode_id
from ai_news_podcast.cli.episode_utils import get_base_url as _get_base_url
from ai_news_podcast.pipeline.runner import run_pipeline
from ai_news_podcast.pipeline.scriptwriter import (
    generate_script,
    generate_show_notes_html,
)
from ai_news_podcast.pipeline.tts_engine import synthesize
from ai_news_podcast.text_utils import clean_tts_text
from ai_news_podcast.utils import load_sources, read_yaml, write_text

log = logging.getLogger("run_daily")


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--sources", default="config/sources.yaml")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument(
        "--force-refresh",
        action="store_true",
        help="强制重新抓取，忽略已有 brief 缓存",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    cfg = read_yaml(root / args.config)
    sources = load_sources(root / args.sources)

    podcast_cfg = cfg.get("podcast", {})
    tts_cfg = cfg.get("tts", {})
    script_cfg = cfg.get("script", {})
    build_cfg = cfg.get("build", {})

    episodes_dir = root / str(build_cfg.get("episodes_dir") or "site/episodes")

    now = datetime.now(tz=timezone.utc)
    if args.date:
        day = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)
    else:
        from zoneinfo import ZoneInfo

        day = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

    episode_id = _episode_id(day)
    episode_title = f"AI 新闻快报 | {episode_id}"

    podcast_title = str(podcast_cfg.get("title") or "AI 新闻播客").strip()

    backend = str(tts_cfg.get("backend") or "edge-tts")
    voices = (
        str(tts_cfg.get("host_a_voice") or "zh-CN-YunxiNeural"),
        str(tts_cfg.get("host_b_voice") or tts_cfg.get("voice") or "zh-CN-XiaoxiaoNeural"),
    )
    rate = str(tts_cfg.get("rate") or "+0%")
    volume = str(tts_cfg.get("volume") or "+0%")
    pitch = str(tts_cfg.get("pitch") or "+0Hz")
    mood_presets = tts_cfg.get("mood_presets")
    chunk_silence_ms = int(tts_cfg.get("chunk_silence_ms", 500))
    bgm_rel = str(tts_cfg.get("bgm_path") or "assets/bgm_placeholder.wav")
    bgm_path = str(root / bgm_rel) if (root / bgm_rel).exists() else None

    base_url = _get_base_url(cfg, args.base_url)

    # ── Stage 1 & 2: 数据基础层（抓取 → 去重 → 聚类 → 打分） ────────────────
    brief = await run_pipeline(
        cfg,
        sources,
        date_str=episode_id,
        data_dir=root / "data",
        force_refresh=args.force_refresh,
    )

    if not brief.get("stories"):
        log.warning("No stories in brief — aborting episode generation")
        return 1

    # ── Stage 3: Script generation ──
    log.info("Stage 3: generating script …")
    llm_cfg = cfg.get("llm", {})
    script_text, warnings = generate_script(
        brief,
        episode_date=day,
        podcast_title=podcast_title,
        script_cfg=script_cfg,
        llm_cfg=llm_cfg,
    )
    for w in warnings:
        log.warning("Script warning: %s", w)

    mp3_path = episodes_dir / f"{episode_id}.mp3"
    notes_path = episodes_dir / f"{episode_id}.html"
    transcript_path = episodes_dir / f"{episode_id}.txt"

    clean_transcript = clean_tts_text(script_text, preserve_ssml=False) + "\n"
    write_text(transcript_path, clean_transcript)
    log.info("Transcript saved: %s (%d chars)", transcript_path, len(clean_transcript))

    # ── Stage 4: TTS ──
    if not args.no_audio:
        log.info("Stage 4: synthesizing audio …")
        await synthesize(
            script_text,
            backend=backend,
            voices=voices,
            output_path=mp3_path,
            bgm_path=bgm_path,
            rate=rate,
            volume=volume,
            pitch=pitch,
            mood_presets=(mood_presets if isinstance(mood_presets, dict) else None),
            chunk_silence_ms=chunk_silence_ms,
            transcript_path=transcript_path,
            cfg=cfg,
            project_root=root,
        )
        log.info("Audio saved: %s", mp3_path)

    # ── Stage 5: Publish ──
    if args.no_audio:
        log.info("Stage 5: show notes only (--no-audio) …")
        notes_html = generate_show_notes_html(brief, episode_title=episode_title, episode_date=day)
        write_text(notes_path, notes_html)
        log.info("--no-audio: deferring feed/site publish to podcast-publish")
        return 0

    log.info("Stage 5: publishing …")
    from ai_news_podcast.cli.publish_episode import publish_episode

    publish_episode(
        root=root,
        cfg=cfg,
        brief=brief,
        episode_id=episode_id,
        episode_title=episode_title,
        episode_date=day,
        base_url=base_url,
        podcast_cfg=podcast_cfg,
        build_cfg=build_cfg,
        now=now,
    )
    log.info("Episode %s published successfully", episode_id)
    return 0


def entrypoint() -> int:
    """Synchronous entrypoint for console_scripts."""
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(entrypoint())
