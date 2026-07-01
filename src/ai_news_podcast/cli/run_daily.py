"""Daily podcast pipeline: fetch → process → script → TTS → publish."""

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

try:
    from ai_news_podcast.cli.daily_report import generate_report
except ImportError:
    generate_report = None  # type: ignore
from ai_news_podcast.text_utils import clean_tts_text
from ai_news_podcast.utils import load_sources, read_yaml, write_text

log = logging.getLogger("run_daily")


def _resolve_date_and_id(
    args: Any,
    cfg: dict[str, Any],
) -> tuple[datetime, str, str, str]:
    """Parse CLI args to resolve episode date, id, title and base URL."""
    if args.date:
        day = datetime.fromisoformat(args.date).replace(tzinfo=UTC)
    else:
        from zoneinfo import ZoneInfo

        day = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

    episode_id = _episode_id(day)
    episode_title = f"AI 新闻快报 | {episode_id}"
    base_url = _get_base_url(cfg, args.base_url)
    return day, episode_id, episode_title, base_url


def _extract_tts_config(
    cfg: dict[str, Any],
    root: Path,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Extract TTS configuration and parameters."""
    podcast_cfg = cfg.get("podcast", {})
    tts_cfg = cfg.get("tts", {})

    podcast_title = str(podcast_cfg.get("title") or "AI 新闻播客").strip()
    bgm_rel = str(tts_cfg.get("bgm_path") or "assets/bgm_placeholder.wav")

    params = {
        "backend": str(tts_cfg.get("backend") or "cosyvoice2"),
        "bgm_path": str(root / bgm_rel) if (root / bgm_rel).exists() else None,
    }
    return podcast_title, params, podcast_cfg


def _maybe_generate_report(
    args: Any, brief: dict[str, Any], day: datetime, episode_id: str, root: Path
) -> None:
    """Generate daily report if --with-report flag is set."""
    if args.with_report and generate_report is not None:
        log.info("Stage 3b: generating daily report …")
        report_dir = root / "data" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        date_str = day.strftime("%Y年%m月%d日")
        llm_cfg = brief.get("llm_cfg", {})
        try:
            generate_report(
                brief,
                date_str=date_str,
                report_id=episode_id,
                outdir=report_dir,
                llm_cfg=llm_cfg,
            )
            log.info("Daily report saved to %s", report_dir)
        except Exception as e:  # noqa: BLE001
            log.warning("Daily report generation failed: %s", e)


async def _synthesize_episode(
    script_text: str,
    tts_params: dict[str, Any],
    mp3_path: Path,
    transcript_path: Path,
    cfg: dict[str, Any],
    root: Path,
) -> None:
    """Synthesize audio for the episode."""
    log.info("Stage 4: synthesizing audio …")
    await synthesize(
        script_text,
        backend=tts_params["backend"],
        output_path=mp3_path,
        bgm_path=tts_params["bgm_path"],
        transcript_path=transcript_path,
        cfg=cfg,
        project_root=root,
    )
    log.info("Audio saved: %s", mp3_path)


async def _publish_or_skip(  # noqa: PLR0913
    args: Any,
    brief: dict[str, Any],
    episode_id: str,
    episode_title: str,
    day: datetime,
    base_url: str,
    podcast_cfg: dict[str, Any],
    build_cfg: dict[str, Any],
    now: datetime,
    root: Path,
    cfg: dict[str, Any],
    notes_path: Path,
) -> int:
    """Publish episode or skip if --no-audio."""
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
        "--with-report",
        action="store_true",
        help="同时生成日报 (daily report)",
    )
    ap.add_argument(
        "--force-refresh",
        action="store_true",
        help="强制重新抓取，忽略已有 brief 缓存",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    cfg = read_yaml(root / args.config)
    sources = load_sources(root / args.sources)

    now = datetime.now(tz=UTC)
    day, episode_id, episode_title, base_url = _resolve_date_and_id(args, cfg)
    build_cfg = cfg.get("build", {})
    script_cfg = cfg.get("script", {})
    podcast_title, tts_params, podcast_cfg = _extract_tts_config(cfg, root)
    episodes_dir = root / str(build_cfg.get("episodes_dir") or "site/episodes")

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

    clean_transcript = clean_tts_text(script_text) + "\n"
    write_text(transcript_path, clean_transcript)
    log.info("Transcript saved: %s (%d chars)", transcript_path, len(clean_transcript))

    _maybe_generate_report(args, brief, day, episode_id, root)

    if not args.no_audio:
        await _synthesize_episode(script_text, tts_params, mp3_path, transcript_path, cfg, root)

    return await _publish_or_skip(
        args,
        brief,
        episode_id,
        episode_title,
        day,
        base_url,
        podcast_cfg,
        build_cfg,
        now,
        root,
        cfg,
        notes_path,
    )


def entrypoint() -> int:
    """Synchronous entrypoint for console_scripts."""
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(entrypoint())
