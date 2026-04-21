"""Daily podcast pipeline: fetch → process → script → TTS → publish."""

import argparse
import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any, Optional

from ai_news_podcast.pipeline.fetcher import fetch_all
from ai_news_podcast.pipeline.processor import process, save_brief
from ai_news_podcast.pipeline.scriptwriter import (
    generate_script,
    generate_show_notes_html,
)
from ai_news_podcast.pipeline.tts_engine import synthesize
from ai_news_podcast.site_builder.html_gen import build_index_html
from ai_news_podcast.site_builder.rss_gen import build_feed_xml
from ai_news_podcast.utils import load_sources, read_json, read_yaml, write_json, write_text

log = logging.getLogger("run_daily")


def _get_base_url(cfg: dict[str, Any], cli_base_url: Optional[str]) -> str:
    env = os.environ
    if cli_base_url:
        return cli_base_url.rstrip("/")
    if env.get("PODCAST_BASE_URL"):
        return env["PODCAST_BASE_URL"].rstrip("/")
    base_url = str(cfg.get("podcast", {}).get("base_url") or "").strip()
    if base_url:
        return base_url.rstrip("/")
    owner = (env.get("GITHUB_REPOSITORY_OWNER") or "").strip()
    repo_full = (env.get("GITHUB_REPOSITORY") or "").strip()
    if owner and repo_full and "/" in repo_full:
        repo = repo_full.split("/", 1)[1]
        if repo == f"{owner}.github.io":
            return f"https://{owner}.github.io".rstrip("/")
        return f"https://{owner}.github.io/{repo}".rstrip("/")
    return "http://localhost"


def _episode_id(day: datetime) -> str:
    return day.strftime("%Y-%m-%d")


def _coerce_episode_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(item)
    return out


def _prune_episodes(
    episodes: list[dict[str, Any]],
    *,
    keep_last: int,
    episodes_dir: Path,
) -> list[dict[str, Any]]:
    def parse_pubdate(ep: dict[str, Any]) -> datetime:
        try:
            return datetime.fromisoformat(str(ep["published_at_iso"]))
        except Exception:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

    sorted_eps = sorted(episodes, key=parse_pubdate, reverse=True)

    keep: list[dict[str, Any]] = []
    for i, ep in enumerate(sorted_eps):
        if i < keep_last:
            keep.append(ep)

    keep_ids = {str(ep.get("id") or "") for ep in keep}

    for i, ep in enumerate(sorted_eps):
        if i < keep_last:
            continue
        eid = str(ep.get("id") or "")
        if not eid:
            continue
        mp3 = episodes_dir / f"{eid}.mp3"
        html = episodes_dir / f"{eid}.html"
        txt = episodes_dir / f"{eid}.txt"
        if mp3.exists():
            mp3.unlink()
        if html.exists():
            html.unlink()
        if txt.exists():
            txt.unlink()

    return [ep for ep in keep if str(ep.get("id") or "") in keep_ids]


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
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    cfg = read_yaml(root / args.config)
    sources = load_sources(root / args.sources)

    podcast_cfg = cfg.get("podcast", {})
    tts_cfg = cfg.get("tts", {})
    fetch_cfg = cfg.get("fetch", {})
    processing_cfg = cfg.get("processing", {})
    script_cfg = cfg.get("script", {})
    build_cfg = cfg.get("build", {})

    site_dir = root / str(build_cfg.get("site_dir") or "site")
    episodes_dir = root / str(build_cfg.get("episodes_dir") or "site/episodes")
    episodes_index = root / str(build_cfg.get("episodes_index") or "data/episodes.json")

    now = datetime.now(tz=timezone.utc)
    if args.date:
        day = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)
    else:
        from zoneinfo import ZoneInfo

        day = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

    episode_id = _episode_id(day)
    episode_title = f"AI 新闻快报 | {episode_id}"

    podcast_title = str(podcast_cfg.get("title") or "AI 新闻播客").strip()
    podcast_description = str(podcast_cfg.get("description") or "").strip()
    podcast_language = str(podcast_cfg.get("language") or "zh-cn").strip()
    podcast_author = str(podcast_cfg.get("author") or "").strip()
    podcast_category = str(podcast_cfg.get("category") or "Technology").strip()
    podcast_explicit = bool(podcast_cfg.get("explicit", False))
    keep_last = int(podcast_cfg.get("keep_last", 30))

    backend = str(tts_cfg.get("backend") or "edge-tts")
    voice = str(tts_cfg.get("voice") or "zh-CN-XiaoxiaoNeural")
    rate = str(tts_cfg.get("rate") or "+0%")
    volume = str(tts_cfg.get("volume") or "+0%")
    pitch = str(tts_cfg.get("pitch") or "+0Hz")
    mood_presets = tts_cfg.get("mood_presets")
    chunk_silence_ms = int(tts_cfg.get("chunk_silence_ms", 500))
    cosyvoice_model_path = str(tts_cfg.get("cosyvoice_model_path") or "")
    cosyvoice_speaker = str(tts_cfg.get("cosyvoice_speaker") or "")

    timeout_seconds = int(fetch_cfg.get("timeout_seconds", 20))
    connect_timeout = int(fetch_cfg.get("connect_timeout", 5))
    user_agent = str(fetch_cfg.get("user_agent") or "ai-news-podcast/0.1")
    max_items_per_feed = int(fetch_cfg.get("max_items_per_feed", 30))
    max_pages = int(processing_cfg.get("max_pages", 80))

    base_url = _get_base_url(cfg, args.base_url)

    # ── Stage 1: Fetch ──
    log.info("Stage 1: fetching RSS feeds …")
    raw_items = await fetch_all(
        sources,
        timeout_seconds=timeout_seconds,
        connect_timeout=connect_timeout,
        user_agent=user_agent,
        max_items_per_feed=max_items_per_feed,
        max_pages=max_pages,
    )
    log.info("Fetched %d raw items from %d sources", len(raw_items), len(sources))

    if not raw_items:
        log.warning("No items fetched — aborting episode generation")
        return 1

    # ── Stage 2: Process (dedup → cluster → score → role assign → thesis) ──
    log.info("Stage 2: processing items …")
    brief = process(raw_items, processing_cfg=processing_cfg)
    brief_path = root / "data" / f"brief_{episode_id}.json"
    save_brief(brief, brief_path)
    log.info(
        "Brief: %d stories (main=%d, supporting=%d, quick=%d)",
        len(brief.get("stories", [])),
        sum(1 for s in brief.get("stories", []) if s.get("role") == "main"),
        sum(1 for s in brief.get("stories", []) if s.get("role") == "supporting"),
        sum(1 for s in brief.get("stories", []) if s.get("role") == "quick"),
    )

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

    clean_transcript = re.sub(r"\[mood:[a-zA-Z0-9_-]+\]\s*", "", script_text)
    clean_transcript = re.sub(r"\[(?:FACT|INFERENCE|OPINION)\]\s*", "", clean_transcript)
    # Fix literal string representation "\n" replacing to actual new lines
    clean_transcript = clean_transcript.replace("\\n", "\n")
    clean_transcript = re.sub(r"\n{3,}", "\n\n", clean_transcript).strip() + "\n"
    write_text(transcript_path, clean_transcript)
    log.info("Transcript saved: %s (%d chars)", transcript_path, len(clean_transcript))

    # ── Stage 4: TTS ──
    if not args.no_audio:
        log.info("Stage 4: synthesizing audio …")
        await synthesize(
            script_text,
            backend=backend,
            voice=voice,
            output_path=mp3_path,
            rate=rate,
            volume=volume,
            pitch=pitch,
            mood_presets=(mood_presets if isinstance(mood_presets, dict) else None),
            chunk_silence_ms=chunk_silence_ms,
            model_path=cosyvoice_model_path,
            speaker=cosyvoice_speaker,
        )
        log.info("Audio saved: %s", mp3_path)

    # ── Stage 5: Publish ──
    log.info("Stage 5: publishing …")
    notes_html = generate_show_notes_html(brief, episode_title=episode_title, episode_date=day)
    write_text(notes_path, notes_html)

    stories = brief.get("stories", [])
    show_desc_lines = [
        f"<p>{podcast_description}</p>" if podcast_description else "",
        "<ol>",
    ]
    for s in stories:
        if s.get("role") == "skip":
            continue
        raw_title = s.get("representative_title") or s.get("title") or ""
        safe_title = str(raw_title).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        items = s.get("items") or []
        first_item = items[0] if items else {}
        raw_source = first_item.get("source_name") or s.get("source_name") or ""
        safe_source = (
            str(raw_source).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        link = str(first_item.get("link") or s.get("link") or "")
        role_emoji = s.get("role_emoji", "")
        show_desc_lines.append(
            f'<li>{role_emoji} <a href="{link}">{safe_title}</a> <small>({safe_source})</small></li>'
        )
    show_desc_lines.append("</ol>")
    description_html = "\n".join([ln for ln in show_desc_lines if ln])

    if args.no_audio:
        log.info("--no-audio: skipping feed.xml update")
        return 0

    enclosure_url = f"{base_url}/episodes/{episode_id}.mp3"
    notes_url = f"{base_url}/episodes/{episode_id}.html"
    enclosure_length = mp3_path.stat().st_size

    published_at_iso = now.isoformat()
    pub_date_rfc = format_datetime(now)
    guid = f"{base_url}/episodes/{episode_id}"

    existing = _coerce_episode_list(read_json(episodes_index))
    existing = [ep for ep in existing if str(ep.get("id") or "") != episode_id]
    existing.append(
        {
            "id": episode_id,
            "title": episode_title,
            "description": description_html,
            "pubDate": pub_date_rfc,
            "guid": guid,
            "link": notes_url,
            "enclosure_url": enclosure_url,
            "enclosure_length": enclosure_length,
            "published_at_iso": published_at_iso,
        }
    )

    pruned = _prune_episodes(existing, keep_last=keep_last, episodes_dir=episodes_dir)
    write_json(episodes_index, pruned)

    sorted_eps = sorted(
        pruned,
        key=lambda ep: datetime.fromisoformat(
            str(ep.get("published_at_iso") or "1970-01-01T00:00:00+00:00")
        ),
        reverse=True,
    )

    feed_xml = build_feed_xml(
        base_url=base_url,
        podcast_title=podcast_title,
        podcast_description=podcast_description,
        podcast_language=podcast_language,
        podcast_author=podcast_author,
        podcast_category=podcast_category,
        podcast_explicit=podcast_explicit,
        episodes=sorted_eps,
    )
    write_text(site_dir / "feed.xml", feed_xml)
    build_index_html(site_dir, podcast_title, sorted_eps, base_url)

    log.info("Episode %s published successfully", episode_id)
    return 0


def entrypoint() -> int:
    """Synchronous entrypoint for console_scripts."""
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(entrypoint())
