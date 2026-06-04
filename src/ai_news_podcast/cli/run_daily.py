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

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ai_news_podcast.pipeline.runner import run_pipeline
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
            transcript_path=transcript_path,
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

    import shutil

    logo_src = root / "assets" / "logo.png"
    if logo_src.exists():
        shutil.copy(logo_src, site_dir / "logo.png")

    # Copy pipeline walkthrough and infographic for site integration
    infographic_src = root / "assets" / "pipeline_infographic.png"
    if infographic_src.exists():
        shutil.copy(infographic_src, site_dir / "pipeline_infographic.png")

    walkthrough_src = root / "docs" / "pipeline_walkthrough.md"
    if walkthrough_src.exists():
        shutil.copy(walkthrough_src, site_dir / "pipeline_walkthrough.md")

    reports_src = root / "data" / "reports"
    reports_dst = site_dir / "reports"
    if reports_src.exists():
        if reports_dst.exists():
            shutil.rmtree(reports_dst)
        shutil.copytree(reports_src, reports_dst)

    log.info("Episode %s published successfully", episode_id)
    return 0


def entrypoint() -> int:
    """Synchronous entrypoint for console_scripts."""
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(entrypoint())
