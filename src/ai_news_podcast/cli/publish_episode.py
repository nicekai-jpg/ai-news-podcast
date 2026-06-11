"""Stage 5: publish episode to site (feed.xml, index.html, assets)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from ai_news_podcast.cli.episode_utils import (
    coerce_episode_list,
    episode_id,
    get_base_url,
    prune_episodes,
)
from ai_news_podcast.pipeline.scriptwriter import generate_show_notes_html
from ai_news_podcast.site_builder.html_gen import build_index_html
from ai_news_podcast.site_builder.rss_gen import build_feed_xml
from ai_news_podcast.utils import load_sources, read_json, read_yaml, write_json, write_text

log = logging.getLogger("publish_episode")


def publish_episode(
    *,
    root: Path,
    cfg: dict[str, Any],
    brief: dict[str, Any],
    episode_id: str,
    episode_title: str,
    episode_date: datetime,
    base_url: str,
    podcast_cfg: dict[str, Any],
    build_cfg: dict[str, Any],
    now: Optional[datetime] = None,
) -> None:
    """Write show notes, update episodes index, feed.xml, and static site."""
    now = now or datetime.now(tz=timezone.utc)

    site_dir = root / str(build_cfg.get("site_dir") or "site")
    episodes_dir = root / str(build_cfg.get("episodes_dir") or "site/episodes")
    episodes_index = root / str(build_cfg.get("episodes_index") or "data/episodes.json")

    podcast_title = str(podcast_cfg.get("title") or "AI 新闻播客").strip()
    podcast_description = str(podcast_cfg.get("description") or "").strip()
    podcast_language = str(podcast_cfg.get("language") or "zh-cn").strip()
    podcast_author = str(podcast_cfg.get("author") or "").strip()
    podcast_category = str(podcast_cfg.get("category") or "Technology").strip()
    podcast_explicit = bool(podcast_cfg.get("explicit", False))
    keep_last = int(podcast_cfg.get("keep_last", 30))

    mp3_path = episodes_dir / f"{episode_id}.mp3"
    notes_path = episodes_dir / f"{episode_id}.html"

    notes_html = generate_show_notes_html(brief, episode_title=episode_title, episode_date=episode_date)
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

    enclosure_url = f"{base_url}/episodes/{episode_id}.mp3"
    notes_url = f"{base_url}/episodes/{episode_id}.html"
    enclosure_length = mp3_path.stat().st_size if mp3_path.exists() else 0

    published_at_iso = now.isoformat()
    pub_date_rfc = format_datetime(now)
    guid = f"{base_url}/episodes/{episode_id}"

    existing = coerce_episode_list(read_json(episodes_index))
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

    pruned = prune_episodes(existing, keep_last=keep_last, episodes_dir=episodes_dir)
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

    logo_src = root / "assets" / "logo.png"
    if logo_src.exists():
        shutil.copy(logo_src, site_dir / "logo.png")

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


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="Publish episode feed and site (Stage 5)")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--sources", default="config/sources.yaml")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--date", default=None, help="Episode date YYYY-MM-DD (Asia/Shanghai)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    cfg = read_yaml(root / args.config)
    load_sources(root / args.sources)  # validate sources file exists

    podcast_cfg = cfg.get("podcast", {})
    build_cfg = cfg.get("build", {})

    if args.date:
        day = datetime.fromisoformat(args.date).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    else:
        day = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

    ep_id = episode_id(day)
    episode_title = f"AI 新闻快报 | {ep_id}"
    base_url = get_base_url(cfg, args.base_url)

    brief_path = root / "data" / "briefs" / f"brief_{ep_id}.json"
    if not brief_path.exists():
        log.error("Brief not found: %s", brief_path)
        return 1

    brief = read_json(brief_path)
    mp3_path = root / str(build_cfg.get("episodes_dir") or "site/episodes") / f"{ep_id}.mp3"
    if not mp3_path.exists():
        log.error("MP3 not found: %s", mp3_path)
        return 1

    publish_episode(
        root=root,
        cfg=cfg,
        brief=brief,
        episode_id=ep_id,
        episode_title=episode_title,
        episode_date=day,
        base_url=base_url,
        podcast_cfg=podcast_cfg,
        build_cfg=build_cfg,
    )
    log.info("Episode %s published", ep_id)
    return 0


def entrypoint() -> int:
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(entrypoint())
