"""Stage 5 CLI entry: podcast-publish.

Publish episode to site (feed.xml, index.html, assets).
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_news_podcast.cli.base import AsyncCommand
from ai_news_podcast.cli.episode_utils import (
    coerce_episode_list,
    episode_id,
    get_base_url,
    prune_episodes,
)
from ai_news_podcast.config.models import AppConfig
from ai_news_podcast.site_builder.html_gen import build_index_html
from ai_news_podcast.site_builder.rss_gen import build_feed_xml
from ai_news_podcast.site_builder.show_notes import generate_show_notes_html
from ai_news_podcast.utils import read_json, write_json, write_text


class PublishCommand(AsyncCommand):
    """Publish episode feed and site."""

    description = "发布播客站点 (Stage 5)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--sources", default="config/sources.yaml")
        parser.add_argument("--base-url", default=None)
        parser.add_argument("--date", default=None, help="Episode date YYYY-MM-DD")

    async def execute_async(self, args: argparse.Namespace, cfg: AppConfig, root: Path) -> int:  # noqa: PLR0915
        from email.utils import format_datetime

        if args.date:
            day = datetime.fromisoformat(args.date).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        else:
            day = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

        ep_id = episode_id(day)
        episode_title = f"AI 新闻快报 | {ep_id}"
        base_url = get_base_url(cfg, args.base_url)

        brief_path = root / "data" / "briefs" / f"brief_{ep_id}.json"
        if not brief_path.exists():
            print(f"Brief not found: {brief_path}")
            return 1

        brief = read_json(brief_path)
        mp3_path = root / cfg.build.episodes_dir / f"{ep_id}.mp3"
        if not mp3_path.exists():
            print(f"MP3 not found: {mp3_path}")
            return 1

        now = datetime.now(tz=UTC)
        site_dir = root / cfg.build.site_dir
        episodes_dir = root / cfg.build.episodes_dir
        episodes_index = root / cfg.build.episodes_index

        podcast_title = cfg.podcast.title
        podcast_description = cfg.podcast.description
        podcast_language = cfg.podcast.language
        podcast_author = cfg.podcast.author
        podcast_category = cfg.podcast.category
        podcast_explicit = cfg.podcast.explicit
        keep_last = cfg.podcast.keep_last

        notes_path = episodes_dir / f"{ep_id}.html"
        notes_html = generate_show_notes_html(brief, episode_title=episode_title, episode_date=day)
        write_text(notes_path, notes_html)

        # Build RSS description
        lines = [f"<p>{podcast_description}</p>" if podcast_description else "", "<ol>"]
        for s in brief.get("stories", []):
            if s.get("role") == "skip":
                continue
            raw_title = s.get("representative_title") or s.get("title") or ""
            items = s.get("items") or []
            first_item = items[0] if items else {}
            raw_source = first_item.get("source_name") or s.get("source_name") or ""
            link = str(first_item.get("link") or s.get("link") or "")
            role_emoji = s.get("role_emoji", "")
            lines.append(
                f'<li>{role_emoji} <a href="{link}">{raw_title}</a>'
                f" <small>({raw_source})</small></li>"
            )
        lines.append("</ol>")
        description_html = "\n".join([ln for ln in lines if ln])

        enclosure_url = f"{base_url}/episodes/{ep_id}.mp3"
        notes_url = f"{base_url}/episodes/{ep_id}.html"
        enclosure_length = mp3_path.stat().st_size if mp3_path.exists() else 0

        published_at_iso = now.isoformat()
        pub_date_rfc = format_datetime(now)
        guid = f"{base_url}/episodes/{ep_id}"

        existing = coerce_episode_list(read_json(episodes_index))
        existing = [ep for ep in existing if str(ep.get("id") or "") != ep_id]
        existing.append(
            {
                "id": ep_id,
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
        build_index_html(site_dir, podcast_title, sorted_eps, base_url, cfg)

        print(f"Episode {ep_id} published")
        return 0


def entrypoint() -> int:
    return PublishCommand().run()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
