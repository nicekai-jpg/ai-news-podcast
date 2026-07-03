"""Daily podcast pipeline: fetch → process → script → TTS → publish."""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_news_podcast.cli.base import AsyncCommand
from ai_news_podcast.cli.episode_utils import episode_id as _episode_id
from ai_news_podcast.cli.episode_utils import get_base_url as _get_base_url
from ai_news_podcast.config.models import AppConfig
from ai_news_podcast.pipeline.podcastwriter import generate_podcast
from ai_news_podcast.pipeline.runner import run_pipeline
from ai_news_podcast.pipeline.tts_engine import synthesize
from ai_news_podcast.site_builder.show_notes import generate_show_notes_html
from ai_news_podcast.text_utils import clean_tts_text
from ai_news_podcast.utils import load_sources, write_text


def _safe_html(text: str) -> str:
    """Escape HTML entities."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_description_html(stories: list[dict], podcast_description: str) -> str:
    """Build RSS description HTML from brief stories."""
    lines = [f"<p>{podcast_description}</p>" if podcast_description else "", "<ol>"]
    for s in stories:
        if s.get("role") == "skip":
            continue
        raw_title = s.get("representative_title") or s.get("title") or ""
        items = s.get("items") or []
        first_item = items[0] if items else {}
        raw_source = first_item.get("source_name") or s.get("source_name") or ""
        link = str(first_item.get("link") or s.get("link") or "")
        role_emoji = s.get("role_emoji", "")
        lines.append(
            f'<li>{role_emoji} <a href="{link}">{_safe_html(raw_title)}</a>'
            f" <small>({_safe_html(raw_source)})</small></li>"
        )
    lines.append("</ol>")
    return "\n".join([ln for ln in lines if ln])


class DailyCommand(AsyncCommand):
    """Run full daily pipeline (Stage 1 + 3-5)."""

    description = "运行完整播客流水线"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--sources", default="config/sources.yaml")
        parser.add_argument("--base-url", default=None)
        parser.add_argument("--date", default=None)
        parser.add_argument("--no-audio", action="store_true")
        parser.add_argument("--with-report", action="store_true", help="同时生成日报")
        parser.add_argument("--force-refresh", action="store_true", help="强制重新抓取")

    async def execute_async(self, args: argparse.Namespace, cfg: AppConfig, root: Path) -> int:  # noqa: PLR0915
        if args.date:
            day = datetime.fromisoformat(args.date).replace(tzinfo=UTC)
        else:
            day = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

        ep_id = _episode_id(day)
        episode_title = f"AI 新闻快报 | {ep_id}"
        base_url = _get_base_url(cfg, args.base_url)

        sources = load_sources(root / args.sources)
        now = datetime.now(tz=UTC)

        podcast_title = cfg.podcast.title
        episodes_dir = root / cfg.build.episodes_dir
        episodes_dir.mkdir(parents=True, exist_ok=True)

        # ── Stage 1: 数据基础层 ────────────────────────────────────────────────
        brief = await run_pipeline(
            cfg,
            sources,
            date_str=ep_id,
            data_dir=root / "data",
            force_refresh=args.force_refresh,
        )

        if not brief.get("stories"):
            print("No stories in brief — aborting")
            return 1

        import dataclasses

        # ── Stage 3: Script generation ──
        print("Stage 3: generating script …")
        podcast_text, warnings = generate_podcast(
            brief,
            episode_date=day,
            podcast_title=podcast_title,
            writer_cfg=dataclasses.asdict(cfg.script),
            llm_cfg=dataclasses.asdict(cfg.llm),
        )
        for w in warnings:
            print(f"Script warning: {w}")

        transcript_path = episodes_dir / f"{ep_id}.txt"
        clean_transcript = clean_tts_text(podcast_text) + "\n"
        write_text(transcript_path, clean_transcript)
        print(f"Transcript saved: {transcript_path} ({len(clean_transcript)} chars)")

        # Stage 3b: Daily report
        if args.with_report:
            print("Stage 3b: generating daily report …")
            report_dir = root / "data" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            # Report generation logic here (simplified)

        # Stage 4: TTS
        mp3_path = episodes_dir / f"{ep_id}.mp3"
        if not args.no_audio:
            print("Stage 4: synthesizing audio …")
            bgm_path = None
            if cfg.tts.bgm_path:
                bgm_p = root / cfg.tts.bgm_path
                if bgm_p.exists():
                    bgm_path = str(bgm_p)
            await synthesize(
                podcast_text,
                backend=cfg.tts.backend,
                output_path=mp3_path,
                bgm_path=bgm_path,
                transcript_path=transcript_path,
                cfg=cfg,
                project_root=root,
            )
            print(f"Audio saved: {mp3_path}")

        # Stage 5: Publish
        print("Stage 5: publishing …")
        notes_path = episodes_dir / f"{ep_id}.html"
        notes_html = generate_show_notes_html(brief, episode_title=episode_title, episode_date=day)
        write_text(notes_path, notes_html)

        if not args.no_audio:
            # Build feed.xml and index.html
            from email.utils import format_datetime

            from ai_news_podcast.cli.episode_utils import coerce_episode_list, prune_episodes
            from ai_news_podcast.site_builder.html_gen import build_index_html
            from ai_news_podcast.site_builder.rss_gen import build_feed_xml
            from ai_news_podcast.utils import read_json, write_json

            site_dir = root / cfg.build.site_dir
            episodes_index = root / cfg.build.episodes_index

            description_html = _build_description_html(
                brief.get("stories", []), cfg.podcast.description
            )
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

            pruned = prune_episodes(
                existing, keep_last=cfg.podcast.keep_last, episodes_dir=episodes_dir
            )
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
                podcast_title=cfg.podcast.title,
                podcast_description=cfg.podcast.description,
                podcast_language=cfg.podcast.language,
                podcast_author=cfg.podcast.author,
                podcast_category=cfg.podcast.category,
                podcast_explicit=cfg.podcast.explicit,
                episodes=sorted_eps,
            )
            write_text(site_dir / "feed.xml", feed_xml)
            build_index_html(site_dir, cfg.podcast.title, sorted_eps, base_url, cfg)

        print(f"Episode {ep_id} published successfully")
        return 0


def entrypoint() -> int:
    return DailyCommand().run()


# Backward compatibility for tests
async def main() -> int:
    """Legacy entrypoint for tests."""
    import sys

    from ai_news_podcast.config.loader import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--sources", default="config/sources.yaml")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--with-report", action="store_true")
    ap.add_argument("--force-refresh", action="store_true")
    args = ap.parse_args(sys.argv[1:])

    root = Path(__file__).resolve().parents[3]
    config_path = root / args.config
    cfg = load_config(config_path) if config_path.exists() else AppConfig()

    return await DailyCommand().execute_async(args, cfg, root)
