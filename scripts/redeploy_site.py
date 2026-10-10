"""Refresh the deployed static site (index.html + feed.xml + reports) from data.

Used by the `Redeploy Site` workflow to push front-end / page changes to GitHub
Pages without re-running the audio pipeline. Audio, per-episode show-notes HTML
and chunk dirs are preserved on gh-pages via the deploy action's keep_files.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ai_news_podcast.data.config.loader_dao import load_config
from ai_news_podcast.data.utils_dao import read_json
from ai_news_podcast.presentation.cli.episode_utils_controller import get_base_url
from ai_news_podcast.presentation.site_builder.html_gen_view import build_index_html
from ai_news_podcast.presentation.site_builder.rss_gen_view import build_feed_xml

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    cfg = load_config(ROOT / "config/config.yaml")
    base_url = get_base_url(cfg, os.environ.get("PODCAST_BASE_URL") or None)

    episodes = read_json(ROOT / str(cfg.build.episodes_index))
    if not isinstance(episodes, list):
        episodes = []
    episodes.sort(key=lambda ep: str(ep.get("published_at_iso") or ""), reverse=True)

    site_dir = ROOT / str(cfg.build.site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)

    build_index_html(site_dir, cfg.podcast.title, episodes, base_url, cfg)

    feed_xml = build_feed_xml(
        base_url=base_url,
        podcast_title=cfg.podcast.title,
        podcast_description=cfg.podcast.description,
        podcast_language=cfg.podcast.language,
        podcast_author=cfg.podcast.author,
        podcast_category=cfg.podcast.category,
        podcast_explicit=cfg.podcast.explicit,
        episodes=episodes,
    )
    (site_dir / "feed.xml").write_text(feed_xml, encoding="utf-8")

    reports_src = ROOT / "data/reports"
    if reports_src.exists():
        shutil.copytree(reports_src, site_dir / "reports", dirs_exist_ok=True)

    radar_src = ROOT / "data/gh_radar"
    if radar_src.exists():
        shutil.copytree(radar_src, site_dir / "radar", dirs_exist_ok=True)

    print(f"redeployed site: {len(episodes)} episodes, base_url={base_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
