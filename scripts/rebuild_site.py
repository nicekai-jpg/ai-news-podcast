import sys
import json
import shutil
import yaml
from pathlib import Path

# Add project src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ai_news_podcast.site_builder.html_gen import build_index_html
from ai_news_podcast.utils import read_json, read_yaml

def rebuild():
    print("Starting site rebuild...")
    cfg = read_yaml(project_root / "config/config.yaml")
    podcast_cfg = cfg.get("podcast", {})
    build_cfg = cfg.get("build", {})
    
    podcast_title = str(podcast_cfg.get("title") or "AI 新闻播客").strip()
    site_dir = project_root / str(build_cfg.get("site_dir") or "site")
    episodes_index = project_root / str(build_cfg.get("episodes_index") or "data/episodes.json")
    
    # Load episodes
    episodes = read_json(episodes_index)
    if not isinstance(episodes, list):
        episodes = []
        
    # Sort episodes by date reverse
    from datetime import datetime
    episodes_sorted = sorted(
        episodes,
        key=lambda ep: datetime.fromisoformat(
            str(ep.get("published_at_iso") or "1970-01-01T00:00:00+00:00")
        ),
        reverse=True,
    )
    
    processed_episodes = []
    for ep in episodes_sorted:
        ep_copy = ep.copy()
        url = ep_copy.get("enclosure_url", "")
        if "episodes/" in url:
            # Rewrite to relative path so it plays correctly in local preview
            ep_copy["enclosure_url"] = "./episodes/" + url.rsplit("/", 1)[-1]
        processed_episodes.append(ep_copy)

    print(f"Rebuilding index.html in {site_dir}...")
    build_index_html(site_dir, podcast_title, processed_episodes, "http://localhost:8000/site")
    
    # Now sync site to data/_preview
    preview_dir = project_root / "data/_preview"
    if preview_dir.exists():
        print(f"Syncing site directory to {preview_dir}...")
        # Copy index.html
        shutil.copy(site_dir / "index.html", preview_dir / "index.html")
        # Copy episodes files
        site_ep_dir = site_dir / "episodes"
        preview_ep_dir = preview_dir / "episodes"
        preview_ep_dir.mkdir(parents=True, exist_ok=True)
        for f in site_ep_dir.glob("*"):
            if f.is_file():
                shutil.copy(f, preview_ep_dir / f.name)
        # Copy reports
        site_rep_dir = site_dir / "reports"
        preview_rep_dir = preview_dir / "reports"
        if site_rep_dir.exists():
            preview_rep_dir.mkdir(parents=True, exist_ok=True)
            for f in site_rep_dir.glob("*"):
                if f.is_file():
                    shutil.copy(f, preview_rep_dir / f.name)
    
    print("Rebuild and sync complete!")

if __name__ == "__main__":
    rebuild()
