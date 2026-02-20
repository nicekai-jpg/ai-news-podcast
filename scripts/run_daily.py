"""Daily podcast pipeline: fetch → process → script → TTS → publish."""

import argparse
import asyncio
import json
import logging
import os
import re
import importlib
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any, Optional

from tts_engine import synthesize
from fetcher import fetch_all
from processor import process, save_brief
from scriptwriter import (
    generate_script,
    generate_show_notes_html,
)

log = logging.getLogger("run_daily")


def _read_yaml(path: Path) -> dict[str, Any]:
    yaml = importlib.import_module("yaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML object at {path}")
    return data


def _read_json(path: Path) -> Any:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_index_html(
    site_dir: Path,
    podcast_title: str,
    episodes: list[dict[str, Any]],
    base_url: str,
) -> None:
    ep_cards = []
    for ep in episodes[:30]:
        ep_id = ep.get("guid", "").rsplit("/", 1)[-1].replace(".mp3", "")
        title = ep.get("title", ep_id)
        desc = ep.get("description", "")
        pub = ep.get("pubDate", "")
        mp3 = ep.get("enclosure_url", f"{base_url}/episodes/{ep_id}.mp3")
        txt = f"{base_url}/episodes/{ep_id}.txt"
        size_mb = round(int(ep.get("enclosure_length", 0)) / 1_048_576, 1)

        desc_html = (
            desc.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

        ep_cards.append(
            f'<article class="ep-card">\n'
            f'  <div class="ep-header">\n'
            f'    <h3 class="ep-title">{title}</h3>\n'
            f'    <span class="ep-meta">{pub}{f" · {size_mb} MB" if size_mb else ""}</span>\n'
            f"  </div>\n"
            f'  <p class="ep-desc">{desc_html}</p>\n'
            f'  <audio controls preload="none" src="{mp3}"></audio>\n'
            f'  <div class="ep-links">\n'
            f'    <a href="{mp3}" download>⬇ 下载音频</a>\n'
            f'    <a href="{txt}" target="_blank">📄 文字稿</a>\n'
            f"  </div>\n"
            f"</article>"
        )

    cards_html = (
        "\n".join(ep_cards)
        if ep_cards
        else '<p class="empty">暂无节目，请稍后再来。</p>'
    )

    html = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{podcast_title}</title>
  <link rel="alternate" type="application/rss+xml" title="{podcast_title}" href="./feed.xml">
  <style>
    :root {{
      --bg: #0f0f14;
      --surface: #1a1a24;
      --surface2: #24243a;
      --accent: #7c6af6;
      --accent-glow: rgba(124,106,246,.25);
      --text: #e8e6f0;
      --text2: #9994b8;
      --border: #2e2e45;
      --radius: 14px;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.7;
      min-height: 100vh;
    }}
    .hero {{
      text-align: center;
      padding: 64px 24px 48px;
      background: linear-gradient(160deg, #1e1b4b 0%, #0f0f14 50%, #1a0a2e 100%);
      border-bottom: 1px solid var(--border);
    }}
    .hero-icon {{
      font-size: 56px;
      margin-bottom: 16px;
      display: block;
    }}
    .hero h1 {{
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -.02em;
      background: linear-gradient(135deg, #c4b5fd, #7c6af6, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    .hero p {{
      color: var(--text2);
      margin-top: 12px;
      font-size: 1.05rem;
    }}
    .hero-actions {{
      margin-top: 28px;
      display: flex;
      justify-content: center;
      gap: 14px;
      flex-wrap: wrap;
    }}
    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 10px 22px;
      border-radius: 999px;
      font-size: .92rem;
      font-weight: 500;
      text-decoration: none;
      transition: all .2s;
    }}
    .btn-primary {{
      background: var(--accent);
      color: #fff;
      box-shadow: 0 0 20px var(--accent-glow);
    }}
    .btn-primary:hover {{ background: #6b5ce7; transform: translateY(-1px); }}
    .btn-outline {{
      border: 1px solid var(--border);
      color: var(--text2);
      background: transparent;
    }}
    .btn-outline:hover {{ border-color: var(--accent); color: var(--accent); }}
    .container {{
      max-width: 780px;
      margin: 0 auto;
      padding: 40px 20px 80px;
    }}
    .section-title {{
      font-size: 1.15rem;
      font-weight: 600;
      color: var(--text2);
      margin-bottom: 24px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }}
    .ep-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      margin-bottom: 18px;
      transition: border-color .2s, box-shadow .2s;
    }}
    .ep-card:hover {{
      border-color: var(--accent);
      box-shadow: 0 0 30px var(--accent-glow);
    }}
    .ep-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .ep-title {{
      font-size: 1.08rem;
      font-weight: 600;
      color: var(--text);
      flex: 1;
    }}
    .ep-meta {{
      font-size: .82rem;
      color: var(--text2);
      white-space: nowrap;
      flex-shrink: 0;
    }}
    .ep-desc {{
      font-size: .9rem;
      color: var(--text2);
      margin: 12px 0 16px;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    audio {{
      width: 100%;
      height: 44px;
      border-radius: 8px;
      outline: none;
    }}
    audio::-webkit-media-controls-panel {{
      background: var(--surface2);
    }}
    .ep-links {{
      display: flex;
      gap: 20px;
      margin-top: 14px;
    }}
    .ep-links a {{
      font-size: .85rem;
      color: var(--accent);
      text-decoration: none;
      transition: opacity .2s;
    }}
    .ep-links a:hover {{ opacity: .8; text-decoration: underline; }}
    .empty {{
      text-align: center;
      color: var(--text2);
      padding: 60px 20px;
      font-size: 1.05rem;
    }}
    .footer {{
      text-align: center;
      padding: 32px 20px;
      font-size: .82rem;
      color: var(--text2);
      border-top: 1px solid var(--border);
    }}
    .footer a {{ color: var(--accent); text-decoration: none; }}
    .footer a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="hero">
    <span class="hero-icon">🧠</span>
    <h1>{podcast_title}</h1>
    <p>每日精选 AI 前沿资讯，用声音连接智能未来</p>
    <div class="hero-actions">
      <a class="btn btn-primary" href="./feed.xml">📡 RSS 订阅</a>
      <a class="btn btn-outline" href="./feed.xml" title="复制订阅链接到播客客户端">📋 复制订阅地址</a>
    </div>
  </div>

  <div class="container">
    <h2 class="section-title">📻 全部节目</h2>
    {cards_html}
  </div>

  <div class="footer">
    <p>{podcast_title} · 每日 08:30 自动更新</p>
    <p>订阅地址: <a href="./feed.xml">{base_url}/feed.xml</a></p>
  </div>
</body>
</html>'''

    _write_text(site_dir / "index.html", html)


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


def _build_feed_xml(
    *,
    base_url: str,
    podcast_title: str,
    podcast_description: str,
    podcast_language: str,
    podcast_author: str,
    podcast_category: str,
    podcast_explicit: bool,
    episodes: list[dict[str, Any]],
) -> str:
    import xml.etree.ElementTree as ET

    ns_atom = "http://www.w3.org/2005/Atom"
    ns_itunes = "http://www.itunes.com/dtds/podcast-1.0.dtd"
    ET.register_namespace("atom", ns_atom)
    ET.register_namespace("itunes", ns_itunes)

    rss = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(rss, "channel")

    ET.SubElement(channel, "title").text = podcast_title
    ET.SubElement(channel, "link").text = base_url + "/"
    ET.SubElement(channel, "description").text = podcast_description
    ET.SubElement(channel, "language").text = podcast_language
    ET.SubElement(channel, "lastBuildDate").text = format_datetime(
        datetime.now(tz=timezone.utc)
    )
    ET.SubElement(channel, f"{{{ns_itunes}}}author").text = podcast_author
    ET.SubElement(channel, f"{{{ns_itunes}}}explicit").text = (
        "yes" if podcast_explicit else "no"
    )
    ET.SubElement(channel, f"{{{ns_itunes}}}category", {"text": podcast_category})
    ET.SubElement(
        channel,
        f"{{{ns_atom}}}link",
        {
            "href": base_url + "/feed.xml",
            "rel": "self",
            "type": "application/rss+xml",
        },
    )

    for ep in episodes:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = str(ep["title"])
        ET.SubElement(item, "description").text = str(ep["description"])
        ET.SubElement(item, "pubDate").text = str(ep["pubDate"])
        guid = ET.SubElement(item, "guid", {"isPermaLink": "false"})
        guid.text = str(ep["guid"])
        ET.SubElement(item, "link").text = str(ep["link"])
        ET.SubElement(
            item,
            "enclosure",
            {
                "url": str(ep["enclosure_url"]),
                "length": str(ep["enclosure_length"]),
                "type": "audio/mpeg",
            },
        )

    xml_bytes = ET.tostring(rss, encoding="utf-8", xml_declaration=True)
    return xml_bytes.decode("utf-8") + "\n"


def _load_sources(sources_path: Path) -> list[dict[str, Any]]:
    data = _read_yaml(sources_path)
    sources = data.get("sources")
    if not isinstance(sources, list):
        raise ValueError("sources.yaml must contain a 'sources' list")
    out: list[dict[str, Any]] = []
    for src in sources:
        if isinstance(src, dict):
            out.append(src)
    return out


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
    keep = sorted_eps[:keep_last]
    keep_ids = {str(ep.get("id") or "") for ep in keep}

    for ep in sorted_eps[keep_last:]:
        eid = str(ep.get("id") or "")
        if not eid:
            continue
        mp3 = episodes_dir / f"{eid}.mp3"
        html = episodes_dir / f"{eid}.html"
        if mp3.exists():
            mp3.unlink()
        if html.exists():
            html.unlink()

    return [ep for ep in keep if str(ep.get("id") or "") in keep_ids]


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sources", default="sources.yaml")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = _read_yaml(root / args.config)
    sources = _load_sources(root / args.sources)

    podcast_cfg = cfg.get("podcast", {})
    tts_cfg = cfg.get("tts", {})
    fetch_cfg = cfg.get("fetch", {})
    processing_cfg = cfg.get("processing", {})
    script_cfg = cfg.get("script", {})
    build_cfg = cfg.get("build", {})

    site_dir = root / str(build_cfg.get("site_dir") or "docs")
    episodes_dir = root / str(build_cfg.get("episodes_dir") or "docs/episodes")
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
    script_text, warnings = generate_script(
        brief, episode_date=day, podcast_title=podcast_title, script_cfg=script_cfg
    )
    for w in warnings:
        log.warning("Script warning: %s", w)

    mp3_path = episodes_dir / f"{episode_id}.mp3"
    notes_path = episodes_dir / f"{episode_id}.html"
    transcript_path = episodes_dir / f"{episode_id}.txt"

    clean_transcript = re.sub(r"\[mood:[a-zA-Z0-9_-]+\]\s*", "", script_text)
    clean_transcript = re.sub(
        r"\[(?:FACT|INFERENCE|OPINION)\]\s*", "", clean_transcript
    )
    clean_transcript = re.sub(r"\n{3,}", "\n\n", clean_transcript).strip() + "\n"
    _write_text(transcript_path, clean_transcript)
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
    notes_html = generate_show_notes_html(
        brief, episode_title=episode_title, episode_date=day
    )
    _write_text(notes_path, notes_html)

    stories = brief.get("stories", [])
    show_desc_lines = [
        f"<p>{podcast_description}</p>" if podcast_description else "",
        "<ol>",
    ]
    for s in stories:
        if s.get("role") == "skip":
            continue
        raw_title = s.get("representative_title") or s.get("title") or ""
        safe_title = (
            str(raw_title)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        items = s.get("items") or []
        first_item = items[0] if items else {}
        raw_source = first_item.get("source_name") or s.get("source_name") or ""
        safe_source = (
            str(raw_source)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
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

    existing = _coerce_episode_list(_read_json(episodes_index))
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
    _write_json(episodes_index, pruned)

    sorted_eps = sorted(
        pruned,
        key=lambda ep: datetime.fromisoformat(
            str(ep.get("published_at_iso") or "1970-01-01T00:00:00+00:00")
        ),
        reverse=True,
    )

    feed_xml = _build_feed_xml(
        base_url=base_url,
        podcast_title=podcast_title,
        podcast_description=podcast_description,
        podcast_language=podcast_language,
        podcast_author=podcast_author,
        podcast_category=podcast_category,
        podcast_explicit=podcast_explicit,
        episodes=sorted_eps,
    )
    _write_text(site_dir / "feed.xml", feed_xml)
    _build_index_html(site_dir, podcast_title, sorted_eps, base_url)

    log.info("Episode %s published successfully", episode_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
