import argparse
import asyncio
import json
import os
import re
import importlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class Story:
    source: str
    title: str
    link: str
    published_at: datetime
    summary: str
    category: str


def _read_yaml(path: Path) -> dict[str, Any]:
    yaml = importlib.import_module("yaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML object at {path}")
    return data


def _read_json(path: Path) -> Any:
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


def _strip_html(s: str) -> str:
    bs4 = importlib.import_module("bs4")
    BeautifulSoup = getattr(bs4, "BeautifulSoup")
    if not s:
        return ""
    if "<" not in s and "&" not in s:
        return re.sub(r"\s+", " ", s).strip()
    soup = BeautifulSoup(s, "html.parser")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _norm_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\W_]+", "", s)
    return s


def _parse_dt(entry: Any) -> datetime:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = getattr(entry, key, None)
        if parsed is not None:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
    return datetime.now(tz=timezone.utc)


def _safe_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    try:
        parsed = urlparse(u)
    except Exception:
        return ""
    if parsed.scheme not in ("http", "https"):
        return ""
    return u


def _fetch_feed(url: str, *, timeout_seconds: int, user_agent: str) -> Any:
    feedparser = importlib.import_module("feedparser")
    requests = importlib.import_module("requests")

    resp = requests.get(
        url,
        timeout=timeout_seconds,
        headers={"User-Agent": user_agent},
    )
    resp.raise_for_status()
    return feedparser.parse(resp.content)


def _collect_stories(
    sources: list[dict[str, Any]],
    *,
    timeout_seconds: int,
    user_agent: str,
    max_items_per_feed: int,
) -> list[Story]:
    out: list[Story] = []
    for src in sources:
        if not src.get("enabled", False):
            continue
        name = str(src.get("name") or "").strip()
        url = str(src.get("url") or "").strip()
        category = str(src.get("category") or "").strip() or "news"
        if not name or not url:
            continue
        try:
            feed = _fetch_feed(
                url, timeout_seconds=timeout_seconds, user_agent=user_agent
            )
        except Exception:
            continue

        entries = list(getattr(feed, "entries", []) or [])[:max_items_per_feed]
        for entry in entries:
            title = _strip_html(str(getattr(entry, "title", "") or "")).strip()
            link = _safe_url(str(getattr(entry, "link", "") or "")).strip()

            summary_raw = str(getattr(entry, "summary", "") or "") or str(
                getattr(entry, "description", "") or ""
            )
            summary = _strip_html(summary_raw)
            if len(summary) > 260:
                summary = summary[:257].rstrip() + "..."

            if not title or not link:
                continue

            published_at = _parse_dt(entry)
            out.append(
                Story(
                    source=name,
                    title=title,
                    link=link,
                    published_at=published_at,
                    summary=summary,
                    category=category,
                )
            )
    return out


def _score_story(story: Story, now: datetime) -> float:
    age_hours = max(0.0, (now - story.published_at).total_seconds() / 3600.0)
    recency = max(0.0, 72.0 - min(72.0, age_hours))
    title = story.title

    bonus = 0.0
    for kw in ("发布", "推出", "开源", "上线", "更新", "版本", "release", "launch"):
        if kw.lower() in title.lower():
            bonus += 3.0
            break
    for kw in ("论文", "arxiv", "benchmark", "sota", "研究", "新方法", "新技术"):
        if kw.lower() in title.lower():
            bonus += 2.0
            break
    for kw in ("访谈", "采访", "podcast", "interview"):
        if kw.lower() in title.lower():
            bonus += 1.5
            break
    for kw in ("政策", "监管", "安全", "合规", "诉讼", "事故"):
        if kw.lower() in title.lower():
            bonus += 1.0
            break

    return recency + bonus


def _select_stories(
    stories: list[Story],
    *,
    now: datetime,
    prefer_recent_hours: int,
    fallback_recent_hours: int,
    max_stories: int,
    per_feed_cap: int,
    include_keywords: list[str],
    exclude_keywords: list[str],
) -> list[Story]:
    def within(hours: int) -> list[Story]:
        cutoff = now - timedelta(hours=hours)
        return [s for s in stories if s.published_at >= cutoff]

    candidates = within(prefer_recent_hours)
    if len(candidates) < max_stories:
        candidates = within(fallback_recent_hours)

    inc = [str(k).strip() for k in include_keywords if str(k).strip()]
    exc = [str(k).strip() for k in exclude_keywords if str(k).strip()]
    if inc or exc:
        filtered: list[Story] = []
        for s in candidates:
            text = (s.title + " " + (s.summary or "")).casefold()
            if inc and not any(k.casefold() in text for k in inc):
                continue
            if exc and any(k.casefold() in text for k in exc):
                continue
            filtered.append(s)
        candidates = filtered

    seen: set[str] = set()
    deduped: list[Story] = []
    for s in sorted(candidates, key=lambda x: x.published_at, reverse=True):
        key = _norm_key(s.link) or _norm_key(s.title)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    scored = sorted(deduped, key=lambda x: _score_story(x, now), reverse=True)
    per_feed: dict[str, int] = {}
    picked: list[Story] = []
    for s in scored:
        if len(picked) >= max_stories:
            break
        per_feed[s.source] = per_feed.get(s.source, 0) + 1
        if per_feed[s.source] > per_feed_cap:
            continue
        picked.append(s)

    return picked


def _cn_date(dt: datetime) -> str:
    return f"{dt.year}年{dt.month}月{dt.day}日"


def _build_script(
    episode_date: datetime, stories: list[Story], podcast_title: str
) -> str:
    lines: list[str] = []
    lines.append(f"欢迎收听{podcast_title}。")
    lines.append(f"今天是{_cn_date(episode_date)}。")
    lines.append(f"下面是今天值得关注的AI动态，共{len(stories)}条。")
    for i, s in enumerate(stories, start=1):
        if i == 1:
            lead = "第一条"
        elif i == 2:
            lead = "第二条"
        elif i == 3:
            lead = "第三条"
        else:
            lead = "接下来"
        lines.append(f"{lead}，来自{s.source}：{s.title}。")
        if s.summary:
            lines.append(f"简要信息：{s.summary}。")
    lines.append("相关链接我都放在节目简介里。")
    lines.append("以上就是今天的更新，感谢收听。")
    return "\n".join(lines).strip() + "\n"


def _build_notes_html(
    episode_title: str, episode_date: datetime, stories: list[Story]
) -> str:
    items = []
    for s in stories:
        pub = s.published_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        summary = (
            (s.summary or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        title = s.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        items.append(
            "".join(
                [
                    "<li>",
                    f'<a href="{s.link}">{title}</a>',
                    f"<div><small>{s.source} · {pub}</small></div>",
                    (f"<div>{summary}</div>" if summary else ""),
                    "</li>",
                ]
            )
        )
    date_text = _cn_date(episode_date)
    items_html = "\n".join(items)
    safe_title = (
        episode_title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return (
        "<!doctype html>\n"
        '<html lang="zh-CN">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{safe_title}</title>\n"
        "  <style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;max-width:860px;margin:24px auto;padding:0 16px;line-height:1.6}li{margin:12px 0}small{color:#555}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{safe_title}</h1>\n"
        f"<p>{date_text}</p>\n"
        "<ol>\n"
        f"{items_html}\n"
        "</ol>\n"
        "</body>\n"
        "</html>\n"
    )


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


async def _tts_to_mp3(
    text: str, *, mp3_path: Path, voice: str, rate: str, volume: str, pitch: str
) -> None:
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = mp3_path.with_name(mp3_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    last_err: Optional[BaseException] = None
    for attempt in range(1, 4):
        edge_tts = importlib.import_module("edge_tts")
        comm = edge_tts.Communicate(
            text, voice=voice, rate=rate, volume=volume, pitch=pitch
        )
        try:
            await comm.save(str(tmp_path))
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("TTS produced empty audio file")
            tmp_path.replace(mp3_path)
            return
        except BaseException as e:
            last_err = e
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < 3:
                await asyncio.sleep(2 * attempt)
            else:
                break

    if last_err is not None:
        raise last_err
    raise RuntimeError("edge-tts failed")


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
    sel_cfg = cfg.get("selection", {})
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
    max_stories = int(podcast_cfg.get("max_stories", 10))

    voice = str(tts_cfg.get("voice") or "zh-CN-XiaoxiaoNeural")
    rate = str(tts_cfg.get("rate") or "+0%")
    volume = str(tts_cfg.get("volume") or "+0%")
    pitch = str(tts_cfg.get("pitch") or "+0Hz")

    timeout_seconds = int(fetch_cfg.get("timeout_seconds", 20))
    user_agent = str(fetch_cfg.get("user_agent") or "ai-news-podcast/0.1")
    max_items_per_feed = int(fetch_cfg.get("max_items_per_feed", 30))

    prefer_recent_hours = int(sel_cfg.get("prefer_recent_hours", 36))
    fallback_recent_hours = int(sel_cfg.get("fallback_recent_hours", 96))
    per_feed_cap = int(sel_cfg.get("per_feed_cap", 6))
    include_keywords = list(sel_cfg.get("include_keywords", []) or [])
    exclude_keywords = list(sel_cfg.get("exclude_keywords", []) or [])

    base_url = _get_base_url(cfg, args.base_url)

    stories_all = _collect_stories(
        sources,
        timeout_seconds=timeout_seconds,
        user_agent=user_agent,
        max_items_per_feed=max_items_per_feed,
    )
    picked = _select_stories(
        stories_all,
        now=now,
        prefer_recent_hours=prefer_recent_hours,
        fallback_recent_hours=fallback_recent_hours,
        max_stories=max_stories,
        per_feed_cap=per_feed_cap,
        include_keywords=include_keywords,
        exclude_keywords=exclude_keywords,
    )

    script_text = _build_script(day, picked, podcast_title)
    mp3_path = episodes_dir / f"{episode_id}.mp3"
    notes_path = episodes_dir / f"{episode_id}.html"
    transcript_path = episodes_dir / f"{episode_id}.txt"

    _write_text(transcript_path, script_text)

    if not args.no_audio:
        await _tts_to_mp3(
            script_text,
            mp3_path=mp3_path,
            voice=voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
        )

    notes_html = _build_notes_html(episode_title, day, picked)
    _write_text(notes_path, notes_html)

    show_desc_lines = [
        f"<p>{podcast_description}</p>" if podcast_description else "",
        "<ol>",
    ]
    for s in picked:
        safe_title = (
            s.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        safe_source = (
            s.source.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        show_desc_lines.append(
            f'<li><a href="{s.link}">{safe_title}</a> <small>({safe_source})</small></li>'
        )
    show_desc_lines.append("</ol>")
    description_html = "\n".join([l for l in show_desc_lines if l])

    if args.no_audio:
        if mp3_path.exists():
            mp3_path.unlink()
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

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
