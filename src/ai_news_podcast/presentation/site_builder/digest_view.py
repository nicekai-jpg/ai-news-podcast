"""每期站点数据聚合：digest_{date}.json / calendar.json。

设计原则：
- 纯函数不读文件（输入全部由调用方传入），方便单测与回放；
- 绝不抛异常到 publish：单板块失败只降级 + warnings；
- 不写时间戳，保证同源重跑零 diff。
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any

from ai_news_podcast.presentation.site_builder.report_digest_view import parse_report_markdown
from ai_news_podcast.presentation.site_builder.sources_digest_view import build_sources_block

log = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_CUE_PATTERNS = (
    re.compile(r"说到这个|接下来|另外|再说"),
    re.compile(r"第[一二三四五六七八九十]条"),
    re.compile(r"还有几条|快速过一下"),
    re.compile(r"最后一条"),
)
_CHAPTER_GAP_SECONDS = 60.0


def _zh_date(date_str: str) -> str:
    dt = datetime.date.fromisoformat(date_str)
    return f"{dt.year}年{dt.month:02d}月{dt.day:02d}日"


def _fmt_duration(seconds: float) -> str:
    total = round(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def resolve_issue(issue_no: int | None) -> dict:
    """期号：只有真实持久化值才显示；v1 不持久化，UI 因此只显示日期（D4）。"""
    if issue_no:
        return {"number": int(issue_no), "label": f"第 {int(issue_no)} 期", "source": "persisted"}
    return {"number": None, "label": None, "source": "unavailable"}


def duration_stats(chunks: list[dict]) -> tuple[float, float]:
    """返回 (时间轴时长, 净语音时长)。"""
    if not chunks:
        return (0.0, 0.0)
    last = chunks[-1]
    timeline = float(last.get("start") or 0) + float(last.get("duration") or 0)
    speech = sum(float(c.get("duration") or 0) for c in chunks)
    return (timeline, speech)


def detect_chapters(chunks: list[dict]) -> list[dict]:
    """按转场词定位章节锚点；不足 2 个锚点时退化为固定三点。"""
    if not chunks:
        return []
    chapters: list[dict] = [
        {
            "title": _short_title(chunks[0]),
            "start": float(chunks[0]["start"]),
            "label": _fmt_duration(float(chunks[0]["start"])),
            "source": "cue_heuristic",
        }
    ]
    for chunk in chunks[1:]:
        text = str(chunk.get("text") or "")
        if not any(pattern.search(text) for pattern in _CUE_PATTERNS):
            continue
        start = float(chunk.get("start") or 0)
        if start - chapters[-1]["start"] < _CHAPTER_GAP_SECONDS:
            continue
        chapters.append(
            {
                "title": _short_title(chunk),
                "start": start,
                "label": _fmt_duration(start),
                "source": "cue_heuristic",
            }
        )
        if len(chapters) == 4:
            break
    if len(chapters) >= 2:
        return chapters
    last = chunks[-1]
    mid = chunks[len(chunks) // 2]
    return [
        {
            "title": "开场",
            "start": float(chunks[0]["start"]),
            "label": _fmt_duration(float(chunks[0]["start"])),
            "source": "fallback",
        },
        {
            "title": "中段",
            "start": float(mid["start"]),
            "label": _fmt_duration(float(mid["start"])),
            "source": "fallback",
        },
        {
            "title": "收尾",
            "start": float(last["start"]),
            "label": _fmt_duration(float(last["start"])),
            "source": "fallback",
        },
    ]


def _short_title(chunk: dict, limit: int = 12) -> str:
    text = re.sub(r"\s+", " ", str(chunk.get("text") or "")).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _section_cfg(cfg: Any, name: str) -> dict:
    """同时支持 dict 与 AppConfig 数据类；返回浅拷贝，避免调用方误改配置。"""
    if isinstance(cfg, dict):
        return dict(cfg.get(name) or {})
    model = getattr(cfg, name, None)
    return dict(vars(model)) if model is not None else {}


def _voice_labels(cfg: Any) -> dict[str, dict[str, str]]:
    from ai_news_podcast.presentation.site_builder.html_gen_view import _voice_labels as labels

    return labels(cfg)


def _hosts(cfg: Any) -> list[dict]:
    names = {"host_a": ("A", "苏晴", "female"), "host_b": ("B", "周航", "male")}
    voice_labels = _voice_labels(cfg)
    hosts = []
    for config_key, (key, name, gender) in names.items():
        variants = [
            {"id": variant, "label": label, "default": index == 0}
            for index, (variant, label) in enumerate(voice_labels.get(config_key, {}).items())
        ]
        hosts.append({"key": key, "name": name, "gender": gender, "variants": variants})
    return hosts


def _metrics(brief: dict, sources: dict, chunks: list[dict]) -> list[dict]:
    metadata = brief.get("metadata") or {}
    publishers = {item["source_name"] for item in sources["items"]}
    return [
        {
            "key": "deduped_items",
            "label": "今日入库",
            "value": int(metadata.get("total_deduped") or 0),
            "unit": "条",
            "source": "brief.metadata.total_deduped",
        },
        {
            "key": "main_stories",
            "label": "核心研判",
            "value": sources["total_main"],
            "unit": "条",
            "source": "brief.stories[role=main]",
        },
        {
            "key": "podcast_segments",
            "label": "双人对谈",
            "value": len(chunks),
            "unit": "段",
            "source": "playlist.chunks",
        },
        {
            "key": "media_outlets",
            "label": "覆盖媒体",
            "value": len(publishers),
            "unit": "家",
            "source": "brief.stories[].items[0].source_name",
        },
    ]


def _radar_block(radar: dict | None, date_str: str, cfg: Any) -> dict:
    rules_cfg = _section_cfg(cfg, "gh_radar")
    rules = {
        "min_stars": int(rules_cfg.get("min_stars", 500)),
        "created_window_days": int(rules_cfg.get("created_window_days", 30)),
        "recent_push_days": int(rules_cfg.get("recent_push_days", 21)),
        "repeat_window_days": int(rules_cfg.get("repeat_window_days", 30)),
        "probe_limit": int(rules_cfg.get("readme_probe_limit", rules_cfg.get("probe_limit", 12))),
        "pick_count": int(rules_cfg.get("pick_count", 1)),
        "runner_up_count": int(rules_cfg.get("runner_up_count", 2)),
    }
    if radar is None:
        return {
            "available": False,
            "status": "unavailable",
            "rules": rules,
            "pick": None,
            "runner_ups": [],
            "yesterday": None,
            "candidates": {
                "total": 0,
                "excluded_recent": 0,
                "fresh": None,
                "semantics": "unknown",
                "detail_available": False,
                "preview": None,
            },
            "message": "该日期暂无项目雷达",
        }
    meta = radar.get("meta") or {}
    projects = radar.get("projects") or []
    meta_keys = set(meta)
    semantics = "v4" if meta_keys & {"fresh_candidates", "pool_by_source"} else "pre-v4"
    yesterday = None
    if meta.get("yesterday_pick"):
        yesterday = {
            "repo": str(meta["yesterday_pick"]),
            "delta_stars": meta.get("yesterday_pick_delta"),
            "date": "",
        }
    status = "pick" if projects else ("degraded" if meta.get("degraded") else "empty")
    message = "今日候选池经多轮筛查，未检出达标项目" if not projects else "今日主推项目"
    return {
        "available": True,
        "status": status,
        "rules": rules,
        "pick": projects[0] if projects else None,
        "runner_ups": projects[1:],
        "yesterday": yesterday,
        "candidates": {
            "total": int(meta.get("candidates") or 0),
            "excluded_recent": int(meta.get("excluded_recent") or 0),
            "fresh": int(meta["fresh_candidates"]) if "fresh_candidates" in meta else None,
            "semantics": semantics,
            "detail_available": False,
            "preview": None,
        },
        "message": message,
    }


def build_digest(
    *,
    date_str: str,
    brief: dict | None,
    report_md: str | None,
    radar: dict | None,
    playlist: dict | None,
    episode: dict | None,
    cfg: Any,
    base_url: str,
) -> dict:
    """组装单期 digest；任何板块失败只降级，不抛异常。"""
    warnings: list[str] = []
    brief = brief or {}
    chunks = list((playlist or {}).get("chunks") or [])
    timeline, speech = duration_stats(chunks)

    try:
        sources = build_sources_block(brief.get("stories") or [])
        warnings.extend(sources.get("warnings") or [])
    except ValueError:
        log.warning("sources block failed for %s", date_str, exc_info=True)
        sources = {
            "total_all": 0,
            "total_main": 0,
            "shown": 0,
            "role_filter": "all",
            "groups": [],
            "items": [],
            "warnings": ["sources_build_failed"],
        }
        warnings.append("sources_build_failed")

    try:
        report = parse_report_markdown(report_md or "")
        if report is not None:
            report["source_file"] = f"data/reports/daily_report_{date_str}.md"
    except ValueError:
        log.warning("report parse failed for %s", date_str, exc_info=True)
        report = None
        warnings.append("report_parse_failed")

    radar_block = _radar_block(radar, date_str, cfg)
    if not radar_block["candidates"]["detail_available"]:
        warnings.append("radar_candidate_detail_unavailable")

    episode = episode or {}
    mp3_url = str(episode.get("enclosure_url") or f"{base_url}/episodes/{date_str}.mp3")
    lead_source = "report_lead" if report and report.get("lead") else "brief_thesis"
    lead_text = (report or {}).get("lead") or brief.get("thesis") or None
    issue = resolve_issue(episode.get("issue_no"))

    hosts = _hosts(cfg)
    variants_ok = all(host["variants"] for host in hosts)
    if not variants_ok:
        warnings.append("voice_variants_incomplete")

    return {
        "schema_version": _SCHEMA_VERSION,
        "date": date_str,
        "date_label": _zh_date(date_str),
        "issue": issue,
        "availability": {
            "podcast": bool(chunks),
            "transcript": bool(chunks),
            "playlist": bool(chunks),
            "report": report is not None,
            "sources": bool(sources["items"]),
            "radar": radar_block["available"],
        },
        "podcast": {
            "title": str(episode.get("title") or f"AI 新闻快报 | {date_str}"),
            "subtitle": brief.get("thesis") or None,
            "mp3_url": mp3_url,
            "duration_seconds": round(timeline, 3),
            "duration_label": _fmt_duration(timeline),
            "speech_seconds": round(speech, 3),
            "segment_count": len(chunks),
            "hosts": hosts,
            "transcript": {
                "format": "playlist",
                "segment_count": len(chunks),
                "segments_url": f"episodes/{date_str}/playlist.json",
                "text_url": f"episodes/{date_str}.txt",
            },
        },
        "lead": (
            {
                "text": lead_text,
                "source": lead_source,
                "source_file": f"data/reports/daily_report_{date_str}.md",
            }
            if lead_text
            else None
        ),
        "metrics": _metrics(brief, sources, chunks),
        "chapters": detect_chapters(chunks),
        "report": report,
        "sources": sources,
        "radar": radar_block,
        "meta": {
            "generated_from": {
                "brief": f"data/briefs/brief_{date_str}.json",
                "report": f"data/reports/daily_report_{date_str}.md",
                "radar": f"data/gh_radar/radar_{date_str}.json",
                "playlist": f"site/episodes/{date_str}/playlist.json",
                "episodes": "data/episodes.json",
            },
            "warnings": warnings,
        },
    }


def build_calendar(episodes: list[dict]) -> dict:
    """日期轴聚合文件：期号 + 标题 + 可用性（可用性由 digest 是否存在推断）。"""
    dates = []
    for episode in episodes:
        date_str = str(episode.get("id") or "")
        if not date_str:
            continue
        dates.append(
            {
                "date": date_str,
                "issue_number": None,  # D4：v1 不显示期号；字段保留给未来真实刊号
                "title": str(episode.get("title") or f"AI 新闻快报 | {date_str}"),
                "availability": {"podcast": True, "report": True, "sources": True, "radar": True},
            }
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "default_date": dates[0]["date"] if dates else None,
        "dates": dates,
    }


def write_digest_outputs(
    root: Path,
    episodes: list[dict],
    cfg: Any,
    base_url: str,
    *,
    site_dir: Path,
    episodes_dir: Path,
) -> list[Path]:
    """为 episodes 里每个日期生成 digest + calendar；单期失败跳过，绝不抛异常。"""
    data_dir = site_dir / "data"
    written: list[Path] = []
    kept: set[str] = set()
    for episode in episodes:
        date_str = str(episode.get("id") or "")
        if not date_str:
            continue
        kept.add(f"digest_{date_str}.json")
        try:
            brief = _read_json_or_none(root / "data" / "briefs" / f"brief_{date_str}.json")
            report_md = _read_text_or_none(
                root / "data" / "reports" / f"daily_report_{date_str}.md"
            )
            radar = _read_json_or_none(root / "data" / "gh_radar" / f"radar_{date_str}.json")
            playlist = _read_json_or_none(episodes_dir / date_str / "playlist.json")
            digest = build_digest(
                date_str=date_str,
                brief=brief,
                report_md=report_md,
                radar=radar,
                playlist=playlist,
                episode=episode,
                cfg=cfg,
                base_url=base_url,
            )
            target = data_dir / f"digest_{date_str}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_dump(digest), encoding="utf-8")
            written.append(target)
        except (OSError, ValueError, TypeError):
            log.warning("digest generation failed for %s", date_str, exc_info=True)

    calendar_path = data_dir / "calendar.json"
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    calendar_path.write_text(_dump(build_calendar(episodes)), encoding="utf-8")
    for stale in data_dir.glob("digest_*.json"):
        if stale.name not in kept:
            stale.unlink(missing_ok=True)
    return written


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _read_json_or_none(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text_or_none(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None
