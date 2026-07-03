"""Show notes generation for episodes.

Extracted from podcastwriter.py to follow Single Responsibility Principle.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _cn_date(dt: datetime) -> str:
    """Format datetime as Chinese date string."""
    return dt.strftime("%Y年%m月%d日")


def generate_show_notes(
    brief: dict[str, Any],
    *,
    episode_title: str,
    episode_date: datetime,
) -> str:
    """Generate Markdown show notes for an episode."""
    stories = brief.get("stories", [])
    thesis = brief.get("thesis", "")
    active = [s for s in stories if s.get("role") != "skip"]

    lines: list[str] = []
    lines.append(f"# {episode_title}")
    lines.append("")
    lines.append(f"**日期**: {_cn_date(episode_date)}")
    lines.append("")

    if thesis:
        lines.append(f"> {thesis}")
        lines.append("")

    for role, label in [
        ("main", "🔴 主要报道"),
        ("supporting", "🟡 支撑消息"),
        ("quick", "🟢 快讯"),
    ]:
        role_stories = [s for s in active if s.get("role") == role]
        if not role_stories:
            continue
        lines.append(f"## {label}")
        lines.append("")
        for story in role_stories:
            title = story.get("representative_title", "")
            context = story.get("context", {})
            summaries = context.get("factual_summary", [])
            total = story.get("total_score", 0)

            lines.append(f"### {title}")
            lines.append("")
            if summaries:
                lines.extend(f"- {s}" for s in summaries)
                lines.append("")
            if story.get("items"):
                lines.append("**来源链接：**")
                lines.append("")
                for item in story.get("items", [])[:5]:
                    name = item.get("source_name", "")
                    link = item.get("link", "")
                    lines.append(f"- [{name}]({link})")
                lines.append("")
            lines.append(f"*综合评分: {total}/15*")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*本期由 AI 自动生成，数据截至 {_cn_date(episode_date)}*")
    lines.append("")
    return "\n".join(lines)


def generate_show_notes_html(
    brief: dict[str, Any],
    *,
    episode_title: str,
    episode_date: datetime,
) -> str:
    """Generate HTML show notes for an episode."""
    stories = brief.get("stories", [])
    active = [s for s in stories if s.get("role") != "skip"]

    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    items_html: list[str] = []
    for story in active:
        title = _esc(story.get("representative_title", ""))
        role_emoji = story.get("role_emoji", "")
        if story.get("items"):
            link = story.get("items", [])[0].get("link", "")
            source = _esc(story.get("items", [])[0].get("source_name", ""))
            items_html.append(
                f'<li>{role_emoji} <a href="{link}">{title}</a> <small>({source})</small></li>'
            )

    date_text = _cn_date(episode_date)
    safe_title = _esc(episode_title)
    body = "\n".join(items_html)

    return (
        "<!doctype html>\n"
        '<html lang="zh-CN">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{safe_title}</title>\n"
        "  <style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;"
        "max-width:860px;margin:24px auto;padding:0 16px;line-height:1.6}"
        "li{margin:12px 0}small{color:#555}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{safe_title}</h1>\n"
        f"<p>{date_text}</p>\n"
        "<ol>\n"
        f"{body}\n"
        "</ol>\n"
        "</body>\n"
        "</html>\n"
    )
