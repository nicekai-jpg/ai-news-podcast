"""Show notes generation for episodes.

Extracted from podcast_writer_service.py to follow Single Responsibility Principle.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _cn_date(dt: datetime) -> str:
    """Format datetime as Chinese date string."""
    return dt.strftime("%Y年%m月%d日")


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

    review = brief.get("podcast_review") or {}
    review_html = ""
    if review.get("show_notes"):
        review_html += f"<p>{_esc(str(review['show_notes']))}</p>\n"
    chapters = [c for c in (review.get("chapters") or []) if isinstance(c, dict)]
    if chapters:
        chapter_html = []
        for chapter in chapters:
            title = _esc(str(chapter.get("title") or ""))
            intro = _esc(str(chapter.get("intro") or ""))
            highlights = [str(h) for h in (chapter.get("highlights") or []) if str(h).strip()]
            if not title and not intro:
                continue
            highlight_html = ""
            if highlights:
                highlight_html = (
                    "<ul>" + "".join(f"<li>{_esc(h)}</li>" for h in highlights) + "</ul>"
                )
            chapter_html.append(
                f"<li><strong>{title}</strong>{(': ' + intro) if intro else ''}{highlight_html}</li>"
            )
        if chapter_html:
            review_html += "<h2>本期章节</h2>\n<ol>\n" + "\n".join(chapter_html) + "\n</ol>\n"

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
        f"{review_html}"
        "<h2>本期提到</h2>\n"
        "<ol>\n"
        f"{body}\n"
        "</ol>\n"
        "</body>\n"
        "</html>\n"
    )
