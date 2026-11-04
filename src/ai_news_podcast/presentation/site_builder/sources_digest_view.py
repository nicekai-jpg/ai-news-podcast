"""brief.stories → digest 的 sources 块（纯函数，无 I/O）。"""

from __future__ import annotations

from collections import Counter
from urllib.parse import urlparse

ROLE_LABELS = {"main": "核心信源", "supporting": "前沿参考", "quick": "快讯"}
_MAX_ITEMS = 150


def _host(link: str) -> str:
    try:
        return (urlparse(link).hostname or "").lower()
    except ValueError:
        return ""


def classify_group(source_name: str, link: str) -> str:
    """按 link host 优先、source_name 兜底，归到设计稿的 6 个筛选组。"""
    host = _host(link)
    name = source_name or ""
    if host.endswith("arxiv.org") or name.startswith("arXiv"):
        return "arXiv 论文"
    if host == "openai.com" or host.endswith(".openai.com") or name.startswith("OpenAI"):
        return "OpenAI"
    if host == "blog.google" or name in ("Google AI Blog", "DeepMind Blog"):
        return "Google"
    if host == "blogs.nvidia.com" or "NVIDIA" in name:
        return "NVIDIA"
    if host == "news.ycombinator.com" or name == "Hacker News":
        return "Hacker News"
    return "科技媒体"


def build_sources_block(stories: list[dict], role_filter: str = "all") -> dict:
    """把非 skip 的 story 转成来源条目；role_filter=all 时保留三类角色。"""
    items: list[dict] = []
    warnings: list[str] = []
    for story in stories:
        role = str(story.get("role") or "")
        if role == "skip":
            continue
        if role_filter not in ("all", role):
            continue
        first = (story.get("items") or [{}])[0]
        title = str(story.get("representative_title") or first.get("title") or "").strip()
        link = str(first.get("link") or "").strip()
        if not title or not link:
            continue
        source_name = str(first.get("source_name") or _host(link)).strip()
        group = classify_group(source_name, link)
        items.append(
            {
                "rank": len(items) + 1,
                "role": role if role in ROLE_LABELS else "quick",
                "role_label": ROLE_LABELS.get(role, "快讯"),
                "publisher": "arXiv 论文" if group == "arXiv 论文" else source_name,
                "group": group,
                "source_name": source_name,
                "category": str(first.get("source_category") or ""),
                "title": title,
                "link": link,
                "published_at": first.get("published_at"),
            }
        )

    if len(items) > _MAX_ITEMS:
        items = items[:_MAX_ITEMS]
        warnings.append("sources_truncated")

    counts = Counter(item["group"] for item in items)
    groups = [
        {"publisher": publisher, "group": publisher, "count": count}
        for publisher, count in counts.most_common()
    ]
    return {
        "total_all": len(items),
        "total_main": sum(1 for item in items if item["role"] == "main"),
        "shown": len(items),
        "role_filter": role_filter,
        "groups": groups,
        "items": items,
        "warnings": warnings,
    }
