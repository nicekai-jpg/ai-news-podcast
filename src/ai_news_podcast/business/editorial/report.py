"""Stage 4 附属产物：02 深度日报 Markdown 渲染。"""

from __future__ import annotations

from ai_news_podcast.business.contracts import Commentary


def render_daily_report(commentaries: list[Commentary], *, date_str: str = "") -> str:
    """把 Stage 4 点评渲染为 02 深度日报 Markdown。"""
    lines = [f"# 深度日报 {date_str}".strip(), ""]
    for index, item in enumerate(commentaries, start=1):
        lines.append(f"## {index}. {item.headline}")
        if item.one_liner:
            lines.append(item.one_liner)
        if item.facts:
            lines.append("")
            lines.append("**事实**")
            for fact in item.facts:
                source_note = "（单一来源）" if fact.single_source else ""
                lines.append(f"- {fact.text}{source_note}")
        for label, value in (
            ("为什么重要", item.why_matters),
            ("背景", item.background),
            ("影响", item.impact),
            ("反方观点", item.counterpoint or "暂无公开反方信息"),
            ("不确定项", item.uncertainty),
        ):
            if value:
                lines.append("")
                lines.append(f"**{label}**：{value}")
        if item.sources:
            lines.append("")
            lines.append(
                "来源："
                + "；".join(
                    str(source.get("label") or source.get("link") or "") for source in item.sources
                )
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"
