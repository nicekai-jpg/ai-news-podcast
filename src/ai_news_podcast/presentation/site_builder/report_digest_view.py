"""日报 Markdown → 结构化 report 块（纯函数，无 I/O）。

只覆盖 2026-09 之后的标准格式；更老的格式（06-04..07-22）没有章节结构，
解析结果会退化为 raw_paragraphs，前端按纯段落渲染。
"""

from __future__ import annotations

import re

_LEAD_RE = re.compile(r"\*\*导语\*\*[：:]\s*(.+)")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
_BULLET_RE = re.compile(r"^[-*]\s+\*\*(.+?)\*\*[：:]\s*(.+)$")
_QUOTE_RE = re.compile(r"^>\s*(.+)$")
_BOLD_POINT_RE = re.compile(r"^\*\*(.+?)\*\*[：:]\s*(.+)$")
_QUOTE_LABEL_RE = re.compile(r"^\*\*.+?\*\*[：:]\s*")

_DEEP_DIVE_H2 = "重磅解读"
_FRONTIER_H2 = "前沿情报"
_RADAR_H2 = "项目雷达"

# 第三段小标题在历史样本里有 9 种写法，统一按关键词宽松匹配。
_IMPACT_KEYWORDS = ("影响", "意义", "冲击")
_BREAKTHROUGH_KEYWORDS = ("突破", "核心", "进展", "创新")


def _classify_point(label: str) -> str:
    if any(keyword in label for keyword in _BREAKTHROUGH_KEYWORDS):
        return "breakthrough"
    if any(keyword in label for keyword in _IMPACT_KEYWORDS):
        return "impact"
    return "background"


class _ReportParser:
    """按行推进的日报解析状态机；每个方法只做一件事，便于单测与维护。"""

    def __init__(self, char_count: int) -> None:
        self.char_count = char_count
        self.title = ""
        self.lead: str | None = None
        self.section = ""
        self.current_dive: dict | None = None
        self.deep_dives: list[dict] = []
        self.frontier: list[dict] = []
        self.raw_paragraphs: list[str] = []
        self.quote_lines: list[str] = []
        self.toc: list[dict] = []

    def feed(self, line: str) -> None:
        if not line or line in ("---", "***"):
            return
        heading = _HEADING_RE.match(line)
        if heading:
            self._feed_heading(len(heading.group(1)), heading.group(2).strip())
            return
        if self.section == _RADAR_H2:
            return
        if self._feed_lead(line) or self._feed_bullet(line):
            return
        if line.startswith(">"):
            self._feed_quote(line)
            return
        if self._feed_point(line):
            return
        self.raw_paragraphs.append(line)

    def _feed_heading(self, level: int, text: str) -> None:
        if level == 1:
            self.title = text
            return
        if level == 2:
            self.current_dive = None
            self.section = self._section_kind(text)
            return
        if self.section == _DEEP_DIVE_H2:
            self.current_dive = {
                "heading": text,
                "background": None,
                "breakthrough": None,
                "impact": None,
                "raw_sections": [],
            }
            self.deep_dives.append(self.current_dive)
            self.toc.append({"anchor": f"deep-dive-{len(self.deep_dives)}", "title": text})

    @staticmethod
    def _section_kind(text: str) -> str:
        if _DEEP_DIVE_H2 in text:
            return _DEEP_DIVE_H2
        if _FRONTIER_H2 in text:
            return _FRONTIER_H2
        if _RADAR_H2 in text:
            return _RADAR_H2
        return "other"

    def _feed_lead(self, line: str) -> bool:
        if self.lead is not None:
            return False
        match = _LEAD_RE.search(line)
        if not match:
            return False
        self.lead = match.group(1).strip()
        return True

    def _feed_bullet(self, line: str) -> bool:
        if self.section != _FRONTIER_H2:
            return False
        match = _BULLET_RE.match(line)
        if match:
            self.frontier.append(
                {"title": match.group(1).strip(), "summary": match.group(2).strip()}
            )
            return True
        if line.startswith(("-", "*")):
            self.frontier.append({"title": line[1:].strip(), "summary": ""})
            return True
        return False

    def _feed_quote(self, line: str) -> None:
        match = _QUOTE_RE.match(line)
        text = match.group(1).strip() if match else ""
        text = _QUOTE_LABEL_RE.sub("", text)
        if text:
            self.quote_lines.append(text)

    def _feed_point(self, line: str) -> bool:
        match = _BOLD_POINT_RE.match(line)
        if not match or self.current_dive is None:
            return False
        key = _classify_point(match.group(1))
        if self.current_dive[key] is None:
            self.current_dive[key] = match.group(2).strip()
        else:
            self.current_dive["raw_sections"].append(f"{match.group(1)}：{match.group(2)}")
        return True

    def result(self) -> dict:
        return {
            "title": self.title,
            "char_count": self.char_count,
            "reading_minutes": max(1, round(self.char_count / 400)),
            "lead": self.lead,
            "deep_dives": self.deep_dives,
            "frontier": self.frontier,
            "editor_verdict": " ".join(self.quote_lines).strip() or None,
            "toc": self.toc,
            "raw_paragraphs": self.raw_paragraphs,
        }


def parse_report_markdown(md: str) -> dict | None:
    """把日报 md 解析为 digest 的 report 块；空内容返回 None。"""
    if not md or not md.strip():
        return None
    parser = _ReportParser(char_count=len(md))
    for raw_line in md.splitlines():
        parser.feed(raw_line.strip())
    return parser.result()
