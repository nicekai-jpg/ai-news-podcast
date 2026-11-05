"""日报 Markdown → 结构化 report 块（纯函数，无 I/O）。

只覆盖 2026-09 之后的标准格式；更老的格式（06-04..07-22）没有章节结构，
解析结果会退化为 raw_paragraphs，前端按纯段落渲染。
"""

from __future__ import annotations

import re

# 兼容两种写法：`**导语**：正文` 与 `**导语：** 正文`（2026-11-05 起报告用了后者）
_LEAD_RE = re.compile(r"\*\*导语[：:]?\*\*[：:]?\s*(.+)")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
_BULLET_RE = re.compile(r"^[-*]\s+\*\*(.+?)\*\*[：:]\s*(.+)$")
_QUOTE_RE = re.compile(r"^>\s*(.+)$")
_BOLD_POINT_RE = re.compile(r"^(?:[-*]\s+)?\*\*(.+?)\*\*[：:]\s*(.+)$")
_QUOTE_LABEL_RE = re.compile(r"^\*\*.+?\*\*[：:]\s*")
# 重磅解读标题可带分类前缀：[模型发布] Gemini 4 Carbon：…（2026-11 起）
_CATEGORY_RE = re.compile(r"^\[([^\]]{1,12})\]\s*(.+)$")

_DEEP_DIVE_H2 = "重磅解读"
_CN_NUMERALS = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")
_NUMBERED_HEADING_RE = re.compile(r"^([0-9]{1,2})[.、]\s*(.+)$")
_FRONTIER_H2 = "前沿情报"
_RADAR_H2 = "项目雷达"

# 第三段小标题在历史样本里有 9 种写法，统一按关键词宽松匹配。
_IMPACT_KEYWORDS = ("影响", "意义", "冲击")
_BREAKTHROUGH_KEYWORDS = ("突破", "核心", "进展", "创新")


def _strip_emphasis(text: str) -> str:
    """去掉要点里的行内 ** 强调标记（前端按纯文本渲染）。"""
    return text.replace("**", "").strip()


def _normalize_dive_heading(text: str) -> str:
    """ "1. 标题" / "1、标题" → "一、标题"（与设计稿编号风格一致）。"""
    match = _NUMBERED_HEADING_RE.match(text)
    if not match:
        return text
    index = int(match.group(1))
    if not 1 <= index <= len(_CN_NUMERALS):
        return text
    return f"{_CN_NUMERALS[index - 1]}、{match.group(2).strip()}"


def _join_rich_lines(lines: list[str]) -> str | None:
    """把引用块/导语拼接成带段落与列表结构的纯文本（前端按 \n 渲染段落/有序列表）。"""
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip() or None


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
        self._lead_open = False
        self._blank = False

    def feed(self, line: str) -> None:
        if not line:
            self._blank = True
            return
        if line in ("---", "***"):
            return
        heading = _HEADING_RE.match(line)
        if heading:
            self._feed_heading(len(heading.group(1)), heading.group(2).strip())
            return
        if self.section == _RADAR_H2:
            return
        if self._feed_content_line(line):
            return
        self._feed_fallback(line)

    def _feed_content_line(self, line: str) -> bool:
        """按优先级匹配内容行；返回 True 表示已消费。"""
        if self._feed_lead(line) or self._feed_lead_continuation(line):
            return True
        if self._feed_bullet(line):
            return True
        if line.startswith(">"):
            self._feed_quote(line)
            return True
        return self._feed_point(line)

    def _feed_fallback(self, line: str) -> None:
        """未匹配任何结构：有当前卡片则作为卡片正文，否则记为全局段落。"""
        if self.current_dive is not None:
            text = _strip_emphasis(line.lstrip("-* ").strip())
            if text:
                self.current_dive["raw_sections"].append(text)
            return
        self.raw_paragraphs.append(line)

    def _feed_heading(self, level: int, text: str) -> None:
        if level == 1:
            self.title = text
            return
        if level == 2:
            self.current_dive = None
            self._lead_open = False
            self.section = self._section_kind(text)
            return
        if self.section == _DEEP_DIVE_H2:
            text = _normalize_dive_heading(text)
            category = None
            # 归一化后形如「一、[模型发布] 标题」，分类前缀可能跟在中文序号后面
            category_match = _CATEGORY_RE.match(text) or re.match(
                r"^([一二三四五六七八九十]+、\s*)\[([^\]]{1,12})\]\s*(.+)$", text
            )
            if category_match:
                groups = category_match.groups()
                if len(groups) == 3:
                    category = groups[1].strip()
                    text = (groups[0] or "") + groups[2].strip()
                else:
                    category = groups[0].strip()
                    text = groups[1].strip()
            self.current_dive = {
                "heading": text,
                "category": category,
                "points": [],
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
        self.lead = _strip_emphasis(match.group(1))
        self._lead_open = True
        self._blank = False
        return True

    def _feed_lead_continuation(self, line: str) -> bool:
        """导语可以写多段（空行分隔），后续普通段落并入导语，直到下一个标题出现。"""
        if not self._lead_open or self.lead is None or self.current_dive is not None:
            return False
        if line.startswith((">", "-", "*", "#")):
            return False
        text = _strip_emphasis(line)
        if not text:
            return False
        self.lead += ("\n\n" if self._blank else " ") + text
        self._blank = False
        return True

    def _feed_bullet(self, line: str) -> bool:
        if self.section != _FRONTIER_H2:
            return False
        match = _BULLET_RE.match(line)
        if match:
            self.frontier.append(
                {
                    "title": _strip_emphasis(match.group(1)),
                    "summary": _strip_emphasis(match.group(2)),
                }
            )
            return True
        if line.startswith(("-", "*")):
            self.frontier.append({"title": _strip_emphasis(line[1:]), "summary": ""})
            return True
        return False

    def _feed_quote(self, line: str) -> None:
        match = _QUOTE_RE.match(line)
        text = match.group(1).strip() if match else ""
        text = _QUOTE_LABEL_RE.sub("", text)
        if not text:
            self.quote_lines.append("")
            self._blank = False
            return
        if self._blank and self.quote_lines and self.quote_lines[-1] != "":
            self.quote_lines.append("")
        self.quote_lines.append(text)
        self._blank = False

    def _feed_point(self, line: str) -> bool:
        match = _BOLD_POINT_RE.match(line)
        if not match or self.current_dive is None:
            return False
        label = _strip_emphasis(match.group(1))
        text = _strip_emphasis(match.group(2))
        self.current_dive["points"].append({"label": label, "text": text})
        key = _classify_point(label)
        if self.current_dive[key] is None:
            self.current_dive[key] = text
        else:
            self.current_dive["raw_sections"].append(f"{label}：{text}")
        return True

    def result(self) -> dict:
        return {
            "title": self.title,
            "char_count": self.char_count,
            "reading_minutes": max(1, round(self.char_count / 400)),
            "lead": self.lead,
            "deep_dives": self.deep_dives,
            "frontier": self.frontier,
            "editor_verdict": _join_rich_lines(self.quote_lines),
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
