"""Stage 2 — Thesis 提炼模块。

从主故事和支撑故事中提炼一句主线论点。
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field

from ai_news_podcast.business.pipeline.processor_types_service import ScoredStory

logger = logging.getLogger(__name__)

_DEFAULT_THESIS_TEMPLATES = [
    "今天的AI领域，{main_topic}正在重塑行业格局",
    "从{main_topic}到{sub_topic}，AI技术持续加速演进",
    "{main_topic}——这可能是本周最值得关注的AI趋势",
    "当{main_topic}遇上实际落地，我们看到了什么",
    "围绕{main_topic}，多方力量正在汇聚",
]


@dataclass
class ThesisTemplateRepository:
    """Repository for thesis templates — replaces module-level global state."""

    templates: list[str] = field(default_factory=lambda: list(_DEFAULT_THESIS_TEMPLATES))

    def get_random(self) -> str:
        """Return a random template."""
        return random.choice(self.templates)

    def set_templates(self, templates: list[str] | None = None) -> None:
        """Set templates from config."""
        self.templates = list(templates) if templates else list(_DEFAULT_THESIS_TEMPLATES)


_TOPIC_SPLIT_RE = re.compile(r"[：:，,。.、—|(/]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _short_topic(title: str, *, max_len: int = 20) -> str:
    """取标题的第一个短语做话题，中文按字符、英文按词边界截断（不再出现 "当Speeding up pla..."）。"""
    text = str(title or "").strip()
    if not text:
        return ""
    topic = _TOPIC_SPLIT_RE.split(text, maxsplit=1)[0].strip() or text
    if len(topic) <= max_len:
        return topic
    if _CJK_RE.search(topic):
        return topic[:max_len].rstrip()
    words = topic.split()
    kept: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + 1 > max_len:
            break
        kept.append(word)
        length += len(word) + 1
    return " ".join(kept) or " ".join(words[:3])


class ThesisExtractor:
    """Extract thesis from scored stories."""

    def __init__(self, template_repo: ThesisTemplateRepository | None = None) -> None:
        self.template_repo = template_repo or ThesisTemplateRepository()

    def extract(self, scored_stories: list[ScoredStory]) -> str:
        """从主故事和支撑故事中提炼一句主线论点。"""
        main_stories = [s for s in scored_stories if s.role == "main"]
        supporting = [s for s in scored_stories if s.role == "supporting"]

        if not main_stories:
            return "今天的AI领域动态丰富，多个方向齐头并进"

        main_topic = _short_topic(main_stories[0].representative_title, max_len=20)
        sub_topic = ""
        if supporting:
            sub_topic = _short_topic(supporting[0].representative_title, max_len=16)

        template = self.template_repo.get_random()
        return template.format(main_topic=main_topic, sub_topic=sub_topic or "产业应用")


# Backward compatibility: module-level functions
def _extract_thesis(scored_stories: list[ScoredStory]) -> str:
    """从主故事和支撑故事中提炼一句主线论点。"""
    return ThesisExtractor().extract(scored_stories)


def set_thesis_templates(templates: list[str] | None = None) -> None:
    """Set thesis templates from config (backward compatible)."""
    ThesisExtractor().template_repo.set_templates(templates)
