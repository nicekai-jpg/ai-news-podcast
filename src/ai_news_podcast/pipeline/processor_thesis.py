"""Stage 2 — Thesis 提炼模块。

从主故事和支撑故事中提炼一句主线论点。
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

from ai_news_podcast.pipeline.processor_types import ScoredStory

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

        main_topic = main_stories[0].representative_title
        # 简化：取前 15 字作为话题
        if len(main_topic) > 15:
            main_topic = main_topic[:15] + "..."

        sub_topic = ""
        if supporting:
            sub_topic = supporting[0].representative_title
            if len(sub_topic) > 12:
                sub_topic = sub_topic[:12] + "..."

        template = self.template_repo.get_random()
        return template.format(main_topic=main_topic, sub_topic=sub_topic or "产业应用")


# Backward compatibility: module-level functions
def _extract_thesis(scored_stories: list[ScoredStory]) -> str:
    """从主故事和支撑故事中提炼一句主线论点。"""
    return ThesisExtractor().extract(scored_stories)


def set_thesis_templates(templates: list[str] | None = None) -> None:
    """Set thesis templates from config (backward compatible)."""
    ThesisExtractor().template_repo.set_templates(templates)
