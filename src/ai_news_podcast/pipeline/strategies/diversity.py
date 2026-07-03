"""Score-based material selection strategy with diversity penalty."""

from __future__ import annotations

from typing import Any, ClassVar

from ai_news_podcast.pipeline.strategies.base import MaterialSelectionStrategy


class DiversityStrategy(MaterialSelectionStrategy):
    """MMR-like diversity strategy: penalize stories with the same entity."""

    # Default companies for entity extraction
    _DEFAULT_COMPANIES: ClassVar[list[str]] = [
        "谷歌",
        "google",
        "openai",
        "微软",
        "microsoft",
        "英伟达",
        "nvidia",
        "苹果",
        "apple",
        "meta",
        "anthropic",
        "claude",
        "字节",
        "腾讯",
        "百度",
        "阿里",
        "华为",
        "奥迪",
        "audi",
        "特斯拉",
        "tesla",
    ]

    def __init__(self, companies: list[str] | None = None, penalty: int = 3) -> None:
        self.companies = companies or self._DEFAULT_COMPANIES
        self.penalty = penalty

    def select(self, stories: list[dict], max_stories: int) -> list[dict]:
        """Select stories with diversity penalty."""
        active = [s for s in stories if isinstance(s, dict) and s.get("role") != "skip"]
        selected: list[dict] = []
        entity_counts: dict[str, int] = {}
        candidates = [dict(s) for s in active]

        while len(selected) < max_stories and candidates:
            for c in candidates:
                orig_score = c.get("total_score", 0)
                c_entities = self._get_story_entities(c)
                penalty = sum(self.penalty * entity_counts.get(ent, 0) for ent in c_entities)
                c["_temp_score"] = orig_score - penalty

            candidates.sort(key=lambda x: x.get("_temp_score", 0), reverse=True)
            best = candidates.pop(0)
            selected.append(best)

            for ent in self._get_story_entities(best):
                entity_counts[ent] = entity_counts.get(ent, 0) + 1

        return selected

    def _get_story_entities(self, story: dict[str, Any]) -> set[str]:
        """Extract company/brand entities from a story title."""
        title = str(story.get("representative_title", "")).lower()
        entities: set[str] = set()
        for c in self.companies:
            if c in title:
                # Normalize entity names
                norm = c
                if c in ("google", "谷歌"):
                    norm = "谷歌"
                elif c in ("microsoft", "微软"):
                    norm = "微软"
                elif c in ("nvidia", "英伟达"):
                    norm = "英伟达"
                elif c in ("apple", "苹果"):
                    norm = "苹果"
                elif c in ("audi", "奥迪"):
                    norm = "奥迪"
                elif c in ("tesla", "特斯拉"):
                    norm = "特斯拉"
                entities.add(norm)
        return entities
