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
        "智谱",
        "zhipu",
        "glm",
        "deepseek",
        "minimax",
        "kimi",
        "moonshot",
        "月之暗面",
        "qwen",
        "通义",
        "百川",
        "baichuan",
        "零一万物",
        "阶跃星辰",
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

    _ENTITY_MAP: ClassVar[dict[str, str]] = {
        "google": "谷歌",
        "谷歌": "谷歌",
        "microsoft": "微软",
        "微软": "微软",
        "nvidia": "英伟达",
        "英伟达": "英伟达",
        "apple": "苹果",
        "苹果": "苹果",
        "audi": "奥迪",
        "奥迪": "奥迪",
        "tesla": "特斯拉",
        "特斯拉": "特斯拉",
        "zhipu": "智谱",
        "智谱": "智谱",
        "glm": "智谱",
        "deepseek": "deepseek",
        "minimax": "minimax",
        "kimi": "月之暗面",
        "moonshot": "月之暗面",
        "月之暗面": "月之暗面",
        "qwen": "通义千问",
        "通义": "通义千问",
        "baichuan": "百川",
        "百川": "百川",
        "grok": "xAI",
        "xai": "xAI",
        "mistral": "Mistral",
        "llama": "Meta",
        "hunyuan": "腾讯",
        "bytedance": "字节跳动",
        "internlm": "上海AI Lab",
        "ernie": "百度",
    }

    def _get_story_entities(self, story: dict[str, Any]) -> set[str]:
        """Extract company/brand entities from a story title."""
        title = str(story.get("representative_title", "")).lower()
        entities: set[str] = set()
        for c in self.companies:
            if c in title:
                norm = self._ENTITY_MAP.get(c, c)
                entities.add(norm)
        return entities
