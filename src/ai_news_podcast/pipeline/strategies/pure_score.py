"""Pure score-based material selection strategy."""

from __future__ import annotations

from ai_news_podcast.pipeline.strategies.base import MaterialSelectionStrategy


class PureScoreStrategy(MaterialSelectionStrategy):
    """Pure score-based selection: sort by total_score descending, take top N."""

    def select(self, stories: list[dict], max_stories: int) -> list[dict]:
        """Select top stories by score."""
        active = [s for s in stories if isinstance(s, dict) and s.get("role") != "skip"]
        active.sort(key=lambda s: s.get("total_score", 0), reverse=True)
        return active[:max_stories]
