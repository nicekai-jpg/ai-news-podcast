"""Pure score-based material selection strategy."""

from __future__ import annotations

from datetime import datetime

from ai_news_podcast.business.pipeline.strategies.base_service import (
    MaterialSelectionStrategy,
    effective_score,
)


class PureScoreStrategy(MaterialSelectionStrategy):
    """Pure score-based selection: sort by score (with freshness penalty) descending, take top N."""

    def select(
        self,
        stories: list[dict],
        max_stories: int,
        reference_time: datetime | None = None,
    ) -> list[dict]:
        """Select top stories by score, penalising stale material."""
        active = [s for s in stories if isinstance(s, dict) and s.get("role") != "skip"]
        active.sort(key=lambda s: effective_score(s, reference_time), reverse=True)
        return active[:max_stories]
