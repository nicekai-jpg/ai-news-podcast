"""Material selection strategy registry.

Usage:
    from ai_news_podcast.business.pipeline.strategies import StrategyRegistry

    strategy = StrategyRegistry.get("score_diversity")
    selected = strategy.select(stories, max_stories=5)
"""

from __future__ import annotations

from typing import ClassVar

from ai_news_podcast.business.pipeline.strategies.base_service import MaterialSelectionStrategy
from ai_news_podcast.business.pipeline.strategies.diversity_service import DiversityStrategy
from ai_news_podcast.business.pipeline.strategies.pure_score_service import PureScoreStrategy


class StrategyRegistry:
    """Registry for material selection strategies."""

    _strategies: ClassVar[dict[str, MaterialSelectionStrategy]] = {
        "score_diversity": DiversityStrategy(),
        "pure_score": PureScoreStrategy(),
    }

    @classmethod
    def get(cls, name: str) -> MaterialSelectionStrategy:
        """Get a strategy by name.

        Args:
            name: Strategy name (e.g. "score_diversity" or "pure_score").

        Returns:
            MaterialSelectionStrategy instance.

        Raises:
            ValueError: If the strategy is not registered.
        """
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name]

    @classmethod
    def register(cls, name: str, strategy: MaterialSelectionStrategy) -> None:
        """Register a new strategy."""
        cls._strategies[name] = strategy
