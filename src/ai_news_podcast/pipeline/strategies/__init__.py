"""Material selection strategies.

Provides StrategyRegistry and built-in strategies for material selection.
"""

from ai_news_podcast.pipeline.strategies.base import MaterialSelectionStrategy
from ai_news_podcast.pipeline.strategies.diversity import DiversityStrategy
from ai_news_podcast.pipeline.strategies.pure_score import PureScoreStrategy
from ai_news_podcast.pipeline.strategies.registry import StrategyRegistry

__all__ = [
    "DiversityStrategy",
    "MaterialSelectionStrategy",
    "PureScoreStrategy",
    "StrategyRegistry",
]
