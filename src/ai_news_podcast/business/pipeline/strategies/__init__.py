"""Material selection strategies.

Provides StrategyRegistry and built-in strategies for material selection.
"""

from ai_news_podcast.business.pipeline.strategies.base_service import MaterialSelectionStrategy
from ai_news_podcast.business.pipeline.strategies.diversity_service import DiversityStrategy
from ai_news_podcast.business.pipeline.strategies.pure_score_service import PureScoreStrategy
from ai_news_podcast.business.pipeline.strategies.registry_service import StrategyRegistry

__all__ = [
    "DiversityStrategy",
    "MaterialSelectionStrategy",
    "PureScoreStrategy",
    "StrategyRegistry",
]
