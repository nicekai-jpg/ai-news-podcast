"""Events package for ai-news-podcast.

Provides event types and event bus for pipeline monitoring.
"""

from ai_news_podcast.events.bus import event_bus
from ai_news_podcast.events.types import (
    EpisodePublished,
    ItemProcessed,
    PipelineEvent,
    ReportGenerated,
    StageCompleted,
    StageFailed,
    StageStarted,
)

__all__ = [
    "EpisodePublished",
    "ItemProcessed",
    "PipelineEvent",
    "ReportGenerated",
    "StageCompleted",
    "StageFailed",
    "StageStarted",
    "event_bus",
]
