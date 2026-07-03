"""Pipeline event types for Observer Pattern.

Events emitted during pipeline execution for monitoring and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class StageStarted:
    """Emitted when a pipeline stage starts."""

    stage: str
    episode_id: str
    timestamp: datetime


@dataclass(frozen=True)
class StageCompleted:
    """Emitted when a pipeline stage completes."""

    stage: str
    episode_id: str
    duration_ms: int
    result: Any | None = None


@dataclass(frozen=True)
class StageFailed:
    """Emitted when a pipeline stage fails."""

    stage: str
    episode_id: str
    error: str
    timestamp: datetime


@dataclass(frozen=True)
class ItemProcessed:
    """Emitted when a single item is processed."""

    stage: str
    item_type: str
    count: int


@dataclass(frozen=True)
class EpisodePublished:
    """Emitted when an episode is published."""

    episode_id: str
    timestamp: datetime


@dataclass(frozen=True)
class ReportGenerated:
    """Emitted when a daily report is generated."""

    episode_id: str
    timestamp: datetime


# Union type for all pipeline events
PipelineEvent = (
    StageStarted | StageCompleted | StageFailed | ItemProcessed | EpisodePublished | ReportGenerated
)
