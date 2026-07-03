"""Event bus — Observer Pattern implementation.

Usage:
    from ai_news_podcast.events.bus import event_bus
    from ai_news_podcast.events.types import StageStarted

    # Subscribe to events
    event_bus.subscribe(StageStarted, lambda e: print(f"Stage {e.stage} started"))

    # Emit events
    event_bus.emit(StageStarted(stage="fetch", episode_id="2024-03-15", timestamp=datetime.now()))
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ai_news_podcast.events.types import PipelineEvent

logger = logging.getLogger(__name__)


class EventBus:
    """Simple event bus for decoupled event communication."""

    def __init__(self) -> None:
        self._listeners: dict[type, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Callable[[Any], None]) -> None:
        """Subscribe a handler to an event type.

        Args:
            event_type: The event type to subscribe to.
            handler: Callback function to call when the event is emitted.
        """
        self._listeners[event_type].append(handler)

    def emit(self, event: PipelineEvent) -> None:
        """Emit an event to all subscribed handlers.

        Args:
            event: The event to emit.
        """
        event_type = type(event)
        for handler in self._listeners.get(event_type, []):
            try:
                handler(event)
            except Exception:
                logger.exception("Event handler failed for %s", event_type.__name__)

    def clear(self) -> None:
        """Remove all listeners."""
        self._listeners.clear()


# Global event bus instance
event_bus = EventBus()
