"""Material selection strategy protocol — Strategy Pattern.

All material selection strategies must implement this protocol.
"""

from __future__ import annotations

from typing import Protocol


class MaterialSelectionStrategy(Protocol):
    """Protocol for material selection strategies."""

    def select(self, stories: list[dict], max_stories: int) -> list[dict]:
        """Select stories from the brief.

        Args:
            stories: List of story dicts from episode_brief.
            max_stories: Maximum number of stories to select.

        Returns:
            Selected stories in priority order.
        """
        ...
