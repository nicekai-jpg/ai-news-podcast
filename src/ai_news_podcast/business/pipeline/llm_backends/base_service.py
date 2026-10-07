"""LLM backend protocol — Adapter Pattern.

All LLM backends must implement this protocol.
"""

from __future__ import annotations

from typing import Protocol


class LLMBackend(Protocol):
    """Protocol for LLM backends."""

    def call(self, prompt: str) -> str | None:
        """Send a prompt to the LLM and return the response text.

        Args:
            prompt: The prompt text to send.

        Returns:
            The LLM response text, or None if the call failed.
        """
        ...
