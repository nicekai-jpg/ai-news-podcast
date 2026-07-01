"""Shared TTS data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DialogueChunk:
    host: str
    text: str
    voice: str | None = None
