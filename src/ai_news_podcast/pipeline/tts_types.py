"""Shared TTS data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DialogueChunk:
    host: str
    text: str
    voice: Optional[str] = None
