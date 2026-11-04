"""Shared TTS data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DialogueChunk:
    host: str
    text: str
    voice: str | None = None
    # 章节标题（来自剧本里的 [CHAPTER] 标记；历史脚本/无标记时为 None）
    chapter: str | None = None
