"""TTS 音频合成引擎：统一入口。

使用 TTSBackendFactory 路由到具体的后端实现。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ai_news_podcast.pipeline.tts_backends import TTSBackendFactory
from ai_news_podcast.pipeline.tts_parser import parse_dialogue_chunks

log = logging.getLogger(__name__)


async def synthesize(
    text: str,
    *,
    backend: str = "cosyvoice2",
    output_path: Path,
    bgm_path: str | None = None,
    **kwargs: Any,
) -> None:
    """Synthesize audio using the specified TTS backend.

    Args:
        text: Podcast script text with [Host A]/[Host B] markers.
        backend: TTS backend name (default: "cosyvoice2").
        output_path: Path to save the output MP3.
        bgm_path: Optional background music path.
        **kwargs: Additional arguments passed to the backend.
    """
    chunks = parse_dialogue_chunks(text)
    if not chunks:
        raise ValueError("Input text is empty after dialogue parsing")

    tts_backend = TTSBackendFactory.create(backend)
    await tts_backend.synthesize(
        text,
        output_path=output_path,
        bgm_path=bgm_path,
        **kwargs,
    )
