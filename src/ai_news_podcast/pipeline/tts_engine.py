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
    import re

    # Check if script already has paralinguistic tags. If not, auto-annotate using Director Agent
    if not re.search(r"<[^>]+>", text):
        log.info("未检测到情感标签，正在启动 Director Agent 进行情感与非语言标签标注...")
        cfg = kwargs.get("cfg")
        if cfg:
            import dataclasses

            from ai_news_podcast.pipeline.llm_client import call_llm
            from ai_news_podcast.prompts import build_director_prompt

            if isinstance(cfg, dict):
                podcast_title = cfg.get("podcast", {}).get("title", "AI 每日先锋")
                llm_cfg = cfg.get("llm", {})
            else:
                podcast_title = "AI 每日先锋"
                if (
                    hasattr(cfg, "podcast")
                    and cfg.podcast
                    and hasattr(cfg.podcast, "title")
                    and cfg.podcast.title
                ):
                    podcast_title = cfg.podcast.title
                llm_cfg = dataclasses.asdict(cfg.llm) if hasattr(cfg, "llm") and cfg.llm else {}

            director_prompt = build_director_prompt(text, podcast_title)
            annotated_text = call_llm(director_prompt, llm_cfg)
            if annotated_text and annotated_text.strip():
                log.info("Director Agent 成功对剧本进行了音频情感标注")
                text = annotated_text
            else:
                log.warning("Director Agent 情感标注失败，降级为原纯净剧本进行合成")
        else:
            log.warning("未检测到 AppConfig 配置，跳过 Director 情感标注")

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
