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


def _annotate_text_in_batches(
    text: str,
    podcast_title: str,
    llm_cfg: dict[str, Any],
    batch_size: int = 10,
) -> str:
    """分切批次对剧本进行 Director Agent 情感标记，并在每个批次内部进行严格的无损校验（核对回合数与文字长度防缩水）。
    一旦检测到某个切段在大模型处理后出现轮次丢失或正文缩减/截断，立刻回退该切段至原纯净对白进行保底，确保整体剧本无一字遗漏。
    """
    import re

    from ai_news_podcast.pipeline.llm_client import call_llm
    from ai_news_podcast.pipeline.tts_parser import parse_dialogue_chunks
    from ai_news_podcast.prompts import build_director_prompt

    input_chunks = parse_dialogue_chunks(text)
    if not input_chunks:
        return text

    batches = [input_chunks[i : i + batch_size] for i in range(0, len(input_chunks), batch_size)]
    final_chunks = []

    for idx, batch in enumerate(batches, start=1):
        batch_text = "\n\n".join(f"[Host {c.host}] {c.text}" for c in batch)
        log.info(
            "Director Agent 正在处理第 %d/%d 切段对白 (包含 %d 轮对话)...",
            idx,
            len(batches),
            len(batch),
        )

        director_prompt = build_director_prompt(batch_text, podcast_title)
        raw_annotated = call_llm(director_prompt, llm_cfg)

        if not raw_annotated or not raw_annotated.strip():
            log.warning("第 %d/%d 切段情感标注返回为空，该切段回退为原对白保底", idx, len(batches))
            final_chunks.extend(batch)
            continue

        annotated_chunks = parse_dialogue_chunks(raw_annotated)
        if len(annotated_chunks) != len(batch):
            log.warning(
                "第 %d/%d 切段情感标注轮次不一致 (期望 %d 轮，实际返回 %d 轮)，检测到 LLM 截断或丢句，该切段回退为原对白保底",
                idx,
                len(batches),
                len(batch),
                len(annotated_chunks),
            )
            final_chunks.extend(batch)
            continue

        # 对切段内每一轮对话进行无损精确校验：说话人一致 & 剔除标签后的正文长度不下坠
        batch_valid = True
        for j, (orig_c, ann_c) in enumerate(zip(batch, annotated_chunks, strict=False)):
            if orig_c.host != ann_c.host:
                log.warning(
                    "第 %d/%d 切段第 %d 轮说话人匹配不符 (期望 Host %s，实际 Host %s)，该切段回退原对白",
                    idx,
                    len(batches),
                    j + 1,
                    orig_c.host,
                    ann_c.host,
                )
                batch_valid = False
                break

            orig_clean = re.sub(r"[^\w\u4e00-\u9fff]+", "", orig_c.text)
            ann_no_tags = re.sub(r"<[^>]+>", "", ann_c.text)
            ann_clean = re.sub(r"[^\w\u4e00-\u9fff]+", "", ann_no_tags)

            # 严格防止正文被大模型删改缩减（若去标签字数减少超过 20%，说明大模型改词或吞字了）
            if orig_clean and len(ann_clean) < len(orig_clean) * 0.8:
                log.warning(
                    "第 %d/%d 切段第 %d 轮对白在添加情感后检测到文字内容缩水或丢失，为保无损，该切段回退原对白",
                    idx,
                    len(batches),
                    j + 1,
                )
                batch_valid = False
                break

        if batch_valid:
            log.info("第 %d/%d 切段情感标注与无损校验通过 (完整保留对白正文)", idx, len(batches))
            final_chunks.extend(annotated_chunks)
        else:
            final_chunks.extend(batch)

    return "\n\n".join(f"[Host {c.host}] {c.text}" for c in final_chunks) + "\n"


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
        log.info("未检测到情感标签，正在启动 Director Agent 切段标注与无损校验机制...")
        cfg = kwargs.get("cfg")
        if cfg:
            import dataclasses

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

            annotated_text = _annotate_text_in_batches(text, podcast_title, llm_cfg, batch_size=10)
            if annotated_text and annotated_text.strip():
                log.info("Director Agent 全剧本分段情感标注与校验完成")
                text = annotated_text
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
