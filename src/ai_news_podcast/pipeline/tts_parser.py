"""TTS 文本解析：对话标记解析与句子切分。"""

from __future__ import annotations

import re

from ai_news_podcast.pipeline.tts_types import DialogueChunk
from ai_news_podcast.text_utils import clean_tts_text


def parse_dialogue_chunks(
    text: str,
) -> list[DialogueChunk]:
    """解析 [Host A]/[Host B] 对话标记为 DialogueChunk 列表。"""
    marker_re = re.compile(r"\[Host\s*([AB])\]", re.IGNORECASE)
    chunks: list[DialogueChunk] = []
    current_host = "A"
    cursor = 0

    for m in marker_re.finditer(text):
        raw = text[cursor:m.start()].strip()
        if raw:
            cleaned = clean_tts_text(raw)
            if cleaned:
                chunks.append(DialogueChunk(host=current_host, text=cleaned))
        current_host = m.group(1).upper()
        cursor = m.end()

    tail = text[cursor:].strip()
    if tail:
        cleaned = clean_tts_text(tail)
        if cleaned:
            chunks.append(DialogueChunk(host=current_host, text=cleaned))
    return chunks


def split_text_into_sentences(text: str, max_chars: int = 80) -> list[str]:
    """将文本切分为较短的句子/短句，避免单次合成文本过长导致 CosyVoice 截断或语速失真。"""
    # 按照常见的标点符号进行切分，保留标点
    pattern = re.compile(r"([^，。！？；、,.!?;\s]+[，。！？；、,.!?;\s]*)")
    parts = pattern.findall(text)
    if not parts:
        return [text] if text.strip() else []

    sentences = []
    current = ""
    for part in parts:
        if len(current) + len(part) <= max_chars:
            current += part
        else:
            if current:
                sentences.append(current.strip())
            current = part
    if current:
        sentences.append(current.strip())
    return sentences
