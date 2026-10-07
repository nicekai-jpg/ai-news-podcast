"""TTS 文本解析：对话标记解析与句子切分。"""

from __future__ import annotations

import re

from ai_news_podcast.pipeline.tts_types import DialogueChunk
from ai_news_podcast.text_utils import clean_tts_text

# 至少含一个可发音字符(汉字/字母/数字)才算有效内容;纯标点/符号/emoji 会被
# CosyVoice 内嵌的 wetext 英文正则化 tokenize 成空列表并触发 AssertionError
# (见 2026-09-30 TTS 事故),因此在切句阶段直接剔除。
_RENDERABLE_RE = re.compile(r"[\w\u4e00-\u9fff]")


def has_renderable_text(text: str) -> bool:
    """文本是否含可发音内容,用于剔除纯标点/符号段。"""
    return bool(_RENDERABLE_RE.search(text))


def _filter_chunk_text(raw: str) -> str:
    """过滤掉 LLM 可能会附带在段落末尾的点评注释或项目符号（如 - Standard opening... 或 * 注：...）。"""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    valid_lines = []
    for line in lines:
        if re.match(r"^[-*#]\s+", line) or re.match(
            r"^（注[：:]|^\(Note[：:]", line, re.IGNORECASE
        ):
            continue
        valid_lines.append(line)
    return clean_tts_text(" ".join(valid_lines))


def parse_dialogue_chunks(
    text: str,
) -> list[DialogueChunk]:
    """解析 [Host A]/[Host B] 对话标记为 DialogueChunk 列表。"""
    marker_re = re.compile(r"\[Host\s*([AB])\]", re.IGNORECASE)
    chunks: list[DialogueChunk] = []
    current_host = "A"
    cursor = 0

    for m in marker_re.finditer(text):
        raw = text[cursor : m.start()].strip()
        if raw:
            cleaned = _filter_chunk_text(raw)
            if cleaned:
                chunks.append(DialogueChunk(host=current_host, text=cleaned))
        current_host = m.group(1).upper()
        cursor = m.end()

    tail = text[cursor:].strip()
    if tail:
        cleaned = _filter_chunk_text(tail)
        if cleaned:
            chunks.append(DialogueChunk(host=current_host, text=cleaned))
    return chunks


def split_text_into_sentences(text: str, max_chars: int = 80) -> list[str]:
    """将文本切分为较短的句子/短句，避免单次合成文本过长导致 CosyVoice 截断或语速失真。"""
    # 按照常见的标点符号进行切分，保留标点
    pattern = re.compile(r"([^，。！？；、,.!?;\s]+[，。！？；、,.!?;\s]*)")
    parts = pattern.findall(text)
    if not parts:
        return [text] if has_renderable_text(text) else []

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

    return [s for s in sentences if has_renderable_text(s)]
