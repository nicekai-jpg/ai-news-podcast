"""TTS 文本解析：对话标记解析与句子切分。"""

from __future__ import annotations

import re

from ai_news_podcast.business.pipeline.tts_types_service import DialogueChunk
from ai_news_podcast.data.text_utils_dao import clean_tts_text

# 至少含一个可发音字符(汉字/字母/数字)才算有效内容;纯标点/符号/emoji 会被
# CosyVoice 内嵌的 wetext 英文正则化 tokenize 成空列表并触发 AssertionError
# (见 2026-09-30 TTS 事故),因此在切句阶段直接剔除。
_RENDERABLE_RE = re.compile(r"[\w\u4e00-\u9fff]")

# 剧本章节标记：单独一行 `[CHAPTER] 标题`（由 Stage 3 writer 产出，标题 ≤12 字）。
_CHAPTER_RE = re.compile(r"^\[CHAPTER\]\s*(.+)$")


def has_renderable_text(text: str) -> bool:
    """文本是否含可发音内容,用于剔除纯标点/符号段。"""
    return bool(_RENDERABLE_RE.search(text))


def _filter_chunk_text(raw: str) -> str:
    """过滤掉 LLM 可能会附带在段落末尾的点评注释或项目符号（如 - Standard opening... 或 * 注：...）。"""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    valid_lines = []
    for line in lines:
        if _CHAPTER_RE.match(line):
            continue
        if re.match(r"^[-*#]\s+", line) or re.match(
            r"^（注[：:]|^\(Note[：:]", line, re.IGNORECASE
        ):
            continue
        valid_lines.append(line)
    return clean_tts_text(" ".join(valid_lines))


def _split_chapter_lines(raw: str) -> tuple[str, str | None]:
    """剥离段落里的 [CHAPTER] 行:返回 (对白正文, 最后一个章节标题)。

    章节行不进 TTS 文本;同一段里出现多条时只有最后一条生效(连续章节标记取最后一个)。
    """
    chapter: str | None = None
    lines: list[str] = []
    for line in raw.splitlines():
        match = _CHAPTER_RE.match(line.strip())
        if match:
            title = match.group(1).strip()
            if title:
                chapter = title
            continue
        lines.append(line)
    return "\n".join(lines), chapter


def parse_dialogue_chunks(
    text: str,
) -> list[DialogueChunk]:
    """解析 [Host A]/[Host B] 对话标记为 DialogueChunk 列表。

    紧邻某个 [Host X] 之前的 `[CHAPTER] 标题` 行归属该 Host 段:标题从对白文本里
    剥离并写入 DialogueChunk.chapter;脚本末尾无后续 Host 的章节行会被忽略。
    """
    marker_re = re.compile(r"\[Host\s*([AB])\]", re.IGNORECASE)
    chunks: list[DialogueChunk] = []
    current_host = "A"
    current_chapter: str | None = None
    cursor = 0

    for m in marker_re.finditer(text):
        raw, pending_chapter = _split_chapter_lines(text[cursor : m.start()])
        if raw.strip():
            cleaned = _filter_chunk_text(raw)
            if cleaned and has_renderable_text(cleaned):
                chunks.append(
                    DialogueChunk(host=current_host, text=cleaned, chapter=current_chapter)
                )
        current_host = m.group(1).upper()
        current_chapter = pending_chapter
        cursor = m.end()

    raw, _trailing_chapter = _split_chapter_lines(text[cursor:])
    if raw.strip():
        cleaned = _filter_chunk_text(raw)
        # 纯符号（如 "/"）会被 CosyVoice 的英文正则化 tokenize 成空列表而报错，
        # 也会在 playlist 里留下 0.1s 的空段：直接丢弃。
        if cleaned and has_renderable_text(cleaned):
            chunks.append(DialogueChunk(host=current_host, text=cleaned, chapter=current_chapter))
    return chunks


def split_text_into_sentences(text: str, max_chars: int = 80) -> list[str]:
    """将文本切分为较短的句子/短句，避免单次合成文本过长导致 CosyVoice 截断或语速失真。"""
    # 按照常见的标点符号进行切分，保留标点
    pattern = re.compile(r"([^，。！？；、,.!?;\s]+[，。！？；、,.!?;\s]*)")
    parts = pattern.findall(text)
    if not parts:
        if not has_renderable_text(text):
            return []
        # 完全没有标点的长文本：直接按 max_chars 硬切（否则整段塞给模型可能被截断）
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    sentences = []
    current = ""
    for part in parts:
        if len(current) + len(part) <= max_chars:
            current += part
            continue
        if current:
            sentences.append(current.strip())
        # 单个 part 本身超长（无标点长文本）时按 max_chars 硬切，避免整段塞给模型
        if len(part) > max_chars:
            for start in range(0, len(part), max_chars):
                piece = part[start : start + max_chars].strip()
                if piece:
                    sentences.append(piece)
            current = ""
        else:
            current = part
    if current:
        sentences.append(current.strip())

    return [s for s in sentences if has_renderable_text(s)]
