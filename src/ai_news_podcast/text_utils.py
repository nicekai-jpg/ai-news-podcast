"""Shared TTS text-cleaning utilities.

Consolidates the text sanitisation logic previously duplicated across
podcastwriter.py, tts_engine.py, and run_daily.py into a single module.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Compiled regex constants — exposed for reuse in other modules
# ---------------------------------------------------------------------------

RE_FACT_TAG = re.compile(r"\[(?:FACT|INFERENCE|OPINION)\]\s*")
RE_MOOD_TAG = re.compile(r"\[mood:[^\]]+\]\s*")
RE_EMOJI_PAREN = re.compile(
    r"[（(][^）)]{0,10}(?:doge|狗头|笑|手动|滑稽|哭|捂脸|bushi|划掉)[^）)]{0,5}[）)]",
    flags=re.IGNORECASE,
)
RE_FANCY_QUOTES = re.compile(r"[「」『』【】]")
RE_EMPTY_PAREN = re.compile(r"[（(]\s*[）)]")
RE_NON_HOST_BRACKET = re.compile(r"\[(?!(Host\s*A|Host\s*B))[^\]]*\]", flags=re.IGNORECASE)
RE_REPEATED_COMMA = re.compile(r"[，,]{2,}")
RE_REPEATED_PERIOD = re.compile(r"[。.]{2,}")
RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_HTML_TAG = re.compile(r"<[^>]+>")

# Thinking-process markers that LLM sometimes outputs instead of actual dialogue
RE_THINKING_MARKERS = re.compile(
    r"(?i)(?:let me|key requirements|requirements and write|"
    r"natural conversation|two hosts:|host a.*host b|"
    r"\d+\.\s*two hosts|here is|below is|i will|"
    r"firstly|secondly|finally|step \d|note:|"
    r"^\s*1\.\s+two hosts|^\s*\d+\.\s*(?:two|host|conversation|natural))"
)


RE_THINK_TAG = re.compile(r"(?is)<think>.*?</think>\s*")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_tts_text(text: str) -> str:
    """Clean TTS-unfriendly artefacts from *text*.

    Parameters
    ----------
    text:
        Raw text produced by the LLM or template fallback.

    Returns
    -------
    str
        Cleaned text safe for TTS consumption.
    """
    if not text:
        return ""

    # 0. Strip <think>...</think> tags and contents
    text = RE_THINK_TAG.sub("", text)

    # 1. Fix escaped newlines (e.g. model outputs literal \\n)
    text = text.replace("\\n", "\n")

    # 2. Remove annotation / mood tags
    text = RE_FACT_TAG.sub("", text)
    text = RE_MOOD_TAG.sub("", text)

    # 3. Remove non-Host square-bracket annotations (keep [Host A]/[Host B])
    text = RE_NON_HOST_BRACKET.sub("", text)

    # 4. Remove emoji parentheticals
    text = RE_EMOJI_PAREN.sub("", text)

    # 5. Remove fancy quotes and brackets
    text = RE_FANCY_QUOTES.sub("", text)

    # 6. Remove empty parentheses
    text = RE_EMPTY_PAREN.sub("", text)

    # 7. Strip HTML tags
    text = RE_HTML_TAG.sub("", text)

    # 8. Detect and strip thinking-process / reasoning blocks
    #    (e.g. "Let me carefully analyze...", "Key requirements:" etc.)
    text = _strip_thinking_process(text)

    # 9. Compress repeated punctuation
    text = RE_REPEATED_COMMA.sub("，", text)
    text = RE_REPEATED_PERIOD.sub("。", text)

    # 10. Normalise whitespace and line breaks
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = RE_MULTI_SPACE.sub(" ", text)
    return RE_MULTI_NEWLINE.sub("\n\n", text).strip()


def _strip_thinking_process(text: str) -> str:
    """如果 LLM 输出了思考/分析过程（非对话内容），过滤掉它们。

    只在文本开头检测思考标记，一旦遇到真实对话就停止过滤。
    """
    lines = text.split("\n")
    result: list[str] = []
    found_dialogue = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if found_dialogue:
                result.append(line)
            continue

        # If we already found real dialogue, keep everything from now on
        if found_dialogue:
            result.append(line)
            continue

        # Before any real dialogue: skip thinking markers
        if re.match(r"^\[Host\s*[AB]\]", stripped, flags=re.IGNORECASE):
            found_dialogue = True
            result.append(line)
            continue

        if RE_THINKING_MARKERS.search(stripped):
            continue  # skip thinking process line

        # Not a host marker, not a thinking marker — could be a real line
        # (like an opening without [Host X] prefix, handled by _normalize_host_tags)
        found_dialogue = True
        result.append(line)

    return "\n".join(result)


def contains_thinking_process(text: str) -> bool:
    """Return True if the text contains LLM reasoning/thinking output."""
    if not text:
        return False
    # Strip <think> tag content first so we check the remaining dialogue text
    text = RE_THINK_TAG.sub("", text)
    # Check if there are lines with thinking markers before any real dialogue
    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # If we hit a real host marker first, it's probably fine
        if re.match(r"^\[Host\s*[AB]\]", stripped, flags=re.IGNORECASE):
            return False
        if RE_THINKING_MARKERS.search(stripped):
            return True
    return False
