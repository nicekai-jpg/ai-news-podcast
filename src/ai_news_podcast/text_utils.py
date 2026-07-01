"""Shared TTS text-cleaning utilities.

Consolidates the text sanitisation logic previously duplicated across
scriptwriter.py, tts_engine.py, and run_daily.py into a single module.
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

    # 8. Compress repeated punctuation
    text = RE_REPEATED_COMMA.sub("，", text)
    text = RE_REPEATED_PERIOD.sub("。", text)

    # 9. Normalise whitespace and line breaks
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NEWLINE.sub("\n\n", text).strip()

    return text
