"""子句切分：引号保护、按标点切，超长子句不硬切（事实层 S2）。"""

from __future__ import annotations

import re

DEFAULT_MIN_CHARS = 4
DEFAULT_MAX_CHARS = 120

_QUOTE_PAIRS = (("“", "”"), ("「", "」"), ("『", "』"), ("（", "）"), ("(", ")"), ("《", "》"))
_SEPS = re.compile(r"[，,；;：:]|——")
_PLACEHOLDER = re.compile(r"\x01(\d+)\x02")


def _protect(text: str) -> tuple[str, list[str]]:
    """把引号/括号内容替换成占位符，避免在引号内切分。"""
    saved: list[str] = []

    def _swap(pair: tuple[str, str]) -> str:
        nonlocal text
        left, right = pair
        out: list[str] = []
        i = 0
        while i < len(text):
            if text.startswith(left, i):
                j = text.find(right, i + len(left))
                if j == -1:
                    out.append(text[i:])
                    break
                saved.append(text[i : j + len(right)])
                out.append(f"\x01{len(saved) - 1}\x02")
                i = j + len(right)
                continue
            out.append(text[i])
            i += 1
        return "".join(out)

    for pair in _QUOTE_PAIRS:
        text = _swap(pair)
    return text, saved


def _restore(text: str, saved: list[str]) -> str:
    value = text
    for _ in range(5):
        if "\x01" not in value:
            break

        def _sub(match: re.Match[str]) -> str:
            index = int(match.group(1))
            return saved[index] if 0 <= index < len(saved) else ""

        replaced = _PLACEHOLDER.sub(_sub, value)
        if replaced == value:
            break
        value = replaced
    return value


def split_clauses(
    sentence: str,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[str]:
    """按 ，；： 与破折号切子句；引号内不切；超长句保留给解析阶段。"""
    value = str(sentence or "").strip()
    if not value:
        return []
    protected, saved = _protect(value)
    clauses: list[str] = []
    for raw in _SEPS.split(protected):
        text = _restore(raw, saved).strip()
        if len(text) < min_chars:
            continue
        clauses.append(text)
    if not clauses and len(value) >= min_chars:
        clauses.append(value)
    return clauses


__all__ = ["DEFAULT_MAX_CHARS", "DEFAULT_MIN_CHARS", "split_clauses"]
