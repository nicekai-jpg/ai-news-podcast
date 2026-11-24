"""事实核对基础工具：数字抽取与逐字匹配。

放在独立模块里，供 S4 点评（``commentary``）与 S6 文稿（``quality``）共用。
``commentary`` 已经依赖 ``quality``，若把这两个函数放在 ``quality`` 里会被反向
导入形成环，所以单独拆出来。
"""

from __future__ import annotations

import re
import unicodedata

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*\s*[%％]?")
_DIGIT_ONLY_RE = re.compile(r"\D")


def normalize(text: str) -> str:
    """NFKC 归一化，统一全角/半角数字与百分号。"""
    return unicodedata.normalize("NFKC", text or "")


def extract_numbers(text: str) -> list[str]:
    """提取文本中的阿拉伯数字 token（含小数/千分位/百分号），保持出现顺序。"""
    return [match.group(0).replace(" ", "") for match in _NUMBER_RE.finditer(normalize(text))]


def number_in_text(number: str, text: str) -> bool:
    """数字是否能在证据文本中找到（逐字匹配；至少两位数字才允许纯数字子串兜底）。"""
    number = normalize(number).replace(" ", "")
    if not number:
        return True
    haystack = normalize(text)
    if number in haystack:
        return True
    digits = _DIGIT_ONLY_RE.sub("", number)
    if len(digits) < 2:
        return False
    return digits in _DIGIT_ONLY_RE.sub("", haystack)


__all__ = ["extract_numbers", "normalize", "number_in_text"]
