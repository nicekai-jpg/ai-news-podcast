"""S6 文稿自动质量检查（纯函数，不发网络请求）。

口径：
- **字数**指口语正文字数：只统计 ``[Host A]`` / ``[Host B]`` 行去掉标记后的非空白字符，
  不含 ``[CHAPTER]`` 行、章节标题与空白（与 15 分钟 ≈ 3000–3600 字的预算一致）。
- **主持占比** = 两位主持人各自正文字数 / 两人正文字数之和，取较小值。
- **未标记行**：非空、既不是主持台词也不是章节标记的行（通常意味着模型输出了
  Markdown / 思考过程 / 舞台提示）。
"""

from __future__ import annotations

import re
from typing import Any

from ai_news_podcast.business.contracts import Commentary

from .config import EditorialConfig, load_config

_HOST_LINE_RE = re.compile(r"^\[Host ([AB])\]\s*(.*)$")
_CHAPTER_LINE_RE = re.compile(r"^\[CHAPTER\]\s*(.*)$")
_WHITESPACE_RE = re.compile(r"\s+")

# 点评各栏位字数（含 facts 正文），用于 150–300 字/条的告警统计
_COMMENTARY_TEXT_FIELDS = (
    "headline",
    "one_liner",
    "why_matters",
    "background",
    "impact",
    "counterpoint",
    "uncertainty",
)


def _compress(text: str) -> str:
    return _WHITESPACE_RE.sub("", text or "")


def count_spoken_chars(script: str) -> int:
    """统计双主持台词的正文总字数（不含标记、章节标题、空白）。"""
    total = 0
    for raw_line in (script or "").splitlines():
        match = _HOST_LINE_RE.match(raw_line.strip())
        if match:
            total += len(_compress(match.group(2)))
    return total


def host_char_counts(script: str) -> tuple[int, int]:
    """按 ``[Host A]`` / ``[Host B]`` 统计两位主持人的正文字数。"""
    counts = {"A": 0, "B": 0}
    for raw_line in (script or "").splitlines():
        match = _HOST_LINE_RE.match(raw_line.strip())
        if match:
            counts[match.group(1)] += len(_compress(match.group(2)))
    return counts["A"], counts["B"]


def host_char_ratio(script: str) -> float:
    """较小一方主持人的正文字数占比；无台词时返回 0。"""
    host_a, host_b = host_char_counts(script)
    total = host_a + host_b
    return (min(host_a, host_b) / total) if total else 0.0


def count_chapters(script: str) -> int:
    """统计 ``[CHAPTER]`` 标记行数量。"""
    return sum(
        1 for raw_line in (script or "").splitlines() if _CHAPTER_LINE_RE.match(raw_line.strip())
    )


def chapter_titles(script: str) -> list[str]:
    """按出现顺序返回章节标题（空标题也算一章，便于发现格式错误）。"""
    titles: list[str] = []
    for raw_line in (script or "").splitlines():
        match = _CHAPTER_LINE_RE.match(raw_line.strip())
        if match:
            titles.append(match.group(1).strip())
    return titles


def find_banned_words(text: str, banned: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """返回文本中命中的禁用词（去重、保持配置顺序）。"""
    banned = banned if banned is not None else load_config().banned_words
    found: list[str] = []
    for banned_word in banned:
        word = str(banned_word)
        if word and word in (text or "") and word not in found:
            found.append(word)
    return found


def unmarked_lines(script: str) -> list[str]:
    """返回非台词语句行（非空、非 ``[Host X]``、非 ``[CHAPTER]``）。"""
    lines: list[str] = []
    for raw_line in (script or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _HOST_LINE_RE.match(line) or _CHAPTER_LINE_RE.match(line):
            continue
        lines.append(line)
    return lines


def check_script(script: str, config: EditorialConfig | None = None) -> dict[str, Any]:
    """检查文稿是否满足 15 分钟双主持播客的硬指标。

    返回可直接打印/落盘的 dict：``ok`` 为总闸门，``issues`` 为人类可读的失败项。
    """
    cfg = config or load_config()
    script = script or ""
    chars = count_spoken_chars(script)
    host_a, host_b = host_char_counts(script)
    ratio = host_char_ratio(script)
    chapters = count_chapters(script)
    marked = unmarked_lines(script)
    total_lines = len([line for line in script.splitlines() if line.strip()])
    unmarked_ratio = (len(marked) / total_lines) if total_lines else 0.0
    banned = find_banned_words(script, cfg.banned_words)

    issues: list[str] = []
    if not chars:
        issues.append("文稿没有任何 [Host A]/[Host B] 台词")
    if not cfg.min_chars <= chars <= cfg.max_chars:
        issues.append(f"正文字数 {chars} 不在 {cfg.min_chars}–{cfg.max_chars} 区间")
    if not cfg.min_chapters <= chapters <= cfg.max_chapters:
        issues.append(f"章节数 {chapters} 不在 {cfg.min_chapters}–{cfg.max_chapters} 区间")
    if chars and ratio < cfg.min_host_char_ratio:
        issues.append(
            f"主持人不均衡：Host A {host_a} 字 / Host B {host_b} 字，"
            f"最低占比 {ratio:.2f} < {cfg.min_host_char_ratio:.2f}"
        )
    if unmarked_ratio > cfg.max_unmarked_ratio:
        issues.append(
            f"未标记行 {len(marked)} 行，占比 {unmarked_ratio:.2f} > "
            f"{cfg.max_unmarked_ratio:.2f}（示例：{marked[0][:40]}）"
        )
    if banned:
        issues.append(f"出现禁用词：{'、'.join(banned)}")

    return {
        "ok": not issues,
        "chars": chars,
        "min_chars": cfg.min_chars,
        "max_chars": cfg.max_chars,
        "chapters": chapters,
        "min_chapters": cfg.min_chapters,
        "max_chapters": cfg.max_chapters,
        "target_chapters": cfg.chapters,
        "chapter_titles": chapter_titles(script),
        "host_a_chars": host_a,
        "host_b_chars": host_b,
        "host_char_ratio": ratio,
        "min_host_char_ratio": cfg.min_host_char_ratio,
        "unmarked_lines": marked,
        "unmarked_ratio": unmarked_ratio,
        "max_unmarked_ratio": cfg.max_unmarked_ratio,
        "banned_words": banned,
        "issues": issues,
    }


def commentary_quality(
    commentary: Commentary,
    config: EditorialConfig | None = None,
) -> dict[str, Any]:
    """检查单条点评的客观性指标（Stage 4 落盘前的审计视图）。

    - facts 是否都带 evidence_ids（无证据事实视为硬错误）；
    - fact 字数是否落在建议区间（仅告警）；
    - 单一来源事实比例；
    - 禁用词与 counterpoint 缺失。
    """
    cfg = config or load_config()
    facts = list(commentary.facts)
    plain_text = " ".join(
        [
            *(str(getattr(commentary, field, "") or "") for field in _COMMENTARY_TEXT_FIELDS),
            *(fact.text for fact in facts),
        ]
    )
    missing_evidence = [fact.text for fact in facts if not fact.evidence_ids]
    fact_chars = [len(_compress(fact.text)) for fact in facts]
    out_of_range = [
        length for length in fact_chars if not cfg.fact_min_chars <= length <= cfg.fact_max_chars
    ]
    single_source = sum(1 for fact in facts if fact.single_source)
    banned = find_banned_words(plain_text, cfg.banned_words)

    warnings: list[str] = []
    if out_of_range:
        warnings.append(
            f"{len(out_of_range)} 条 fact 字数不在建议区间 "
            f"{cfg.fact_min_chars}–{cfg.fact_max_chars}"
        )
    if single_source:
        warnings.append(
            f"单一来源事实 {single_source}/{len(facts)} 条（占比 {single_source / len(facts):.2f}）"
        )
    if not commentary.counterpoint.strip():
        warnings.append("counterpoint 为空（应写「暂无公开反方信息」）")
    if not commentary.sources:
        warnings.append("sources 为空")

    return {
        "ok": not missing_evidence and not banned and bool(facts),
        "event_id": commentary.event_id,
        "chars": len(_compress(plain_text)),
        "facts": len(facts),
        "facts_without_evidence": missing_evidence,
        "fact_chars": fact_chars,
        "fact_chars_in_range": not out_of_range,
        "single_source_facts": single_source,
        "single_source_ratio": (single_source / len(facts)) if facts else 0.0,
        "banned_words": banned,
        "warnings": warnings,
    }
