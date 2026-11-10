"""Stage 5 播客文稿：LLM 写作 → 格式归一化 → 质量闸门 / 重写 / 截断。

设计要点：
- **只认一种台词格式**：``[Host A]`` / ``[Host B]`` / ``[CHAPTER] 标题``。模型常见的
  「林夏：」「Host A:」「[A]」等写法会被归一化为 ``[Host X]``，Markdown/思考过程行被丢弃，
  未标记的续行并入上一位主持人的台词，保证「每段台词独占一行」。
- **质量闸门**：字数 3000–3600、章节 3–4、双主持占比 ≥25%、无禁用词、未标记行 ≤20%；
  不达标时最多重写 ``max_rewrites`` 次（保留更好的一版），仍超长则按章节/行边界截断。
- **事实来源**：只用 Stage 4 点评（headline/facts）；``item_evidence`` 为接口预留，
  当前不注入文稿 prompt，避免引入点评之外的新事实。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ai_news_podcast.business.contracts import Event, ShortlistEntry

from .config import DEFAULT_CONFIG_PATH, EditorialConfig, LLMCall, load_config
from .prompts import build_script_prompt, build_script_repair_prompt
from .quality import check_script

logger = logging.getLogger(__name__)

_HOST_LINE_RE = re.compile(r"^\[Host ([AB])\]\s*(.*)$")
_CHAPTER_RE = re.compile(r"^\[(?:chapter|章节)\]\s*(.*)$", re.IGNORECASE)
_TAG_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\[host\s*([ab])\]\s*[:：]?\s*(.*)$", re.IGNORECASE),
    re.compile(r"^host\s*([ab])\s*[:：]\s*(.*)$", re.IGNORECASE),
    re.compile(r"^\[([ab])\]\s*[:：]?\s*(.*)$", re.IGNORECASE),
)
_META_LINE_RE = re.compile(
    r"^(?:#|```|\||---|(?:（|\(|注[:：]|字数|说明[:：]|输出|以下是|写作思路|Let me\b|I'?ll\b|"
    r"Here is\b))",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^(?:[-*+>]\s*)+")
_NUMBERING_RE = re.compile(r"^(?:\d+[.、)]\s*|[一二三四五六七八九十]+[、.]\s*)")
_CHAPTER_TRAILING_RE = re.compile(r"[。.!！?？；;：:]+$")
_CHAPTER_MAX_CHARS = 24
_INLINE_TAG_RE = re.compile(r"\s*(?=\[(?:Host\s*[AB]|CHAPTER|章节))", re.IGNORECASE)


def _prepare_lines(raw: str) -> list[str]:
    """统一换行；模型偶尔把整篇挤成一行时，在台词/章节标记前补换行。"""
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    if text.count("\n") < 3:
        text = _INLINE_TAG_RE.sub("\n", text)
    return text.split("\n")


def _match_chapter(line: str) -> str | None:
    """命中章节行时返回标题（可能为空串），否则返回 None。"""
    match = _CHAPTER_RE.match(line)
    return match.group(1).strip() if match else None


def _clean_chapter_title(title: str) -> str:
    clean = str(title or "").strip().strip("#").strip()
    clean = _NUMBERING_RE.sub("", clean).strip()
    clean = _CHAPTER_TRAILING_RE.sub("", clean).strip()
    return clean[:_CHAPTER_MAX_CHARS]


def _clean_content(text: str) -> str:
    clean = str(text or "").replace("**", "").replace("`", "")
    return re.sub(r"[ \t]{2,}", " ", clean).strip()


def _split_speaker(line: str, config: EditorialConfig) -> tuple[str | None, str]:
    """识别一行台词的说话人；返回 ``(A/B, 内容)``，识别不出返回 ``(None, 原文)``。"""
    cleaned = _BULLET_RE.sub("", line).replace("**", "")
    for pattern in _TAG_RES:
        match = pattern.match(cleaned)
        if match:
            return match.group(1).upper(), match.group(2).strip()
    for name, key in ((config.host_a, "A"), (config.host_b, "B")):
        bracket = re.compile(rf"^\[{re.escape(name)}\]\s*[:：]?\s*(.*)$")
        named = re.compile(rf"^{re.escape(name)}\s*[:：]\s*(.*)$")
        for pattern in (bracket, named):
            match = pattern.match(cleaned)
            if match:
                return key, match.group(1).strip()
    return None, line


def normalize_script(raw: str, config: EditorialConfig | None = None) -> str:
    """把模型输出的任意脚本整理成标准双主持文稿格式。"""
    cfg = config or load_config()
    out: list[str] = []
    current: str | None = None
    pending: str | None = None
    for raw_line in _prepare_lines(raw):
        line = raw_line.strip()
        if not line:
            continue
        chapter = _match_chapter(line)
        if chapter is not None:
            title = _clean_chapter_title(chapter)
            if title:
                out.append(f"[CHAPTER] {title}")
            current, pending = None, None
            continue
        speaker, content = _split_speaker(line, cfg)
        if speaker is not None:
            content = _clean_content(content)
            if content:
                out.append(f"[Host {speaker}] {content}")
                current, pending = speaker, None
            else:
                pending = speaker
            continue
        if _META_LINE_RE.match(line):
            continue
        continuation = _clean_content(_BULLET_RE.sub("", line))
        if not continuation:
            continue
        # 未标记行视为上一位主持人的续行；章前无归属内容直接丢弃
        if pending is not None:
            out.append(f"[Host {pending}] {continuation}")
            current, pending = pending, None
        elif current is not None and out:
            out[-1] = out[-1] + continuation
    return "\n".join(out) + ("\n" if out else "")


def _line_spoken_chars(line: str) -> int:
    match = _HOST_LINE_RE.match(line)
    return len(re.sub(r"\s+", "", match.group(2))) if match else 0


def _trim_to_max(script: str, max_chars: int) -> str:
    """超长时按整行截断到 max_chars；丢弃末尾没有台词的章节标题。"""
    kept: list[str] = []
    total = 0
    for line in script.splitlines():
        if line.startswith("[CHAPTER]"):
            kept.append(line)
            continue
        chars = _line_spoken_chars(line)
        if total + chars > max_chars:
            break
        kept.append(line)
        total += chars
    while kept and kept[-1].startswith("[CHAPTER]"):
        kept.pop()
    return "\n".join(kept) + ("\n" if kept else "")


def _defect_score(report: dict[str, Any]) -> tuple[int, int]:
    """缺陷越少越好；同缺陷数时越接近目标字数中点越好。"""
    mid = (int(report["min_chars"]) + int(report["max_chars"])) // 2
    return (len(report["issues"]), abs(int(report["chars"]) - mid))


def build_script(
    shortlist: Sequence[ShortlistEntry],
    events: Sequence[Event],
    item_evidence: Mapping[str, str] | None,
    llm_call: LLMCall,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> str:
    """根据事实底稿（Stage 2 全文证据）生成 15 分钟双主持播客文稿。

    - ``item_evidence`` 是 item_id -> 正文，用于构造每事件的事实底稿；
    - 返回标准化后的纯文本（``[Host A]/[Host B]/[CHAPTER]``），可直接写
      ``site/episodes/{date}.txt``；不满足字数/章节等硬指标时返回尽力而为的版本，
      具体缺陷由 ``quality()`` 报告；
    - LLM 调用异常直接抛出，由上层决定重试/降级。
    """
    config = load_config(config_path)
    logger.debug("S6 使用配置 %s，item_evidence=%s", config_path, item_evidence is not None)
    prompt = build_script_prompt(shortlist, list(events), item_evidence, config=config)
    raw = llm_call(prompt)
    script = normalize_script(raw, config)
    report = check_script(script, config)

    attempts = 0
    while not report["ok"] and attempts < config.max_rewrites:
        attempts += 1
        logger.info("文稿未达标（第 %s 次重写）：%s", attempts, "；".join(report["issues"]))
        raw = llm_call(build_script_repair_prompt(script, report, config=config))
        candidate = normalize_script(raw, config)
        candidate_report = check_script(candidate, config)
        if _defect_score(candidate_report) <= _defect_score(report):
            script, report = candidate, candidate_report
        else:
            logger.warning("重写结果更差，保留原稿：%s", "；".join(candidate_report["issues"]))
            break

    if not report["ok"]:
        logger.warning(
            "文稿质量为 %s（重写 %s 次）：%s", "未达标", attempts, "；".join(report["issues"])
        )
    if report["chars"] > config.max_chars:
        script = _trim_to_max(script, config.max_chars)
        final_report = check_script(script, config)
        logger.warning(
            "文稿超长已截断：%s → %s 字，剩余问题：%s",
            report["chars"],
            final_report["chars"],
            "；".join(final_report["issues"]) or "无",
        )
    return script


def quality(script: str, *, config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """面向调用方的质量报告入口（加载配置后委托 ``quality.check_script``）。"""
    return check_script(script, load_config(config_path))
