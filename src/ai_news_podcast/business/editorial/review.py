"""S5 日报点评 / S7 播客点评：对已生成结果的审读。

- S5：输入 S4 已生成的日报（Commentary 列表）+ 证据正文，输出编辑点评（供 02 页面）；
- S7：输入 S6 已生成的播客文稿 + 日报要点，输出 show notes / 章节导语 / 点评（供 01 页面）。
点评失败不阻断主流程：返回空点评并记录日志。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ai_news_podcast.business.contracts import (
    Commentary,
    EditorialReview,
    Event,
    PodcastReview,
    ReviewItem,
)

from .commentary import EditorialValidationError, extract_json_object, sanitize_text
from .config import DEFAULT_CONFIG_PATH, EditorialConfig, LLMCall, load_config
from .prompts import build_podcast_review_prompt, build_report_review_prompt

logger = logging.getLogger(__name__)


def _sanitize(value: Any, config: EditorialConfig) -> str:
    return sanitize_text(str(value or ""), config.banned_words)


def _sanitize_list(raw: Any, config: EditorialConfig, *, limit: int = 8) -> list[str]:
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for item in raw[:limit]:
        text = _sanitize(item, config)
        if text:
            result.append(text)
    return result


def _parse_review_items(raw: Any, config: EditorialConfig) -> list[ReviewItem]:
    if not isinstance(raw, list):
        return []
    items: list[ReviewItem] = []
    for row in raw:
        if not isinstance(row, Mapping):
            continue
        items.append(
            ReviewItem(
                event_id=str(row.get("event_id") or ""),
                title=_sanitize(row.get("title"), config),
                verdict=_sanitize(row.get("verdict"), config),
                comment=_sanitize(row.get("comment"), config),
                corrections=_sanitize_list(row.get("corrections"), config),
                highlights=_sanitize_list(row.get("highlights"), config),
            )
        )
    return items


def parse_report_review(payload: Mapping[str, Any], config: EditorialConfig) -> EditorialReview:
    """把 S5 LLM JSON 解析为 EditorialReview（缺字段时宽容处理）。"""
    if not isinstance(payload, Mapping):
        raise EditorialValidationError("日报点评 JSON 顶层必须是对象")
    return EditorialReview(
        overall=_sanitize(payload.get("overall"), config),
        items=_parse_review_items(payload.get("items"), config),
        risks=_sanitize_list(payload.get("risks"), config),
    )


def build_report_review(
    report_rows: Sequence[Commentary],
    events: Sequence[Event],
    llm_call: LLMCall,
    *,
    item_evidence: Mapping[str, str] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> EditorialReview:
    """S5：对已生成的日报做编辑审读。"""
    config = load_config(config_path)
    if not report_rows:
        return EditorialReview()
    prompt = build_report_review_prompt(
        report_rows, events, item_evidence=item_evidence, config=config
    )
    try:
        raw = llm_call(prompt)
        payload = extract_json_object(raw)
        review = parse_report_review(payload, config)
    except EditorialValidationError as exc:
        logger.warning("日报点评未通过解析，已跳过：%s", exc)
        return EditorialReview()
    except Exception:
        logger.exception("日报点评 LLM 调用失败，已跳过")
        return EditorialReview()
    logger.info("S5 日报点评完成：%s 条", len(review.items))
    return review


def parse_podcast_review(payload: Mapping[str, Any], config: EditorialConfig) -> PodcastReview:
    """把 S7 LLM JSON 解析为 PodcastReview。"""
    if not isinstance(payload, Mapping):
        raise EditorialValidationError("播客点评 JSON 顶层必须是对象")
    chapters_raw = payload.get("chapters")
    chapters: list[dict[str, Any]] = []
    if isinstance(chapters_raw, list):
        for chapter in chapters_raw:
            if not isinstance(chapter, Mapping):
                continue
            chapters.append(
                {
                    "title": _sanitize(chapter.get("title"), config),
                    "intro": _sanitize(chapter.get("intro"), config),
                    "highlights": _sanitize_list(chapter.get("highlights"), config),
                }
            )
    return PodcastReview(
        overall=_sanitize(payload.get("overall"), config),
        show_notes=_sanitize(payload.get("show_notes"), config),
        chapters=chapters,
        items=_parse_review_items(payload.get("items"), config),
        fact_check_notes=_sanitize_list(payload.get("fact_check_notes"), config),
    )


def _normalize_title(text: str) -> str:
    """章节标题归一化：去空白与标点，用于宽松比对。"""
    import re

    return re.sub(r"[\s\-—·:：,，。.!！?？\"'“”‘’()（）\[\]【】|]", "", str(text or "")).lower()


def _align_chapters(
    review: PodcastReview, script_titles: list[str]
) -> tuple[PodcastReview, list[str]]:
    """让 S7 的章节标题与文稿 ``[CHAPTER]`` 严格对齐。

    章节标题是页面上章节导航的唯一来源，必须与听众实际听到的一致。
    之前只靠 prompt 约束 + 事后记一条 note，标题被改写时页面就会漂移。
    这里改为程序对齐：
    - 数量一致 → 逐条比对，不匹配的用文稿标题覆盖（保留 LLM 写的 intro）；
    - 数量不一致 → 以文稿为准重建章节骨架（保留能对上的 intro），并记 note。
    """
    notes: list[str] = []
    if not script_titles:
        return review, notes
    chapters = list(review.chapters or [])
    if not chapters:
        notes.append("点评未给章节，已按文稿 [CHAPTER] 生成骨架")
        review.chapters = [
            {"title": title, "intro": "", "highlights": []} for title in script_titles
        ]
        return review, notes

    if len(chapters) == len(script_titles):
        drifted: list[str] = []
        for chapter, title in zip(chapters, script_titles, strict=True):
            current = str(chapter.get("title") or "")
            if _normalize_title(current) != _normalize_title(title):
                drifted.append(f"{current or '(空)'} → {title}")
                chapter["title"] = title
        if drifted:
            notes.append("章节标题已按文稿校正：" + "；".join(drifted))
        return review, notes

    notes.append(
        f"章节数与文稿不一致（点评 {len(chapters)} 章 / 文稿 {len(script_titles)} 章），已按文稿重建"
    )
    aligned: list[dict[str, Any]] = []
    for index, title in enumerate(script_titles):
        source = chapters[index] if index < len(chapters) else {}
        aligned.append(
            {
                "title": title,
                "intro": str(source.get("intro") or ""),
                "highlights": list(source.get("highlights") or []),
            }
        )
    review.chapters = aligned
    return review, notes


def _deterministic_fact_notes(
    script: str,
    config: EditorialConfig,
    evidence_text: str,
) -> tuple[list[str], list[str]]:
    """用稿件与证据做确定性核对，产出可复现的 fact_check_notes。

    LLM 自评的 ``fact_check_notes`` 没有任何校验价值；这里改为程序核对：
    - 文稿里的声明数字是否都能在事实底稿中逐字找到；
    - 点评章节数与文稿 ``[CHAPTER]`` 是否一致。
    """
    from .quality import chapter_titles, check_script

    notes: list[str] = []
    if evidence_text.strip():
        report = check_script(script, config, evidence_text=evidence_text)
        if report["unverified_numbers"]:
            notes.append("以下数字未在事实底稿中找到：" + "、".join(report["unverified_numbers"]))
        else:
            notes.append(f"文稿内 {len(report['claim_numbers'])} 个数字均可在事实底稿核对")
    else:
        notes.append("未提供事实底稿，跳过数字核对")
    return notes, chapter_titles(script)


def build_podcast_review(
    script: str,
    report_rows: Sequence[Commentary],
    events: Sequence[Event],
    llm_call: LLMCall,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    evidence_text: str = "",
) -> PodcastReview:
    """S7：对已生成的播客文稿做审读并生成 01 页面 show notes。

    ``fact_check_notes`` 不采用 LLM 自评，改用 ``_deterministic_fact_notes``
    对稿件与证据做程序核对，结果可复现。
    """
    config = load_config(config_path)
    if not script.strip():
        return PodcastReview()
    prompt = build_podcast_review_prompt(script, report_rows, events, config=config)
    try:
        raw = llm_call(prompt)
        payload = extract_json_object(raw)
        review = parse_podcast_review(payload, config)
    except EditorialValidationError as exc:
        logger.warning("播客点评未通过解析，已跳过：%s", exc)
        return PodcastReview()
    except Exception:
        logger.exception("播客点评 LLM 调用失败，已跳过")
        return PodcastReview()

    notes, script_chapters = _deterministic_fact_notes(script, config, evidence_text)
    review, chapter_notes = _align_chapters(review, script_chapters)
    review.fact_check_notes = [*notes, *chapter_notes]
    logger.info("S7 播客点评完成：%s 章，%s 条点评", len(review.chapters), len(review.items))
    return review
