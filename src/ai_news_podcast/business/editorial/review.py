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
    if review.chapters and len(review.chapters) != len(script_chapters):
        notes.append(
            f"章节数与文稿不一致：文稿 {len(script_chapters)} 章，点评 {len(review.chapters)} 章"
        )
    review.fact_check_notes = notes
    logger.info("S7 播客点评完成：%s 章，%s 条点评", len(review.chapters), len(review.items))
    return review
