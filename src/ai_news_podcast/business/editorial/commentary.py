"""Stage 4 编辑点评：LLM 生成 → JSON 解析 → 客观性校验。

校验是「硬闸门」，任何一条不满足都拒收对应 fact（必要时整条点评拒收）：

1. facts 必须带 ``evidence_ids``，且 ID 必须是该事件可引用的证据条目；
2. fact 中出现的阿拉伯数字/百分比必须能在证据正文里逐字找到（版本号、金额、日期同规则）；
3. fact 中不得出现禁用营销词；
4. 观点字段（why_matters/impact/...）若含禁用词会被清洗，但不会被当成事实；
5. counterpoint 为空时补「暂无公开反方信息」，避免点评单边化。

证据来源：
- 默认 ``item_evidence=None``：语料 = ``representative_title + summary``，证据 ID = ``item_ids``
  （为空回退 ``event_id``）；
- 传入 ``item_evidence``（item_id -> 正文）：语料 = 对应条目正文，证据 ID 必须同时属于
  ``event.item_ids`` 且存在于 ``item_evidence``。
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ai_news_podcast.business.contracts import Commentary, Event, Fact, ShortlistEntry

from .config import DEFAULT_CONFIG_PATH, EditorialConfig, LLMCall, load_config
from .prompts import build_commentary_prompt
from .quality import find_banned_words

logger = logging.getLogger(__name__)

COUNTERPOINT_FALLBACK = "暂无公开反方信息"

# 数字 token：整数/小数/千分位 + 可选百分号（全角在比较前统一 NFKC）
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*\s*[%％]?")


class EditorialValidationError(ValueError):
    """LLM 输出不满足 Stage 4 的格式/客观性约束。"""


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出里提取第一个 JSON 对象（容忍代码块围栏与前后解释）。"""
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", clean)
        clean = re.sub(r"\s*```\s*$", "", clean)
    try:
        loaded = json.loads(clean)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict):
        return loaded
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", clean):
        try:
            candidate, _ = decoder.raw_decode(clean[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    raise EditorialValidationError("LLM 输出中找不到合法 JSON 对象")


def extract_numbers(text: str) -> list[str]:
    """提取文本中的阿拉伯数字 token（含小数/千分位/百分号），保持出现顺序。"""
    return [match.group(0).replace(" ", "") for match in _NUMBER_RE.finditer(_normalize(text))]


def number_in_text(number: str, text: str) -> bool:
    """数字是否能在证据文本中找到（逐字匹配；至少两位数字才允许纯数字子串兜底）。"""
    number = _normalize(number).replace(" ", "")
    if not number:
        return True
    haystack = _normalize(text)
    if number in haystack:
        return True
    digits = re.sub(r"\D", "", number)
    if len(digits) < 2:
        return False
    return digits in re.sub(r"\D", "", haystack)


def sanitize_text(text: str, banned: Sequence[str]) -> str:
    """移除观点文本中的禁用营销词，避免它们进入播客文稿。"""
    clean = str(text or "")
    for word in banned:
        if word:
            clean = clean.replace(word, "")
    return re.sub(r"[ \t]{2,}", " ", clean).strip()


def _event_evidence_ids(event: Event) -> list[str]:
    ids = [str(item_id) for item_id in event.item_ids if item_id]
    return ids or [str(event.event_id)]


def resolve_evidence(
    event: Event,
    item_evidence: Mapping[str, str] | None = None,
) -> tuple[list[str], str, dict[str, str]]:
    """解析某事件可引用的证据 ID、用于数字核对的语料、以及 prompt 用正文片段。

    返回 ``(allowed_ids, corpus_text, snippets)``。
    """
    title_summary = f"{event.representative_title}\n{event.summary}".strip()
    if item_evidence is None:
        return _event_evidence_ids(event), title_summary, {}

    item_ids = [str(item_id) for item_id in event.item_ids if item_id]
    allowed = [item_id for item_id in item_ids if item_id in item_evidence]
    if not allowed and not item_ids and str(event.event_id) in item_evidence:
        allowed = [str(event.event_id)]
    if not allowed:
        logger.warning(
            "事件 %s 在 item_evidence 中没有可引用的证据正文，点评 facts 将被拒收",
            event.event_id,
        )
        return [], title_summary, {}
    snippets = {item_id: str(item_evidence[item_id]) for item_id in allowed}
    corpus = "\n\n".join(snippets.values()).strip() or title_summary
    return allowed, corpus, snippets


def _clean_sources(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    sources: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        link = str(item.get("link") or "").strip()
        if not label and not link:
            continue
        sources.append(
            {"label": label, "link": link, "official": bool(item.get("official", False))}
        )
    return sources


def _parse_fact(
    raw: Any,
    event: Event,
    allowed_ids: Sequence[str],
    corpus: str,
    banned: Sequence[str],
) -> Fact | None:
    """解析并校验单条 fact；不满足客观性要求时返回 None 并记日志。"""
    if not isinstance(raw, dict):
        logger.warning("事件 %s 的 fact 不是对象，已丢弃", event.event_id)
        return None
    text = str(raw.get("text") or "").strip()
    if not text:
        logger.warning("事件 %s 存在空 fact，已丢弃", event.event_id)
        return None

    banned_hits = find_banned_words(text, banned)
    if banned_hits:
        logger.warning(
            "事件 %s 的 fact 含禁用词 %s，已拒收：%s", event.event_id, banned_hits, text[:60]
        )
        return None

    raw_ids = raw.get("evidence_ids")
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    if not isinstance(raw_ids, list):
        raw_ids = []
    valid_ids: list[str] = []
    for raw_id in raw_ids:
        candidate_id = str(raw_id)
        if candidate_id in allowed_ids and candidate_id not in valid_ids:
            valid_ids.append(candidate_id)
    if not valid_ids:
        logger.warning(
            "事件 %s 的 fact 缺少有效 evidence_ids（允许：%s），已拒收：%s",
            event.event_id,
            allowed_ids,
            text[:60],
        )
        return None

    missing = [number for number in extract_numbers(text) if not number_in_text(number, corpus)]
    if missing:
        logger.warning(
            "事件 %s 的 fact 中数字 %s 在证据文本中找不到，已拒收：%s",
            event.event_id,
            missing,
            text[:60],
        )
        return None

    single_source = bool(raw.get("single_source", False)) or event.independent_sources <= 1
    return Fact(text=text, evidence_ids=valid_ids, single_source=single_source)


def parse_commentary(
    payload: Mapping[str, Any],
    event: Event,
    *,
    config: EditorialConfig,
    item_evidence: Mapping[str, str] | None = None,
) -> Commentary:
    """把 LLM 返回的 JSON 映射校验并转换为 ``Commentary``。

    校验失败（无合法 facts / 结构错误）抛 ``EditorialValidationError``；
    单条 fact 不合格只会被丢弃。
    """
    if not isinstance(payload, Mapping):
        raise EditorialValidationError("点评 JSON 顶层必须是对象")

    allowed_ids, corpus, _ = resolve_evidence(event, item_evidence)
    logger.debug("事件 %s 可引用证据 %s", event.event_id, allowed_ids)

    raw_facts = payload.get("facts")
    if not isinstance(raw_facts, list):
        raise EditorialValidationError("点评缺少 facts 数组")
    facts = [
        fact
        for raw in raw_facts
        if (fact := _parse_fact(raw, event, allowed_ids, corpus, config.banned_words)) is not None
    ]
    if not facts:
        raise EditorialValidationError(f"事件 {event.event_id} 没有通过客观性校验的 facts")

    headline = sanitize_text(
        str(payload.get("headline") or event.representative_title), config.banned_words
    )
    if not headline:
        headline = event.representative_title
    counterpoint = (
        sanitize_text(str(payload.get("counterpoint") or ""), config.banned_words)
        or COUNTERPOINT_FALLBACK
    )
    event_id = str(payload.get("event_id") or event.event_id)
    if event_id != event.event_id:
        logger.warning("点评 event_id=%s 与事件 %s 不一致，已按事件修正", event_id, event.event_id)

    return Commentary(
        event_id=event.event_id,
        headline=headline,
        one_liner=sanitize_text(str(payload.get("one_liner") or ""), config.banned_words),
        facts=facts,
        why_matters=sanitize_text(str(payload.get("why_matters") or ""), config.banned_words),
        background=sanitize_text(str(payload.get("background") or ""), config.banned_words),
        impact=sanitize_text(str(payload.get("impact") or ""), config.banned_words),
        counterpoint=counterpoint,
        uncertainty=sanitize_text(str(payload.get("uncertainty") or ""), config.banned_words),
        sources=_clean_sources(payload.get("sources")),
    )


def build_commentaries(
    shortlist: Sequence[ShortlistEntry],
    events: Sequence[Event],
    llm_call: LLMCall,
    *,
    item_evidence: Mapping[str, str] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[Commentary]:
    """为精排 shortlist 的前 ``report_top_n`` 条事件生成编辑点评。

    - ``shortlist`` 的顺序即 rank，不再重排；同一 event 只处理一次；
    - 单条事件失败（event 缺失 / LLM 异常 / 客观性校验不过）记日志后跳过，
      不阻断其余事件；
    - ``item_evidence`` 可选：item_id -> 正文，用于数字核对与 prompt 证据片段。
    """
    config = load_config(config_path)
    event_by_id = {str(event.event_id): event for event in events}
    selected: list[ShortlistEntry] = []
    seen: set[str] = set()
    for entry in shortlist:
        event_id = str(entry.event_id)
        if event_id in seen:
            continue
        seen.add(event_id)
        if event_id not in event_by_id:
            logger.warning("shortlist 中的事件 %s 不在 events 里，跳过点评", event_id)
            continue
        selected.append(entry)
        if len(selected) >= config.report_top_n:
            break

    commentaries: list[Commentary] = []
    for entry in selected:
        event = event_by_id[str(entry.event_id)]
        allowed_ids, _, snippets = resolve_evidence(event, item_evidence)
        prompt = build_commentary_prompt(
            entry,
            event,
            config=config,
            evidence_ids=allowed_ids,
            evidence_snippets=snippets,
        )
        try:
            raw_output = llm_call(prompt)
        except Exception:
            logger.exception("事件 %s 点评 LLM 调用失败，已跳过", event.event_id)
            continue
        try:
            payload = extract_json_object(raw_output)
            commentary = parse_commentary(
                payload, event, config=config, item_evidence=item_evidence
            )
        except EditorialValidationError as exc:
            logger.warning("事件 %s 点评未通过校验，已跳过：%s", event.event_id, exc)
            continue
        logger.info("事件 %s 点评完成：%s", event.event_id, commentary.headline)
        commentaries.append(commentary)
    return commentaries
