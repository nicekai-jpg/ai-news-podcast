"""Stage 2 连续性识别：与历史事件匹配、新进展判定、sequence/continuity 分配。

规则（v3 §2.5 / v4 §2 / v5 修订）：
- 匹配：主体 Jaccard ≥ 阈值，或代表标题相似 ≥ 阈值（实体命中但标题过低时用 floor 兜底）；
- 无匹配 → 新事件（sequence=1，新 continuity_id）；
- 有匹配且无新进展 → 同一事件（sequence 不变，is_update=False）；
- 有匹配且新进展（新版本/新数字/新动作）→ sequence+1，is_update=True；
- 已覆盖（coverage ledger）+ 无新进展 → 过滤，避免旧闻重播；未覆盖 → 保留并标记 reappeared。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai_news_podcast.business.contracts import Event
from ai_news_podcast.business.coverage.ledger import CoverageEntry, is_covered
from ai_news_podcast.business.ocean.config import cfg_get
from ai_news_podcast.business.ocean.labels import infer_lifecycle, match_subjects
from ai_news_podcast.business.ocean.textutils import (
    days_between,
    extract_novelty_numbers,
    extract_version_strings,
    normalize_title,
    parse_time,
    title_similarity,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchResult:
    event: Event
    title_similarity: float
    subject_jaccard: float
    score: float


@dataclass(frozen=True)
class ProgressResult:
    is_new: bool
    reasons: tuple[str, ...] = ()


@dataclass
class ContinuityOutcome:
    """连续性处理结果：保留事件、被覆盖过滤事件、事件线。"""

    events: list[Event] = field(default_factory=list)
    dropped: list[Event] = field(default_factory=list)
    drop_reasons: dict[str, str] = field(default_factory=dict)
    matched_events: dict[str, str] = field(default_factory=dict)
    lines: list[dict[str, Any]] = field(default_factory=list)
    updates: int = 0
    new_events: int = 0


# ---------------------------------------------------------------------------
# 历史事件工具
# ---------------------------------------------------------------------------


def _as_event(value: Event | dict[str, Any]) -> Event:
    return value if isinstance(value, Event) else Event.from_dict(value)


def _event_stamp(event: Event) -> datetime | None:
    return parse_time(event.last_update_at) or parse_time(event.first_seen_at)


def filter_history_window(
    history_events: list[Event | dict[str, Any]],
    date_str: str,
    history_days: int,
) -> list[Event]:
    """只保留最近 history_days 天内的历史事件（按 last_update/first_seen）。"""
    anchor_date = date_str[:10] if date_str else ""
    anchor = parse_time(anchor_date) if anchor_date else None
    result: list[Event] = []
    for raw in history_events:
        event = _as_event(raw)
        if anchor is None:
            result.append(event)
            continue
        stamp = _event_stamp(event)
        if stamp is None:
            result.append(event)
            continue
        delta_days = (anchor - stamp).total_seconds() / 86400.0
        if -1.0 <= delta_days <= history_days:
            result.append(event)
    return result


# ---------------------------------------------------------------------------
# 历史匹配
# ---------------------------------------------------------------------------


def match_history_event(
    event: Event,
    history_events: list[Event],
    config: dict[str, Any],
) -> MatchResult | None:
    """在历史事件中找最佳匹配（标题相似 / 主体 Jaccard 双通道）。"""
    title_threshold = float(cfg_get(config, "continuity.title_threshold", 0.75))
    entity_threshold = float(cfg_get(config, "continuity.entity_jaccard_threshold", 0.60))
    title_floor = float(cfg_get(config, "continuity.entity_title_floor", 0.35))
    single_floor = float(cfg_get(config, "continuity.entity_title_floor_single", 0.50))

    best: MatchResult | None = None
    for history in history_events:
        title_sim = title_similarity(event.representative_title, history.representative_title)
        jaccard, shared = match_subjects(event.labels, history.labels)
        subjects_available = bool(event.labels.subjects or history.labels.subjects)
        title_ok = title_sim >= title_threshold and (shared > 0 or not subjects_available)
        floor = title_floor if shared >= 2 else single_floor
        entity_ok = jaccard >= entity_threshold and title_sim >= floor and shared > 0
        if not (title_ok or entity_ok):
            continue
        score = 0.6 * title_sim + 0.4 * jaccard
        candidate = MatchResult(
            event=history,
            title_similarity=title_sim,
            subject_jaccard=jaccard,
            score=score,
        )
        if best is None or _match_sort_key(candidate) > _match_sort_key(best):
            best = candidate
    return best


def _match_sort_key(match: MatchResult) -> tuple[int, int, str, str]:
    """同分数档内优先取同事件线的“最新 sequence”，避免命中最老的历史成员。"""
    stamp = match.event.last_update_at or match.event.first_seen_at or ""
    return (int(match.score / 0.05), match.event.sequence, stamp, match.event.event_id)


# ---------------------------------------------------------------------------
# 新进展检测
# ---------------------------------------------------------------------------

_DEFAULT_PROGRESS_NUMBER_TYPES = ("version", "money", "percent", "score", "count")


def _number_keys(numbers: list[dict[str, Any]], allowed_types: set[str]) -> set[tuple[str, str]]:
    return {
        (str(number.get("type", "")), str(number.get("value", "")))
        for number in numbers
        if str(number.get("type", "")) in allowed_types and number.get("value")
    }


def detect_new_progress(
    current_text: str,
    previous_texts: list[str],
    *,
    current_actions: list[str] | tuple[str, ...] = (),
    previous_actions: list[str] | tuple[str, ...] | None = None,
    current_numbers: list[dict[str, Any]] | None = None,
    previous_numbers: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> ProgressResult:
    """检测相对历史事件是否出现新版本/新数字/新动作。"""
    cfg = config or {}
    allowed_types = {
        str(value)
        for value in cfg_get(
            cfg, "continuity.progress_number_types", _DEFAULT_PROGRESS_NUMBER_TYPES
        )
    }
    reasons: list[str] = []

    previous_version_text = " ".join(previous_texts or [])
    previous_versions = extract_version_strings(previous_version_text)
    if previous_numbers:
        previous_versions |= {
            str(number.get("value", ""))
            for number in previous_numbers
            if str(number.get("type", "")) == "version"
        }
    new_versions = extract_version_strings(current_text) - previous_versions
    reasons.extend(f"new_version:{version}" for version in sorted(new_versions))

    current_number_keys = _number_keys(
        current_numbers if current_numbers is not None else extract_novelty_numbers(current_text),
        allowed_types,
    )
    previous_number_keys = _number_keys(
        previous_numbers
        if previous_numbers is not None
        else extract_novelty_numbers(previous_version_text),
        allowed_types,
    )
    previous_number_keys |= set(_number_keys(previous_numbers or [], allowed_types))
    new_numbers = current_number_keys - previous_number_keys
    reasons.extend(f"new_number:{kind}:{value}" for kind, value in sorted(new_numbers))

    if bool(cfg_get(cfg, "continuity.new_progress_actions", True)):
        previous_action_ids = set(previous_actions or [])
        if previous_action_ids:
            new_actions = set(current_actions) - previous_action_ids
            reasons.extend(f"new_action:{action}" for action in sorted(new_actions))

    return ProgressResult(is_new=bool(reasons), reasons=tuple(reasons))


# ---------------------------------------------------------------------------
# 覆盖账本
# ---------------------------------------------------------------------------


def _ledger_entries(ledger: dict[str, Any]) -> list[CoverageEntry]:
    return [
        CoverageEntry.from_dict(raw) for raw in ledger.get("entries") or [] if isinstance(raw, dict)
    ]


def is_event_covered(
    matched: Event,
    ledger: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    """历史事件是否已被覆盖（账本 event_id/continuity_id/标题 + 快照 covered_before）。"""
    if matched.covered_before:
        return True
    if is_covered(ledger, event_id=matched.event_id):
        return True
    title_threshold = float(cfg_get(config, "continuity.covered_title_threshold", 0.85))
    for entry in _ledger_entries(ledger):
        if matched.continuity_id and entry.continuity_id == matched.continuity_id:
            return True
        if (
            entry.title
            and title_similarity(entry.title, matched.representative_title) >= title_threshold
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# continuity 分配
# ---------------------------------------------------------------------------


def _new_continuity_id(event: Event, date_str: str) -> str:
    subject_ids = sorted(
        str(subject.get("id", "")) for subject in event.labels.subjects if subject.get("id")
    )
    key = f"{normalize_title(event.representative_title)}|{date_str}|{','.join(subject_ids)}"
    return "c_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _fallback_continuity_id(history: Event) -> str:
    return "c_" + hashlib.sha256(history.event_id.encode("utf-8")).hexdigest()[:16]


def _min_stamp(left: str, right: str) -> str:
    left_time = parse_time(left)
    right_time = parse_time(right)
    if left_time is None:
        return right
    if right_time is None:
        return left
    return left if left_time <= right_time else right


def _max_stamp(left: str, right: str) -> str:
    left_time = parse_time(left)
    right_time = parse_time(right)
    if left_time is None:
        return right
    if right_time is None:
        return left
    return left if left_time >= right_time else right


def _text_of(event: Event) -> str:
    return f"{event.representative_title} {event.summary}".strip()


def _assign_lifecycle(
    event: Event,
    *,
    matched: Event | None,
    config: dict[str, Any],
) -> None:
    first_seen = parse_time(event.first_seen_at)
    last_update = parse_time(event.last_update_at)
    age_days = days_between(first_seen, last_update) if first_seen and last_update else 0.0
    gap_days = 0.0
    if matched is not None:
        market_stamp = _event_stamp(matched)
        current_stamp = last_update
        if market_stamp is not None and current_stamp is not None:
            gap_days = days_between(market_stamp, current_stamp)
    event.labels.lifecycle = infer_lifecycle(
        sequence=event.sequence,
        source_count=event.source_count,
        is_update=event.is_update,
        age_days=age_days,
        gap_days=gap_days,
        config=config,
    )


def _handle_matched_event(
    event: Event,
    match: MatchResult,
    ledger: dict[str, Any],
    config: dict[str, Any],
) -> str | None:
    """处理命中历史的单个事件，返回过滤原因（None 表示保留）。"""
    history = match.event
    progress = detect_new_progress(
        _text_of(event),
        [_text_of(history)],
        current_actions=event.labels.actions,
        previous_actions=history.labels.actions,
        current_numbers=event.labels.novelty_numbers,
        previous_numbers=history.labels.novelty_numbers,
        config=config,
    )
    covered = is_event_covered(history, ledger, config)
    continuity_id = history.continuity_id or _fallback_continuity_id(history)
    event.continuity_id = continuity_id
    event.first_seen_at = _min_stamp(
        event.first_seen_at or event.last_update_at, history.first_seen_at
    )
    event.last_update_at = _max_stamp(event.last_update_at, history.last_update_at)
    if progress.is_new:
        max_sequence = int(cfg_get(config, "continuity.max_sequence", 999))
        event.sequence = min(history.sequence + 1, max_sequence)
        event.is_update = True
        event.covered_before = covered
        event.reappeared = False
        log.info(
            "Continuity update for '%s': sequence %s→%s (%s)",
            event.representative_title,
            history.sequence,
            event.sequence,
            ", ".join(progress.reasons),
        )
        return None
    event.sequence = history.sequence
    event.is_update = False
    event.covered_before = covered
    if covered:
        return "covered_before"
    event.reappeared = True
    return None


def assign_continuity(
    events: list[Event],
    *,
    history_events: list[Event | dict[str, Any]] | None = None,
    ledger: dict[str, Any] | None = None,
    date_str: str = "",
    config: dict[str, Any] | None = None,
) -> ContinuityOutcome:
    """为今日事件分配 continuity_id/sequence，并过滤“已覆盖且无新进展”的事件。"""
    cfg = config or {}
    ledger = ledger or {"version": 1, "entries": []}
    history_days = int(cfg_get(cfg, "continuity.history_days", 30))
    history = filter_history_window(list(history_events or []), date_str, history_days)

    outcome = ContinuityOutcome()
    for event in events:
        match = match_history_event(event, history, cfg)
        if match is None:
            event.continuity_id = _new_continuity_id(event, date_str)
            event.sequence = 1
            event.is_update = False
            event.covered_before = False
            _assign_lifecycle(event, matched=None, config=cfg)
            outcome.events.append(event)
            outcome.new_events += 1
            continue
        outcome.matched_events[event.event_id] = match.event.event_id
        drop_reason = _handle_matched_event(event, match, ledger, cfg)
        _assign_lifecycle(event, matched=match.event, config=cfg)
        if drop_reason is None:
            outcome.events.append(event)
            outcome.updates += int(event.is_update)
        else:
            outcome.dropped.append(event)
            outcome.drop_reasons[event.event_id] = drop_reason
            log.info(
                "Filtered covered event '%s' (matched history %s, no new progress).",
                event.representative_title,
                match.event.event_id,
            )
    outcome.lines = build_continuity_lines(outcome.events, history)
    return outcome


def build_continuity_lines(
    events: list[Event],
    history_events: list[Event],
) -> list[dict[str, Any]]:
    """按 continuity_id 汇总事件线（包含历史成员）。"""
    history_by_continuity: dict[str, list[Event]] = {}
    for event in history_events:
        if event.continuity_id:
            history_by_continuity.setdefault(event.continuity_id, []).append(event)

    lines: dict[str, dict[str, Any]] = {}
    for event in events:
        continuity_id = event.continuity_id or event.event_id
        line = lines.setdefault(
            continuity_id,
            {
                "continuity_id": continuity_id,
                "event_id": event.event_id,
                "representative_title": event.representative_title,
                "sequence": event.sequence,
                "first_seen_at": event.first_seen_at,
                "last_update_at": event.last_update_at,
                "event_ids": [],
                "history_event_ids": [],
                "is_update": event.is_update,
                "subjects": [
                    str(subject.get("id", ""))
                    for subject in event.labels.subjects
                    if subject.get("id")
                ],
                "topics": list(event.labels.topics),
            },
        )
        if event.event_id not in line["event_ids"]:
            line["event_ids"].append(event.event_id)
        line["sequence"] = max(int(line["sequence"]), event.sequence)
        line["first_seen_at"] = _min_stamp(line["first_seen_at"], event.first_seen_at)
        line["last_update_at"] = _max_stamp(line["last_update_at"], event.last_update_at)
        line["is_update"] = bool(line["is_update"]) or event.is_update
        for history in history_by_continuity.get(continuity_id, []):
            if history.event_id not in line["history_event_ids"]:
                line["history_event_ids"].append(history.event_id)
            if history.event_id not in line["event_ids"]:
                line["event_ids"].append(history.event_id)
            line["first_seen_at"] = _min_stamp(line["first_seen_at"], history.first_seen_at)
            line["last_update_at"] = _max_stamp(line["last_update_at"], history.last_update_at)

    return sorted(
        lines.values(),
        key=lambda row: (
            -int(row["sequence"]),
            str(row["first_seen_at"]),
            str(row["continuity_id"]),
        ),
    )
