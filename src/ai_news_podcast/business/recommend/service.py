"""Stage 3 选题推荐编排：召回 → 粗排 → 精排（规则 + LLM）→ 重排 → top N。

对外主接口::

    recommend(events, *, config_path="config/recommend_v3.yaml",
              llm_judge=None, covered_event_ids=None) -> list[ShortlistEntry]

`llm_judge` 是可注入的 LLM rubric 打分函数（生产由 Lead 用 llm_client_service 适配，
单测注入假函数，不联网）。约定输入 `list[Event]`，输出::

    {
        "<event_id>": {
            "impact": 4, "novelty": 3, "momentum": 4,
            "audience_fit": 5, "narratability": 4, "evidence_risk": 4,
            "reason": "一句话中文理由",
        },
        ...
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_news_podcast.business.contracts import Event, ShortlistEntry
from ai_news_podcast.business.recommend.base import ensure_utc
from ai_news_podcast.business.recommend.coarse import CoarseOutcome, coarse_rank
from ai_news_podcast.business.recommend.config import (
    DEFAULT_CONFIG_PATH,
    RecommendConfig,
)
from ai_news_podcast.business.recommend.fine import (
    FineScore,
    LLMJudge,
    apply_judgements,
    score_events,
)
from ai_news_podcast.business.recommend.recall import (
    RecallCandidate,
    format_channels,
    recall_events,
)
from ai_news_podcast.business.recommend.rerank import RerankedItem, rerank

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecommendationResult:
    """推荐全链路结果（含各阶段中间产物，便于审计/回测）。"""

    shortlist: list[ShortlistEntry]
    config: RecommendConfig
    recalled: list[RecallCandidate]
    coarse: CoarseOutcome
    fine_scores: list[FineScore]
    reranked: list[RerankedItem]


def recommend(
    events: list[Event],
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    llm_judge: LLMJudge | None = None,
    covered_event_ids: set[str] | None = None,
    reference_time: datetime | None = None,
) -> list[ShortlistEntry]:
    """Stage 3 入口：给定当日事件，返回按排名排序的短名单。"""
    result = recommend_with_audit(
        events,
        config_path=config_path,
        llm_judge=llm_judge,
        covered_event_ids=covered_event_ids,
        reference_time=reference_time,
    )
    return result.shortlist


def recommend_with_audit(
    events: list[Event],
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    llm_judge: LLMJudge | None = None,
    covered_event_ids: set[str] | None = None,
    reference_time: datetime | None = None,
) -> RecommendationResult:
    """与 `recommend` 相同，但额外返回召回/粗排/精排/重排的中间结果。"""
    config = RecommendConfig.load(config_path)
    reference = ensure_utc(reference_time) if reference_time is not None else datetime.now(tz=UTC)
    covered = set(covered_event_ids or ())
    normalized_events = _coerce_events(events)

    recalled = recall_events(normalized_events, config.recall, reference_time=reference)
    coarse = coarse_rank(
        recalled,
        config.coarse,
        covered_event_ids=covered,
        reference_time=reference,
    )
    scored = score_events([candidate.event for candidate in coarse.candidates], config.fine)
    scored = _apply_llm_judge(scored, llm_judge, config)
    reranked = rerank(
        scored,
        config.rerank,
        covered_event_ids=covered,
        reference_time=reference,
    )
    channels_by_id = {
        candidate.event.event_id: candidate.channels for candidate in coarse.candidates
    }
    shortlist = [
        _to_shortlist_entry(rank, item, channels_by_id.get(item.event.event_id, ()))
        for rank, item in enumerate(reranked, 1)
    ]
    logger.info(
        "推荐完成: 事件 %d → 召回 %d → 粗排 %d → 精排 %d → 输出 %d",
        len(normalized_events),
        len(recalled),
        len(coarse.candidates),
        len(scored),
        len(shortlist),
    )
    return RecommendationResult(
        shortlist=shortlist,
        config=config,
        recalled=recalled,
        coarse=coarse,
        fine_scores=scored,
        reranked=reranked,
    )


def load_events(path: str | Path) -> list[Event]:
    """读取 Stage 2 的 `events.json`（`list[Event.to_dict()]` 或 `{"events": [...]}`）。"""
    event_path = Path(path)
    if not event_path.exists():
        raise FileNotFoundError(f"事件文件不存在: {event_path}")
    with event_path.open("r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    rows: Any = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError(f"events.json 必须是事件列表或含 events 键的对象: {event_path}")
    return [Event.from_dict(row) for row in rows if isinstance(row, dict)]


def _coerce_events(events: list[Event]) -> list[Event]:
    """兼容 list[Event] / list[dict]；非法元素跳过并告警。"""
    normalized: list[Event] = []
    for row in events or []:
        if isinstance(row, Event):
            normalized.append(row)
        elif isinstance(row, dict):
            normalized.append(Event.from_dict(row))
        else:
            logger.warning("忽略非法事件类型: %r", type(row))
    return normalized


def _apply_llm_judge(
    scored: list[FineScore],
    llm_judge: LLMJudge | None,
    config: RecommendConfig,
) -> list[FineScore]:
    """只对精排前 llm_top_k 条调用 LLM；失败降级为纯规则分，不影响推荐结果。"""
    if llm_judge is None or not scored:
        return scored
    judged_events = [item.event for item in scored[: config.fine.llm_top_k]]
    samples = max(1, int(config.fine.llm_samples))
    results: list[dict] = []
    for index in range(samples):
        try:
            judgements = llm_judge(judged_events) or {}
        except Exception:
            logger.warning("LLM judge 第 %d 次调用失败，跳过该样本", index + 1, exc_info=True)
            continue
        if isinstance(judgements, dict) and judgements:
            results.append(judgements)
    if not results:
        logger.warning("LLM judge 无有效结果，降级为纯规则分")
        return scored
    judgements = results[0] if len(results) == 1 else _median_judgements(results)
    logger.debug("LLM judge 评分 %d 条（%d/%d 次采样）", len(judgements), len(results), samples)
    if not isinstance(judgements, dict) or not judgements:
        logger.warning("LLM judge 返回空结果，降级为纯规则分")
        return scored
    logger.debug("LLM judge 评分 %d 条", len(judgements))
    return apply_judgements(scored, judgements, config.fine)


def _median_judgements(samples: list[dict]) -> dict[str, dict]:
    """多次 LLM 采样按维度取中位数，降低单次打分波动。"""
    from ai_news_podcast.business.recommend.base import JUDGE_DIMENSIONS

    merged: dict[str, dict] = {}
    event_ids: set[str] = set()
    for sample in samples:
        event_ids.update(str(key) for key in sample)
    for event_id in event_ids:
        rows = [sample[event_id] for sample in samples if isinstance(sample.get(event_id), dict)]
        if not rows:
            continue
        row: dict = {}
        for dimension in JUDGE_DIMENSIONS:
            values = []
            for item in rows:
                try:
                    values.append(float(item.get(dimension)))
                except (TypeError, ValueError):
                    continue
            if values:
                values.sort()
                row[dimension] = values[len(values) // 2]
        reason = next((str(item.get("reason") or "") for item in rows if item.get("reason")), "")
        if reason:
            row["reason"] = reason
        merged[event_id] = row
    return merged


def _to_shortlist_entry(rank: int, item: RerankedItem, channels: tuple[str, ...]) -> ShortlistEntry:
    event = item.event
    fine = item.fine
    reasons = [f"命中召回通道：{format_channels(channels)}", *fine.reasons]
    reasons.extend(item.reasons)
    if fine.llm_reason:
        reasons.append(f"LLM 理由：{fine.llm_reason}")
    components = {key: round(float(value), 3) for key, value in fine.components.items()}
    components["rule_score"] = round(fine.rule_score, 3)
    components["llm_score"] = round(fine.llm_score or 0.0, 3)
    components["freshness"] = round(item.freshness, 3)
    components["mmr_utility"] = round(item.mmr_utility, 3)
    return ShortlistEntry(
        rank=rank,
        event_id=event.event_id,
        score=round(item.final_score, 3),
        rule_score=round(fine.rule_score, 3),
        llm_score=round(fine.llm_score or 0.0, 3),
        reasons=reasons[:12],
        evidence_item_ids=list(event.item_ids),
        components=components,
        rejected_reason="",
    )


__all__ = [
    "RecommendationResult",
    "load_events",
    "recommend",
    "recommend_with_audit",
]
