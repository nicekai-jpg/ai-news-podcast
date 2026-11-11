"""Stage 2「新闻潮汐」编排：raw artifact → 去重 → 聚类 → 证据 → 标注 → 连续性 → 快照。

输出目录（默认）：`data/ocean/{date}/`
- `events.json`：`list[contracts.Event.to_dict()]`
- `items.json`：条目级（含 item 字段 + event_id/continuity_id/labels/dup_of/dup_reason）
- `continuity.json`：事件线
- `manifest.json`：参数/统计/输入
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ai_news_podcast.business.contracts import (
    Event,
    Item,
    canonical_link,
    dumps,
    make_event_id,
    utc_now_iso,
)
from ai_news_podcast.business.contracts.artifact import read_artifact
from ai_news_podcast.business.coverage.ledger import load_ledger
from ai_news_podcast.business.ocean.cluster import ClusterResult, cluster_items
from ai_news_podcast.business.ocean.config import cfg_get, load_ocean_config
from ai_news_podcast.business.ocean.continuity import assign_continuity
from ai_news_podcast.business.ocean.dedupe import DUP_CROSS_DAY, dedupe_items
from ai_news_podcast.business.ocean.evidence import score_event
from ai_news_podcast.business.ocean.labels import (
    LabeledItem,
    label_item,
    load_lexicon,
    merge_labels,
)
from ai_news_podcast.business.ocean.source_tier import best_tier
from ai_news_podcast.business.ocean.textutils import clean_text, parse_time

log = logging.getLogger(__name__)


@dataclass
class OceanSnapshot:
    """一次 build_ocean 的结果（events 为 contracts.Event，items/continuity 为可写 JSON）。"""

    date: str = ""
    events: list[Event] = field(default_factory=list)
    items: list[dict[str, Any]] = field(default_factory=list)
    continuity: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "events": [event.to_dict() for event in self.events],
            "items": self.items,
            "continuity": self.continuity,
            "manifest": self.manifest,
        }


def _resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _clean_item(item: Item) -> Item:
    return replace(
        item,
        title=clean_text(item.title),
        summary=clean_text(item.summary),
        normalized_link=item.normalized_link or canonical_link(item.link),
    )


def _date_dir_value(name: str) -> str:
    return name.strip()[:10]


def _load_history_events(
    ocean_root: Path,
    date_str: str,
    history_days: int,
) -> list[Event]:
    """扫描最近 history_days 天的 `{date}/events.json`（跳过当天/未来目录）。"""
    if not ocean_root.exists():
        return []
    anchor = parse_time(date_str[:10])
    events: list[Event] = []
    for child in sorted(ocean_root.iterdir()):
        if not child.is_dir():
            continue
        directory_date = _date_dir_value(child.name)
        stamp = parse_time(directory_date)
        if stamp is None or not is_date_dirname(directory_date):
            continue
        # 跳过当天与未来目录：当天快照是本次输出，不能当作历史（保证重跑幂等）
        if anchor is not None and stamp >= anchor:
            continue
        if anchor is not None and (anchor - stamp).days > history_days:
            continue
        events_path = child / "events.json"
        if not events_path.exists():
            continue
        try:
            rows = json.loads(events_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Failed to read history events at %s: %s", events_path, exc)
            continue
        if isinstance(rows, list):
            events.extend(Event.from_dict(row) for row in rows if isinstance(row, dict))
    return events


def is_date_dirname(name: str) -> bool:
    """目录名是否为 YYYY-MM-DD。"""
    if len(name) != 10 or name[4] != "-" or name[7] != "-":
        return False
    return name[:4].isdigit() and name[5:7].isdigit() and name[8:10].isdigit()


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(dumps(payload) + "\n", encoding="utf-8")
    temp.replace(path)
    return path


def _build_event(
    cluster: ClusterResult,
    items_by_id: dict[str, Item],
    labeled: dict[str, LabeledItem],
    config: dict[str, Any],
) -> Event:
    members = [items_by_id[item_id] for item_id in cluster.item_ids]
    evidence = score_event(members, config=config)
    labels = merge_labels(
        [labeled[item_id] for item_id in cluster.item_ids],
        evidence_tier=evidence.tier,
        config=config,
    )
    labels.confidence["fact_consistency"] = evidence.fact_consistency
    representative = items_by_id[cluster.representative_item_id]
    published = [
        parsed
        for parsed in (parse_time(item.published_at) for item in members)
        if parsed is not None
    ]
    summary = representative.summary or clean_text(representative.full_text)[:240]
    return Event(
        event_id=make_event_id(cluster.item_ids),
        representative_title=representative.title,
        summary=summary,
        item_ids=list(cluster.item_ids),
        source_count=evidence.source_count,
        independent_sources=evidence.independent_sources,
        first_seen_at=min(published).isoformat() if published else utc_now_iso(),
        last_update_at=max(published).isoformat() if published else utc_now_iso(),
        continuity_id="",
        sequence=1,
        is_update=False,
        labels=labels,
        source_categories=sorted({str(item.category or "news") for item in members}),
        best_source_tier=best_tier(members),
        evidence_score=evidence.score,
        momentum_growth=evidence.momentum_growth,
        reappeared=False,
        covered_before=False,
    )


def _item_rows(
    items: list[Item],
    labeled: dict[str, LabeledItem],
    events: list[Event],
    dropped_events: list[Event],
    drop_reasons: dict[str, str],
    matched_events: dict[str, str],
    duplicates: dict[str, Any],
    reappeared: set[str],
) -> list[dict[str, Any]]:
    """条目级输出：保留原始 item 字段，附加事件/标签/去重信息。"""
    event_by_item: dict[str, Event] = {}
    for event in events:
        for item_id in event.item_ids:
            event_by_item[item_id] = event
    dropped_by_item: dict[str, Event] = {}
    for event in dropped_events:
        for item_id in event.item_ids:
            dropped_by_item[item_id] = event

    rows: list[dict[str, Any]] = []
    for item in items:
        duplicate = duplicates.get(item.item_id)
        row: dict[str, Any] = {
            **item.to_dict(),
            "event_id": "",
            "continuity_id": "",
            "labels": {},
            "dup_of": duplicate.dup_of if duplicate else "",
            "dup_reason": duplicate.reason if duplicate else "",
            "reappeared": item.item_id in reappeared,
            "covered_before": False,
        }
        event = event_by_item.get(item.item_id)
        if event is not None:
            label = labeled.get(item.item_id)
            row.update(
                event_id=event.event_id,
                continuity_id=event.continuity_id,
                labels=label.labels.to_dict() if label else {},
                covered_before=event.covered_before,
                reappeared=event.reappeared,
            )
        elif duplicate is not None and duplicate.reason != DUP_CROSS_DAY:
            survivor_event = event_by_item.get(str(duplicate.dup_of))
            if survivor_event is not None:
                row.update(
                    event_id=survivor_event.event_id,
                    continuity_id=survivor_event.continuity_id,
                )
        elif item.item_id in dropped_by_item:
            dropped = dropped_by_item[item.item_id]
            row.update(
                dup_of=matched_events.get(dropped.event_id, row["dup_of"]),
                dup_reason=drop_reasons.get(dropped.event_id, "covered_before"),
                covered_before=True,
            )
        rows.append(row)
    return rows


def _params_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    keys = ("dedupe", "cluster", "continuity", "evidence", "preprocess")
    return {key: config.get(key) for key in keys}


def _stats_snapshot(
    items: list[Item],
    dedup: Any,
    events: list[Event],
    dropped: list[Event],
    continuity_updates: int,
    new_events: int,
) -> dict[str, Any]:
    return {
        "input_items": len(items),
        "after_dedupe": len(dedup.survivors),
        "duplicates": len(dedup.duplicates),
        "events": len(events),
        "filtered_covered": len(dropped),
        "continuity_updates": continuity_updates,
        "new_events": new_events,
        "reappeared_events": sum(1 for event in events if event.reappeared),
        "covered_events": sum(1 for event in events if event.covered_before),
        "dedupe_layers": dict(dedup.stats),
    }


def build_ocean(
    raw_artifact_path: str | Path,
    *,
    date_str: str,
    history_dir: str | Path | None = None,
    ledger_path: str | Path | dict[str, Any] | None = None,
    config_path: str | Path = "config/ocean_v3.yaml",
) -> OceanSnapshot:
    """Stage 2 主入口：读取 raw artifact，产出 `data/ocean/{date}/` 快照。"""
    if not date_str or not is_date_dirname(date_str[:10]):
        raise ValueError(f"date_str must be YYYY-MM-DD, got {date_str!r}")
    artifact_path = Path(raw_artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Raw artifact not found: {artifact_path}")

    resolved_config = _resolve_config_path(config_path)
    config = load_ocean_config(resolved_config)
    nlp_dir = Path(str(config.get("_nlp_dir") or resolved_config.parent / "nlp"))
    lexicon = load_lexicon(nlp_dir)

    manifest, raw_items = read_artifact(artifact_path)
    items = [_clean_item(item) for item in raw_items]
    log.info("Stage 2 input: %s items from %s", len(items), artifact_path)

    ocean_root = (
        Path(history_dir)
        if history_dir is not None
        else artifact_path.resolve().parent.parent / "ocean"
    )
    history_days = int(cfg_get(config, "continuity.history_days", 30))
    history_events = _load_history_events(ocean_root, date_str, history_days)

    if isinstance(ledger_path, dict):
        ledger = ledger_path
    elif ledger_path is not None:
        ledger = load_ledger(Path(ledger_path))
    else:
        ledger = {"version": 1, "entries": []}

    labeled: dict[str, LabeledItem] = {
        item.item_id: label_item(item, lexicon, config) for item in items
    }
    dedup = dedupe_items(
        items,
        labeled=labeled,
        history_events=history_events,
        ledger=ledger,
        config=config,
    )
    items_by_id = {item.item_id: item for item in dedup.survivors}
    cluster_outcome = cluster_items(
        dedup.survivors,
        labeled={item_id: labeled[item_id] for item_id in items_by_id},
        config=config,
    )
    events = [
        _build_event(cluster, items_by_id, labeled, config) for cluster in cluster_outcome.clusters
    ]
    reappeared_ids = dedup.reappeared
    for event in events:
        event.reappeared = any(item_id in reappeared_ids for item_id in event.item_ids)

    continuity = assign_continuity(
        events,
        history_events=history_events,
        ledger=ledger,
        date_str=date_str,
        config=config,
    )
    rows = _item_rows(
        items,
        labeled,
        continuity.events,
        continuity.dropped,
        continuity.drop_reasons,
        continuity.matched_events,
        dedup.duplicates,
        reappeared_ids,
    )
    stats = _stats_snapshot(
        items,
        dedup,
        continuity.events,
        continuity.dropped,
        continuity.updates,
        continuity.new_events,
    )
    manifest_payload = {
        "date": date_str,
        "generated_at": utc_now_iso(),
        "artifact": str(artifact_path),
        "config": str(resolved_config),
        "nlp_dir": str(nlp_dir),
        "history_days": history_days,
        "history_events": len(history_events),
        "parameters": _params_snapshot(config),
        "stats": stats,
        "history_input": manifest,
        "notes": [
            "evidence_tier/lifecycle only feed Stage 3 coarse recall & commentary (v5).",
            "labels.confidence.fact_consistency exposes cross-source fact consistency.",
            "items.json keeps full_text for downstream evidence packs.",
        ],
    }

    output_dir = ocean_root / date_str[:10]
    _write_json(output_dir / "events.json", [event.to_dict() for event in continuity.events])
    _write_json(output_dir / "continuity.json", continuity.lines)
    _write_json(output_dir / "items.json", rows)
    _write_json(output_dir / "manifest.json", manifest_payload)
    log.info(
        "Stage 2 snapshot written: %s events, %s items, %s continuity lines → %s",
        len(continuity.events),
        len(rows),
        len(continuity.lines),
        output_dir,
    )
    return OceanSnapshot(
        date=date_str,
        events=continuity.events,
        items=rows,
        continuity=continuity.lines,
        manifest=manifest_payload,
    )
