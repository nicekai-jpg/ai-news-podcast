"""事实卡归并与落盘（事实层 S7）。"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from ai_news_podcast.business.kg.types import DayStats, FactCard

logger = logging.getLogger(__name__)

FACTS_DIRNAME = "facts"


def _slug(text: str) -> str:
    return "".join(str(text or "").split()).lower()


def fact_key(
    date: str,
    subject_id: str,
    subject_name: str,
    relation: str,
    object_text: str,
    peer_text: str = "",
) -> str:
    """事实去重键：日期 + 主体 + 关系 + 对象 + 合作方。"""
    raw = "|".join(
        [
            date,
            _slug(subject_id or subject_name),
            relation,
            _slug(object_text),
            _slug(peer_text),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _merge_two(old: FactCard, new: FactCard) -> FactCard:
    """同 fact_id 合并：档位取高者为主，来源累加；理由/证据跟随主卡。"""
    winner = old
    if new.grade == "A" and old.grade != "A":
        winner = new
    sources = tuple(sorted(set(old.sources) | set(new.sources)))
    return FactCard(
        fact_id=old.fact_id,
        date=old.date,
        subject=winner.subject,
        role=winner.role,
        relation=winner.relation,
        relation_raw=winner.relation_raw,
        slots=winner.slots,
        polarity=winner.polarity,
        modality=winner.modality,
        grade=winner.grade,
        grade_reasons=winner.grade_reasons,
        evidence=winner.evidence,
        source=winner.source,
        continuity_id=old.continuity_id,
        is_new=bool(old.is_new or new.is_new),
        sources=sources,
        created_at=old.created_at,
    )


def merge_cards(cards: list[FactCard]) -> list[FactCard]:
    """同 fact_id 合并：来源累加、证据取最早、档位取高。"""
    by_id: dict[str, FactCard] = {}
    for card in cards:
        existing = by_id.get(card.fact_id)
        if existing is None:
            by_id[card.fact_id] = card
        else:
            by_id[card.fact_id] = _merge_two(existing, card)
    return sorted(
        by_id.values(),
        key=lambda card: (
            str(card.subject.get("id") or card.subject.get("name") or ""),
            card.relation,
            card.fact_id,
        ),
    )


def facts_path(facts_dir: Path, date: str) -> Path:
    return Path(facts_dir) / f"facts-{date}.jsonl"


def manifest_path(facts_dir: Path, date: str) -> Path:
    return Path(facts_dir) / f"manifest-{date}.json"


def c_samples_path(debug_dir: Path, date: str) -> Path:
    return Path(debug_dir) / f"c-{date}.jsonl"


def write_jsonl_atomic(path: Path, rows: list[dict]) -> Path:
    """先写 .tmp 再原子改名，避免半截文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(target)
    return target


def write_day(
    facts_dir: Path,
    date: str,
    cards: list[FactCard],
    *,
    rules_version: str,
    stats: DayStats,
    partial: bool,
    elapsed_ms: int,
    alert: str = "",
) -> Path:
    """写一天的事实文件 + manifest。"""
    payload = [card.to_dict() for card in cards]
    path = write_jsonl_atomic(facts_path(facts_dir, date), payload)
    manifest = {
        "date": date,
        "generated_at": _now_iso(),
        "rules_version": rules_version,
        "stats": stats.to_dict(),
        "partial": bool(partial),
        "elapsed_ms": int(elapsed_ms),
        "alert": alert,
    }
    write_jsonl_atomic(manifest_path(facts_dir, date).with_suffix(".json"), [manifest])
    return path


def _now_iso() -> str:
    from ai_news_podcast.business.contracts import utc_now_iso

    return utc_now_iso()


def load_day(facts_dir: Path, date: str) -> list[dict]:
    path = facts_path(facts_dir, date)
    if not path.is_file():
        return []
    rows: list[dict] = []
    for raw_line in path.open(encoding="utf-8"):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("事实文件坏行：%s", path)
    return rows


def read_all_facts(facts_dir: Path, *, days: int = 0) -> list[dict]:
    """读取全部事实；days>0 时只看最近 N 天。"""
    directory = Path(facts_dir)
    if not directory.is_dir():
        return []
    files = sorted(directory.glob("facts-*.jsonl"))
    if days > 0:
        files = files[-days:]
    rows: list[dict] = []
    for path in files:
        date = path.stem.replace("facts-", "")
        rows.extend(load_day(directory, date))
    return rows


def write_c_samples(debug_dir: Path, date: str, samples: list[dict]) -> Path | None:
    if not samples:
        return None
    return write_jsonl_atomic(c_samples_path(debug_dir, date), samples)


__all__ = [
    "FACTS_DIRNAME",
    "c_samples_path",
    "fact_key",
    "facts_path",
    "load_day",
    "manifest_path",
    "merge_cards",
    "read_all_facts",
    "write_c_samples",
    "write_day",
]
