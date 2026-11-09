"""跨日覆盖账本：记录“已选中并播报过”的事件，供 Stage 2/3 做跨日去重。

账本很小（每期 5–8 条），可长期保留在 git：
`data/coverage/ledger.json`。它只记录“已覆盖”的事件，不记录所有抓过的条目。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_news_podcast.business.contracts import canonical_link


@dataclass
class CoverageEntry:
    event_id: str
    title: str = ""
    links: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    first_covered: str = ""
    last_covered: str = ""
    sequence: int = 1
    continuity_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "links": self.links,
            "subjects": self.subjects,
            "keywords": self.keywords,
            "first_covered": self.first_covered,
            "last_covered": self.last_covered,
            "sequence": self.sequence,
            "continuity_id": self.continuity_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageEntry:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def load_ledger(path: Path) -> dict[str, Any]:
    """读取账本；不存在返回空结构。"""
    path = Path(path)
    if not path.exists():
        return {"version": 1, "entries": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("version", 1)
    data.setdefault("entries", [])
    return data


def save_ledger(path: Path, ledger: dict[str, Any]) -> Path:
    """原子写入账本（先写临时文件再替换）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(path)
    return path


def is_covered(
    ledger: dict[str, Any], *, links: list[str] | None = None, event_id: str = ""
) -> bool:
    """事件/链接是否已被覆盖。"""
    norm = {canonical_link(u) for u in (links or []) if u}
    for raw in ledger.get("entries") or []:
        entry = CoverageEntry.from_dict(raw)
        if event_id and entry.event_id == event_id:
            return True
        if norm and norm & {canonical_link(u) for u in entry.links if u}:
            return True
    return False


def append_entry(ledger: dict[str, Any], entry: CoverageEntry) -> dict[str, Any]:
    """幂等追加：同 event_id 已存在则更新 last_covered/sequence。"""
    entries = ledger.setdefault("entries", [])
    for raw in entries:
        existing = CoverageEntry.from_dict(raw)
        if existing.event_id == entry.event_id:
            existing.last_covered = entry.last_covered or existing.last_covered
            existing.sequence = max(existing.sequence, entry.sequence)
            raw.update(existing.to_dict())
            return ledger
    entries.append(entry.to_dict())
    return ledger
