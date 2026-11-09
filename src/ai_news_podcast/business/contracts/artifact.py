"""Stage 1 采集产物（raw artifact）的读写与校验。

artifact 是阶段间唯一传递通道：Stage 1 写入，Stage 2 读取；
不经过 git，默认放 GHA artifact 或本地 data/raw/。
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_news_podcast.business.contracts import Item, utc_now_iso


@dataclass
class ArtifactManifest:
    date: str
    generated_at: str = ""
    window_hours: int = 72
    source_stats: list[dict[str, Any]] = field(default_factory=list)
    item_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "generated_at": self.generated_at or utc_now_iso(),
            "window_hours": self.window_hours,
            "source_stats": self.source_stats,
            "item_count": self.item_count,
        }


def write_artifact(
    path: Path,
    items: list[Item],
    *,
    date_str: str,
    window_hours: int = 72,
    source_stats: list[dict[str, Any]] | None = None,
) -> Path:
    """写入 gzip JSON artifact（`.json.gz`），返回路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = ArtifactManifest(
        date=date_str,
        window_hours=window_hours,
        source_stats=source_stats or [],
        item_count=len(items),
    ).to_dict()
    payload = {"manifest": manifest, "items": [item.to_dict() for item in items]}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with gzip.open(path, "wb") as fh:
        fh.write(raw)
    return path


def read_artifact(path: Path) -> tuple[dict[str, Any], list[Item]]:
    """读取 artifact，返回 (manifest, items)。"""
    with gzip.open(Path(path), "rb") as fh:
        payload = json.loads(fh.read().decode("utf-8"))
    manifest = payload.get("manifest") or {}
    items = [Item.from_dict(row) for row in payload.get("items") or []]
    return manifest, items


def latest_artifact(data_dir: Path) -> Path | None:
    """返回 data_dir 下最新的 raw-*.json.gz（按文件名日期排序）。"""
    files = sorted(Path(data_dir).glob("raw-*.json.gz"))
    return files[-1] if files else None
