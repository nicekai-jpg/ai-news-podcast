#!/usr/bin/env python3
"""把规则预标注转成评测标注文件，并抽一份人工抽查样本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def to_eval_label(row: dict[str, Any]) -> dict[str, Any]:
    relevance = str(row.get("prelabel_ai_relevance") or "unknown")
    worth = True if relevance == "core" else (False if relevance == "none" else None)
    return {
        "title": row.get("title", ""),
        "summary": row.get("summary", ""),
        "ai_relevance": relevance,
        "worth_publishing": worth,
        "label_source": "machine_prelabel",
        "reviewed": False,
        "note": "规则预标注，未人工复核；仅冒烟",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="机器预标注 → 评测标注")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--spotcheck-out", default="")
    parser.add_argument("--spotcheck-size", type=int, default=50)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.queue).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    labels = [to_eval_label(row) for row in rows]
    Path(args.out).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in labels) + "\n",
        encoding="utf-8",
    )
    print(f"机器标注 {len(labels)} 条 → {args.out}")
    if args.spotcheck_out:
        picked = stratified(labels, args.spotcheck_size)
        Path(args.spotcheck_out).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in picked) + "\n",
            encoding="utf-8",
        )
        print(f"抽查样本 {len(picked)} 条 → {args.spotcheck_out}")
    return 0


def stratified(labels: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        groups.setdefault(str(label["ai_relevance"]), []).append(label)
    picked: list[dict[str, Any]] = []
    while len(picked) < size and any(groups.values()):
        for key in list(groups):
            if groups[key] and len(picked) < size:
                picked.append(groups[key].pop(0))
    return picked


if __name__ == "__main__":
    raise SystemExit(main())
