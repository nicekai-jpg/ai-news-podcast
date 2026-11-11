#!/usr/bin/env python3
"""交互式标注评测队列（终端问答，逐条写盘、可断点续标）。

用法：
    uv run python scripts/label_queue.py \
        --queue data/eval/recommend/days/2026-11-11.jsonl \
        --out data/eval/recommend/labels_2026-11-11.jsonl \
        --limit 50

每行会显示机器建议，直接回车表示接受建议。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

RELEVANCE_KEYS = {"c": "core", "a": "adjacent", "n": "none", "u": "unknown"}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _prompt_relevance(default: str) -> str:
    while True:
        raw = input(f"AI 相关性 [c/a/n/u，回车={default}，s=跳过]: ").strip().lower()
        if raw == "":
            return default
        if raw == "s":
            return "skip"
        if raw in RELEVANCE_KEYS:
            return RELEVANCE_KEYS[raw]
        if raw in RELEVANCE_KEYS.values():
            return raw


def _prompt_worth(default: bool) -> bool:
    while True:
        raw = input(f"值得上节目 [y/n，回车={'y' if default else 'n'}，s=跳过]: ").strip().lower()
        if raw == "":
            return default
        if raw == "s":
            return False
        if raw in ("y", "n"):
            return raw == "y"


def main() -> int:
    parser = argparse.ArgumentParser(description="交互式标注评测队列")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="本次最多标多少条；0=全部")
    args = parser.parse_args()

    queue = _load_jsonl(Path(args.queue))
    out_path = Path(args.out)
    done_titles = {row.get("title") for row in _load_jsonl(out_path)}
    todo = [row for row in queue if row.get("title") not in done_titles]
    if args.limit:
        todo = todo[: args.limit]

    print(f"队列 {len(queue)} 条，已标 {len(done_titles)} 条，本次 {len(todo)} 条")
    counts: Counter[str] = Counter()
    skipped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("a", encoding="utf-8") as handle:
            for index, row in enumerate(todo, start=1):
                print(f"\n[{index}/{len(todo)}] {row.get('date', '')}")
                print(row.get("title", ""))
                summary = str(row.get("summary") or "")
                if summary:
                    print(f"摘要：{summary[:220]}")
                default = str(row.get("suggested_ai_relevance") or "unknown")
                relevance = _prompt_relevance(default)
                if relevance == "skip":
                    skipped += 1
                    continue
                worth_default = relevance in ("core", "adjacent")
                worth = _prompt_worth(worth_default)
                note = input("备注（可空）: ").strip()
                payload = {
                    **row,
                    "ai_relevance": relevance,
                    "worth_publishing": worth,
                    "label_source": "human",
                    "reviewed": True,
                    "note": note,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                counts[relevance] += 1
    except (KeyboardInterrupt, EOFError):
        print("\n已中断，进度已保存。")

    print(f"\n本次完成 {sum(counts.values())} 条，跳过 {skipped} 条：{dict(counts)}")
    print(f"输出：{out_path}")
    return 0 if len(todo) == sum(counts.values()) + skipped else 130


if __name__ == "__main__":
    sys.exit(main())
