#!/usr/bin/env python3
"""从 events artifact 抽样生成人工标注模板（JSONL）。

用法：
    uv run python scripts/build_eval_template.py \
        --events data/ocean/2026-11-11/events.json \
        --out data/eval/recommend/review_queue.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="生成推荐评测人工标注模板")
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--stride", type=int, default=0, help="抽样步长；默认按 limit 均分")
    args = parser.parse_args()

    events = json.loads(Path(args.events).read_text(encoding="utf-8"))
    events = [row for row in events if row.get("representative_title")]
    stride = args.stride or max(1, len(events) // max(1, args.limit))
    sampled = events[::stride][: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for event in sampled:
            row = {
                "title": event.get("representative_title", ""),
                "summary": event.get("summary", ""),
                "ai_relevance": "",
                "worth_publishing": None,
                "label_source": "human",
                "reviewed": False,
                "note": "",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"写出 {len(sampled)} 条标注模板：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
