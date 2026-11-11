#!/usr/bin/env python3
"""按天拆分评测标注队列，便于分次人工标注。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_news_podcast.business.eval.sampling import split_by_day


def main() -> int:
    parser = argparse.ArgumentParser(description="按天拆分标注队列")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.queue).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = split_by_day(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for date, day_rows in groups.items():
        path = out_dir / f"{date}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in day_rows) + "\n",
            encoding="utf-8",
        )
        print(f"{date}: {len(day_rows)} 条 -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
