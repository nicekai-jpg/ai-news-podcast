#!/usr/bin/env python3
"""标签覆盖率审计：缺词/空白率一眼看清。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_news_podcast.business.contracts import Event
from ai_news_podcast.business.eval.label_coverage import label_coverage, render_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="标签覆盖率")
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument(
        "--max-empty-rate", type=float, default=0.0, help="超过则退出码 1（0=不检查）"
    )
    args = parser.parse_args()
    events = []
    for path_str in args.events:
        path = Path(path_str)
        files = sorted(path.glob("*/events.json")) if path.is_dir() else [path]
        for file_path in files:
            events.extend(
                Event.from_dict(row) for row in json.loads(file_path.read_text(encoding="utf-8"))
            )
    report = label_coverage(events)
    markdown = render_markdown(report)
    print(markdown)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    if args.max_empty_rate > 0:
        exceeded = [
            field
            for field, row in report["fields"].items()
            if field not in {"sensitivity"} and row["empty_rate"] > args.max_empty_rate
        ]
        if exceeded:
            print("覆盖率超阈值: " + "、".join(exceeded))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
