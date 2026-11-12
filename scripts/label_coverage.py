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
    args = parser.parse_args()
    events = []
    for path_str in args.events:
        path = Path(path_str)
        files = sorted(path.glob("*/events.json")) if path.is_dir() else [path]
        for file_path in files:
            events.extend(
                Event.from_dict(row) for row in json.loads(file_path.read_text(encoding="utf-8"))
            )
    markdown = render_markdown(label_coverage(events))
    print(markdown)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
