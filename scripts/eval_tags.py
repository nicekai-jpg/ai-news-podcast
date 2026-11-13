#!/usr/bin/env python3
"""用金标准评测 Stage 2 标签。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_news_podcast.business.contracts import Event
from ai_news_podcast.business.eval.tagging_eval import evaluate_tagging, load_gold, render_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="标签金标准评测")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()
    events: dict[str, Event] = {}
    for path_str in args.events:
        path = Path(path_str)
        files = sorted(path.glob("*/events.json")) if path.is_dir() else [path]
        for file_path in files:
            for row in json.loads(file_path.read_text(encoding="utf-8")):
                event = Event.from_dict(row)
                events[event.event_id] = event
    report = evaluate_tagging(load_gold(args.gold), events)
    markdown = render_markdown(report)
    print(markdown)
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
