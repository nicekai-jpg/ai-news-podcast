#!/usr/bin/env python3
"""推荐流程离线评测 CLI：读事件 artifact + 标注 JSONL，输出分阶段指标。"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from ai_news_podcast.business.contracts import Event
from ai_news_podcast.business.eval.recommend_eval import (
    evaluate,
    load_labels,
    render_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="推荐流程离线评测")
    parser.add_argument("--events", required=True, help="events.json 路径（Stage 2 产物）")
    parser.add_argument("--labels", required=True, help="标注 JSONL 路径")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--reference-time", default="", help="ISO 时间，默认当前")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    events = [
        Event.from_dict(row) for row in json.loads(Path(args.events).read_text(encoding="utf-8"))
    ]
    labels = load_labels(args.labels)
    reference = (
        datetime.fromisoformat(args.reference_time.replace("Z", "+00:00"))
        if args.reference_time
        else datetime.now(tz=UTC)
    )
    report = evaluate(
        events,
        labels,
        config_path=Path(args.config_dir) / "recommend_v3.yaml",
        reference_time=reference,
    )
    markdown = render_markdown(report)
    print(markdown)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
