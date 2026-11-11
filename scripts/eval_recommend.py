#!/usr/bin/env python3
"""推荐流程离线评测 CLI：支持多天 events artifact + 标注 JSONL。"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from ai_news_podcast.business.contracts import Event
from ai_news_podcast.business.eval.recommend_eval import (
    aggregate_reports,
    evaluate,
    load_labels,
    render_markdown,
)


def _event_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for path_str in paths:
        path = Path(path_str)
        files.extend(sorted(path.glob("*/events.json")) if path.is_dir() else [path])
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="推荐流程离线评测")
    parser.add_argument("--events", nargs="+", required=True, help="events.json 文件或目录")
    parser.add_argument("--labels", required=True, help="标注 JSONL 路径")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--reference-time", default="", help="ISO 时间，默认当前")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    reference = (
        datetime.fromisoformat(args.reference_time.replace("Z", "+00:00"))
        if args.reference_time
        else datetime.now(tz=UTC)
    )
    reports = []
    for file_path in _event_files(args.events):
        events = [Event.from_dict(row) for row in json.loads(file_path.read_text(encoding="utf-8"))]
        reports.append(
            evaluate(
                events,
                labels,
                config_path=Path(args.config_dir) / "recommend_v3.yaml",
                reference_time=reference,
            )
        )
    aggregate = aggregate_reports(reports)
    markdown = render_markdown(aggregate)
    markdown += (
        "\n## 分天明细\n\n| 日期 | 事件数 | top8 正样本 | top8 负样本 |\n|---|---|---|---|\n"
    )
    for file_path, report in zip(_event_files(args.events), reports, strict=True):
        stage = report["stages"]["shortlist"]
        markdown += f"| {file_path.parent.name} | {report['events']} | {stage['positives']} | {stage['negatives']} |\n"
    print(markdown)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
