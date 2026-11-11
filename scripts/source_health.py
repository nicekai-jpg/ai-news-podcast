#!/usr/bin/env python3
"""聚合多天 raw artifact，输出采集源健康度报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_news_podcast.business.collect.health import aggregate_source_health, render_markdown
from ai_news_podcast.business.contracts.artifact import read_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="采集源健康度")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--days", type=int, default=7, help="只看最近 N 个 artifact")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    files = sorted(Path(args.raw_dir).glob("raw-*.json.gz"))[-args.days :]
    snapshots = [read_artifact(path) for path in files]
    healths = aggregate_source_health(snapshots)
    markdown = render_markdown(healths)
    print(markdown)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps([health.to_dict() for health in healths], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
