#!/usr/bin/env python3
"""主题域覆盖缺口报告：看 unknown/other 里还有哪些没覆盖的标题。

用法：
    uv run python scripts/domain_coverage.py \
        --events data/ocean/2026-11-12/events.json \
        --out-md data/domain_coverage/report.md \
        --out-queue data/domain_coverage/review.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ai_news_podcast.business.ocean.domain import classify_domain


def main() -> int:
    parser = argparse.ArgumentParser(description="主题域覆盖缺口")
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-queue", default="")
    args = parser.parse_args()

    rows = []
    for path_str in args.events:
        path = Path(path_str)
        files = sorted(path.glob("*/events.json")) if path.is_dir() else [path]
        for file_path in files:
            for event in json.loads(file_path.read_text(encoding="utf-8")):
                title = str(event.get("representative_title") or "").strip()
                if not title:
                    continue
                domain, evidence = classify_domain(title)
                rows.append({"title": title, "domain": domain, "evidence": evidence})

    counts = Counter(row["domain"] for row in rows)
    unknown = [row for row in rows if row["domain"] == "unknown"]
    other = [row for row in rows if row["domain"] == "other"]

    lines = [
        "# 主题域覆盖报告",
        "",
        f"- 事件数：{len(rows)}",
        f"- ai_core：{counts.get('ai_core', 0)}",
        f"- ai_infra：{counts.get('ai_infra', 0)}",
        f"- other：{counts.get('other', 0)}",
        f"- unknown：{counts.get('unknown', 0)}",
        "",
        "## unknown 标题（最可能需要补词）",
        "",
    ]
    lines += [f"- {row['title']}" for row in unknown[:100]]
    lines += ["", "## other 标题（抽查是否误杀）", ""]
    lines += [f"- {row['title']}" for row in other[:100]]
    markdown = "\n".join(lines) + "\n"
    print(markdown[:2000])

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    if args.out_queue:
        Path(args.out_queue).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_queue).open("w", encoding="utf-8") as handle:
            for row in unknown:
                handle.write(
                    json.dumps(
                        {
                            "title": row["title"],
                            "current_domain": "unknown",
                            "suggested_action": "",
                            "note": "",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
