#!/usr/bin/env python3
"""从多天 events artifact 分层抽样生成人工标注模板（JSONL）。

示例：
    uv run python scripts/build_eval_template.py \
        --events data/ocean/2026-11-09/events.json data/ocean/2026-11-10/events.json \
        --out data/eval/recommend/review_queue.jsonl \
        --quota core=200,adjacent=100,none=200,unknown=100

每行带 suggested_ai_relevance（机器建议，只是加快人工标注，不是金标准）：
人工确认/修改 ai_relevance，填 worth_publishing，把 reviewed 改成 true。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_news_podcast.business.eval.sampling import classify_suggestion, stratified_sample


def _load_events(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    seen_titles: set[str] = set()
    for path_str in paths:
        path = Path(path_str)
        files = sorted(path.glob("*/events.json")) if path.is_dir() else [path]
        for file_path in files:
            date = file_path.parent.name
            for event in json.loads(file_path.read_text(encoding="utf-8")):
                title = str(event.get("representative_title") or "").strip()
                if not title:
                    continue
                summary = str(event.get("summary") or "")
                title_key = " ".join(title.lower().split())
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                rows.append(
                    {
                        "date": date,
                        "title": title,
                        "summary": summary[:200],
                        "group": classify_suggestion(title, summary),
                    }
                )
    return rows


def _parse_quota(text: str) -> dict[str, int]:
    quota: dict[str, int] = {}
    for raw in text.split(","):
        part = raw.strip()
        if not part:
            continue
        key, _, value = part.partition("=")
        quota[key.strip()] = int(value.strip())
    return quota


def main() -> int:
    parser = argparse.ArgumentParser(description="生成推荐评测人工标注模板")
    parser.add_argument("--events", nargs="+", required=True, help="events.json 文件或目录")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--quota",
        default="core=200,adjacent=100,none=200,unknown=100",
        help="分组配额，如 core=200,adjacent=100,none=200,unknown=100",
    )
    args = parser.parse_args()

    rows = _load_events(args.events)
    quota = _parse_quota(args.quota)
    sampled = stratified_sample(rows, quota, total_limit=sum(quota.values()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in sampled:
            payload = {
                "date": row["date"],
                "title": row["title"],
                "summary": row["summary"],
                "suggested_ai_relevance": row["group"],
                "suggested_worth_publishing": (
                    None if row["group"] == "unknown" else row["group"] in ("core", "adjacent")
                ),
                "ai_relevance": "",
                "worth_publishing": None,
                "label_source": "human",
                "reviewed": False,
                "note": "",
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for row in sampled:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    print(f"写出 {len(sampled)} 条：{out}")
    print("分组分布：", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
