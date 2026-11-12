#!/usr/bin/env python3
"""每日运维快照：规模、推荐、点评质量、覆盖率、源健康度。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from ai_news_podcast.business.editorial.review_quality import review_quality
from ai_news_podcast.business.eval.label_coverage import label_coverage


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _size_kb(path: Path) -> int:
    return round(path.stat().st_size / 1024) if path.is_file() else 0


def build_snapshot(data_dir: Path, date_str: str) -> dict[str, Any]:
    ocean = data_dir / "ocean" / date_str
    events = _load_json(ocean / "events.json", [])
    shortlist = _load_json(data_dir / "selection" / date_str / "shortlist.json", [])
    editorial = data_dir / "editorial" / date_str
    report_review = _load_json(editorial / "report_review.json", {})
    podcast_review = _load_json(editorial / "podcast_review.json", {})
    coverage = label_coverage(events) if events else {"warnings": [], "fields": {}}
    health_path = ocean / "source_health.json"
    health = (
        yaml.safe_load(health_path.read_text(encoding="utf-8")) if health_path.is_file() else {}
    )
    return {
        "date": date_str,
        "events": len(events),
        "shortlist": len(shortlist),
        "events_kb": _size_kb(ocean / "events.json"),
        "report_quality": review_quality(report_review, kind="report") if report_review else None,
        "podcast_quality": (
            review_quality(podcast_review, kind="podcast") if podcast_review else None
        ),
        "coverage_warnings": coverage.get("warnings", []),
        "source_health": health.get("summary") if isinstance(health, dict) else None,
    }


def render(snapshot: dict[str, Any]) -> str:
    lines = [f"# 运维快照 {snapshot['date']}", ""]
    lines.append(f"- 事件数：{snapshot['events']}")
    lines.append(f"- 推荐条数：{snapshot['shortlist']}")
    lines.append(f"- events.json：{snapshot['events_kb']} KB")
    for key, label in (("report_quality", "日报点评"), ("podcast_quality", "播客点评")):
        quality = snapshot.get(key)
        if quality:
            lines.append(f"- {label}质量：{quality['score']}（问题 {len(quality['issues'])} 个）")
    warnings = snapshot.get("coverage_warnings") or []
    lines.append(f"- 覆盖率告警：{len(warnings)} 个")
    if snapshot.get("source_health"):
        lines.append(f"- 源健康度：{snapshot['source_health']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="每日运维快照")
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()
    text = render(build_snapshot(Path(args.data_dir), args.date))
    print(text)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
