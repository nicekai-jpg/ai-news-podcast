#!/usr/bin/env python3
"""每日运维快照：规模、推荐、点评质量、覆盖率、源健康度。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from ai_news_podcast.business.contracts import Commentary, Fact
from ai_news_podcast.business.editorial.quality import commentary_quality
from ai_news_podcast.business.editorial.review_quality import review_quality
from ai_news_podcast.business.eval.label_coverage import label_coverage


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _commentary_audit(rows: list[dict]) -> dict[str, Any] | None:
    """S4 日报客观性审计：无证据事实、单源占比、缺出处。

    `commentary_quality` 早已实现但从未接入流水线；这里让它进入每日快照，
    否则「事实没有出处」这类问题永远不可见。
    """
    if not rows:
        return None
    facts = single = no_evidence = no_sources = 0
    for row in rows:
        payload = {k: v for k, v in row.items() if k in Commentary.__dataclass_fields__}
        payload["facts"] = [
            Fact(**f) if isinstance(f, dict) else f for f in payload.get("facts") or []
        ]
        try:
            commentary = Commentary(**payload)
        except (TypeError, ValueError):
            continue
        report = commentary_quality(commentary)
        facts += report["facts"]
        single += report["single_source_facts"]
        no_evidence += len(report["facts_without_evidence"])
        if not commentary.sources:
            no_sources += 1
    return {
        "items": len(rows),
        "facts": facts,
        "single_source_facts": single,
        "single_source_ratio": round(single / facts, 2) if facts else None,
        "facts_without_evidence": no_evidence,
        "items_without_sources": no_sources,
    }


def _script_chapters(path: Path) -> list[str]:
    """读取文稿里的 [CHAPTER] 标题（缺失返回空）。"""
    if not path.is_file():
        return []
    titles: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[CHAPTER]"):
            titles.append(stripped[len("[CHAPTER]") :].strip())
    return titles


def _size_kb(path: Path) -> int:
    return round(path.stat().st_size / 1024) if path.is_file() else 0


def build_snapshot(data_dir: Path, date_str: str) -> dict[str, Any]:
    ocean = data_dir / "ocean" / date_str
    events = _load_json(ocean / "events.json", [])
    shortlist = _load_json(data_dir / "selection" / date_str / "shortlist.json", [])
    editorial = data_dir / "editorial" / date_str
    report_review = _load_json(editorial / "report_review.json", {})
    podcast_review = _load_json(editorial / "podcast_review.json", {})
    script_chapters = _script_chapters(Path("site") / "episodes" / f"{date_str}.txt")
    coverage = label_coverage(events) if events else {"warnings": [], "fields": {}}
    report_rows = _load_json(editorial / "report.json", [])
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
            review_quality(podcast_review, kind="podcast", script_chapters=script_chapters)
            if podcast_review
            else None
        ),
        "commentary_audit": _commentary_audit(report_rows),
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
            lines.append(
                f"- {label}结构完整度：{quality['structure_score']}"
                f"（问题 {len(quality['issues'])} 个）"
            )
    audit = snapshot.get("commentary_audit")
    if audit:
        lines.append(
            f"- 日报事实：{audit['facts']} 条，单源 {audit['single_source_facts']}"
            f"（{audit['single_source_ratio']}），无证据 {audit['facts_without_evidence']} 条"
        )
        lines.append(f"- 日报缺出处：{audit['items_without_sources']}/{audit['items']} 条")
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
