"""日报/播客点评质量检查（P6）：结构完整性与证据提示。"""

from __future__ import annotations

from typing import Any


def review_quality(review: dict[str, Any], *, kind: str = "report") -> dict[str, Any]:
    """返回 {score, issues, checks}；score 越低越需要人工复核。"""
    issues: list[str] = []
    checks: dict[str, bool] = {}
    score_name = "structure_score"
    overall = str(review.get("overall") or "").strip()
    checks["overall"] = len(overall) >= 50
    if not checks["overall"]:
        issues.append("overall 缺失或过短")

    items = review.get("items") or []
    checks["items"] = bool(items)
    if not items:
        issues.append("items 为空")

    if kind == "podcast":
        show_notes = str(review.get("show_notes") or "").strip()
        checks["show_notes"] = len(show_notes) >= 100
        if not checks["show_notes"]:
            issues.append("show_notes 缺失或过短")
        chapters = review.get("chapters") or []
        checks["chapters"] = len(chapters) >= 3 and all(
            isinstance(chapter, dict) and chapter.get("title") and chapter.get("intro")
            for chapter in chapters
        )
        if not checks["chapters"]:
            issues.append("chapters 少于 3 或缺少 title/intro")
    else:
        checks["item_comments"] = all(item.get("comment") for item in items)
        if not checks["item_comments"]:
            issues.append("存在缺少 comment 的点评项")

    passed = sum(1 for value in checks.values() if value)
    score = round(passed / len(checks), 3) if checks else 0.0
    return {
        "metric": "structure_completeness",
        "score_name": score_name,
        "structure_score": score,
        "issues": issues,
        "checks": checks,
    }


__all__ = ["review_quality"]
