"""Tests for editorial/quality.py."""

from __future__ import annotations

from ai_news_podcast.business.editorial.review_quality import review_quality


def _report() -> dict:
    return {
        "overall": "整体评价" * 20,
        "items": [{"event_id": "e1", "comment": "有依据"}],
    }


def test_report_review_ok() -> None:
    result = review_quality(_report(), kind="report")
    assert result["structure_score"] == 1.0
    assert result["issues"] == []


def test_missing_comment_flagged() -> None:
    review = _report()
    review["items"] = [{"event_id": "e1"}]
    result = review_quality(review, kind="report")
    assert result["structure_score"] < 1.0
    assert any("comment" in issue for issue in result["issues"])


def test_podcast_requires_show_notes_and_chapters() -> None:
    review = {
        "overall": "整体评价" * 20,
        "items": [],
        "show_notes": "短",
        "chapters": [],
    }
    result = review_quality(review, kind="podcast")
    assert result["structure_score"] <= 0.25
    assert len(result["issues"]) >= 3
