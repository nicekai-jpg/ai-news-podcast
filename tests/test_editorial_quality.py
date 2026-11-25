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


def test_claim_numbers_skip_dates_and_ordinals() -> None:
    from ai_news_podcast.business.editorial.quality import script_claim_numbers

    script = (
        "[Host A] 2026年11月16日，第3章提到 5.5 版本，涨了 35%，卖出 1,000 台。\n"
        "[Host B] 只有 3 个要点。\n"
    )
    numbers = script_claim_numbers(script)
    assert "5.5" in numbers
    assert "35%" in numbers
    assert "1,000" in numbers
    # 日期与序号不进核对集合
    assert "2026" not in numbers and "11" not in numbers and "16" not in numbers
    # 单数字太常见，不当作事实声明
    assert "3" not in numbers


def test_check_script_flags_number_missing_from_evidence() -> None:
    from ai_news_podcast.business.editorial.quality import check_script

    script = "[Host A] " + "内容" * 40 + " 融资 99 亿美元。\n[Host B] " + "对白" * 40 + "\n"
    report = check_script(script, evidence_text="该公司融资 10 亿美元。")
    assert "99" in report["unverified_numbers"]
    assert report["fact_checked"] is True
    assert any("底稿" in issue for issue in report["issues"])


def test_check_script_accepts_numbers_present_in_evidence() -> None:
    from ai_news_podcast.business.editorial.quality import check_script

    script = "[Host A] " + "内容" * 40 + " 融资 10 亿美元。\n[Host B] " + "对白" * 40 + "\n"
    report = check_script(script, evidence_text="该公司融资 10 亿美元。")
    assert report["unverified_numbers"] == []


def test_check_script_skips_fact_check_without_evidence() -> None:
    from ai_news_podcast.business.editorial.quality import check_script

    report = check_script("[Host A] 融资 99 亿美元。\n")
    assert report["fact_checked"] is False
    assert report["unverified_numbers"] == []


def test_ops_snapshot_reports_commentary_audit() -> None:
    """日报客观性审计必须进入运维快照，否则「缺出处」永远不可见。"""
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "ops_snapshot.py"
    spec = importlib.util.spec_from_file_location("ops_snapshot", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = [
        {
            "event_id": "e1",
            "headline": "标题",
            "one_liner": "",
            "why_matters": "",
            "background": "",
            "impact": "",
            "counterpoint": "暂无公开反方信息",
            "uncertainty": "",
            "facts": [
                {"text": "该公司发布 2.1 版本。", "evidence_ids": ["i1"], "single_source": True}
            ],
            "sources": [],
        }
    ]
    audit = mod._commentary_audit(rows)  # noqa: SLF001 - 按路径加载的脚本内部函数
    assert audit["items"] == 1
    assert audit["facts"] == 1
    assert audit["single_source_facts"] == 1
    assert audit["facts_without_evidence"] == 0
    assert audit["items_without_sources"] == 1
