"""Tests for ocean/eval_cues.py — 评价词到刻度。"""

from __future__ import annotations

from ai_news_podcast.business.ocean.eval_cues import evaluate_cues


def test_positive_impact() -> None:
    result = evaluate_cues("OpenAI 发布重磅新模型")
    assert result.impact >= 3
    assert "重磅" in result.evidence


def test_negation_flips_sentiment() -> None:
    result = evaluate_cues("这次发布不太成功")
    assert result.sentiment < 0


def test_intensifier_increases_strength() -> None:
    result = evaluate_cues("极其重磅的发布")
    assert result.impact >= 3


def test_controversy() -> None:
    result = evaluate_cues("该模型因版权问题引发争议与诉讼")
    assert result.controversy >= 3


def test_no_cue_returns_zero() -> None:
    result = evaluate_cues("公司发布季度公告")
    assert result.sentiment == 0
    assert result.impact == 0
