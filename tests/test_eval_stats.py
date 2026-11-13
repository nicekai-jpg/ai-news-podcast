"""Tests for eval/stats.py."""

from __future__ import annotations

from ai_news_podcast.business.eval.stats import (
    bootstrap_ci,
    cohen_kappa,
    expected_calibration_error,
    wilson_interval,
)


def test_wilson_small_sample_widens() -> None:
    low, high = wilson_interval(3, 4)
    assert low < 0.75 < high
    assert high - low > 0.2


def test_bootstrap_ci_brackets_mean() -> None:
    low, high = bootstrap_ci([1.0] * 10 + [0.0] * 10, samples=200)
    assert low <= 0.5 <= high


def test_cohen_kappa_perfect_and_disagree() -> None:
    assert cohen_kappa(["a", "b"], ["a", "b"]) == 1.0
    assert cohen_kappa(["a", "a"], ["a", "b"]) < 1.0


def test_ece_zero_when_confidences_match_accuracy() -> None:
    assert expected_calibration_error([1.0, 1.0, 0.0, 0.0], [True, True, False, False]) == 0.0
