"""Tests for business/pipeline/llm_usage_service.py."""

from __future__ import annotations

from ai_news_podcast.business.pipeline import llm_usage_service as usage


def setup_function() -> None:
    usage.reset()


def test_record_and_snapshot() -> None:
    usage.record_usage("m1", prompt_tokens=100, completion_tokens=50)
    usage.record_usage("m1", prompt_tokens=10, completion_tokens=5)
    usage.record_usage("m2", prompt_tokens=1, completion_tokens=2)
    snap = usage.snapshot()
    assert snap["m1"] == {
        "calls": 2,
        "prompt_tokens": 110,
        "completion_tokens": 55,
        "total_tokens": 165,
    }
    assert snap["m2"]["calls"] == 1


def test_reset() -> None:
    usage.record_usage("m", prompt_tokens=1, completion_tokens=1)
    usage.reset()
    assert usage.snapshot() == {}


def test_estimate_cost() -> None:
    snap = {"m": {"calls": 1, "prompt_tokens": 1_000_000, "completion_tokens": 500_000}}
    estimate = usage.estimate_cost(snap, {"m": {"input": 1.0, "output": 4.0}})
    assert estimate["total_cny"] == 3.0
    assert estimate["by_model"]["m"]["cost_cny"] == 3.0


def test_estimate_cost_without_price() -> None:
    snap = {"m": {"calls": 1, "prompt_tokens": 10, "completion_tokens": 10}}
    estimate = usage.estimate_cost(snap, {})
    assert estimate["total_cny"] is None
    assert estimate["by_model"]["m"]["cost_cny"] is None
