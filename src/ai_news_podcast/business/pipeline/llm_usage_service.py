"""进程内 LLM token 用量收集与费用估算。

后端每次成功调用后 `record_usage()` 累加；流水线结束时 `snapshot()` 落盘，
用于观察每期真实 token / 费用（价格表来自 config/pipeline_v3.yaml 的 llm.pricing）。
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_usage: dict[str, dict[str, int]] = {}


def reset() -> None:
    """清空统计（测试/单次运行开始时调用）。"""
    with _lock:
        _usage.clear()


def record_usage(
    model: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """累加一次调用的 token 用量。"""
    model = str(model or "unknown")
    prompt_tokens = max(0, int(prompt_tokens or 0))
    completion_tokens = max(0, int(completion_tokens or 0))
    with _lock:
        row = _usage.setdefault(
            model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        row["calls"] += 1
        row["prompt_tokens"] += prompt_tokens
        row["completion_tokens"] += completion_tokens
        row["total_tokens"] += prompt_tokens + completion_tokens


def snapshot() -> dict[str, dict[str, int]]:
    """返回当前累计用量（拷贝）。"""
    with _lock:
        return {model: dict(row) for model, row in _usage.items()}


def _price_for(prices: dict[str, Any], model: str) -> dict[str, float]:
    raw = prices.get(model) if isinstance(prices, dict) else None
    if not isinstance(raw, dict):
        return {}
    try:
        return {"input": float(raw.get("input", 0)), "output": float(raw.get("output", 0))}
    except (TypeError, ValueError):
        return {}


def estimate_cost(usage: dict[str, dict[str, int]], prices: dict[str, Any]) -> dict[str, Any]:
    """按「元/百万 tokens」估算费用。

    返回 ``{"total_cny": float, "by_model": {model: {...}}}``；无价格时费用为 None。
    """
    by_model: dict[str, dict[str, Any]] = {}
    total = 0.0
    priced = False
    for model, row in usage.items():
        price = _price_for(prices, model)
        prompt = int(row.get("prompt_tokens", 0))
        completion = int(row.get("completion_tokens", 0))
        if not price:
            by_model[model] = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cost_cny": None,
            }
            continue
        cost = prompt / 1_000_000 * price["input"] + completion / 1_000_000 * price["output"]
        total += cost
        priced = True
        by_model[model] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "input_price": price["input"],
            "output_price": price["output"],
            "cost_cny": round(cost, 6),
        }
    return {"total_cny": round(total, 6) if priced else None, "by_model": by_model}
