"""评测统计工具：置信区间、一致性、校准。"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间（小样本比正态近似稳）。"""
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_ci(
    values: Sequence[float],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20261113,
) -> tuple[float, float]:
    """均值的 bootstrap 置信区间。"""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    alpha = (1 - confidence) / 2
    low = means[int(alpha * samples)]
    high = means[min(samples - 1, int((1 - alpha) * samples))]
    return (round(low, 4), round(high, 4))


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """两标注者名义一致性（Cohen's kappa）。"""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("两列标注长度必须相同且非空")
    total = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / total
    counts_a: dict[str, int] = {}
    counts_b: dict[str, int] = {}
    for a, b in zip(labels_a, labels_b, strict=True):
        counts_a[a] = counts_a.get(a, 0) + 1
        counts_b[b] = counts_b.get(b, 0) + 1
    expected = sum(counts_a[key] * counts_b.get(key, 0) for key in counts_a) / (total * total)
    if expected >= 1:
        return 1.0
    return round((observed - expected) / (1 - expected), 4)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = 10,
) -> float:
    """ECE：置信度与实际准确率的加权差；样本少时只作诊断。"""
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("置信度与正确性长度必须相同且非空")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, is_correct in zip(confidences, correct, strict=True):
        index = min(bins - 1, max(0, int(confidence * bins)))
        buckets[index].append((confidence, is_correct))
    total = len(confidences)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_conf = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(1 for item in bucket if item[1]) / len(bucket)
        error += (len(bucket) / total) * abs(avg_conf - accuracy)
    return round(error, 4)


__all__ = ["bootstrap_ci", "cohen_kappa", "expected_calibration_error", "wilson_interval"]
