"""统一打分：严格三元组匹配、误差归因、候选召回、bootstrap 置信区间。

口径固定为：
- 关系必须归一后完全一致；
- 端点默认 `exact`（归一化后相等），另报 `contain` 作为宽松口径；
- 每个文档内做最大二分匹配，避免贪心把后面的正确项挤掉。
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .standard import GoldItem, Triple, normalize_text


def endpoint_match(left: str, right: str, mode: str = "exact") -> bool:
    a, b = normalize_text(left), normalize_text(right)
    if not a or not b:
        return False
    if a == b:
        return True
    if mode != "contain":
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 2:
        return False
    return shorter in longer


def compatible(gold: Triple, pred: Triple, mode: str = "exact") -> bool:
    return (
        gold.relation == pred.relation
        and endpoint_match(gold.subject, pred.subject, mode)
        and endpoint_match(gold.object, pred.object, mode)
    )


def _max_matching(edges: list[tuple[int, int]], n_left: int, n_right: int) -> list[tuple[int, int]]:
    """Kuhn 算法求最大二分匹配；图很小（每篇 < 100 条）。"""
    adj: list[list[int]] = [[] for _ in range(n_left)]
    for left, right in edges:
        adj[left].append(right)
    match_right = [-1] * n_right
    result: list[tuple[int, int]] = []

    def try_kuhn(left: int, seen: list[bool]) -> bool:
        for right in adj[left]:
            if seen[right]:
                continue
            seen[right] = True
            if match_right[right] == -1 or try_kuhn(match_right[right], seen):
                match_right[right] = left
                return True
        return False

    for left in range(n_left):
        try_kuhn(left, [False] * n_right)
    for right, left in enumerate(match_right):
        if left != -1:
            result.append((left, right))
    return result


@dataclass(slots=True)
class DocScore:
    item_id: str
    lang: str
    split: str
    gold: int = 0
    pred: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    attribution: dict[str, int] = field(default_factory=dict)
    pool_hit: int = 0
    grounded: int = 0


@dataclass(slots=True)
class Score:
    docs: list[DocScore]
    mode: str

    @property
    def gold(self) -> int:
        return sum(d.gold for d in self.docs)

    @property
    def pred(self) -> int:
        return sum(d.pred for d in self.docs)

    @property
    def tp(self) -> int:
        return sum(d.tp for d in self.docs)

    @property
    def fp(self) -> int:
        return sum(d.fp for d in self.docs)

    @property
    def fn(self) -> int:
        return sum(d.fn for d in self.docs)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def pool_recall(self) -> float:
        total = sum(d.gold for d in self.docs)
        hit = sum(d.pool_hit for d in self.docs)
        return hit / total if total else 0.0

    @property
    def grounded(self) -> int:
        return sum(d.grounded for d in self.docs)

    @property
    def grounded_rate(self) -> float:
        return self.grounded / self.pred if self.pred else 0.0

    def attribution(self) -> dict[str, int]:
        out = {"full": 0, "relation_only": 0, "endpoint_only": 0, "no_candidate": 0}
        for doc in self.docs:
            for key, value in doc.attribution.items():
                out[key] = out.get(key, 0) + value
        return out

    def by_lang(self) -> dict[str, dict[str, float]]:
        groups: dict[str, list[DocScore]] = {}
        for doc in self.docs:
            groups.setdefault(doc.lang or "unknown", []).append(doc)
        out: dict[str, dict[str, float]] = {}
        for lang, rows in groups.items():
            sub = Score(rows, self.mode)
            out[lang] = {
                "gold": sub.gold,
                "pred": sub.pred,
                "tp": sub.tp,
                "precision": sub.precision,
                "recall": sub.recall,
                "f1": sub.f1,
            }
        return out

    def by_split(self, split: str) -> Score:
        return Score([d for d in self.docs if d.split == split], self.mode)

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "gold": self.gold,
            "pred": self.pred,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "pool_recall": round(self.pool_recall, 4),
            "grounded_rate": round(self.grounded_rate, 4),
            "attribution": self.attribution(),
            "by_lang": {
                lang: {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
                for lang, row in self.by_lang().items()
            },
            "docs": [d.item_id for d in self.docs],
        }


def _endpoint_present(value: str, text: str) -> bool:
    return normalize_text(value) in normalize_text(text)


def score_document(
    gold: GoldItem,
    preds: list[Triple],
    *,
    text: str = "",
    mode: str = "exact",
) -> DocScore:
    row = DocScore(
        item_id=gold.item_id,
        lang=gold.lang,
        split=gold.split,
        gold=len(gold.triples),
        pred=len(preds),
    )
    edges = [
        (gi, pi)
        for gi, g in enumerate(gold.triples)
        for pi, p in enumerate(preds)
        if compatible(g, p, mode)
    ]
    matched = _max_matching(edges, len(gold.triples), len(preds))
    matched_gold = {gi for gi, _ in matched}
    matched_pred = {pi for _, pi in matched}
    row.tp = len(matched)
    row.fp = len(preds) - len(matched_pred)
    row.fn = len(gold.triples) - len(matched_gold)

    for gi, g in enumerate(gold.triples):
        if gi in matched_gold:
            row.attribution["full"] = row.attribution.get("full", 0) + 1
            row.pool_hit += 1
            continue
        if any(
            g.relation == p.relation and endpoint_match(g.subject, p.subject, mode) for p in preds
        ) or any(
            g.relation == p.relation and endpoint_match(g.object, p.object, mode) for p in preds
        ):
            row.attribution["relation_only"] = row.attribution.get("relation_only", 0) + 1
        elif any(
            endpoint_match(g.subject, p.subject, mode) and endpoint_match(g.object, p.object, mode)
            for p in preds
        ):
            row.attribution["endpoint_only"] = row.attribution.get("endpoint_only", 0) + 1
        else:
            row.attribution["no_candidate"] = row.attribution.get("no_candidate", 0) + 1
        if any(compatible(g, p, mode) for p in preds):
            row.pool_hit += 1

    if text:
        for p in preds:
            if _endpoint_present(p.subject, text) and _endpoint_present(p.object, text):
                row.grounded += 1
    return row


def score_all(
    gold_items: Iterable[GoldItem],
    predictions: dict[str, list[Triple]],
    *,
    texts: dict[str, str] | None = None,
    mode: str = "exact",
    split: str | None = None,
) -> Score:
    texts = texts or {}
    rows = [
        score_document(
            item,
            predictions.get(item.item_id, []),
            text=texts.get(item.item_id, ""),
            mode=mode,
        )
        for item in gold_items
        if split is None or item.split == split
    ]
    return Score(rows, mode)


def bootstrap_f1(
    score: Score,
    *,
    rounds: int = 1000,
    seed: int = 13,
) -> tuple[float, float, float]:
    """按文档重采样，返回 (mean, low95, high95)。"""
    rng = random.Random(seed)
    docs = score.docs
    if not docs:
        return 0.0, 0.0, 0.0
    values: list[float] = []
    for _ in range(rounds):
        sample = [docs[rng.randrange(len(docs))] for _ in range(len(docs))]
        sub = Score(sample, score.mode)
        values.append(sub.f1)
    values.sort()
    mean = sum(values) / len(values)
    low = values[int(0.025 * len(values))]
    high = values[int(0.975 * len(values)) - 1]
    return mean, low, high


def paired_bootstrap_diff(
    left: Score,
    right: Score,
    *,
    rounds: int = 1000,
    seed: int = 13,
) -> tuple[float, float, float, bool]:
    """配对重采样 F1 差值（left - right）。

    两个 Score 必须基于同一批文档（同序）。返回 (mean, low95, high95, significant)，
    significant 表示 95% 区间不含 0。
    """
    docs_left = {doc.item_id: doc for doc in left.docs}
    docs_right = {doc.item_id: doc for doc in right.docs}
    shared = [item for item in docs_left if item in docs_right]
    if not shared:
        return 0.0, 0.0, 0.0, False
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(rounds):
        picked = [shared[rng.randrange(len(shared))] for _ in range(len(shared))]
        sub_left = Score([docs_left[item] for item in picked], left.mode)
        sub_right = Score([docs_right[item] for item in picked], right.mode)
        values.append(sub_left.f1 - sub_right.f1)
    values.sort()
    mean = sum(values) / len(values)
    low = values[int(0.025 * len(values))]
    high = values[int(0.975 * len(values)) - 1]
    return mean, low, high, (low > 0 or high < 0)
