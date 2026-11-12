"""评价线索词 → 极性/强度（形容词刻度的“输入侧词表”）。

刻度不是替代词表：同义词仍要尽量补全，只是归并到同一刻度，
再叠加否定词/程度词组合，得到可比较的分数与证据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_BASE_DIR = Path(__file__).resolve().parents[4] / "config" / "taxonomy"


@dataclass(frozen=True)
class CueHit:
    cue: str
    dimension: str
    value: int
    intensity: int
    negated: bool


@dataclass
class CueScore:
    sentiment: float = 0.0
    impact: float = 0.0
    controversy: float = 0.0
    hits: list[CueHit] = field(default_factory=list)

    @property
    def evidence(self) -> list[str]:
        return [hit.cue for hit in self.hits]


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def load_eval_cues() -> dict[str, Any]:
    return _load(_BASE_DIR / "eval_cues.yaml")


def evaluate_cues(text: str, *, lexicon: dict[str, Any] | None = None) -> CueScore:
    """匹配评价词，叠加前置否定词/程度词，返回三维分数与证据。"""
    raw = lexicon or load_eval_cues()
    cues: list[dict[str, Any]] = list(raw.get("cues") or [])
    negators = [str(item) for item in raw.get("negators") or []]
    intensifiers: dict[int, list[str]] = {
        int(level): [str(item) for item in words]
        for level, words in (raw.get("intensifiers") or {}).items()
    }
    text_lower = str(text or "").lower()
    score = CueScore()
    for cue in sorted(cues, key=lambda item: len(str(item.get("cue") or "")), reverse=True):
        word = str(cue.get("cue") or "")
        if not word or word.lower() not in text_lower:
            continue
        position = text_lower.find(word.lower())
        prefix = text_lower[max(0, position - 6) : position]
        negated = any(negator in prefix for negator in negators)
        value = int(cue.get("value") or 0)
        intensity = int(cue.get("intensity") or 1)
        for level, words in intensifiers.items():
            if any(item in prefix for item in words):
                intensity = max(intensity, level)
                break
        if negated:
            value = -value if value else value
        dimension = str(cue.get("dimension") or "sentiment")
        weighted = value * intensity
        if dimension == "sentiment":
            score.sentiment += weighted
        elif dimension == "impact":
            score.impact = max(score.impact, float(weighted))
        elif dimension == "controversy":
            score.controversy = max(score.controversy, float(weighted))
        score.hits.append(
            CueHit(
                cue=word,
                dimension=dimension,
                value=value,
                intensity=intensity,
                negated=negated,
            )
        )
    return score


__all__ = ["CueHit", "CueScore", "evaluate_cues", "load_eval_cues"]
