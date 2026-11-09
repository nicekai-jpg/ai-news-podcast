"""Stage 2「新闻潮汐」配置加载。

`config/ocean_v3.yaml` 是唯一运行时配置源；代码内保留一份结构一致的默认值，
缺项时回退默认，保证配置演进与单测都不易碎。
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 3,
    "preprocess": {"max_full_text_chars": 4000, "keyword_top_k": 15},
    "dedupe": {
        "title_threshold": 0.90,
        "semantic_threshold": 0.80,
        "semantic_embedding_threshold": 0.80,
        "cross_day_title_threshold": 0.88,
        "cross_day_semantic_threshold": 0.40,
        "cross_day_new_progress": True,
        "min_title_chars": 6,
        "max_title_candidates": 300,
    },
    "cluster": {
        "time_window_hours": 48,
        "semantic_threshold": 0.55,
        "entity_jaccard_threshold": 0.50,
        "title_threshold": 0.85,
        "min_shared_keywords": 2,
        "method": "components",
        "hdbscan_min_cluster_size": 2,
        "max_candidate_pairs": 50000,
        "max_candidates_per_item": 200,
        "reject_distinct_versions": True,
    },
    "continuity": {
        "history_days": 30,
        "entity_jaccard_threshold": 0.60,
        "title_threshold": 0.75,
        "entity_title_floor": 0.35,
        "entity_title_floor_single": 0.50,
        "progress_number_types": ["version", "money", "percent", "score", "count"],
        "new_progress_actions": True,
        "covered_title_threshold": 0.85,
        "max_sequence": 999,
    },
    "evidence": {
        "weights": {
            "official_source": 2.0,
            "source_count": 1.0,
            "fact_consistency": 1.0,
            "full_text_ratio": 0.5,
            "single_source": -1.0,
            "rumor": -1.5,
        },
        "source_count_cap": 5,
        "min_full_text_chars": 300,
        "official_categories": ["official"],
        "rumor_keywords": [
            "传闻",
            "据悉",
            "消息称",
            "爆料",
            "据报道",
            "未经证实",
            "rumor",
            "reportedly",
            "unconfirmed",
        ],
        "rumor_max_sources": 2,
        "momentum_window_hours": 6,
        "tiers": {"strong": 4.0, "medium": 2.5, "weak": 1.0},
    },
    "labels": {
        "nlp_dir": "nlp",
        "lifecycle": {
            "burst_sources": 3,
            "burst_window_hours": 24,
            "long_tail_days": 7,
            "fading_days": 3,
        },
        "novelty": {"max_values": 12},
        "quote_markers": [
            "官方表示",
            "官方声明",
            "在声明中",
            "新闻稿",
            "发言人",
            "says",
            "said",
            "announced",
            "according to",
            "press release",
        ],
        "min_alias_length": 2,
        "context_window_chars": 40,
        "confidence": {
            "subjects": 0.95,
            "object_type": 0.75,
            "actions": 0.85,
            "tech_stack": 0.85,
            "regions": 0.65,
            "topics": 0.65,
            "evidence_tier": 0.90,
            "novelty_numbers": 0.80,
            "narrative_angle": 0.70,
            "audience_persona": 0.65,
            "sensitivity": 0.75,
            "quote_availability": 0.70,
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并：override 覆盖 base，dict 逐层合并。"""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def cfg_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """按 `a.b.c` 读取配置，缺失时返回默认值。"""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_nlp_dir(config_path: Path, cfg: dict[str, Any]) -> Path:
    """把 labels.nlp_dir 解析为目录：绝对路径直接用，相对路径相对配置文件所在目录。"""
    raw = Path(str(cfg_get(cfg, "labels.nlp_dir", "nlp")))
    if raw.is_absolute():
        return raw
    return Path(config_path).resolve().parent / raw


def load_ocean_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载 ocean 配置；文件缺失/为空时回退默认值（并打 warning）。"""
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)
    if not config_path.exists():
        log.warning("Ocean config not found at %s; falling back to defaults.", config_path)
        return deepcopy(DEFAULT_CONFIG)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Invalid ocean config YAML at {config_path}")
    merged = _deep_merge(DEFAULT_CONFIG, raw)
    merged["_config_path"] = str(config_path)
    merged["_nlp_dir"] = str(resolve_nlp_dir(config_path, merged))
    return merged
