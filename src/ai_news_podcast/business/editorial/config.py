"""Stage 4/5 编辑流水线的配置加载与校验。

配置默认来自 ``config/editorial_v3.yaml``；文件缺失或字段缺失时回落到内置默认值，
保证在任意工作目录下也可用。也兼容 ``stage4:`` / ``stage5:`` / ``commentary:``
子节写法（同名键子节优先）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/editorial_v3.yaml"

# LLM 调用约定：prompt -> 模型原文；测试与集成均注入假函数。
LLMCall = Callable[[str], str]

DEFAULT_BANNED_WORDS: tuple[str, ...] = (
    "废话不多说",
    "众所周知",
    "颠覆",
    "炸裂",
    "重磅",
    "王炸",
    "杀疯了",
    "遥遥领先",
    "细思极恐",
)


@dataclass(frozen=True)
class EditorialConfig:
    """Stage 4/5 的运行参数（不可变，便于测试注入）。"""

    report_top_n: int = 5
    target_minutes: int = 15
    min_chars: int = 3000
    max_chars: int = 3600
    chapters: int = 4
    min_chapters: int = 3
    max_chapters: int = 4
    min_host_char_ratio: float = 0.25
    max_unmarked_ratio: float = 0.2
    max_rewrites: int = 1
    host_a: str = "林夏"
    host_b: str = "陈默"
    fact_min_chars: int = 150
    fact_max_chars: int = 300
    banned_words: tuple[str, ...] = DEFAULT_BANNED_WORDS

    @property
    def host_a_tag(self) -> str:
        return "[Host A]"

    @property
    def host_b_tag(self) -> str:
        return "[Host B]"

    @property
    def host_tags(self) -> tuple[str, str]:
        return (self.host_a_tag, self.host_b_tag)


def _pick_int(flat: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(flat.get(key, default))
    except (TypeError, ValueError):
        logger.warning("编辑配置 %s 不是整数，使用默认值 %s", key, default)
        return default


def _pick_float(flat: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(flat.get(key, default))
    except (TypeError, ValueError):
        logger.warning("编辑配置 %s 不是数字，使用默认值 %s", key, default)
        return default


def _pick_banned_words(flat: dict[str, Any]) -> tuple[str, ...]:
    raw = flat.get("banned_words")
    if not isinstance(raw, list):
        return DEFAULT_BANNED_WORDS
    words = tuple(str(item).strip() for item in raw if str(item).strip())
    return words or DEFAULT_BANNED_WORDS


def _merge_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """展开 stage4/stage5/commentary 子节；子节键覆盖同名顶层键。"""
    flat: dict[str, Any] = dict(data)
    for section in ("stage4", "stage5", "commentary"):
        sub = data.get(section)
        if isinstance(sub, dict):
            flat.update(sub)
    return flat


def config_from_mapping(data: dict[str, Any] | None = None) -> EditorialConfig:
    """从已解析的映射构造配置（测试可直接传 dict，不必落盘）。"""
    data = data or {}
    flat = _merge_mapping(data)
    hosts = data.get("hosts") if isinstance(data.get("hosts"), dict) else {}
    host_a = str(hosts.get("host_a") or flat.get("host_a") or "林夏")
    host_b = str(hosts.get("host_b") or flat.get("host_b") or "陈默")
    cfg = EditorialConfig(
        report_top_n=_pick_int(flat, "report_top_n", 5),
        target_minutes=_pick_int(flat, "target_minutes", 15),
        min_chars=_pick_int(flat, "min_chars", 3000),
        max_chars=_pick_int(flat, "max_chars", 3600),
        chapters=_pick_int(flat, "chapters", 4),
        min_chapters=_pick_int(flat, "min_chapters", 3),
        max_chapters=_pick_int(flat, "max_chapters", 4),
        min_host_char_ratio=_pick_float(flat, "min_host_char_ratio", 0.25),
        max_unmarked_ratio=_pick_float(flat, "max_unmarked_ratio", 0.2),
        max_rewrites=_pick_int(flat, "max_rewrites", 1),
        host_a=host_a,
        host_b=host_b,
        fact_min_chars=_pick_int(flat, "fact_min_chars", _pick_int(flat, "min_fact_chars", 150)),
        fact_max_chars=_pick_int(flat, "fact_max_chars", _pick_int(flat, "max_fact_chars", 300)),
        banned_words=_pick_banned_words(flat),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: EditorialConfig) -> None:
    """配置自相矛盾时尽早报错，而不是产出不可用的文稿。"""
    if cfg.report_top_n < 1:
        raise ValueError("editorial.report_top_n 必须 >= 1")
    if cfg.min_chars >= cfg.max_chars:
        raise ValueError(
            f"editorial.min_chars({cfg.min_chars}) 必须小于 max_chars({cfg.max_chars})"
        )
    if not 1 <= cfg.min_chapters <= cfg.chapters <= cfg.max_chapters:
        raise ValueError(
            "editorial 章节配置必须满足 1 <= min_chapters <= chapters <= max_chapters，"
            f"当前为 {cfg.min_chapters}/{cfg.chapters}/{cfg.max_chapters}"
        )
    if not 0.0 < cfg.min_host_char_ratio < 0.5:
        raise ValueError("editorial.min_host_char_ratio 必须在 (0, 0.5) 之间")
    if not 0.0 <= cfg.max_unmarked_ratio <= 1.0:
        raise ValueError("editorial.max_unmarked_ratio 必须在 [0, 1] 之间")
    if cfg.max_rewrites < 0:
        raise ValueError("editorial.max_rewrites 必须 >= 0")
    if not cfg.host_a or not cfg.host_b or cfg.host_a == cfg.host_b:
        raise ValueError("editorial.hosts 需要两个不同的主持人名字")


def load_config(config_path: str | Path | None = DEFAULT_CONFIG_PATH) -> EditorialConfig:
    """读取 YAML 配置；文件不存在时记 warning 并回落到内置默认值。"""
    path = Path(config_path) if config_path else Path(DEFAULT_CONFIG_PATH)
    if not path.is_file():
        logger.warning("编辑配置不存在，使用内置默认值: %s", path)
        return config_from_mapping({})
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"读取编辑配置失败: {path}: {exc}") from exc
    if raw is None:
        return config_from_mapping({})
    if not isinstance(raw, dict):
        raise TypeError(f"编辑配置必须是 YAML 映射: {path}")
    return config_from_mapping(raw)
