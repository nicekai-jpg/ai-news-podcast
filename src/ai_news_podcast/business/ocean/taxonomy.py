"""标签体系加载与校验（config/taxonomy/*.yaml）。

设计：先定骨架（维度/枚举/主体/行业），再补词。
- labels.yaml：决策标签的受控枚举（只允许枚举值）；
- entities.yaml：公司/实验室/人物/产品，别名映射到规范 id；
- industries.yaml：行业树，关键词按标题映射。

本模块只做加载/校验/查询，不产生标签；标签由规则或 LLM 产出后必须映射到这里的 id。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parents[4] / "config" / "taxonomy"
_AI_RELATED = {"core", "infra", "application", "none"}
_ENTITY_TYPES = {"company", "lab", "person", "product", "org", "government"}


class TaxonomyError(ValueError):
    """标签体系配置不合法。"""


@dataclass
class Taxonomy:
    dimensions: dict[str, dict[str, Any]] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)
    industries: list[dict[str, Any]] = field(default_factory=list)
    entity_aliases: dict[str, dict[str, Any]] = field(default_factory=dict)
    industry_keywords: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def enum_values(self, dimension: str) -> tuple[str, ...]:
        values = self.dimensions.get(dimension, {}).get("values") or ()
        return tuple(str(value) for value in values)

    def canonical_entity(self, text: str) -> dict[str, Any] | None:
        normalized = _normalize(text)
        if not normalized:
            return None
        return self.entity_aliases.get(normalized)

    def industry_for_title(self, title: str) -> dict[str, Any] | None:
        for keyword, industry in self.industry_keywords:
            if keyword and keyword in _normalize(title):
                return industry
        return None

    def industry_match(self, text: str) -> tuple[dict[str, Any], str] | None:
        """返回 (行业, 命中关键词)，用于写证据。"""
        normalized = _normalize(text)
        for keyword, industry in self.industry_keywords:
            if keyword and keyword in normalized:
                return industry, keyword
        return None


def _normalize(text: str) -> str:
    return "".join(str(text or "").lower().split())


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TaxonomyError(f"缺少标签体系文件: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise TaxonomyError(f"读取失败 {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TaxonomyError(f"{path} 顶层必须是对象")
    return data


def _validate_dimensions(labels: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dimensions = labels.get("dimensions") or {}
    if not isinstance(dimensions, dict) or not dimensions:
        raise TaxonomyError("labels.yaml 缺少 dimensions")
    for name, spec in dimensions.items():
        values = (spec or {}).get("values") or []
        if not values:
            raise TaxonomyError(f"维度 {name} 缺少枚举值")
        if name in ("ai_centrality", "primary_domain") and "unknown" not in values:
            raise TaxonomyError(f"维度 {name} 必须包含 unknown")
    return {str(name): dict(spec or {}) for name, spec in dimensions.items()}


def _parse_entities(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    entities = raw.get("entities") or []
    seen: set[str] = set()
    aliases: dict[str, dict[str, Any]] = {}
    for entity in entities:
        entity_id = str(entity.get("id") or "")
        if not entity_id or entity_id in seen:
            raise TaxonomyError(f"实体 id 非法或重复: {entity_id!r}")
        seen.add(entity_id)
        if str(entity.get("type")) not in _ENTITY_TYPES:
            raise TaxonomyError(f"实体 {entity_id} 的 type 非法")
        if not isinstance(entity.get("ai_related"), bool):
            raise TaxonomyError(f"实体 {entity_id} 缺少 ai_related")
        for alias in [entity.get("name"), *(entity.get("aliases") or [])]:
            key = _normalize(str(alias or ""))
            if key:
                aliases.setdefault(key, entity)
    return list(entities), aliases


def _parse_industries(
    raw: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    industries = raw.get("industries") or []
    seen: set[str] = set()
    keywords: list[tuple[str, dict[str, Any]]] = []
    for industry in industries:
        industry_id = str(industry.get("id") or "")
        if not industry_id or industry_id in seen:
            raise TaxonomyError(f"行业 id 非法或重复: {industry_id!r}")
        seen.add(industry_id)
        if str(industry.get("ai_related")) not in _AI_RELATED:
            raise TaxonomyError(f"行业 {industry_id} 的 ai_related 非法")
        for keyword in industry.get("keywords") or []:
            normalized = _normalize(str(keyword))
            if normalized:
                keywords.append((normalized, industry))
    for industry in industries:
        parent = industry.get("parent")
        if parent and parent not in seen:
            raise TaxonomyError(f"行业 {industry.get('id')} 的 parent 不存在: {parent}")
    keywords.sort(key=lambda item: len(item[0]), reverse=True)
    return list(industries), keywords


def load_taxonomy(base_dir: str | Path | None = None) -> Taxonomy:
    """加载并校验标签体系；失败抛 TaxonomyError。"""
    root = Path(base_dir) if base_dir else _BASE_DIR
    dimensions = _validate_dimensions(_read_yaml(root / "labels.yaml"))
    entities, aliases = _parse_entities(_read_yaml(root / "entities.yaml"))
    industries, keywords = _parse_industries(_read_yaml(root / "industries.yaml"))
    logger.debug(
        "标签体系加载: %d 维度, %d 实体, %d 行业", len(dimensions), len(entities), len(industries)
    )
    return Taxonomy(
        dimensions=dimensions,
        entities=entities,
        industries=industries,
        entity_aliases=aliases,
        industry_keywords=keywords,
    )


@lru_cache(maxsize=1)
def default_taxonomy() -> Taxonomy:
    return load_taxonomy()


__all__ = ["Taxonomy", "TaxonomyError", "default_taxonomy", "load_taxonomy"]
