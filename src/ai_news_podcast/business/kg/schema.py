"""事件 schema：21 类的槽位定义加载与校验（事实层 S4）。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from ai_news_podcast.business.kg.relations import CANONICAL_LABELS

DEFAULT_SCHEMA_PATH = "config/nlp/event_schemas.yaml"


@dataclass(frozen=True)
class EventSchema:
    """一类事件允许的主体类型、对象类型、必填槽、可选槽、填充器。"""

    relation: str
    subject_types: tuple[str, ...]
    object_types: tuple[str, ...]
    required: tuple[str, ...]
    optional: tuple[str, ...]
    filler: str = "generic"


def _as_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(x) for x in value)
    if isinstance(value, str) and value:
        return (value,)
    return ()


@lru_cache(maxsize=8)
def _load_schemas_cached(path: str) -> tuple[EventSchema, ...]:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"事件 schema 文件不存在：{path}")
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    rows: list[EventSchema] = []
    for key, raw in (data.get("schemas") or {}).items():
        relation = str(key).strip()
        if relation not in CANONICAL_LABELS:
            raise ValueError(f"事件 schema 含未知关系：{relation}")
        if not isinstance(raw, dict):
            raise TypeError(f"事件 schema 格式非法：{relation}")
        required = _as_tuple(raw.get("required"))
        filler = str(raw.get("filler") or "generic")
        if not required and filler != "none":
            raise ValueError(f"事件 schema 必填槽为空：{relation}")
        rows.append(
            EventSchema(
                relation=relation,
                subject_types=_as_tuple(raw.get("subject")),
                object_types=_as_tuple(raw.get("object")),
                required=required,
                optional=_as_tuple(raw.get("optional")),
                filler=filler,
            )
        )
    if not rows:
        raise ValueError("事件 schema 为空")
    return tuple(rows)


def load_schemas(path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, EventSchema]:
    """加载 21 类 schema；未知关系/空必填槽直接报错。"""
    return {row.relation: row for row in _load_schemas_cached(str(Path(path)))}


__all__ = ["DEFAULT_SCHEMA_PATH", "EventSchema", "load_schemas"]
