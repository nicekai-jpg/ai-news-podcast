"""Stage 2.5：按事件抽知识图谱（零 LLM 调用，可失败）。

- 输入：Stage 2 的 `events` 与 `items`（内存里就有，不用重新读文件）；
- 处理：按 `event.item_ids` 拼全文，逐事件调用 `business.kg.build_kg`；
- 输出：`data/kg/{date}/kg.json`，失败只记日志，不阻断 Stage 3。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml

from ai_news_podcast.business.contracts import Event, utc_now_iso
from ai_news_podcast.business.kg import KGlexicon, build_kg
from ai_news_podcast.business.kg.dep import DepParser
from ai_news_podcast.business.kg.service import build_facts
from ai_news_podcast.business.kg.store import write_c_samples, write_day
from ai_news_podcast.business.kg.types import DayStats, FactsBuildResult, FactsConfig

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 6000
DEFAULT_MIN_CONFIDENCE = 0.4


def build_kg_artifact(
    events: list[Event],
    items: list[dict[str, Any]],
    *,
    date_str: str,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_chars: int = DEFAULT_MAX_CHARS,
    lexicon: KGlexicon | None = None,
    nlp: Any | None = None,
) -> dict[str, Any]:
    """逐事件抽图，返回可写盘的 payload。"""
    lex = lexicon or KGlexicon.load()
    texts = _item_texts(items)
    rows: list[dict[str, Any]] = []
    for event in events or []:
        text = _event_text(event, texts, max_chars)
        if not text:
            rows.append(
                {
                    "event_id": event.event_id,
                    "title": event.representative_title,
                    "nodes": [],
                    "edges": [],
                    "stats": {"sentences": 0, "candidates": 0, "kept": 0, "nodes": 0, "edges": 0},
                }
            )
            continue
        doc = build_kg(
            text,
            event.representative_title,
            lexicon=lex,
            nlp=nlp,
            min_confidence=min_confidence,
        )
        rows.append({"event_id": event.event_id, **doc.to_dict()})

    stats = {
        "events": len(rows),
        "events_with_edges": sum(1 for row in rows if row["edges"]),
        "nodes": sum(len(row["nodes"]) for row in rows),
        "edges": sum(len(row["edges"]) for row in rows),
        "high_confidence": sum(row["stats"].get("high_confidence", 0) for row in rows),
    }
    logger.debug("知识图谱抽取完成: %s", stats)
    return {
        "date": date_str,
        "generated_at": utc_now_iso(),
        "parameters": {"min_confidence": min_confidence, "max_chars": max_chars},
        "stats": stats,
        "events": rows,
    }


def write_kg_artifact(path: str | Path, payload: dict[str, Any]) -> Path:
    """把 payload 写到 `kg.json`（自动建目录）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _item_texts(items: list[dict[str, Any]]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for row in items or []:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("item_id") or "")
        if not item_id:
            continue
        text = str(row.get("full_text") or row.get("summary") or "").strip()
        if text:
            texts[item_id] = text
    return texts


def _event_text(event: Event, texts: dict[str, str], max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for item_id in event.item_ids:
        text = texts.get(str(item_id))
        if not text:
            continue
        parts.append(text)
        used += len(text)
        if used >= max_chars:
            break
    return " ".join(parts)[:max_chars]


__all__ = ["build_kg_artifact", "write_kg_artifact"]


# ---------------------------------------------------------------------------
# V2 事实层：Stage 2.5 入口（run_facts_stage）
# ---------------------------------------------------------------------------

DEFAULT_FACTS_DIR = "data/facts"
DEFAULT_DEBUG_DIR = "data/kg_debug"
_RULES_FILES = (
    "nlp/actions.yaml",
    "nlp/entities.yaml",
    "nlp/objects.yaml",
    "nlp/event_schemas.yaml",
    "nlp/noise.yaml",
    "nlp/negation.yaml",
)


def _load_facts_settings(config_root: Path) -> dict[str, Any]:
    path = Path(config_root) / "pipeline_v3.yaml"
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    stage = data.get("stage2_5") if isinstance(data, dict) else {}
    return stage if isinstance(stage, dict) else {}


def _clamp(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _env_enabled() -> bool | None:
    raw = os.environ.get("KG_FACTS_ENABLED", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def load_facts_config(config_root: Path) -> FactsConfig:
    """读 stage2_5 配置；非法值回退默认；环境变量可覆盖开关（演练期用）。"""
    settings = _load_facts_settings(config_root)
    enabled = bool(settings.get("enabled", True))
    override = _env_enabled()
    if override is not None:
        enabled = override
    return FactsConfig(
        enabled=enabled,
        max_sentences_guard=_clamp(settings.get("max_sentences_guard"), 500, 50, 1000),
        write_b=bool(settings.get("write_b", True)),
        n_process=_clamp(settings.get("n_process"), 2, 1, 4),
        max_seconds=_clamp(settings.get("max_seconds"), 600, 60, 1800),
        debug_kg=bool(settings.get("debug_kg", False)),
    )


def rules_version(config_root: Path) -> str:
    """词表/规则版本：对 config/nlp 关键文件取哈希。"""
    digest = hashlib.sha1()
    for name in _RULES_FILES:
        path = Path(config_root) / name
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def build_facts_artifact(
    items: list[dict[str, Any]],
    events: list[Event],
    *,
    date_str: str,
    config_root: str | Path = "config",
    config: FactsConfig | None = None,
    lexicon: KGlexicon | None = None,
    parser: DepParser | None = None,
) -> FactsBuildResult:
    """纯计算版事实层入口，供测试与 CLI 复用。"""
    root = Path(config_root)
    cfg = config or load_facts_config(root)
    lex = lexicon or KGlexicon.load(root / "nlp")
    schema_path = root / "nlp/event_schemas.yaml"
    rules_path = root / "nlp/negation.yaml"
    return build_facts(
        items,
        events,
        date_str=date_str,
        config=cfg,
        lexicon=lex,
        parser=parser,
        schema_path=schema_path if schema_path.is_file() else None,
        rules_path=rules_path if rules_path.is_file() else None,
    )


def run_facts_stage(
    items: list[dict[str, Any]],
    events: list[Event],
    *,
    date_str: str,
    config_root: str | Path = "config",
    data_root: str | Path = "data",
    config: FactsConfig | None = None,
    lexicon: KGlexicon | None = None,
    parser: DepParser | None = None,
) -> dict[str, Any]:
    """跑 Stage 2.5 事实层：写 facts + manifest，任何异常只告警。"""
    root = Path(config_root)
    cfg = config or load_facts_config(root)
    if not cfg.enabled:
        logger.info("Stage 2.5 事实层：配置关闭，跳过")
        return {"enabled": False}
    facts_dir = Path(data_root) / "facts"
    debug_dir = Path(data_root) / "kg_debug"
    started = time.monotonic()
    try:
        lex = lexicon or KGlexicon.load(root / "nlp")
        dep_parser = parser or DepParser.load(lex)
        if dep_parser is None:
            alert = "parser_missing"
            logger.warning("Stage 2.5 事实层：解析模型不可用，跳过")
            write_day(
                facts_dir,
                date_str,
                [],
                rules_version=rules_version(root),
                stats=DayStats(),
                partial=False,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                alert=alert,
            )
            return {"enabled": True, "ok": False, "error": alert}
        result = build_facts(
            items,
            events,
            date_str=date_str,
            config=cfg,
            lexicon=lex,
            parser=dep_parser,
            schema_path=root / "nlp/event_schemas.yaml",
            rules_path=root / "nlp/negation.yaml",
        )
        cards = result.cards
        alert = ""
        if not cards:
            alert = "zero_cards"
        partial = False
        write_day(
            facts_dir,
            date_str,
            cards,
            rules_version=rules_version(root),
            stats=result.stats,
            partial=partial,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            alert=alert,
        )
        write_c_samples(debug_dir, date_str, result.c_samples)
        stats = result.stats.to_dict()
        logger.info(
            "Stage 2.5 事实层：%s 个事件，%s 条 A / %s 条 B",
            result.stats.events,
            result.stats.cards_a,
            result.stats.cards_b,
        )
        return {"enabled": True, "ok": True, "path": str(facts_dir), **stats}
    except Exception:
        logger.warning("Stage 2.5 事实层失败，跳过（不影响主链）", exc_info=True)
        try:
            write_day(
                facts_dir,
                date_str,
                [],
                rules_version=rules_version(root),
                stats=DayStats(),
                partial=False,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                alert="facts_failed",
            )
        except Exception:
            logger.warning("事实层失败后写告警 manifest 也失败", exc_info=True)
        return {"enabled": True, "ok": False, "error": "facts_failed"}


__all__ = [
    "build_facts_artifact",
    "build_kg_artifact",
    "load_facts_config",
    "rules_version",
    "run_facts_stage",
    "write_kg_artifact",
]
