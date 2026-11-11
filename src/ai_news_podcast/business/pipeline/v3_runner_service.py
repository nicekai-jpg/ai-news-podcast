"""Stage 1–5 新流水线编排（v3）。

旧版 ``runner_service.run_pipeline`` 保持不变；本模块只负责：
- 按约定路径串联 Stage 1→4，读写 artifact / JSON；
- 把“已选中事件”写入跨日覆盖账本；
- 通过注入的 ``llm_call`` / ``llm_judge`` 与 LLM 解耦，便于测试与灰度。

阶段产物：
    data/raw/raw-{date}.json.gz
    data/ocean/{date}/events.json|items.json|continuity.json
    data/selection/{date}/shortlist.json
    data/editorial/{date}/commentary.json
    site/episodes/{date}.txt
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ai_news_podcast.business.contracts import Commentary, Event, ShortlistEntry, utc_now_iso

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_CONFIG_DIR = Path("config")

LLMCall = Callable[[str], str]
LLMJudge = Callable[[list[Event]], dict[str, dict[str, Any]]]
LLMCallFactory = Callable[[str], LLMCall]


@dataclass
class V3RunResult:
    """一次 v3 流水线运行的产物路径。"""

    date: str
    raw_path: Path
    ocean_dir: Path
    events_path: Path
    shortlist_path: Path
    commentary_path: Path  # S4 日报 JSON（保留旧字段名，内容为 report.json）
    report_path: Path | None = None
    report_review_path: Path | None = None
    podcast_review_path: Path | None = None
    script_path: Path | None = None
    brief_path: Path | None = None


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_item_evidence(ocean_dir: Path) -> dict[str, str]:
    """读取 items.json，构造 item_id -> 正文（供 S4 日报/S6 文稿的事实底稿）。"""
    path = Path(ocean_dir) / "items.json"
    if not path.exists():
        return {}
    rows = _read_json(path)
    evidence: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("item_id") or "")
        if not item_id:
            continue
        evidence[item_id] = str(row.get("full_text") or row.get("summary") or "")
    return evidence


def make_llm_call(
    llm_cfg: Any,
    stage: str,
    options: dict[str, Any] | None = None,
) -> LLMCall:
    """用现有 llm_client_service 构造某 Stage 的 prompt→text 调用。

    ``llm_cfg`` 可以是配置 dataclass（支持 ``resolve_model(stage)``）或 dict。
    """
    from ai_news_podcast.business.pipeline.llm_client_service import call_llm

    if dataclasses.is_dataclass(llm_cfg) and not isinstance(llm_cfg, type):
        cfg: dict[str, Any] = dataclasses.asdict(llm_cfg)
        resolver = getattr(llm_cfg, "resolve_model", None)
        if callable(resolver):
            cfg["model"] = resolver(stage)
    else:
        cfg = dict(llm_cfg or {})
        cfg.setdefault("model", cfg.get("model", "MiniMax-M3"))
    if options:
        cfg.update(options)

    def _call(prompt: str) -> str:
        result = call_llm(prompt, cfg)
        if result is None:
            raise RuntimeError(f"LLM 调用失败（stage={stage}）")
        return result

    return _call


def _extract_json_object(text: str) -> dict[str, Any]:
    """从 LLM 输出中提取第一个 JSON 对象；失败返回空 dict。"""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def make_llm_judge(llm_call: LLMCall) -> LLMJudge:
    """构造 Stage 3 的 LLM rubric judge（prompt 来自 recommend.prompts）。"""
    from ai_news_podcast.business.recommend.prompts import build_rubric_prompt

    def judge(events: list[Event]) -> dict[str, dict[str, Any]]:
        if not events:
            return {}
        raw = llm_call(build_rubric_prompt(events))
        data = _extract_json_object(raw)
        result: dict[str, dict[str, Any]] = {}
        for event_id, scores in data.items():
            if isinstance(scores, dict):
                result[str(event_id)] = scores
        return result

    return judge


def _append_coverage(
    *,
    ledger_path: Path,
    date_str: str,
    shortlist: list[ShortlistEntry],
    events: list[Event],
    ocean_dir: Path,
    event_ids: list[str] | None = None,
) -> None:
    """把本次选中的事件写入跨日覆盖账本（幂等）。"""
    from ai_news_podcast.business.coverage.ledger import (
        CoverageEntry,
        append_entry,
        load_ledger,
        save_ledger,
    )

    by_id = {str(event.event_id): event for event in events}
    links_by_event: dict[str, list[str]] = {}
    items_path = Path(ocean_dir) / "items.json"
    if items_path.exists():
        link_by_item: dict[str, str] = {}
        for row in _read_json(items_path):
            if isinstance(row, dict) and row.get("item_id"):
                link_by_item[str(row["item_id"])] = str(row.get("link") or "")
        for event_id, event in by_id.items():
            links_by_event[event_id] = [
                link_by_item[item_id] for item_id in event.item_ids if link_by_item.get(item_id)
            ]

    ledger = load_ledger(ledger_path)
    allowed = {str(x) for x in event_ids} if event_ids is not None else None
    for entry in shortlist:
        if allowed is not None and str(entry.event_id) not in allowed:
            continue
        event = by_id.get(str(entry.event_id))
        if event is None:
            continue
        append_entry(
            ledger,
            CoverageEntry(
                event_id=str(event.event_id),
                title=event.representative_title,
                links=links_by_event.get(str(event.event_id), []),
                subjects=[str(s.get("name") or "") for s in event.labels.subjects if s.get("name")],
                keywords=event.labels.topics,
                first_covered=date_str,
                last_covered=date_str,
                sequence=event.sequence,
                continuity_id=event.continuity_id,
            ),
        )
    save_ledger(ledger_path, ledger)


def load_ocean_items(ocean_dir: Path) -> list[dict[str, Any]]:
    """读取 Stage 2 的 items.json（item_id -> 完整条目，含正文）。"""
    path = Path(ocean_dir) / "items.json"
    if not path.exists():
        return []
    rows = _read_json(path)
    return [row for row in rows if isinstance(row, dict)]


@dataclass
class _EditorialOutputs:
    """S4–S7 的产物与中间结果。"""

    report_rows: list[Commentary]
    report_json_path: Path
    report_review: Any
    report_path: Path
    report_review_path: Path
    script_path: Path | None
    podcast_review: Any | None
    podcast_review_path: Path | None


def _load_stage1_settings(config_root: Path) -> dict[str, Any]:
    path = config_root / "pipeline_v3.yaml"
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    stage1 = data.get("stage1") if isinstance(data, dict) else {}
    return stage1 if isinstance(stage1, dict) else {}


def _load_pricing(config_root: Path) -> dict[str, Any]:
    path = config_root / "pipeline_v3.yaml"
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    llm = data.get("llm") if isinstance(data, dict) else {}
    return (llm or {}).get("pricing") or {}


def _write_llm_usage(*, config_root: Path, editorial_root: Path, date_str: str) -> None:
    """把本次运行的 LLM token 用量与费用估算写入 editorial 目录（进 artifact）。"""
    from ai_news_podcast.business.pipeline.llm_usage_service import estimate_cost, snapshot

    usage = snapshot()
    if not usage:
        return
    payload = {
        "usage": usage,
        "estimate": estimate_cost(usage, _load_pricing(config_root)),
    }
    _write_json(editorial_root / date_str / "llm_usage.json", payload)
    logger.info("LLM 用量：%s", payload["estimate"])


def _run_editorial_stages(
    *,
    shortlist: list[ShortlistEntry],
    events: list[Event],
    item_evidence: dict[str, str],
    editorial_root: Path,
    date_str: str,
    config_root: Path,
    llm_for: Callable[[str], LLMCall],
    skip_script: bool,
) -> _EditorialOutputs:
    """S4 日报生成 → S5 日报点评 → S6 文稿生成 → S7 播客点评。"""
    from ai_news_podcast.business.editorial.commentary import build_commentaries
    from ai_news_podcast.business.editorial.report import render_daily_report
    from ai_news_podcast.business.editorial.review import (
        build_podcast_review,
        build_report_review,
    )
    from ai_news_podcast.business.editorial.script import build_script

    editorial_dir = editorial_root / date_str
    editorial_cfg = config_root / "editorial_v3.yaml"

    # S4：日报正文（事实底稿来自 Stage 2 全文证据）
    report_rows: list[Commentary] = build_commentaries(
        shortlist,
        events,
        llm_for("stage4"),
        item_evidence=item_evidence,
        config_path=editorial_cfg,
    )
    report_json_path = _write_json(
        editorial_dir / "report.json", [row.to_dict() for row in report_rows]
    )
    report_path = editorial_dir / f"daily_report_{date_str}.md"
    report_path.write_text(render_daily_report(report_rows, date_str=date_str), encoding="utf-8")
    logger.info("S4 完成：日报 %s 条，%s", len(report_rows), report_path)

    # S5：对已生成的日报做点评
    report_review = build_report_review(
        report_rows,
        events,
        llm_for("stage5"),
        item_evidence=item_evidence,
        config_path=editorial_cfg,
    )
    report_review_path = _write_json(editorial_dir / "report_review.json", report_review.to_dict())
    logger.info("S5 完成：日报点评 %s 条", len(report_review.items))

    # S6：用事实底稿生成播客文稿（不消费点评）
    script_path: Path | None = None
    podcast_review_path: Path | None = None
    podcast_review: Any | None = None
    if not skip_script:
        script_text = build_script(
            shortlist,
            events,
            item_evidence,
            llm_for("stage6"),
            config_path=editorial_cfg,
        )
        script_path = Path("site") / "episodes" / f"{date_str}.txt"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_text + "\n", encoding="utf-8")
        logger.info("S6 完成：%s", script_path)

        # S7：对已生成的文稿做点评 + show notes
        podcast_review = build_podcast_review(
            script_text,
            report_rows,
            events,
            llm_for("stage7"),
            config_path=editorial_cfg,
        )
        podcast_review_path = _write_json(
            editorial_dir / "podcast_review.json", podcast_review.to_dict()
        )
        logger.info("S7 完成：播客点评 %s 章", len(podcast_review.chapters))

    return _EditorialOutputs(
        report_rows=report_rows,
        report_json_path=report_json_path,
        report_review=report_review,
        report_path=report_path,
        report_review_path=report_review_path,
        script_path=script_path,
        podcast_review=podcast_review,
        podcast_review_path=podcast_review_path,
    )


def build_brief_compat(
    *,
    date_str: str,
    events: list[Event],
    shortlist: list[ShortlistEntry],
    ocean_dir: Path,
    report_rows: list[Commentary] | None = None,
    report_review: Any | None = None,
    podcast_review: Any | None = None,
) -> dict[str, Any]:
    """把 Stage 2/3/4 产物转成旧 publish 能消费的 brief 结构。

    旧 `podcast-publish` 依赖 `data/briefs/brief_{date}.json` 的 thesis/stories/metadata/radar；
    这里只做兼容转换，不改 publish 逻辑，保证新链可以直接复用 Stage 8/9。
    """
    items_by_id = {str(row.get("item_id")): row for row in load_ocean_items(ocean_dir)}
    event_by_id = {str(event.event_id): event for event in events}
    stories: list[dict[str, Any]] = []
    for index, entry in enumerate(shortlist):
        event = event_by_id.get(str(entry.event_id))
        if event is None:
            continue
        member_items: list[dict[str, Any]] = []
        for item_id in event.item_ids:
            row = items_by_id.get(str(item_id))
            if not row:
                continue
            member_items.append(
                {
                    "id": str(item_id),
                    "title": str(row.get("title") or ""),
                    "link": str(row.get("link") or ""),
                    "source_name": str(row.get("source_label") or row.get("source") or ""),
                    "published_at": str(row.get("published_at") or ""),
                    "summary": str(row.get("summary") or ""),
                }
            )
        components = entry.components or {}
        scores = {
            "impact_scope": round(float(components.get("impact", 0.0))),
            "novelty": round(float(components.get("novelty", 0.0))),
            "explainability": round(float(components.get("narratability", 0.0))),
            "listener_relevance": round(float(components.get("audience_fit", 0.0))),
            "source_richness": round(float(components.get("evidence", 0.0))),
        }
        stories.append(
            {
                "cluster_id": index,
                "representative_title": event.representative_title,
                "role": "main" if index == 0 else "support",
                "role_emoji": "🔴" if index == 0 else "⚪",
                "total_score": int(sum(scores.values())),
                "scores": scores,
                "context": {
                    "factual_summary": [event.summary or event.representative_title],
                    "historical_background": "",
                    "sources_ranked": [
                        {"name": item["source_name"], "link": item["link"]}
                        for item in member_items[:5]
                    ],
                },
                "items": member_items,
            }
        )
    return {
        "thesis": stories[0]["representative_title"] if stories else "",
        "stories": stories,
        "metadata": {
            "generated_at": utc_now_iso(),
            "total_raw": sum(max(1, event.source_count) for event in events),
            "total_deduped": len(events),
            "total_clusters": len(events),
            "pipeline": "v3",
            "shortlist": len(shortlist),
        },
        "radar": {"date": date_str, "projects": []},
        "report_rows": [row.to_dict() for row in (report_rows or [])],
        "report_review": report_review.to_dict() if report_review else {},
        "podcast_review": podcast_review.to_dict() if podcast_review else {},
    }


def run_v3_pipeline(
    date_str: str,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    rsshub_base: str | None = None,
    window_hours: int = 72,
    fetch_fulltext: bool = True,
    max_workers: int = 8,
    llm_cfg: Any | None = None,
    llm_call_factory: LLMCallFactory | None = None,
    stage_options: dict[str, dict[str, Any]] | None = None,
    llm_judge: LLMJudge | None = None,
    skip_script: bool = False,
    update_ledger: bool = True,
) -> V3RunResult:
    """串联 Stage 1→7（S1–S7），返回产物路径。"""
    from ai_news_podcast.business.pipeline.llm_usage_service import reset as _reset_usage

    _reset_usage()
    data_root = Path(data_dir)
    config_root = Path(config_dir)
    raw_dir = data_root / "raw"
    ocean_root = data_root / "ocean"
    selection_root = data_root / "selection"
    editorial_root = data_root / "editorial"
    ledger_path = data_root / "coverage" / "ledger.json"
    ocean_dir = ocean_root / date_str

    # ---- Stage 1 采集入海 ----
    from ai_news_podcast.business.collect.runner import collect_and_write

    stage1_settings = _load_stage1_settings(config_root)
    raw_path = collect_and_write(
        config_root / "sources_v3.yaml",
        date_str,
        raw_dir / f"raw-{date_str}.json.gz",
        rsshub_base=rsshub_base,
        window_hours=window_hours,
        fetch_fulltext=fetch_fulltext,
        max_workers=max_workers,
        request_interval=float(stage1_settings.get("request_interval", 1.0)),
    )
    logger.info("Stage 1 完成：%s", raw_path)

    # ---- Stage 2 新闻潮汐 ----
    from ai_news_podcast.business.ocean.build import build_ocean

    snapshot = build_ocean(
        raw_path,
        date_str=date_str,
        history_dir=ocean_root,
        ledger_path=ledger_path,
        config_path=config_root / "ocean_v3.yaml",
    )
    events: list[Event] = list(snapshot.events)
    events_path = ocean_dir / "events.json"
    logger.info("Stage 2 完成：%s 个事件", len(events))

    def _llm_for_factory(stage: str) -> LLMCall:
        if llm_call_factory is not None:
            return llm_call_factory(stage)
        if llm_cfg is None:
            raise ValueError("需要提供 llm_cfg 或 llm_call_factory 才能调用 LLM")
        return make_llm_call(llm_cfg, stage, (stage_options or {}).get(stage))

    _llm_for = _llm_for_factory

    # ---- Stage 3 选题推荐 ----
    from ai_news_podcast.business.recommend.service import recommend

    if llm_judge is None and (llm_cfg is not None or llm_call_factory is not None):
        llm_judge = make_llm_judge(_llm_for_factory("stage3"))

    shortlist = recommend(
        events,
        config_path=config_root / "recommend_v3.yaml",
        llm_judge=llm_judge,
    )
    shortlist_path = _write_json(
        selection_root / date_str / "shortlist.json",
        [entry.to_dict() for entry in shortlist],
    )
    logger.info("Stage 3 完成：shortlist=%s", len(shortlist))

    editorial = _run_editorial_stages(
        shortlist=shortlist,
        events=events,
        item_evidence=load_item_evidence(ocean_dir),
        editorial_root=editorial_root,
        date_str=date_str,
        config_root=config_root,
        llm_for=_llm_for,
        skip_script=skip_script,
    )
    report_rows = editorial.report_rows

    _write_llm_usage(config_root=config_root, editorial_root=editorial_root, date_str=date_str)

    brief = build_brief_compat(
        date_str=date_str,
        events=events,
        shortlist=shortlist,
        ocean_dir=ocean_dir,
        report_rows=report_rows,
        report_review=editorial.report_review,
        podcast_review=editorial.podcast_review,
    )
    brief_path = _write_json(data_root / "briefs" / f"brief_{date_str}.json", brief)
    logger.info("S4–S7 兼容 brief：%s", brief_path)

    if update_ledger:
        # 只记录真正进入点评/文稿的事件，避免“入选但未播报”被跨日过滤
        _append_coverage(
            ledger_path=ledger_path,
            date_str=date_str,
            shortlist=shortlist,
            events=events,
            ocean_dir=ocean_dir,
            event_ids=[str(row.event_id) for row in report_rows],
        )

    return V3RunResult(
        date=date_str,
        raw_path=Path(raw_path),
        ocean_dir=ocean_dir,
        events_path=events_path,
        shortlist_path=shortlist_path,
        commentary_path=editorial.report_json_path,
        report_path=editorial.report_path,
        report_review_path=editorial.report_review_path,
        podcast_review_path=editorial.podcast_review_path,
        script_path=editorial.script_path,
        brief_path=brief_path,
    )
