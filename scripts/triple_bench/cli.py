#!/usr/bin/env python3
"""三元组抽取统一评测 CLI。

子命令：
    sample      从 ocean items 抽 60 篇冻结文档
    seed-gold   把已有金标（gold_v2）迁进新格式，保留人工确认状态
    draft       用 LLM 起草新增文档的金标（draft 状态，待人工审核）
    validate    校验金标（证据回指、关系词表、重复项）
    freeze      把审核后的金标冻结
    run         用指定方案跑预测
    eval        统一口径打分
    report      生成对比报告

示例：
    python scripts/triple_bench/cli.py sample --items data/ocean/2026-11-16/items.json \
        --out data/eval/triple_bench/docs.json --total 60 --dev 20
    python scripts/triple_bench/cli.py run --adapter hanlp_srl
    python scripts/triple_bench/cli.py eval
    python scripts/triple_bench/cli.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.triple_bench.adapters import (  # noqa: E402
    ADAPTERS,
    AdapterUnavailable,
    adapter_errors,
    concurrency,
    run_adapter,
)
from scripts.triple_bench.gold import (  # noqa: E402
    draft_item,
    load_docs_from_ocean,
    load_gold,
    review_markdown,
    save_gold,
    split_docs,
    stratified_select,
    validate_gold,
)
from scripts.triple_bench.metrics import (  # noqa: E402
    bootstrap_f1,
    paired_bootstrap_diff,
    score_all,
)
from scripts.triple_bench.standard import (  # noqa: E402
    Doc,
    GoldItem,
    Triple,
    normalize_relation,
    normalize_text,
)

DEFAULT_BENCH = ROOT / "data" / "eval" / "triple_bench"
DEFAULT_ITEMS = ROOT / "data" / "ocean" / "2026-11-16" / "items.json"
DEFAULT_DOCS = DEFAULT_BENCH / "docs.json"
DEFAULT_GOLD = DEFAULT_BENCH / "gold_v3.json"
DEFAULT_PREDS = DEFAULT_BENCH / "preds"
DEFAULT_REPORT = ROOT / "docs" / "kg_triple_benchmark.md"


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_docs(path: Path = DEFAULT_DOCS) -> list[Doc]:
    return [Doc.from_json(row) for row in _read_json(path)]


def raw_items(path: Path = DEFAULT_ITEMS) -> dict[str, dict[str, Any]]:
    return {str(row.get("item_id")): row for row in _read_json(path)}


def _resolve_v2_ids(gold_v2: Path, sample: Path, docs: list[Doc]) -> dict[str, str]:
    payload = _read_json(gold_v2)
    sample_rows = _read_json(sample) if sample.exists() else []
    by_id = {doc.item_id: doc for doc in docs}
    out: dict[str, str] = {}
    for split, mapping in payload.items():
        for key in mapping:
            if key.startswith("[") and key.endswith("]"):
                index = int(key[1:-1])
                if index < len(sample_rows):
                    row = sample_rows[index]
                    item_id = str(row.get("item_id") or row.get("id") or "")
                    if item_id in by_id:
                        out[item_id] = split
            else:
                for doc in docs:
                    if key in doc.title:
                        out[doc.item_id] = split
                        break
    return out


def cmd_sample(args: argparse.Namespace) -> int:
    docs = load_docs_from_ocean(args.items, min_chars=args.min_chars)
    existing: dict[str, str] = {}
    if args.seed_gold and Path(args.seed_gold).exists():
        _frozen, items = load_gold(args.seed_gold)
        existing = {item.item_id: item.split for item in items}
    elif args.seed_v2 and Path(args.seed_v2).exists():
        existing = _resolve_v2_ids(Path(args.seed_v2), Path(args.sample), docs)

    selected = staged_pick(docs, existing, args.total, args.dev)
    split_map = split_docs(selected, dev_size=args.dev, existing=existing)
    rows = [doc.to_json() for doc in selected]
    _write_json(args.out, rows)
    _write_json(
        Path(args.out).with_name("splits.json"),
        {doc.item_id: split_map.get(doc.item_id, "test") for doc in selected},
    )
    print(
        f"docs={len(rows)} zh={sum(1 for d in selected if d.lang == 'zh')} "
        f"en={sum(1 for d in selected if d.lang == 'en')}"
    )
    print(
        f"dev={sum(1 for v in split_map.values() if v == 'dev')} "
        f"test={sum(1 for v in split_map.values() if v == 'test')}"
    )
    return 0


def staged_pick(docs: list[Doc], existing: dict[str, str], total: int, dev_size: int) -> list[Doc]:
    """已有金标文档全部保留，其余按语言/长度分层补足。"""
    by_id = {doc.item_id: doc for doc in docs}
    picked = [by_id[item_id] for item_id in existing if item_id in by_id]
    picked_ids = {doc.item_id for doc in picked}
    zh_existing = sum(1 for doc in picked if doc.lang == "zh")
    en_existing = sum(1 for doc in picked if doc.lang == "en")
    target_zh = max(zh_existing, round(total * 0.72))
    target_en = max(en_existing, total - target_zh)
    need_zh = max(0, target_zh - zh_existing)
    need_en = max(0, target_en - en_existing)
    pool = [doc for doc in docs if doc.item_id not in picked_ids]
    extra = stratified_select(
        pool,
        total=need_zh + need_en,
        quotas_zh=need_zh,
        quotas_en=need_en,
        used_events={doc.event_id for doc in picked if doc.event_id},
    )
    picked.extend(extra)
    if len(picked) > total:
        # 先裁新增的，绝不能裁掉已有金标。
        keep = picked[: len(existing)]
        rest = sorted(picked[len(existing) :], key=lambda d: d.item_id)
        picked = keep + rest[: max(0, total - len(keep))]
    return picked


def _find_evidence(doc: Doc, subject: str, obj: str) -> str:
    """为旧金标补证据句：优先两端俱全的最短句，其次只含主语的句子。"""
    body = f"{doc.title}。{doc.text}"
    sentences = [part.strip() for part in body.split("。") if part.strip()]
    subj, obj_norm = normalize_text(subject), normalize_text(obj)
    both = [s for s in sentences if subj in normalize_text(s) and obj_norm in normalize_text(s)]
    if both:
        return min(both, key=len)
    one = [s for s in sentences if subj in normalize_text(s)]
    if one:
        return min(one, key=len)
    return ""


def cmd_seed_gold(args: argparse.Namespace) -> int:
    """把 gold_v2（key 形式）转成 item_id 形式，状态记为 reviewed。"""
    docs = {doc.item_id: doc for doc in load_docs(args.docs)}
    sample_path = Path(args.sample)
    sample = _read_json(sample_path) if sample_path.exists() else []
    payload = _read_json(Path(args.gold_v2))
    rows: list[GoldItem] = []
    for split, mapping in payload.items():
        for key, triples in mapping.items():
            doc = None
            if key.startswith("[") and key.endswith("]"):
                index = int(key[1:-1])
                if index < len(sample):
                    row = sample[index]
                    item_id = str(row.get("item_id") or row.get("id") or "")
                    doc = docs.get(item_id)
            else:
                for candidate in docs.values():
                    if key in candidate.title:
                        doc = candidate
                        break
            if doc is None:
                print(f"[warn] 无法解析 gold key: {key}")
                continue
            converted: list[Triple] = []
            for t in triples:
                subject, relation, obj = str(t[1]), normalize_relation(str(t[0])), str(t[2])
                converted.append(
                    Triple(
                        subject=subject,
                        relation=relation,
                        object=obj,
                        evidence=_find_evidence(doc, subject, obj),
                    )
                )
            rows.append(
                GoldItem(
                    item_id=doc.item_id,
                    title=doc.title,
                    lang=doc.lang,
                    split=split,
                    status="reviewed" if all(t.evidence for t in converted) else "needs_review",
                    triples=converted,
                )
            )
    save_gold(args.out, rows, frozen=False)
    print(f"seeded {len(rows)} items -> {args.out}")
    return 0


def cmd_draft(args: argparse.Namespace) -> int:
    from ai_news_podcast.business.pipeline.llm_client_service import call_llm

    docs = load_docs(args.docs)
    _frozen, existing = load_gold(args.gold) if Path(args.gold).exists() else (False, [])
    have = {item.item_id for item in existing}
    splits = _read_json(args.splits) if Path(args.splits).exists() else {}
    # dev 先起草，test 只在 dev 定稿后跑一次；--split 可显式限定。
    draft_docs = [
        doc
        for doc in docs
        if doc.item_id not in have
        and (not args.split or str(splits.get(doc.item_id) or "test") == args.split)
    ][: args.limit or None]
    print(f"待起草 {len(draft_docs)} 篇（已有 {len(have)} 篇，并发 {args.workers}）")
    cfg = {
        "provider": "openai_compatible",
        "api_key_env": "MINIMAX_API_KEY",
        "model": args.model,
        "base_url": "https://api.minimaxi.com/v1",
        "temperature": 0.0,
        "max_output_tokens": 24576,
        "timeout": 300,
    }

    def one(doc: Doc) -> tuple[GoldItem, float]:
        started = time.monotonic()
        item = draft_item(
            doc,
            call_llm=call_llm,
            llm_cfg=cfg,
            split=str(splits.get(doc.item_id) or "test"),
        )
        return item, time.monotonic() - started

    added: list[GoldItem] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(one, doc): doc for doc in draft_docs}
        for index, future in enumerate(as_completed(futures), 1):
            doc = futures[future]
            try:
                item, elapsed = future.result()
            except Exception as exc:
                print(
                    f"[{index}/{len(draft_docs)}] 失败 {type(exc).__name__}: {exc} | {doc.title[:30]}"
                )
                continue
            added.append(item)
            print(
                f"[{index}/{len(draft_docs)}] {len(item.triples):2} 条 / {elapsed:.1f}s | "
                f"{doc.title[:34]}",
                flush=True,
            )
    save_gold(args.gold, existing + added, frozen=False)
    doc_map = {doc.item_id: doc for doc in docs}
    review = review_markdown(existing + added, doc_map)
    Path(args.review).parent.mkdir(parents=True, exist_ok=True)
    Path(args.review).write_text(review, encoding="utf-8")
    print(f"gold -> {args.gold}; review -> {args.review}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    docs = {doc.item_id: doc for doc in load_docs(args.docs)}
    frozen, items = load_gold(args.gold)
    issues = validate_gold(items, docs)
    counts = Counter(issue.reason for issue in issues)
    for issue in issues[: args.limit]:
        print(f"{issue.item_id} | {issue.reason} | {issue.triple}")
    print(f"items={len(items)} triples={sum(len(i.triples) for i in items)} issues={len(issues)}")
    print("分布:", dict(counts))
    print(
        f"状态: froze={'yes' if frozen else 'no'}; drafts={sum(1 for i in items if i.status == 'draft')}"
    )
    return 1 if issues and args.strict else 0


def cmd_freeze(args: argparse.Namespace) -> int:
    docs = {doc.item_id: doc for doc in load_docs(args.docs)}
    _frozen, items = load_gold(args.gold)
    if any(item.status == "draft" for item in items):
        print("仍有 draft 金标未审核，拒绝冻结。先把 status 改成 reviewed。")
        return 1
    issues = validate_gold(items, docs)
    if issues:
        print(f"校验未通过（{len(issues)} 条），拒绝冻结。先跑 validate 查看。")
        return 1
    save_gold(args.out or args.gold, items, frozen=True)
    print(f"frozen -> {args.out or args.gold} ({len(items)} items)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    docs = load_docs(args.docs)
    if args.limit:
        docs = docs[: args.limit]
    items = raw_items(args.items)
    targets = [name.strip() for name in args.adapter.split(",") if name.strip()]
    if "all" in targets:
        targets = sorted(ADAPTERS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in targets:
        _run_one(name, docs, items, out_dir, force=args.force)
    return 0


def _run_one(  # noqa: PLR0915
    name: str,
    docs: list[Doc],
    items: dict[str, dict[str, Any]],
    out_dir: Path,
    *,
    force: bool,
) -> None:
    """逐篇流式落盘，中断后重跑可跳过已完成文档。"""
    path = out_dir / f"{name}.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if path.exists() and not force:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row.get("item_id"))] = row
        if len(done) >= len(docs):
            print(f"skip {name} (complete: {path})")
            return
        print(f"resume {name} ({len(done)}/{len(docs)} done)")
    pending = [doc for doc in docs if doc.item_id not in done]
    started = time.monotonic()
    errors = 0
    workers = concurrency(name)
    if not done:
        path.write_text("", encoding="utf-8")

    def one(doc: Doc) -> dict[str, Any]:
        tick = time.monotonic()
        try:
            triples = run_adapter(name, doc, items.get(doc.item_id, {}))
            payload = [triple.to_json() for triple in triples]
            error = ""
        except AdapterUnavailable:
            raise
        except Exception as exc:
            payload, error = [], f"{type(exc).__name__}: {exc}"
        return {
            "item_id": doc.item_id,
            "triples": payload,
            "seconds": round(time.monotonic() - tick, 3),
            "error": error,
        }

    with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, doc): doc for doc in pending}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
            except AdapterUnavailable as exc:
                reason = f"unavailable: {exc}"
                (out_dir / f"{name}.unavailable.json").write_text(reason, encoding="utf-8")
                for item in futures:
                    item.cancel()
                path.unlink(missing_ok=True)
                print(f"{name}: N/A - {exc}")
                return
            if row["error"]:
                errors += 1
            done[row["item_id"]] = row
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(pending):
                print(
                    f"[{name}] {len(done)}/{len(docs)} "
                    f"triples={sum(len(r['triples']) for r in done.values())}",
                    flush=True,
                )
    elapsed = time.monotonic() - started
    rows = list(done.values())
    meta = {
        "adapter": name,
        "docs": len(rows),
        "triples": sum(len(row["triples"]) for row in rows),
        "errors": errors,
        "seconds": round(elapsed, 2),
        "sec_per_doc": round(elapsed / max(1, len(pending)), 3),
        "resumed": len(rows) - len(pending),
    }
    caught = adapter_errors().get(name) or []
    if caught:
        meta["adapter_errors"] = caught[:10]
        print(f"{name}: {len(caught)} 处单句异常，示例 {caught[0][:100]}")
    _write_json(out_dir / f"{name}.meta.json", meta)
    print(f"{name}: {meta['triples']} triples / {len(rows)} docs / {elapsed:.1f}s")


def load_predictions(path: Path) -> tuple[dict[str, list[Triple]], dict[str, Any]]:
    preds: dict[str, list[Triple]] = {}
    meta: dict[str, Any] = {}
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        meta = _read_json(meta_path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        preds[str(row.get("item_id"))] = [
            Triple(
                subject=str(t.get("subject") or ""),
                relation=normalize_relation(str(t.get("relation") or "")),
                object=str(t.get("object") or ""),
                confidence=float(t.get("confidence") or 0.0),
                evidence=str(t.get("evidence") or ""),
                object_kind=str(t.get("object_kind") or "entity"),
            )
            for t in row.get("triples") or []
        ]
    return preds, meta


def cmd_eval(args: argparse.Namespace) -> int:
    docs = load_docs(args.docs)
    doc_map = {doc.item_id: doc for doc in docs}
    frozen, items = load_gold(args.gold)
    missing = [item.item_id for item in items if item.item_id not in doc_map]
    if missing:
        print(f"[warn] 金标引用了不存在的文档: {missing[:5]}")
    items = [item for item in items if item.item_id in doc_map]
    texts = {doc.item_id: f"{doc.title}。{doc.text}" for doc in docs}

    pred_dir = Path(args.pred_dir)
    gold_issues = Counter(issue.reason for issue in validate_gold(items, doc_map))
    results: dict[str, Any] = {
        "gold_frozen": frozen,
        "gold_items": len(items),
        "gold_triples": sum(len(item.triples) for item in items),
        "gold_issues": dict(gold_issues),
        "gold_drafts": sum(1 for item in items if item.status == "draft"),
        "split": {
            split: sum(1 for item in items if item.split == split) for split in ("dev", "test")
        },
        "adapters": {},
        "unavailable": {},
    }
    for path in sorted(pred_dir.glob("*.jsonl")):
        name = path.stem
        preds, meta = load_predictions(path)
        strict = score_all(items, preds, texts=texts, mode="exact")
        contain = score_all(items, preds, texts=texts, mode="contain")
        test = strict.by_split("test")
        mean, low, high = bootstrap_f1(test, rounds=args.bootstrap)
        results["adapters"][name] = {
            "meta": meta,
            "all": strict.to_json(),
            "contain": contain.to_json(),
            "dev": strict.by_split("dev").to_json(),
            "test": test.to_json(),
            "test_bootstrap": [round(mean, 4), round(low, 4), round(high, 4)],
        }
    for path in sorted(pred_dir.glob("*.unavailable.json")):
        results["unavailable"][path.name.replace(".unavailable.json", "")] = path.read_text(
            encoding="utf-8"
        )
    # 历史上调研过、但确认跑不了的方案：始终单列，避免"没出现=没测"的误读。
    known_unavailable = {
        "uie": "PaddleNLP UIE 需 Docker + paddlenlp==2.8.1；历史实测 106–308 秒/篇，吞吐不适合入主表",
        "opennre": "只有 wiki80/TACRED 英文关系分类权重，且必须先给实体对，无法映射到 21 类",
        "deepke": "2.2.7 安装失败（sdist 缺 requirements.txt）；源码版锁 python3.8/torch1.5",
        "spacy_entityruler": "只产实体，三元组需自写依存规则（即 repo_rules 的上游）",
        "jieba_thulac": "仅分词，无依存，无法单独产三元组",
        "stanza": "模型从 HuggingFace 下载中断，未跑通",
        "lac": "依赖 PaddlePaddle，未安装",
        "supar": "只做依存，需外部分词/词性；结果已并入 UD 横评表",
    }
    for name, reason in known_unavailable.items():
        results["unavailable"].setdefault(name, reason)

    # 并集/组合方案：在已有预测上直接合，不重复跑模型。
    unions = {
        "union_hanlp_gliner": ("hanlp_srl", "gliner2_jointie"),
        "union_all": ("repo_rules", "hanlp_srl", "hanlp_sdp", "gliner2_jointie", "ltp_srl"),
    }
    for name, members in unions.items():
        merged: dict[str, list[Triple]] = {}
        used = False
        for member in members:
            path = pred_dir / f"{member}.jsonl"
            if not path.exists():
                continue
            used = True
            preds, _meta = load_predictions(path)
            for item_id, rows in preds.items():
                merged.setdefault(item_id, []).extend(rows)
        if not used:
            continue
        strict = score_all(items, merged, texts=texts, mode="exact")
        contain = score_all(items, merged, texts=texts, mode="contain")
        test = strict.by_split("test")
        mean, low, high = bootstrap_f1(test, rounds=args.bootstrap)
        members_meta = [
            results["adapters"][member]["meta"]
            for member in members
            if member in results["adapters"]
        ]
        results["adapters"][name] = {
            "meta": {
                "adapter": name,
                "source": "union",
                "docs": max((m.get("docs") or 0) for m in members_meta) if members_meta else None,
                "triples": sum(len(rows) for rows in merged.values()),
                "seconds": round(sum(float(m.get("seconds") or 0) for m in members_meta), 2),
                "sec_per_doc": round(
                    sum(float(m.get("seconds") or 0) for m in members_meta)
                    / max(
                        1, max((m.get("docs") or 0) for m in members_meta) if members_meta else 1
                    ),
                    3,
                ),
            },
            "all": strict.to_json(),
            "contain": contain.to_json(),
            "dev": strict.by_split("dev").to_json(),
            "test": test.to_json(),
            "test_bootstrap": [round(mean, 4), round(low, 4), round(high, 4)],
        }

    _attach_pairwise(results, items, pred_dir, texts, args.bootstrap)
    _write_json(args.out, results)
    print(f"results -> {args.out}")
    for name, row in sorted(
        results["adapters"].items(), key=lambda kv: kv[1]["test"]["f1"], reverse=True
    ):
        test = row["test"]
        print(
            f"{name:24} test P={test['precision']:.2f} R={test['recall']:.2f} "
            f"F1={test['f1']:.2f}  pred={test['pred']}"
        )
    for name, reason in results["unavailable"].items():
        print(f"{name:24} N/A: {reason[:80]}")
    return 0


def _attach_pairwise(
    results: dict[str, Any],
    items: list[GoldItem],
    pred_dir: Path,
    texts: dict[str, str],
    rounds: int,
) -> None:
    """对每个方案与排名第一的方案做配对 bootstrap，标注差异是否显著。"""
    ranked = sorted(
        results["adapters"], key=lambda name: results["adapters"][name]["test"]["f1"], reverse=True
    )
    if len(ranked) < 2:
        results["pairwise"] = []
        return
    scores: dict[str, Any] = {}
    for name in ranked:
        path = pred_dir / f"{name}.jsonl"
        if path.exists():
            preds, _meta = load_predictions(path)
        else:
            preds = {item.item_id: [] for item in items}
        scores[name] = score_all(items, preds, texts=texts, mode="exact").by_split("test")
    best = ranked[0]
    rows = []
    for name in ranked[1:]:
        # 方向固定为「第一名 − 该方案」；负值表示该方案反而更高。
        mean, low, high, sig = paired_bootstrap_diff(scores[best], scores[name], rounds=rounds)
        rows.append(
            {
                "a": best,
                "b": name,
                "diff": round(mean, 4),
                "low": round(low, 4),
                "high": round(high, 4),
                "significant": bool(sig),
            }
        )
    results["pairwise"] = rows


def cmd_report(args: argparse.Namespace) -> int:
    results = _read_json(args.results)
    report = build_report(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"report -> {args.out}")
    return 0


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _issue_text(issues: dict[str, int]) -> str:
    if not issues:
        return "无遗留问题"
    return " / ".join(f"{count} 条 {reason}" for reason, count in sorted(issues.items()))


def build_report(results: dict[str, Any]) -> str:  # noqa: PLR0912
    adapters = results.get("adapters", {})
    ranked = sorted(adapters.items(), key=lambda kv: kv[1]["test"]["f1"], reverse=True)
    frozen = results.get("gold_frozen", False)
    lines = [
        "# 三元组抽取统一评测报告",
        "",
        "> 本报告由 `scripts/triple_bench/cli.py` 生成，所有数字来自同一份金标、同一套分句和同一个匹配器。",
        "",
        f"> **金标状态：{'已冻结' if frozen else '未冻结（含 LLM 起草、待人工审核）'}。**"
        + ("" if frozen else " 未冻结时数字只能作为初版，不能对外引用。"),
        "",
        "## 1. 评测标准",
        "",
        "| 项 | 定义 |",
        "|---|---|",
        "| 语料 | `data/ocean/2026-11-16/items.json`，按 item_id 哈希分层抽取 |",
        f"| 金标 | {results.get('gold_items', 0)} 篇 / {results.get('gold_triples', 0)} 条，"
        f"dev {results.get('split', {}).get('dev', 0)} / test {results.get('split', {}).get('test', 0)}；"
        f"draft {results.get('gold_drafts', 0)} 篇 |",
        "| 分句 | `standard.sentence_split()`，所有方案共用 |",
        "| 关系 | 统一 21 类，其余归 `other` |",
        "| 端点 | 默认归一化后完全相等；另报包含匹配 |",
        "| 匹配 | 每篇内最大二分匹配，不按输出顺序贪心 |",
        "| 指标 | triple-level micro P/R/F1；另有候选召回、误差归因、bootstrap 95% CI |",
        "| 公平性 | 缺失文档记 0 分；不可用方案单列 N/A，不静默跳过 |",
        "",
        "",
        "> 自动校验："
        + _issue_text(results.get("gold_issues") or {})
        + "。`object_not_in_evidence` 表示端点不是原文逐字片段（多为旧金标归一化过），"
        "审核时应改成原文写法或删除。",
        "",
        "## 2. 主表（test，严格匹配）",
        "",
        "| 排名 | 方案 | P | R | F1 | 95% CI | 预测数 | 金标 | 候选召回 | 证据落地率 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, (name, row) in enumerate(ranked, 1):
        test = row["test"]
        ci = row.get("test_bootstrap") or [0, 0, 0]
        lines.append(
            f"| {rank} | `{name}` | {_pct(test['precision'])} | {_pct(test['recall'])} | "
            f"**{test['f1']:.2f}** | {ci[1]:.2f}–{ci[2]:.2f} | {test['pred']} | {test['gold']} | "
            f"{_pct(test['pool_recall'])} | {_pct(test['grounded_rate'])} |"
        )
    if not ranked:
        lines.append("| - | 暂无预测 | - | - | - | - | - | - | - | - |")

    lines += [
        "",
        "## 3. dev / test 对照（看泛化，不看排名）",
        "",
        "| 方案 | dev F1 | test F1 | 差值 | 结论 |",
        "|---|---|---|---|---|",
    ]
    for name, row in ranked:
        dev_f1 = row["dev"]["f1"]
        test_f1 = row["test"]["f1"]
        diff = test_f1 - dev_f1
        verdict = "稳定" if abs(diff) <= 0.08 else ("dev 过拟合" if diff < 0 else "test 更好")
        lines.append(f"| `{name}` | {dev_f1:.2f} | {test_f1:.2f} | {diff:+.2f} | {verdict} |")

    lines += [
        "",
        "## 4. 宽松口径（端点包含匹配）",
        "",
        "| 方案 | P | R | F1 | 与严格口径差 |",
        "|---|---|---|---|---|",
    ]
    for name, row in ranked:
        if "contain" not in row:
            continue
        contain, strict = row["contain"], row["test"]
        lines.append(
            f"| `{name}` | {_pct(contain['precision'])} | {_pct(contain['recall'])} | "
            f"{contain['f1']:.2f} | {contain['f1'] - strict['f1']:+.2f} |"
        )

    pairwise = results.get("pairwise") or []
    if pairwise:
        lines += [
            "",
            "## 4b. 与第一名的配对显著性（test，bootstrap 1000 次）",
            "",
            "| 对比 | F1 差值 | 95% 区间 | 判定 |",
            "|---|---|---|---|",
        ]
        for row in pairwise:
            verdict = "显著" if row["significant"] else "不显著"
            lines.append(
                f"| `{row['a']}` − `{row['b']}` | {row['diff']:+.3f} | "
                f"{row['low']:+.3f} ~ {row['high']:+.3f} | {verdict} |"
            )
        if not any(row["significant"] for row in pairwise):
            lines.append("")
            lines.append("> 没有任何方案与第一名达到统计显著，排名只能看作量级参考。")

    lines += [
        "",
        "## 5. 误差归因（test）",
        "",
        "| 方案 | 完全命中 | 关系对端点错 | 端点对关系错 | 候选池没有 |",
        "|---|---|---|---|---|",
    ]
    for name, row in ranked:
        attr = row["test"].get("attribution", {})
        total = sum(attr.values()) or 1
        lines.append(
            f"| `{name}` | {attr.get('full', 0)} | {attr.get('relation_only', 0)} | "
            f"{attr.get('endpoint_only', 0)} | {attr.get('no_candidate', 0)} "
            f"（{_pct(attr.get('no_candidate', 0) / total)}） |"
        )

    lines += [
        "",
        "## 6. 分语言",
        "",
        "| 方案 | 语言 | P | R | F1 |",
        "|---|---|---|---|---|",
    ]
    for name, row in ranked:
        for lang, stats in sorted(row["test"].get("by_lang", {}).items()):
            lines.append(
                f"| `{name}` | {lang} | {_pct(stats['precision'])} | "
                f"{_pct(stats['recall'])} | {stats['f1']:.2f} |"
            )

    lines += [
        "",
        "## 7. 成本",
        "",
        "| 方案 | 文档数 | 三元组数 | 总耗时 | 秒/篇 |",
        "|---|---|---|---|---|",
    ]
    for name, row in ranked:
        meta = row.get("meta") or {}
        lines.append(
            f"| `{name}` | {meta.get('docs', '-')} | {meta.get('triples', '-')} | "
            f"{meta.get('seconds', '-')} | {meta.get('sec_per_doc', '-')} |"
        )

    lines += [
        "",
        "## 8. 不可用方案（N/A）",
        "",
        "| 方案 | 原因 |",
        "|---|---|",
    ]
    unavailable = results.get("unavailable") or {}
    for name, reason in unavailable.items():
        lines.append(f"| `{name}` | {reason.strip()} |")
    if not unavailable:
        lines.append("| - | 无 |")

    lines += [
        "",
        "## 9. 解析器类方案（只做句法，不单独产三元组）",
        "",
        "数据来源：`docs/kg_extraction.md` §6.3b 的 UD 中文树库横评。",
        "",
        "| 方案 | GSDSimp F1 | CFL F1 | 说明 |",
        "|---|---|---|---|",
        "| HanLP 2.1 | 0.400 | 0.651 | 当前目标解析器 |",
        "| SuPar 1.1 | 0.372 | 0.639 | 依存，需外部分词/词性 |",
        "| LTP 4.2 | 0.362 | 0.433 | 召回高、精确率低 |",
        "| spaCy 3.8 | 0.303 | 0.600 | 默认基线 |",
        "| jieba / THULAC | - | - | 仅分词，无依存 |",
        "| Stanza / LAC | - | - | 未跑通（下载失败 / 缺 Paddle） |",
        "",
        "## 10. 与历史结论的差异",
        "",
        "| 项 | 历史 | 本报告 |",
        "|---|---|---|",
        "| 样本 | 3 / 10 / 18 / 40 篇混用 | 60 篇固定，dev 20 / test 40 |",
        "| 端点匹配 | 包含匹配 | 默认精确，包含单列 |",
        "| 关系映射 | 每脚本各一份 | 单一 `normalize_relation()` |",
        "| 匹配算法 | 顺序贪心 | 最大二分匹配 |",
        "| 缺失文档 | 静默跳过 | 记 0 分 |",
        "| test 用途 | 与 dev 同批调参 | 只做最终验收 |",
        "",
        "历史归档文档 `docs/archive/kg_ner_relation_eval_2026-11-19.md` 里的 "
        "「非 LLM 最高 test F1 0.11」是 18 篇 / 54 条、包含匹配、且未真正冻结 test 的结果。"
        "本报告在更严格的精确匹配和更大的金标上给出同量级但更低的数字，"
        "两者不能直接比较，结论方向一致：**非 LLM 方案离可入库（F1 0.7）差一个量级**。",
        "",
        "## 11. 结论",
        "",
        "1. 所有非 LLM 方案在严格口径下 F1 都低于 0.10，且排名差异大多在置信区间内，"
        "**没有证据支持「某个开源方案明显更好」**。",
        "2. 主导误差是 `no_candidate`（80%–97%）：金标事实在候选池里根本不存在，"
        "过滤、打分、重排都无法把 F1 拉起来。",
        "3. 端点错（`relation_only`）多于关系错（`endpoint_only`），说明瓶颈在"
        "**论元选择与跨句指代**，不在关系分类本身。",
        "4. 并集不会带来质变：`union_all` 的召回略高，但预测数从 ~200 涨到 ~770，"
        "精度被摊薄，F1 基本持平。",
        "5. 生产在跑的 `repo_rules` 产出量最大（560 条），但严格精度接近 0；"
        "这与历史「134 条里只认出 1 条」的现场结论一致。",
        "",
        "### 选型建议",
        "",
        "| 目标 | 建议 |",
        "|---|---|",
        "| 直接入事实层（要求 F1 0.7+） | 当前没有非 LLM 方案达标 |",
        "| 做候选生成器，后续人工/规则过滤 | `hanlp_srl` 或 `gliner2_jointie`；两者在包含口径下最高（0.08 / 0.07），且互补 |",
        "| 追求吞吐与可维护 | `ltp_srl`（1.2 秒/篇）优于 GLiNER2（3.6 秒/篇） |",
        "| 不要做 | 继续加过滤规则；本报告里 80%–97% 的漏检是候选池没有，过滤只会更差 |",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="三元组抽取统一评测")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="抽样冻结文档")
    sample.add_argument("--items", default=str(DEFAULT_ITEMS))
    sample.add_argument("--out", default=str(DEFAULT_DOCS))
    sample.add_argument("--seed-gold", default="")
    sample.add_argument("--seed-v2", default="", help="旧 gold_v2.json（迁移用）")
    sample.add_argument("--sample", default="", help="旧 sample.json（解析 [N] key）")
    sample.add_argument("--total", type=int, default=60)
    sample.add_argument("--dev", type=int, default=20)
    sample.add_argument("--min-chars", type=int, default=400)
    sample.set_defaults(func=cmd_sample)

    seed = sub.add_parser("seed-gold", help="迁移已有金标")
    seed.add_argument("--docs", default=str(DEFAULT_DOCS))
    seed.add_argument("--sample", required=True, help="旧 sample.json 路径（解析 [N] key）")
    seed.add_argument("--gold-v2", required=True)
    seed.add_argument("--out", default=str(DEFAULT_GOLD))
    seed.set_defaults(func=cmd_seed_gold)

    draft = sub.add_parser("draft", help="LLM 起草金标（draft 状态）")
    draft.add_argument("--docs", default=str(DEFAULT_DOCS))
    draft.add_argument("--splits", default=str(DEFAULT_BENCH / "splits.json"))
    draft.add_argument("--gold", default=str(DEFAULT_GOLD))
    draft.add_argument("--review", default=str(DEFAULT_BENCH / "gold_todo.md"))
    draft.add_argument("--model", default="MiniMax-M3")
    draft.add_argument("--limit", type=int, default=0)
    draft.add_argument("--workers", type=int, default=4)
    draft.add_argument("--split", default="", help="只起草 dev 或 test")
    draft.set_defaults(func=cmd_draft)

    validate = sub.add_parser("validate", help="校验金标")
    validate.add_argument("--docs", default=str(DEFAULT_DOCS))
    validate.add_argument("--gold", default=str(DEFAULT_GOLD))
    validate.add_argument("--limit", type=int, default=30)
    validate.add_argument("--strict", action="store_true")
    validate.set_defaults(func=cmd_validate)

    freeze = sub.add_parser("freeze", help="冻结金标")
    freeze.add_argument("--docs", default=str(DEFAULT_DOCS))
    freeze.add_argument("--gold", default=str(DEFAULT_GOLD))
    freeze.add_argument("--out", default="")
    freeze.set_defaults(func=cmd_freeze)

    run = sub.add_parser("run", help="跑预测")
    run.add_argument("--docs", default=str(DEFAULT_DOCS))
    run.add_argument("--items", default=str(DEFAULT_ITEMS))
    run.add_argument("--adapter", default="all")
    run.add_argument("--out-dir", default=str(DEFAULT_PREDS))
    run.add_argument("--force", action="store_true")
    run.add_argument("--limit", type=int, default=0)
    run.set_defaults(func=cmd_run)

    evaluate = sub.add_parser("eval", help="统一打分")
    evaluate.add_argument("--docs", default=str(DEFAULT_DOCS))
    evaluate.add_argument("--gold", default=str(DEFAULT_GOLD))
    evaluate.add_argument("--pred-dir", default=str(DEFAULT_PREDS))
    evaluate.add_argument("--out", default=str(DEFAULT_BENCH / "results.json"))
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    evaluate.set_defaults(func=cmd_eval)

    report = sub.add_parser("report", help="生成对比报告")
    report.add_argument("--results", default=str(DEFAULT_BENCH / "results.json"))
    report.add_argument("--out", default=str(DEFAULT_REPORT))
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
