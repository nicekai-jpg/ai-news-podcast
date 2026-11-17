#!/usr/bin/env python3
"""事实层 CLI：build / backfill / report / eval / query。

用法：
    python scripts/kg_facts.py build --date 2026-11-16
    python scripts/kg_facts.py backfill --days 8
    python scripts/kg_facts.py report --days 28 --out-md data/facts/report.md
    python scripts/kg_facts.py eval --gold data/eval/kg_facts_gold.jsonl --min-precision 0.85
    python scripts/kg_facts.py query --subject 英伟达 --relation personnel
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_news_podcast.business.contracts import Event  # noqa: E402
from ai_news_podcast.business.kg.dep import DepParser, extract_predicates  # noqa: E402
from ai_news_podcast.business.kg.fill import reporting_verbs  # noqa: E402
from ai_news_podcast.business.kg.lexicon import KGlexicon  # noqa: E402
from ai_news_podcast.business.kg.relations import CANONICAL_LABELS  # noqa: E402
from ai_news_podcast.business.kg.service import build_facts  # noqa: E402
from ai_news_podcast.business.kg.store import (  # noqa: E402
    read_all_facts,
)
from ai_news_podcast.business.pipeline.kg_stage_service import (  # noqa: E402
    load_facts_config,
    run_facts_stage,
)

RELATIONS = tuple(CANONICAL_LABELS)


def _load_day(date_str: str, data_dir: Path) -> tuple[list[dict], list[Event]]:
    ocean = data_dir / "ocean" / date_str
    items = json.loads((ocean / "items.json").read_text(encoding="utf-8"))
    events = [
        Event.from_dict(row)
        for row in json.loads((ocean / "events.json").read_text(encoding="utf-8"))
    ]
    return items, events


def cmd_build(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    items, events = _load_day(args.date, data_dir)
    stats = run_facts_stage(
        items,
        events,
        date_str=args.date,
        config_root=Path(args.config_dir),
        data_root=data_dir,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if stats.get("ok", True) else 1


def cmd_backfill(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    ocean_root = data_dir / "ocean"
    if not ocean_root.is_dir():
        print(f"没有历史数据：{ocean_root}", file=sys.stderr)
        return 2
    dates = sorted(p.name for p in ocean_root.iterdir() if p.is_dir())[-args.days :]
    done = 0
    for date_str in dates:
        if not args.force and (data_dir / "facts" / f"facts-{date_str}.jsonl").exists():
            print(f"{date_str}: 已存在，跳过")
            continue
        items, events = _load_day(date_str, data_dir)
        stats = run_facts_stage(
            items,
            events,
            date_str=date_str,
            config_root=Path(args.config_dir),
            data_root=data_dir,
        )
        print(
            f"{date_str}: events={stats.get('events', 0)} A={stats.get('cards_a', 0)} B={stats.get('cards_b', 0)}"
        )
        done += 1
    print(f"回填完成：{done} 天")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    facts = read_all_facts(Path(args.data_dir) / "facts")
    rows = []
    for row in facts:
        if args.subject and args.subject not in str(row.get("subject", {}).get("name", "")):
            continue
        if args.relation and row.get("relation") != args.relation:
            continue
        if args.since and str(row.get("date", "")) < args.since:
            continue
        rows.append(row)
    for row in rows[: args.limit]:
        subject = row.get("subject", {}).get("name", "")
        slots = row.get("slots", {})
        object_text = ""
        for key in ("object", "person", "role"):
            if slots.get(key):
                object_text = str(slots[key].get("text") or "")
                break
        print(
            f"{row.get('date')} [{row.get('grade')}] {subject} "
            f"--{row.get('relation')}--> {object_text} | {row.get('evidence', {}).get('sentence', '')[:60]}"
        )
    print(f"共 {len(rows)} 条")
    return 0


def _iso_weeks(days: int) -> list[str]:
    today = date.today()
    weeks: list[str] = []
    for offset in range(days, -1, -1):
        day = today - timedelta(days=offset)
        year, week, _ = day.isocalendar()
        label = f"{year}-W{week:02d}"
        if label not in weeks:
            weeks.append(label)
    return weeks[-4:]


def cmd_report(args: argparse.Namespace) -> int:  # noqa: PLR0912
    data_dir = Path(args.data_dir)
    facts = read_all_facts(data_dir / "facts", days=args.days)
    if not facts:
        print("没有事实数据", file=sys.stderr)
        return 2
    weeks = _iso_weeks(args.days)
    relation_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    kind_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in facts:
        if row.get("is_new") is False:
            continue
        try:
            week = datetime.strptime(str(row.get("date")), "%Y-%m-%d").date().isocalendar()
        except ValueError:
            continue
        label = f"{week[0]}-W{week[1]:02d}"
        if label not in weeks:
            continue
        relation = str(row.get("relation") or "other")
        relation_matrix[relation][label] += 1
        kind = str(row.get("subject", {}).get("kind") or "other")
        kind_matrix[kind][label] += 1
    lines = [f"# 事实层周报（最近 {args.days} 天）", ""]
    new_rows = [row for row in facts if row.get("is_new") is not False]
    lines += [
        "## 概览",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 事实总数 | {len(facts)} 条 |",
        f"| 新事件事实 | {len(new_rows)} 条 |",
        f"| A 档 | {sum(1 for r in new_rows if r.get('grade') == 'A')} 条 |",
        f"| B 档 | {sum(1 for r in new_rows if r.get('grade') == 'B')} 条 |",
        "",
        "## 事件类型矩阵",
        "",
        "| 关系 | " + " | ".join(weeks) + " |",
        "|---|" + "---|" * len(weeks),
    ]
    for relation in RELATIONS:
        counts = relation_matrix.get(relation, Counter())
        lines.append(
            "| "
            + CANONICAL_LABELS[relation]
            + " | "
            + " | ".join(str(counts.get(w, 0)) for w in weeks)
            + " |"
        )
    lines += [
        "",
        "## 实体域矩阵",
        "",
        "| 实体类型 | " + " | ".join(weeks) + " |",
        "|---|" + "---|" * len(weeks),
    ]
    for kind in ("company", "person", "org", "product", "tech"):
        counts = kind_matrix.get(kind, Counter())
        lines.append(f"| {kind} | " + " | ".join(str(counts.get(w, 0)) for w in weeks) + " |")
    gap_relations = [
        CANONICAL_LABELS[r]
        for r in RELATIONS
        if sum(relation_matrix.get(r, Counter()).values()) == 0
    ]
    gap_kinds = [
        kind
        for kind in ("company", "person", "org", "product", "tech")
        if sum(kind_matrix.get(kind, Counter()).values()) == 0
    ]
    lines += [
        "",
        "## 缺口清单",
        "",
        f"- 近 {args.days} 天 0 条的关系：{'、'.join(gap_relations) or '无'}",
        f"- 0 条的实体类型：{'、'.join(gap_kinds) or '无'}",
    ]
    unmatched: Counter[str] = Counter()
    for manifest_path in sorted((data_dir / "facts").glob("manifest-*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8").strip().splitlines()[0])
        except (ValueError, IndexError):
            continue
        for key in ("unmatched_verbs", "unmatched_objects", "unmatched_subjects"):
            for name, count in manifest.get("stats", {}).get(key, []):
                unmatched[f"{key}:{name}"] += int(count)
    lines += ["", "## 未匹配词 Top 20", ""]
    for name, count in unmatched.most_common(20):
        lines.append(f"- {name}：{count}")
    report = "\n".join(lines) + "\n"
    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"报告已写入 {out}")
    else:
        print(report)
    return 0


def _norm(text: str) -> str:
    value = re.sub(r"[\s，。；：、（）()「」“”\"'·-]", "", str(text or "")).lower()
    for prefix in ("公司",):
        value = value.replace(prefix, "")
    return value


def _subject_match(card_subject: str, expected_subject: str) -> bool:
    a, b = _norm(card_subject), _norm(expected_subject)
    return bool(a and b) and (a == b or a in b or b in a)


def _object_match(card_object: str, expected_object: str) -> bool:
    a, b = _norm(card_object), _norm(expected_object)
    if not a or not b:
        return a == b
    return a == b or a in b or b in a


def _match_fact(card: dict, expected: dict) -> bool:
    if card.get("relation") != expected.get("relation"):
        return False
    slots = card.get("slots", {})
    object_text = ""
    for key in ("object", "person", "role"):
        if slots.get(key):
            object_text = str(slots[key].get("text") or "")
            if key in ("object", "role"):
                break
    subject = str(card.get("subject", {}).get("name") or "")
    return _subject_match(subject, str(expected.get("subject") or "")) and _object_match(
        object_text, str(expected.get("object") or "")
    )


ALIASES = {
    "三方合作": "三方",
    "一个漏洞": "漏洞",
    "技术报告、模型和代码": "开源",
}


def _in_scope(fact: dict, sentence: str) -> bool:
    """金标事实的主语/宾语是否在句内出现过（含别名归一）；越界的单列不计召回。"""
    haystack = _norm(sentence)
    subject = _norm(fact.get("subject"))
    obj = _norm(fact.get("object"))
    if not subject or subject not in haystack:
        return False
    if not obj or obj in haystack:
        return True
    return any(_norm(key) == obj and _norm(value) in haystack for key, value in ALIASES.items())


def _make_eval_item(sample: dict, index: int) -> tuple[dict, Event]:
    sentence = "".join(sample.get("sentences") or [])
    text = str(sentence or "")
    lang = (
        "en" if re.search(r"[A-Za-z]", text) and not re.search(r"[\u4e00-\u9fff]", text) else "zh"
    )
    item = {
        "item_id": f"eval-{index:04d}",
        "source": str(sample.get("source") or "eval"),
        "source_label": str(sample.get("source_name") or ""),
        "category": str(sample.get("category") or "news"),
        "title": str(sample.get("title") or ""),
        "link": "",
        "published_at": str(sample.get("published_at") or ""),
        "summary": "",
        "full_text": text,
        "full_text_status": "ok",
        "lang": lang,
        "dup_of": "",
    }
    event = Event(
        event_id=f"eval-{index:04d}",
        representative_title=str(sample.get("title") or ""),
        item_ids=[item["item_id"]],
        sequence=1,
    )
    return item, event


def _parse_diagnostic(
    samples: list[dict], lex: KGlexicon, models: dict[str, str]
) -> dict[str, dict[str, float | int]]:
    """银标主谓宾诊断：sm/md 对比。"""
    result: dict[str, dict[str, float | int]] = {}
    reporting = reporting_verbs()
    for label, model_spec in models.items():
        models_cfg = {"zh": model_spec, "en": "en_core_web_sm"}
        if model_spec == "md":
            models_cfg = {"zh": "zh_core_web_md", "en": "en_core_web_md"}
        elif model_spec == "sm":
            models_cfg = {"zh": "zh_core_web_sm", "en": "en_core_web_sm"}
        parser = DepParser.load(lex, models=models_cfg)
        if parser is None:
            result[label] = {"skipped": 1}
            continue
        total = 0
        subject_ok = 0
        object_ok = 0
        predicate_ok = 0
        for sample in samples:
            roles = sample.get("parse_roles") or {}
            if not roles:
                continue
            sentence = "".join(sample.get("sentences") or [])
            lang = (
                "en"
                if re.search(r"[A-Za-z]", sentence) and not re.search(r"[\u4e00-\u9fff]", sentence)
                else "zh"
            )
            try:
                parse = parser.analyze(sentence, lang)
            except Exception:
                continue
            preds = [p for p in extract_predicates(parse, reporting) if p.subjects]
            if not preds:
                continue
            total += 1
            gold_subject = _norm(str(roles.get("subject") or ""))
            gold_object = _norm(str(roles.get("object") or ""))
            gold_predicate = _norm(str(roles.get("predicate") or ""))
            subjects = [_norm(parse.tokens[i].text) for pred in preds for i in pred.subjects]
            objects = [_norm(parse.tokens[i].text) for pred in preds for i in pred.objects]
            predicates = [_norm(parse.tokens[p.verb_index].text) for p in preds]
            if gold_subject and any(_subject_match(s, gold_subject) for s in subjects):
                subject_ok += 1
            if gold_object and any(_object_match(o, gold_object) for o in objects):
                object_ok += 1
            if gold_predicate and any(_object_match(p, gold_predicate) for p in predicates):
                predicate_ok += 1
        result[label] = {
            "sentences": total,
            "subject_accuracy": round(subject_ok / total, 3) if total else 0.0,
            "object_accuracy": round(object_ok / total, 3) if total else 0.0,
            "predicate_accuracy": round(predicate_ok / total, 3) if total else 0.0,
        }
    return result


def cmd_eval(args: argparse.Namespace) -> int:  # noqa: PLR0915
    gold_path = Path(args.gold)
    if not gold_path.is_file():
        print(f"金标不存在：{gold_path}", file=sys.stderr)
        return 2
    samples = [
        json.loads(line)
        for line in gold_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = load_facts_config(Path(args.config_dir))
    lex = KGlexicon.load(Path(args.config_dir) / "nlp")
    parser = DepParser.load(lex)
    if parser is None:
        print("解析模型不可用", file=sys.stderr)
        return 2
    cards_all: list[dict] = []
    expected_total = 0
    out_of_scope = 0
    matched_expected = 0
    noise_false = 0
    for index, sample in enumerate(samples):
        item, event = _make_eval_item(sample, index)
        result = build_facts(
            [item], [event], date_str="2026-11-16", config=config, lexicon=lex, parser=parser
        )
        cards = [card.to_dict() for card in result.cards]
        for card in cards:
            card["_sample_id"] = sample.get("sample_id")
            card["_noise"] = bool(sample.get("noise"))
        cards_all.extend(cards)
        expected = sample.get("expected") or []
        sentence_text = "".join(sample.get("sentences") or [])
        for fact in expected:
            if not _in_scope(fact, sentence_text):
                out_of_scope += 1
                continue
            expected_total += 1
            if any(_match_fact(card, fact) for card in cards):
                matched_expected += 1
        if sample.get("noise"):
            noise_false += sum(1 for card in cards if card.get("grade") == "A")
    a_cards = [card for card in cards_all if card.get("grade") == "A"]
    a_correct = 0
    for card in a_cards:
        expected = next(
            (s.get("expected") or [] for s in samples if s.get("sample_id") == card["_sample_id"]),
            [],
        )
        if any(_match_fact(card, fact) for fact in expected):
            a_correct += 1
    precision = round(a_correct / len(a_cards), 3) if a_cards else 1.0
    recall = round(matched_expected / expected_total, 3) if expected_total else 0.0
    evidence_ok = sum(1 for card in cards_all if card.get("evidence", {}).get("sentence"))
    print("== 金标评测 ==")
    print(f"样本数：{len(samples)}")
    print(
        f"抽出事实：{len(cards_all)} 条（A 档 {len(a_cards)} / B 档 {len(cards_all) - len(a_cards)}）"
    )
    print(f"A 档准确率：{precision}（{a_correct}/{len(a_cards)}）")
    print(f"事件召回：{recall}（{matched_expected}/{expected_total}）")
    print(f"金标越界（句内找不到主语/宾语，不计召回）：{out_of_scope}")
    print(f"证据率：{round(evidence_ok / len(cards_all), 3) if cards_all else 0.0}")
    print(f"噪声误抽：{noise_false}")
    if args.diagnose:
        print("== 解析诊断（sm/md 对比）==")
        diag = _parse_diagnostic(samples, lex, {"sm": "sm", "md": "md"})
        for label, metrics in diag.items():
            print(f"{label}: {metrics}")
    ok = precision >= args.min_precision
    print(f"门槛 {args.min_precision}：{'通过' if ok else '未通过'}")
    return 0 if ok else 1


def _fact_object_text(row: dict) -> str:
    slots = row.get("slots") or {}
    for key in ("object", "role", "person"):
        value = slots.get(key)
        if isinstance(value, dict) and value.get("text"):
            return str(value["text"])
    return ""


def _expected_fact_id(row: dict) -> str:
    import hashlib

    def slug(text: str) -> str:
        return "".join(str(text or "").split()).lower()

    subject = row.get("subject") or {}
    slots = row.get("slots") or {}
    peer = ""
    for key in ("peer", "company", "investor"):
        value = slots.get(key)
        if isinstance(value, dict) and value.get("text"):
            peer = str(value["text"])
            break
    raw = "|".join(
        [
            str(row.get("date") or ""),
            slug(str(subject.get("id") or subject.get("name") or "")),
            str(row.get("relation") or ""),
            slug(_fact_object_text(row)),
            slug(peer),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def cmd_prune(args: argparse.Namespace) -> int:
    """B 档轮转：只保留最近 N 天的 B，A 永久保留。默认 dry-run，--apply 才真删。"""
    from datetime import date, timedelta

    data_dir = Path(args.data_dir)
    facts_dir = data_dir / "facts"
    cutoff = date.today() - timedelta(days=args.keep_b_days)
    total_before = total_after = 0
    for path in sorted(facts_dir.glob("facts-*.jsonl")):
        date_str = path.stem.replace("facts-", "")
        try:
            file_date = date.fromisoformat(date_str)
        except ValueError:
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        total_before += len(rows)
        if file_date >= cutoff:
            total_after += len(rows)
            continue
        kept = [row for row in rows if row.get("grade") == "A"]
        total_after += len(kept)
        removed = len(rows) - len(kept)
        print(f"{date_str}: 保留 {len(kept)} 条 A，删除 {removed} 条 B")
        if args.apply and removed:
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                for row in kept:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp.replace(path)
            manifest_path = facts_dir / f"manifest-{date_str}.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.setdefault("stats", {})
                manifest["stats"]["cards_a"] = sum(1 for r in kept if r["grade"] == "A")
                manifest["stats"]["cards_b"] = 0
                manifest["stats"]["cards_new"] = sum(
                    1 for r in kept if r.get("is_new") and r["grade"] == "A"
                )
                manifest["pruned_b_at"] = date.today().isoformat()
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
    print(f"合计：{total_before} → {total_after} 条（dry-run={not args.apply}）")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:  # noqa: C901, PLR0912, PLR0915
    """独立校验：字段/档位/事实 id/证据可回溯/重复。"""
    data_dir = Path(args.data_dir)
    facts_dir = data_dir / "facts"
    errors: list[str] = []
    checked = 0
    allowed = {"company", "person", "org", "product", "tech", "other"}
    for path in sorted(facts_dir.glob("facts-*.jsonl")):
        date_str = path.stem.replace("facts-", "")
        ocean = data_dir / "ocean" / date_str
        items: dict[str, dict] = {}
        if (ocean / "items.json").is_file():
            items = {
                str(row.get("item_id")): row
                for row in json.loads((ocean / "items.json").read_text(encoding="utf-8"))
            }
        seen: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            checked += 1
            row = json.loads(line)
            fact_id = str(row.get("fact_id") or "")
            if not fact_id:
                errors.append(f"{date_str}: 缺 fact_id")
                continue
            if fact_id in seen:
                errors.append(f"{date_str}: fact_id 重复 {fact_id}")
            seen.add(fact_id)
            if row.get("date") != date_str:
                errors.append(f"{date_str}: 日期字段不一致 {fact_id}")
            if row.get("grade") not in ("A", "B"):
                errors.append(f"{date_str}: 档位非法 {fact_id}")
            if row.get("relation") not in RELATIONS:
                errors.append(f"{date_str}: 关系非法 {fact_id}")
            subject = row.get("subject") or {}
            if not str(subject.get("name") or ""):
                errors.append(f"{date_str}: 主体为空 {fact_id}")
            if subject.get("kind") not in allowed:
                errors.append(f"{date_str}: 主体类型非法 {fact_id}")
            evidence = row.get("evidence") or {}
            sentence = str(evidence.get("sentence") or "")
            item_id = str(evidence.get("item_id") or "")
            item = items.get(item_id)
            if item is None:
                errors.append(f"{date_str}: 证据条目缺失 {fact_id}")
            else:
                haystack = " ".join(
                    str(item.get(key) or "") for key in ("title", "summary", "full_text")
                )
                if sentence and sentence not in haystack:
                    errors.append(f"{date_str}: 证据句回溯失败 {fact_id}")
            if _expected_fact_id(row) != fact_id:
                errors.append(f"{date_str}: fact_id 不可重算 {fact_id}")
            if not (row.get("sources") or []):
                errors.append(f"{date_str}: sources 为空 {fact_id}")
            if row.get("grade") == "A":
                if not str(subject.get("id") or ""):
                    errors.append(f"{date_str}: A 档主体无实体 id {fact_id}")
                if "dep_confirmed" not in (row.get("grade_reasons") or []):
                    errors.append(f"{date_str}: A 档缺 dep_confirmed {fact_id}")
    print(f"校验事实 {checked} 条，错误 {len(errors)} 处")
    for item in errors[:30]:
        print(" -", item)
    return 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="事实层 CLI")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config-dir", default="config")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="单日构建")
    build.add_argument("--date", required=True)
    build.set_defaults(func=cmd_build)

    backfill = sub.add_parser("backfill", help="回填最近 N 天")
    backfill.add_argument("--days", type=int, default=8)
    backfill.add_argument("--force", action="store_true")
    backfill.set_defaults(func=cmd_backfill)

    report = sub.add_parser("report", help="覆盖周报")
    report.add_argument("--days", type=int, default=28)
    report.add_argument("--out-md", default="")
    report.set_defaults(func=cmd_report)

    ev = sub.add_parser("eval", help="金标评测")
    ev.add_argument("--gold", default="data/eval/kg_facts_gold.jsonl")
    ev.add_argument("--min-precision", type=float, default=0.85)
    ev.add_argument("--diagnose", action="store_true")
    ev.set_defaults(func=cmd_eval)

    prune = sub.add_parser("prune", help="B 档轮转（A 永久保留）")
    prune.add_argument("--keep-b-days", type=int, default=90)
    prune.add_argument("--apply", action="store_true")
    prune.set_defaults(func=cmd_prune)

    validate = sub.add_parser("validate", help="独立校验事实")
    validate.set_defaults(func=cmd_validate)

    query = sub.add_parser("query", help="查事实")
    query.add_argument("--subject", default="")
    query.add_argument("--relation", default="")
    query.add_argument("--since", default="")
    query.add_argument("--limit", type=int, default=50)
    query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
