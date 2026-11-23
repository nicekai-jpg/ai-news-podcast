"""金标清洗：把 LLM 起草的 draft 按可机器判定的规则收紧，再交人工复核。

只做「明显错误」的剔除，不做语义改写；不确定的一律保留并标记，交人工决定。
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.triple_bench.gold import validate_gold  # noqa: E402
from scripts.triple_bench.standard import (  # noqa: E402
    Doc,
    GoldItem,
    Triple,
    normalize_relation,
    normalize_text,
)

# 泛指主语/宾语：不构成可入库事实。
GENERIC = {
    "双方",
    "多方",
    "官方",
    "该公司",
    "该团队",
    "研究团队",
    "研究人员",
    "开发者",
    "用户",
    "玩家",
    "市场",
    "行业",
    "人们",
    "the premium tier",
    "experimental ai dj",
}

# 这些关系要求宾语是数值或可核验对象，宾语是泛指时基本是误判。
VALUE_RELATIONS = {"pricing", "benchmark", "funding"}

# 表对比/并列的连词，出现在证据句里说明该句不是单一事实。
CONTRAST = ("相比之下", "与此同时", "此外", "另外", "而", "但是", "不过", "相较于")

MAX_ENDPOINT = 24


def _too_long(value: str) -> bool:
    return len(value) > MAX_ENDPOINT


def _is_generic(value: str) -> bool:
    return normalize_text(value) in {normalize_text(item) for item in GENERIC}


def _has_role_list(value: str) -> bool:
    """职务串（含顿号/逗号 + 职衔）不适合当宾语。"""
    return bool(re.search(r"[、,，]", value)) and len(value) > 12


def clean_triples(item: GoldItem, doc: Doc) -> tuple[list[Triple], Counter[str]]:
    dropped: Counter[str] = Counter()
    kept: list[Triple] = []
    for triple in item.triples:
        subject, obj = triple.subject.strip(), triple.object.strip()
        if _is_generic(subject) or _is_generic(obj):
            dropped["generic_endpoint"] += 1
            continue
        if _too_long(subject) or _too_long(obj):
            dropped["endpoint_too_long"] += 1
            continue
        if _has_role_list(obj):
            dropped["role_list_object"] += 1
            continue
        if triple.relation == "other":
            dropped["other_relation"] += 1
            continue
        if triple.relation in VALUE_RELATIONS and not re.search(r"\d", obj):
            dropped["value_relation_without_number"] += 1
            continue
        evidence = triple.evidence.strip()
        if any(word in evidence for word in CONTRAST) and len(evidence) > 40:
            dropped["contrast_sentence"] += 1
            continue
        kept.append(
            Triple(
                subject=subject,
                relation=normalize_relation(triple.relation),
                object=obj,
                confidence=triple.confidence,
                evidence=evidence,
                object_kind=triple.object_kind,
            )
        )
    return kept, dropped


def main(argv: list[str]) -> int:
    from argparse import ArgumentParser

    parser = ArgumentParser(description="清洗 draft 金标")
    parser.add_argument("--docs", default=str(ROOT / "data/eval/triple_bench/docs.json"))
    parser.add_argument("--gold", default=str(ROOT / "data/eval/triple_bench/gold_v3.json"))
    parser.add_argument("--out", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    docs = {row["item_id"]: Doc.from_json(row) for row in json.loads(Path(args.docs).read_text())}
    payload = json.loads(Path(args.gold).read_text())
    items = [GoldItem.from_json(row) for row in payload["items"]]

    total_dropped: Counter[str] = Counter()
    out: list[GoldItem] = []
    touched = 0
    for item in items:
        doc = docs.get(item.item_id)
        if doc is None or item.status != "draft":
            out.append(item)
            continue
        kept, dropped = clean_triples(item, doc)
        if dropped:
            touched += 1
            total_dropped.update(dropped)
        out.append(
            GoldItem(
                item_id=item.item_id,
                title=item.title,
                lang=item.lang,
                split=item.split,
                status="draft",
                triples=kept,
            )
        )

    issues = validate_gold(out, docs)
    print(f"items={len(out)} drafts={sum(1 for i in out if i.status == 'draft')}")
    print(f"清洗涉及 {touched} 篇；剔除统计 {dict(total_dropped)}")
    print(f"剩余校验问题 {len(issues)}: {dict(Counter(i.reason for i in issues))}")
    print(f"三元组总数 {sum(len(i.triples) for i in out)}")

    if not args.dry_run:
        target = Path(args.out or args.gold)
        payload["items"] = [item.to_json() for item in out]
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"written -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
