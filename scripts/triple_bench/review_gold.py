"""金标自动审核：只做可客观判定的剔除与证据修正，不做语义改写。

判定分层（宁可漏放，不可误杀）：
1. 剔除「事实不成立」——端点不在原文、泛指、职务串、端点过长、关系为 other。
2. 剔除「无法证实」——既找不到同句证据，也找不到跨句共现证据。
3. 只修正证据——事实成立但 evidence 指错句子时，重新定位最小证据句。
4. 其余一律通过，交人工复核。

所有剔除都会打印原因和原文，便于逐条复核。
"""

from __future__ import annotations

import json
import re
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.triple_bench.standard import (  # noqa: E402
    Doc,
    GoldItem,
    Triple,
    normalize_relation,
    normalize_text,
)

# 泛指成分：不能作为事实主体。
GENERIC = {
    "双方",
    "多方",
    "官方",
    "该公司",
    "该系统",
    "该团队",
    "此类",
    "这类",
    "研究团队",
    "研究人员",
    "开发者",
    "用户",
    "玩家",
    "市场",
    "行业",
    "人们",
    "她",
    "他",
    "它",
    "团队",
    "公司",
    "平台",
}

# 宾语应含数值的关系。funding 的宾语常是投资方机构，不强制数值。
VALUE_RELATIONS = {"pricing", "benchmark", "funding"}
NUMERIC_RELATIONS = {"pricing", "benchmark"}
CONTRAST = ("相比之下", "与此同时", "此外", "另外", "但是", "不过", "相较于", "反之")

# 允许较长的宾语：产品线、系列名天然长（Radeon HD 2000、HD 3000系列显卡）。
MAX_ENDPOINT = 32


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；\n])", text or "")
    return [part.strip() for part in parts if part.strip()]


def locate_evidence(sentences: list[str], subject: str, obj: str) -> str:
    """重新定位最小证据句：优先两端共现，其次主语共现。"""
    sub, ob = normalize_text(subject), normalize_text(obj)
    both = [
        sentence
        for sentence in sentences
        if sub in normalize_text(sentence) and ob in normalize_text(sentence)
    ]
    if both:
        return min(both, key=len)
    one = [sentence for sentence in sentences if sub in normalize_text(sentence)]
    if one:
        return min(one, key=len)
    return ""


def review_triple(  # noqa: PLR0911, PLR0912
    triple: Triple, doc: Doc, sentences: list[str]
) -> tuple[Triple | None, str]:
    """返回 (清洗后的三元组 或 None, 剔除原因)。"""
    subject, obj = triple.subject.strip(), triple.object.strip()
    body = normalize_text(f"{doc.title}。{doc.text}")
    sub, ob = normalize_text(subject), normalize_text(obj)

    if not sub or not ob:
        return None, "empty_endpoint"
    if sub == ob:
        return None, "same_endpoint"
    if triple.relation == "other":
        return None, "other_relation"
    if sub in GENERIC or ob in GENERIC:
        return None, "generic_endpoint"
    if len(subject) > MAX_ENDPOINT or len(obj) > MAX_ENDPOINT:
        return None, "endpoint_too_long"
    # 职务串只对人事关系有害；产品系列名里的顿号是正常的。
    if triple.relation == "personnel" and re.search(r"[、,，]", obj) and len(obj) > 12:
        return None, "role_list_object"
    if triple.relation in NUMERIC_RELATIONS and not re.search(r"\d", obj):
        return None, "value_relation_without_number"

    # 端点必须能在原文找到（归一化后包含匹配）。做不到说明是改写过的规范名。
    if sub not in body or ob not in body:
        return None, "endpoint_not_in_text"

    evidence = triple.evidence.strip()
    ev_norm = normalize_text(evidence)
    same_sentence = any(
        sub in normalize_text(sentence) and ob in normalize_text(sentence) for sentence in sentences
    )
    ok = bool(evidence) and ev_norm in body and sub in ev_norm and ob in ev_norm

    if not ok and same_sentence:
        evidence = locate_evidence(sentences, subject, obj)
    elif not ok and not same_sentence:
        # 跨句事实（"该工作""团队"指代前文）：两端都在原文即保留，标记供人工确认。
        if not any(sub in normalize_text(sentence) for sentence in sentences):
            return None, "subject_not_verifiable"
        if not any(ob in normalize_text(sentence) for sentence in sentences):
            return None, "object_not_verifiable"
        anchor = locate_evidence(sentences, subject, "") or triple.evidence
        return (
            Triple(
                subject=subject,
                relation=normalize_relation(triple.relation),
                object=obj,
                confidence=triple.confidence,
                evidence=anchor,
                object_kind=triple.object_kind,
                meta={**triple.meta, "review": "cross_sentence"},
            ),
            "",
        )

    if evidence and any(word in evidence for word in CONTRAST) and len(evidence) > 40:
        return None, "contrast_sentence"

    return (
        Triple(
            subject=subject,
            relation=normalize_relation(triple.relation),
            object=obj,
            confidence=triple.confidence,
            evidence=evidence,
            object_kind=triple.object_kind,
            meta={**triple.meta, "review": "auto"},
        ),
        "",
    )


def main(argv: list[str]) -> int:
    parser = ArgumentParser(description="金标自动审核")
    parser.add_argument("--docs", default=str(ROOT / "data/eval/triple_bench/docs.json"))
    parser.add_argument("--gold", default=str(ROOT / "data/eval/triple_bench/gold_v3.json"))
    parser.add_argument("--out", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    docs = {row["item_id"]: Doc.from_json(row) for row in json.loads(Path(args.docs).read_text())}
    payload = json.loads(Path(args.gold).read_text())
    items = [GoldItem.from_json(row) for row in payload["items"]]

    dropped: Counter[str] = Counter()
    reevidenced = 0
    out: list[GoldItem] = []
    touched_docs = 0
    for item in items:
        doc = docs.get(item.item_id)
        if doc is None:
            dropped["missing_doc"] += 1
            continue
        sentences = [doc.title, *split_sentences(doc.text)]
        sentences = [s for s in sentences if s]
        kept: list[Triple] = []
        before = len(item.triples)
        for triple in item.triples:
            cleaned, reason = review_triple(triple, doc, sentences)
            if cleaned is None:
                dropped[reason] += 1
                if args.verbose:
                    print(
                        f"[drop:{reason}] {item.item_id[:12]} {triple.subject} | "
                        f"{triple.relation} | {triple.object}"
                    )
                continue
            if cleaned.evidence != triple.evidence and cleaned.evidence:
                reevidenced += 1
            kept.append(cleaned)
        if len(kept) != before:
            touched_docs += 1
        out.append(
            GoldItem(
                item_id=item.item_id,
                title=item.title,
                lang=item.lang,
                split=item.split,
                status="reviewed",
                triples=kept,
            )
        )

    total = sum(len(i.triples) for i in out)
    print(f"文档 {len(out)}（涉及修改 {touched_docs}）三元组 {total}")
    print(f"剔除 {sum(dropped.values())} 条：{dict(dropped)}")
    print(f"证据重定位 {reevidenced} 条")
    empty = [i.item_id for i in out if not i.triples]
    print(f"无三元组文档 {len(empty)}")

    if not args.dry_run:
        target = Path(args.out or args.gold)
        payload["items"] = [item.to_json() for item in out]
        payload["frozen"] = False
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"written -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
