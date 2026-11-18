#!/usr/bin/env python3
"""在 UD 中文树库上评测解析器的三元组质量（不引入事实层规则）。

用法：
    git clone --depth 1 https://github.com/UniversalDependencies/UD_Chinese-GSD /tmp/UD_Chinese-GSD
    python scripts/kg_ud_eval.py --ud-dir /tmp/UD_Chinese-GSD --split dev+test

金标三元组由 UD 标注导出：VERB 节点 + nsubj 子节点 + obj 子节点。
"""

# ruff: noqa: B007, E741, PERF102, PERF401

import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import torch

_orig = torch.load


def _load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig(*a, **k)


torch.load = _load
import argparse  # noqa: E402

import hanlp  # noqa: E402
from ltp import LTP  # noqa: E402
from supar import Parser  # noqa: E402

from ai_news_podcast.business.kg.dep import DepParser  # noqa: E402
from ai_news_podcast.business.kg.lexicon import KGlexicon  # noqa: E402


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ud-dir", default="/tmp/UD_Chinese-GSD")
    ap.add_argument("--split", default="dev+test")
    ap.add_argument("--tools", default="spacy,ltp,hanlp,supar")
    return ap.parse_args()


def load_conllu(path):
    sents = []
    toks = []
    text = ""
    with Path(path).open(encoding="utf-8") as fh:
        lines = [raw.rstrip("\n") for raw in fh]
    for line in lines:
        if line.startswith("# text = "):
            text = line[9:]
        elif line.startswith("#"):
            continue
        elif not line.strip():
            if toks:
                sents.append({"text": text, "tokens": toks})
            toks = []
            text = ""
        else:
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            toks.append(
                {
                    "id": int(cols[0]),
                    "form": cols[1],
                    "upos": cols[3],
                    "head": int(cols[6]),
                    "deprel": cols[7],
                }
            )
    if toks:
        sents.append({"text": text, "tokens": toks})
    return [s for s in sents if s["text"]]


def gold_spans(sent):
    """给每个 token 算字符区间（按 text 顺序查找，处理重复用游标）。"""
    spans = {}
    cur = 0
    for t in sent["tokens"]:
        i = sent["text"].find(t["form"], cur)
        if i < 0:
            i = sent["text"].find(t["form"])
        spans[t["id"]] = (i, i + len(t["form"]))
        cur = i + len(t["form"])
    return spans


def gold_triples(sent, spans):
    triples = []
    for t in sent["tokens"]:
        if t["upos"] != "VERB":
            continue
        subs = [
            c for c in sent["tokens"] if c["head"] == t["id"] and c["deprel"].startswith("nsubj")
        ]
        objs = [c for c in sent["tokens"] if c["head"] == t["id"] and c["deprel"] == "obj"]
        for s in subs:
            for o in objs:
                triples.append((spans[s["id"]], spans[t["id"]], spans[o["id"]]))
    return triples


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def cover(gt, p):
    inter = overlap(gt, p)
    return inter / max(1, gt[1] - gt[0])


def align(gt_span, ptoks):
    best = None
    bestr = 0
    for t in ptoks:
        r = cover(gt_span, (t.idx, t.idx + len(t.text)))
        if r > bestr:
            bestr = r
            best = t
    return best if bestr >= 0.5 else None


def eval_tool(name, sents, parse_fn, subj_labels, obj_labels):
    t0 = time.time()
    total = found = 0
    tok_hit = tok_gold = 0
    for sent in sents:
        spans = gold_spans(sent)
        triples = gold_triples(sent, spans)
        try:
            ptoks = parse_fn(sent["text"])
        except Exception:
            continue
        if not ptoks:
            continue
        tok_gold += len(spans)
        for gid, gspan in spans.items():
            if align(gspan, ptoks):
                tok_hit += 1
        for ss, vs, os_ in triples:
            total += 1
            sp = align(ss, ptoks)
            vp = align(vs, ptoks)
            op = align(os_, ptoks)
            if not (sp and vp and op):
                continue
            s_ok = sp.dep in subj_labels and sp.head == vp.i
            o_ok = op.dep in obj_labels and op.head == vp.i
            if s_ok and o_ok:
                found += 1
    sec = time.time() - t0
    print(
        f"{name}|金标三元组={total}|命中={found}|召回={found / max(1, total):.3f}|token对齐={tok_hit}/{tok_gold}|耗时={sec:.1f}s",
        flush=True,
    )


LEX = KGlexicon.load()
SP = DepParser.load(LEX)
SPC = DepParser.load(LEX, custom_tokenizer=True)
LTP_E = LTP("LTP/small")
HAN = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
SUP = Parser.load("biaffine-dep-zh")
args = _args()
ROOT = args.ud_dir
files = []
for part in args.split.split("+"):
    files.append(f"{ROOT}/zh_gsd-ud-{part}.conllu")
sents = []
for f in files:
    sents.extend(load_conllu(f))
print(
    "UD 句子数",
    len(sents),
    "金标三元组",
    sum(len(gold_triples(s, gold_spans(s))) for s in sents),
    flush=True,
)


def sp_parse(s):
    p = SP.analyze(s, "zh")
    toks = []
    for t in p.tokens:
        toks.append(
            type("T", (), {"text": t.text, "idx": t.idx, "dep": t.dep, "head": t.head, "i": t.i})()
        )
    return toks


def spc_parse(s):
    p = SPC.analyze(s, "zh")
    toks = []
    for t in p.tokens:
        toks.append(
            type("T", (), {"text": t.text, "idx": t.idx, "dep": t.dep, "head": t.head, "i": t.i})()
        )
    return toks


def ltp_parse(s):
    out = LTP_E.pipeline([s], tasks=["cws", "pos", "dep"])
    words = [w.strip() for w in out.cws[0] if w.strip()]
    heads = out.dep[0]["head"]
    labels = out.dep[0]["label"]
    toks = []
    cur = 0
    for i, w in enumerate(words):
        idx = s.find(w, cur)
        idx = idx if idx >= 0 else cur
        cur = idx + len(w)
        toks.append(
            type(
                "T",
                (),
                {
                    "text": w,
                    "idx": idx,
                    "dep": labels[i],
                    "head": (heads[i] - 1 if heads[i] > 0 else i),
                    "i": i,
                },
            )()
        )
    return toks


def han_parse(s):
    out = HAN(s, tasks=["tok/fine", "pos/ctb", "dep"])
    words = out["tok/fine"]
    deps = out["dep"]
    toks = []
    cur = 0
    for i, w in enumerate(words):
        idx = s.find(w, cur)
        idx = idx if idx >= 0 else cur
        cur = idx + len(w)
        h, l = deps[i]
        toks.append(
            type(
                "T", (), {"text": w, "idx": idx, "dep": l, "head": (h - 1 if h > 0 else i), "i": i}
            )()
        )
    return toks


def sup_parse(s):
    words = [t.text for t in SP.nlps["zh"].tokenizer(s) if t.text.strip()]
    sent = SUP.predict([words])[0]
    ws = list(sent.values[1])
    heads = [int(h) - 1 if int(h) > 0 else i for i, h in enumerate(sent.values[6])]
    labels = list(sent.values[7])
    toks = []
    cur = 0
    for i, w in enumerate(ws):
        idx = s.find(w, cur)
        idx = idx if idx >= 0 else cur
        cur = idx + len(w)
        toks.append(
            type("T", (), {"text": w, "idx": idx, "dep": labels[i], "head": heads[i], "i": i})()
        )
    return toks


tools = set(args.tools.split(","))
if "spacy" in tools:
    eval_tool("spaCy", sents, sp_parse, {"nsubj", "nsubjpass"}, {"dobj", "obj"})
if "custom" in tools:
    eval_tool("spaCy+custom", sents, spc_parse, {"nsubj", "nsubjpass"}, {"dobj", "obj"})
if "ltp" in tools:
    eval_tool("LTP", sents, ltp_parse, {"SBV"}, {"VOB", "FOB", "IOB"})
if "hanlp" in tools:
    eval_tool("HanLP", sents, han_parse, {"nsubj"}, {"dobj"})
if "supar" in tools:
    eval_tool("SuPar", sents, sup_parse, {"nsubj"}, {"dobj", "obj"})
