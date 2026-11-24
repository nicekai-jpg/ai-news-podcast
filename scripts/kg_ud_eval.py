#!/usr/bin/env python3
"""在 UD 简体中文树库上评测解析器三元组质量（不引入事实层规则）。

用法：
    git clone --depth 1 https://github.com/UniversalDependencies/UD_Chinese-GSDSimp.git /tmp/UD_Chinese-GSDSimp
    python scripts/kg_ud_eval.py

输出：三元组召回/精确率/F1 + UAS；金标由 UD 的 VERB+nsubj+obj 导出。
"""

# ruff: noqa: C901, E402, E741, N806, PERF401, PLR0912, PLR0915

import sys
import time

sys.path.insert(0, "src")
import torch

_orig = torch.load


def _load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig(*a, **k)


torch.load = _load
import argparse
from collections import Counter
from pathlib import Path

import hanlp
import spacy
from ltp import LTP
from supar import Parser

_ap = argparse.ArgumentParser()
_ap.add_argument("--ud-dir", default="/tmp/UD_Chinese-GSDSimp")
_ap.add_argument("--prefix", default="zh_gsdsimp")
ROOT = _ap.parse_args().ud_dir
PREFIX = _ap.parse_args().prefix


def load_conllu(path):
    s = []
    toks = []
    text = ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("# text = "):
            text = line[9:]
        elif line.startswith("#"):
            continue
        elif not line.strip():
            if toks:
                s.append({"text": text, "tokens": toks})
            toks = []
            text = ""
        else:
            c = line.split("\t")
            if "-" in c[0] or "." in c[0]:
                continue
            toks.append(
                {"id": int(c[0]), "form": c[1], "upos": c[3], "head": int(c[6]), "deprel": c[7]}
            )
    return [x for x in s if x["text"] and x["tokens"]]


_files = [f"{ROOT}/{PREFIX}-ud-{part}.conllu" for part in ("dev", "test")]
_files = [path for path in _files if Path(path).is_file()]
S = []
for _path in _files:
    S.extend(load_conllu(_path))
dep = Counter(t["deprel"] for s in S for t in s["tokens"])
print("句数", len(S), "deprel:", dep.most_common(6))


def spans(sent):
    out = {}
    cur = 0
    for t in sent["tokens"]:
        i = sent["text"].find(t["form"], cur)
        if i < 0:
            i = sent["text"].find(t["form"])
        out[t["id"]] = (i, i + len(t["form"]))
        cur = i + len(t["form"])
    return out


def ov(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def cover(g, p):
    return ov(g, p) / max(1, g[1] - g[0])


def gold_triples(sent, sp):
    out = []
    for t in sent["tokens"]:
        if t["upos"] != "VERB":
            continue
        subs = [
            c for c in sent["tokens"] if c["head"] == t["id"] and c["deprel"].startswith("nsubj")
        ]
        objs = [c for c in sent["tokens"] if c["head"] == t["id"] and c["deprel"] == "obj"]
        for s_ in subs:
            for o in objs:
                out.append((sp[s_["id"]], sp[t["id"]], sp[o["id"]]))
    return out


def evaluate(name, parse_fn, subj_labels, obj_labels):
    t0 = time.time()
    gt_n = gf = pt_n = pm = uas_t = uas_h = 0
    for sent in S:
        sp = spans(sent)
        gt = gold_triples(sent, sp)
        gt_n += len(gt)
        try:
            pt = parse_fn(sent["text"])
        except Exception:
            pt = []
        if not pt:
            continue
        # parser token -> gold token（按字符重叠最大）
        p2g = {}
        for p in pt:
            best = None
            bestr = 0
            for t in sent["tokens"]:
                r = cover(sp[t["id"]], (p["idx"], p["idx"] + len(p["text"])))
                if r > bestr:
                    bestr = r
                    best = t["id"]
            if bestr >= 0.5:
                p2g[p["i"]] = best
        by = {t["id"]: t for t in sent["tokens"]}
        for p in pt:
            if p["i"] not in p2g:
                continue
            g = by[p2g[p["i"]]]
            if g["head"] not in p2g.values():
                continue
            # 找对齐到 gold head 的 parser 头 token
            heads = [q for q in pt if p2g.get(q["i"]) == g["head"]]
            if not heads:
                continue
            uas_t += 1
            if any(q["i"] == p["head"] for q in heads):
                uas_h += 1
        kids = {}
        for p in pt:
            kids.setdefault(p["head"], []).append(p)
        for p in pt:
            subs = [c for c in kids.get(p["i"], []) if c["dep"] in subj_labels]
            objs = [c for c in kids.get(p["i"], []) if c["dep"] in obj_labels]
            for s_ in subs:
                for o in objs:
                    pt_n += 1
                    ps = (s_["idx"], s_["idx"] + len(s_["text"]))
                    pv = (p["idx"], p["idx"] + len(p["text"]))
                    po = (o["idx"], o["idx"] + len(o["text"]))
                    if any(
                        cover(gs, ps) >= 0.5 and cover(gv, pv) >= 0.5 and cover(go, po) >= 0.5
                        for gs, gv, go in gt
                    ):
                        pm += 1
        for gs, gv, go in gt:
            hit = False
            for p in pt:
                if cover(gv, (p["idx"], p["idx"] + len(p["text"]))) < 0.5:
                    continue
                for s_ in kids.get(p["i"], []):
                    if (
                        s_["dep"] in subj_labels
                        and cover(gs, (s_["idx"], s_["idx"] + len(s_["text"]))) >= 0.5
                    ):
                        for o in kids.get(p["i"], []):
                            if (
                                o["dep"] in obj_labels
                                and cover(go, (o["idx"], o["idx"] + len(o["text"]))) >= 0.5
                            ):
                                hit = True
            if hit:
                gf += 1
    rec = gf / max(1, gt_n)
    prec = pm / max(1, pt_n)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    print(
        f"{name}|gold={gt_n}|R={rec:.3f}|P={prec:.3f}|F1={f1:.3f}|UAS={uas_h / max(1, uas_t):.3f}|parserTriples={pt_n}|{time.time() - t0:.1f}s",
        flush=True,
    )


raw = spacy.load("zh_core_web_sm")
raw.select_pipes(enable=["tok2vec", "tagger", "attribute_ruler", "parser", "ner"])
LTP_E = LTP("LTP/small")
HAN = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
SUP = Parser.load("biaffine-dep-zh")


def sp_parse(s):
    return [
        {"text": x.text, "idx": x.idx, "dep": x.dep_, "head": x.head.i, "i": x.i} for x in raw(s)
    ]


def ltp_parse(s):
    out = LTP_E.pipeline([s], tasks=["cws", "pos", "dep"])
    words = [w.strip() for w in out.cws[0] if w.strip()]
    H = out.dep[0]["head"]
    L = out.dep[0]["label"]
    res = []
    cur = 0
    for i, w in enumerate(words):
        idx = s.find(w, cur)
        idx = idx if idx >= 0 else cur
        cur = idx + len(w)
        res.append(
            {"text": w, "idx": idx, "dep": L[i], "head": (H[i] - 1 if H[i] > 0 else i), "i": i}
        )
    return res


def han_parse(s):
    out = HAN(s, tasks=["tok/fine", "dep"])
    words = out["tok/fine"]
    deps = out["dep"]
    res = []
    cur = 0
    for i, w in enumerate(words):
        idx = s.find(w, cur)
        idx = idx if idx >= 0 else cur
        cur = idx + len(w)
        h, l = deps[i]
        res.append({"text": w, "idx": idx, "dep": l, "head": (h - 1 if h > 0 else i), "i": i})
    return res


def sup_parse(s):
    words = [t.text for t in raw.tokenizer(s) if t.text.strip()]
    sent = SUP.predict([words])[0]
    ws = list(sent.values[1])
    H = [int(h) - 1 if int(h) > 0 else i for i, h in enumerate(sent.values[6])]
    L = list(sent.values[7])
    res = []
    cur = 0
    for i, w in enumerate(ws):
        idx = s.find(w, cur)
        idx = idx if idx >= 0 else cur
        cur = idx + len(w)
        res.append({"text": w, "idx": idx, "dep": L[i], "head": H[i], "i": i})
    return res


evaluate("spaCy", sp_parse, {"nsubj", "nsubjpass"}, {"dobj", "obj", "attr"})
evaluate("LTP", ltp_parse, {"SBV"}, {"VOB", "FOB", "IOB"})
evaluate("HanLP", han_parse, {"nsubj"}, {"dobj"})
evaluate("SuPar", sup_parse, {"nsubj"}, {"dobj", "obj"})
