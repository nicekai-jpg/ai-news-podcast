#!/usr/bin/env python3
"""金标分词喂入（oracle tokenization）评测：排除分词干扰，只比句法结构与三元组。

用法：
    git clone --depth 1 https://github.com/UniversalDependencies/UD_Chinese-GSDSimp.git /tmp/UD_Chinese-GSDSimp
    python scripts/kg_ud_oracle_eval.py
"""

# ruff: noqa: B007, B905, C901, E402, E741, N806, PERF401, PLR0912

import sys
import time

sys.path.insert(0, "src")
import torch

_orig = torch.load


def _load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig(*a, **k)


torch.load = _load
from pathlib import Path

import hanlp
import spacy
from ltp import LTP
from spacy.tokens import Doc
from supar import Parser

ROOT = "/tmp/UD_Chinese-GSDSimp"


def load(path):
    out = []
    toks = []
    text = ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("# text = "):
            text = line[9:]
        elif line.startswith("#"):
            continue
        elif not line.strip():
            if toks:
                out.append({"text": text, "tokens": toks})
            toks = []
            text = ""
        else:
            c = line.split("\t")
            if "-" in c[0] or "." in c[0]:
                continue
            toks.append(
                {"id": int(c[0]), "form": c[1], "upos": c[3], "head": int(c[6]), "deprel": c[7]}
            )
    return [x for x in out if x["text"] and x["tokens"]]


S = load(f"{ROOT}/zh_gsdsimp-ud-dev.conllu") + load(f"{ROOT}/zh_gsdsimp-ud-test.conllu")
print("句数", len(S), flush=True)

UD = {
    "nsubj": "nsubj",
    "nsubjpass": "nsubj",
    "nsubj:pass": "nsubj",
    "dobj": "obj",
    "obj": "obj",
    "root": "root",
    "ROOT": "root",
    "SBV": "nsubj",
    "VOB": "obj",
    "FOB": "obj",
    "IOB": "obj",
    "HED": "root",
    "nn": "compound",
    "compound:nn": "compound",
    "amod": "amod",
    "ATT": "amod",
    "advmod": "advmod",
    "ADV": "advmod",
    "case": "case",
    "DE": "case",
    "DI": "case",
    "mark": "mark",
    "aux": "aux",
    "AUX": "aux",
    "cc": "cc",
    "CC": "cc",
    "conj": "conj",
    "COO": "conj",
    "punct": "punct",
    "PUN": "punct",
    "WP": "punct",
    "nummod": "nummod",
    "det": "det",
    "nmod": "nmod",
    "nmod:prep": "nmod",
    "nmod:assmod": "nmod",
    "nmod:tmod": "nmod",
    "assmod": "nmod",
    "assm": "nmod",
    "prep": "case",
    "pobj": "nmod",
    "acl": "acl",
    "acl:relcl": "acl",
    "rcmod": "acl",
    "advcl": "advcl",
    "LAD": "advcl",
    "WADV": "advcl",
    "ccomp": "ccomp",
    "xcomp": "xcomp",
    "parataxis": "parataxis",
    "dep": "dep",
    "appos": "appos",
    "flat": "flat",
    "compound": "compound",
    "comp": "compound",
    "cop": "cop",
    "CMP": "ccomp",
    "neg": "neg",
    "discourse": "discourse",
    "vocative": "vocative",
    "list": "list",
    "orphan": "orphan",
    "RAD": "acl",
    "POB": "nmod",
    "BA": "case",
    "BEI": "aux:pass",
    "LB": "case",
    "LC": "nmod",
    "M": "nummod",
    "CD": "nummod",
    "OD": "nummod",
    "NN": "compound",
    "VA": "aux",
    "VC": "aux",
    "VE": "aux",
    "VV": "root",
    "f": "compound",
    "s": "nmod",
    "t": "nmod",
    "q": "nummod",
    "j": "compound",
    "n": "compound",
    "m": "nummod",
    "r": "nsubj",
    "p": "case",
    "c": "cc",
    "d": "advmod",
    "a": "amod",
    "b": "amod",
}


def to_ud(label):
    return UD.get(str(label), "dep")


def gold_triples(sent):
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
                out.append((s_["id"] - 1, t["id"] - 1, o["id"] - 1))
    return out


def score(name, parse_fn, subj_labels, obj_labels):
    t0 = time.time()
    ut = uh = lt = lh = pt = pm = 0
    mapped = totlab = 0
    gt_all = 0
    gf = 0
    for sent in S:
        words = [t["form"] for t in sent["tokens"]]
        try:
            pred = parse_fn(words)
        except Exception:
            continue
        if not pred or len(pred) != len(words):
            continue
        heads = [p["head"] for p in pred]
        labels = [p["dep"] for p in pred]
        gold = sent["tokens"]
        for i, (g, p) in enumerate(zip(gold, pred)):
            ut += 1
            if heads[i] == g["head"] - 1:
                uh += 1
            totlab += 1
            if UD.get(str(labels[i])) is not None:
                mapped += 1
            lt += 1
            if heads[i] == g["head"] - 1 and to_ud(labels[i]) == g["deprel"]:
                lh += 1
        gt = gold_triples(sent)
        gt_all += len(gt)
        kids = {}
        for i, h in enumerate(heads):
            kids.setdefault(h, []).append(i)
        for i in range(len(words)):
            subs = [c for c in kids.get(i, []) if labels[c] in subj_labels]
            objs = [c for c in kids.get(i, []) if labels[c] in obj_labels]
            for s_ in subs:
                for o in objs:
                    pt += 1
                    if (s_ - 1 + 1 - 1, i, o) and any(
                        s_ == gs and i == gv and o == go for gs, gv, go in gt
                    ):
                        pm += 1
        for gs, gv, go in gt:
            hit = False
            for s_ in kids.get(gv, []):
                if labels[s_] in subj_labels and s_ == gs:
                    for o in kids.get(gv, []):
                        if labels[o] in obj_labels and o == go:
                            hit = True
            if hit:
                gf += 1
    rec = gf / max(1, gt_all)
    prec = pm / max(1, pt)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    print(
        f"{name}|sent={len(S)}|UAS={uh / max(1, ut):.3f}|LAS={lh / max(1, lt):.3f}|label覆盖={mapped / max(1, totlab):.2f}|triple R={rec:.3f} P={prec:.3f} F1={f1:.3f}|{time.time() - t0:.1f}s",
        flush=True,
    )


raw = spacy.load("zh_core_web_sm")
nlp = spacy.load("zh_core_web_sm")
nlp.select_pipes(enable=["tok2vec", "tagger", "attribute_ruler", "parser"])
LTP_E = LTP("LTP/small")
SUP = Parser.load("biaffine-dep-zh")
HAN = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)


def sp_parse(words):
    doc = Doc(nlp.vocab, words=words)
    for _, proc in nlp.pipeline:
        doc = proc(doc)
    return [{"dep": t.dep_, "head": t.head.i} for t in doc]


def ltp_parse(words):
    out = LTP_E.pipeline([words], tasks=["pos", "dep"])
    H = out.dep[0]["head"]
    L = out.dep[0]["label"]
    return [{"dep": L[i], "head": (H[i] - 1 if H[i] > 0 else i)} for i in range(len(words))]


def sup_parse(words):
    sent = SUP.predict([words])[0]
    ws = list(sent.values[1])
    if len(ws) != len(words):
        raise ValueError("token mismatch")
    H = [int(h) - 1 if int(h) > 0 else i for i, h in enumerate(sent.values[6])]
    L = list(sent.values[7])
    return [{"dep": L[i], "head": H[i]} for i in range(len(words))]


def han_parse(words):
    out = HAN(" ".join(words), tasks=["tok/fine", "dep"])
    ws = out["tok/fine"]
    if len(ws) != len(words):
        raise ValueError("token mismatch")
    deps = out["dep"]
    return [{"dep": l, "head": (h - 1 if h > 0 else i)} for i, (h, l) in enumerate(deps)]


score("spaCy", sp_parse, {"nsubj", "nsubjpass"}, {"dobj", "obj", "attr"})
score("LTP", ltp_parse, {"SBV"}, {"VOB", "FOB", "IOB"})
score("SuPar", sup_parse, {"nsubj"}, {"dobj", "obj"})
score("HanLP", han_parse, {"nsubj"}, {"dobj"})
