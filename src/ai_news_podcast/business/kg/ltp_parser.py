"""LTP 解析器适配（可选）：把 LTP 的 cws/pos/dep 转成事实层统一的 ParseResult。

- 只处理中文；英文走 fallback（spaCy）。
- LTP 依赖标签是 SDGP 体系，这里映射成 spaCy 风格的标签，便于 fill/grade 复用。
- ltp 是可选依赖（`kg-ltp` extra），未安装时 import 才报错，不影响默认链路。
"""

from __future__ import annotations

import logging

from ai_news_podcast.business.kg.dep import DepParser, detect_spans
from ai_news_podcast.business.kg.lexicon import KGlexicon
from ai_news_podcast.business.kg.types import DepToken, ParseResult

logger = logging.getLogger(__name__)

_LTP_POS = {
    "a": "ADJ",
    "b": "NOUN",
    "c": "CCONJ",
    "d": "ADV",
    "e": "INTJ",
    "g": "NOUN",
    "h": "PART",
    "i": "NOUN",
    "j": "PROPN",
    "k": "PART",
    "m": "NUM",
    "n": "NOUN",
    "nd": "NOUN",
    "nh": "PROPN",
    "ni": "PROPN",
    "nl": "NOUN",
    "ns": "PROPN",
    "nt": "PROPN",
    "nz": "PROPN",
    "o": "NUM",
    "p": "ADP",
    "q": "NOUN",
    "r": "PRON",
    "u": "PART",
    "v": "VERB",
    "wp": "PUNCT",
    "ws": "SPACE",
    "x": "X",
    "z": "NOUN",
    "f": "NOUN",
    "s": "NOUN",
    "t": "NOUN",
}

_LTP_DEP = {
    "HED": "ROOT",
    "SBV": "nsubj",
    "VOB": "dobj",
    "FOB": "dobj",
    "IOB": "dobj",
    "POB": "nmod:prep",
    "ADV": "advmod",
    "CMP": "nmod:prep",
    "COO": "conj",
    "PUN": "punct",
    "RAD": "acl",
    "LAD": "advcl",
    "DE": "case",
    "DI": "case",
    "BA": "nsubj",
    "BEI": "nsubjpass",
    "TMP": "nmod:tmod",
    "IS": "parataxis",
}


def convert_ltp_output(
    text: str, words: list[str], pos_tags: list[str], heads: list[int], labels: list[str]
) -> list[DepToken]:
    """把 LTP 的单句输出转成 DepToken 列表（纯函数，便于单测）。"""
    kept = [
        (i, str(words[i] or "").strip(), pos_tags[i], int(heads[i]), labels[i])
        for i in range(len(words))
        if str(words[i] or "").strip()
    ]
    position = {raw_index: pos for pos, (raw_index, *_rest) in enumerate(kept)}
    tokens: list[DepToken] = []
    cursor = 0
    for pos, (_raw_index, word, tag, raw_head, label) in enumerate(kept):
        pos_tag = _LTP_POS.get(str(tag), "NOUN")
        if str(label) == "ATT":
            dep = "amod" if pos_tag == "ADJ" else "compound:nn"
        else:
            dep = _LTP_DEP.get(str(label), "dep")
        head = pos if raw_head <= 0 or (raw_head - 1) not in position else position[raw_head - 1]
        idx = text.find(word, cursor)
        if idx < 0:
            idx = cursor
        cursor = idx + len(word)
        tokens.append(
            DepToken(i=pos, text=word, idx=idx, dep=dep, head=head, pos=pos_tag, ent_type="")
        )
    return tokens


class LTPParser:
    """LTP 4 解析器；中文用 LTP，其他语言回退到 spaCy。"""

    def __init__(
        self,
        lexicon: KGlexicon,
        *,
        fallback: DepParser | None = None,
        model: str = "LTP/small",
    ) -> None:
        from ltp import LTP

        self.lexicon = lexicon
        self.fallback = fallback
        self.engine = LTP(model)

    def analyze(self, text: str, lang: str = "zh", title: str = "") -> ParseResult:
        if lang != "zh":
            if self.fallback is not None:
                return self.fallback.analyze(text, lang, title)
            raise RuntimeError("LTP 只支持中文解析，且没有可用的 fallback")
        out = self.engine.pipeline([text], tasks=["cws", "pos", "dep"])
        row = out.dep[0]
        tokens = convert_ltp_output(text, out.cws[0], out.pos[0], row["head"], row["label"])
        spans = detect_spans(self.lexicon, text, lang, title)
        root_index = next((t.i for t in tokens if t.dep == "ROOT"), -1)
        return ParseResult(text=text, tokens=tokens, spans=spans, root_index=root_index, ents=[])


__all__ = ["LTPParser", "convert_ltp_output"]
