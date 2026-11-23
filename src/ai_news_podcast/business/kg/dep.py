"""依存解析：模型加载、词表 span 合并、谓词抽取（事实层 S3）。"""

from __future__ import annotations

import logging
import re
from typing import Any

from ai_news_podcast.business.kg.entities import detect_open_entities
from ai_news_podcast.business.kg.lexicon import KGlexicon
from ai_news_podcast.business.kg.types import DepToken, LexiconSpan, ParseResult, Predicate

logger = logging.getLogger(__name__)

_KEEP_PIPES = ("tok2vec", "tagger", "attribute_ruler", "parser", "ner")
_SUBJECT_DEPS = ("nsubj", "nsubjpass", "nsubj:xsubj")
# nmod:range：带版本号的产品名常被 spaCy 标成范围修饰（如 "Seedance 2.5 flash"）。
_OBJECT_DEPS = ("dobj", "obj", "attr", "nmod:range")
_PREP_DEPS = ("nmod:prep", "nmod", "obl", "prep")
_COORD_DEPS = ("conj",)
_DESCEND_DEPS = ("ccomp", "xcomp", "parataxis")
_SIDE_VERB_DEPS = ("acl", "advcl", "relcl", "conj", "dep", "amod", "nmod")
_VERB_POS = ("VERB", "VA", "VC", "VE", "AUX")
_VERSION_RE = re.compile(r"v?\d+(?:\.\d+)+", re.IGNORECASE)


def detect_spans(
    lexicon: KGlexicon, text: str, lang: str = "zh", title: str = ""
) -> list[LexiconSpan]:
    """词典实体 + 未登录实体，按“起点升序、长词优先”去重叠。"""
    spans: list[LexiconSpan] = lexicon.find_spans(text)
    if lang == "zh" and text:
        try:
            opened = detect_open_entities(text, title)
        except Exception:
            opened = []
        for item in opened:
            if any(item.start < span.end and span.start < item.end for span in spans):
                continue
            spans.append(LexiconSpan(item.start, item.end, item.text, "", item.kind, "open"))
    spans.sort(key=lambda row: (row.start, -(row.end - row.start)))
    deduped: list[LexiconSpan] = []
    last_end = -1
    for span in spans:
        if span.start < last_end:
            continue
        deduped.append(span)
        last_end = span.end
    return deduped


def _entity_vocab(lexicon: KGlexicon) -> set[str]:
    """把词典里的实体/对象名称和别名收集成分词词表（只收 ≥2 字）。"""
    words: set[str] = set()
    for row in lexicon.lexicon.entities:
        for name in (row.name, *row.aliases):
            text = str(name or "").strip()
            if len(text) >= 2:
                words.add(text)
    for row in lexicon.lexicon.objects:
        for name in (row.name, *row.aliases):
            text = str(name or "").strip()
            if len(text) >= 2:
                words.add(text)
    return words


class _EntityMergeTokenizer:
    """默认分词器外面包一层：把“相邻且合起来是词表词/版本号”的 token 合并。

    只合并不拆分，尽量不破坏原模型（pkuseg）的分词分布。
    """

    def __init__(self, base: Any, vocab: set[str]) -> None:
        self.base = base
        self.vocab = vocab

    def __call__(self, text: str) -> Any:
        from spacy.tokens import Doc

        doc = self.base(text)
        toks = list(doc)
        words: list[str] = []
        spaces: list[bool] = []
        i = 0
        while i < len(toks):
            best = i
            acc = toks[i].text
            for j in range(i + 1, min(i + 6, len(toks))):
                if toks[j].idx != toks[j - 1].idx + len(toks[j - 1].text):
                    break
                acc += toks[j].text
                if acc in self.vocab or _VERSION_RE.fullmatch(acc):
                    best = j
            words.append("".join(t.text for t in toks[i : best + 1]))
            end = toks[best].idx + len(toks[best].text)
            spaces.append(end < len(text) and text[end].isspace())
            i = best + 1
        return Doc(doc.vocab, words=words, spaces=spaces)


class DepParser:
    """中英双模型解析器；模型缺失时对应语言不可用。"""

    def __init__(
        self, nlps: dict[str, Any], lexicon: KGlexicon, *, custom_tokenizer: bool = False
    ) -> None:
        self.nlps = nlps
        self.lexicon = lexicon
        self.custom_tokenizer = custom_tokenizer
        if custom_tokenizer:
            vocab = _entity_vocab(lexicon)
            for lang, nlp in nlps.items():
                if lang == "zh":
                    nlp.tokenizer = _EntityMergeTokenizer(nlp.tokenizer, vocab)

    @classmethod
    def load(
        cls,
        lexicon: KGlexicon,
        *,
        models: dict[str, str] | None = None,
        custom_tokenizer: bool = False,
    ) -> DepParser | None:
        import spacy

        model_names = models or {"zh": "zh_core_web_sm", "en": "en_core_web_sm"}
        nlps: dict[str, Any] = {}
        for lang, name in model_names.items():
            try:
                nlp = spacy.load(name)
                nlp.select_pipes(enable=list(_KEEP_PIPES))
                nlps[lang] = nlp
            except Exception:
                logger.warning("spaCy 模型加载失败：%s（%s）", name, lang, exc_info=True)
        if not nlps:
            return None
        return cls(nlps, lexicon, custom_tokenizer=custom_tokenizer)

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(self.nlps)

    def analyze(self, text: str, lang: str = "zh", title: str = "") -> ParseResult:
        """解析一个子句：先合并词表实体/未登录实体，再跑依存。"""
        nlp = self.nlps.get(lang) or next(iter(self.nlps.values()))
        spans = detect_spans(self.lexicon, text, lang, title)
        doc = nlp.tokenizer(text)
        self._merge_spans(doc, spans)
        for _, proc in nlp.pipeline:
            doc = proc(doc)
        tokens = [
            DepToken(
                i=token.i,
                text=token.text,
                idx=token.idx,
                dep=token.dep_,
                head=token.head.i,
                pos=token.pos_,
                ent_type=getattr(token, "ent_type_", "") or "",
            )
            for token in doc
        ]
        root_index = next((t.i for t in tokens if t.dep == "ROOT"), -1)
        ents = [
            (ent.text, int(ent.start_char), int(ent.end_char), str(ent.label_))
            for ent in getattr(doc, "ents", [])
        ]
        return ParseResult(text=text, tokens=tokens, spans=spans, root_index=root_index, ents=ents)

    def analyze_many(self, items: list[tuple[str, str]]) -> list[ParseResult | None]:
        """批量解析 [(text, lang)]；单条失败返回 None。"""
        results: list[ParseResult | None] = []
        for text, lang in items:
            try:
                results.append(self.analyze(text, lang))
            except Exception:
                logger.debug("子句解析失败：%s", text[:50], exc_info=True)
                results.append(None)
        return results

    @staticmethod
    def _merge_spans(doc: Any, spans: list[LexiconSpan]) -> None:
        merges: list[tuple[int, int]] = []
        for span in spans:
            covered = [t for t in doc if t.idx < span.end and t.idx + len(t.text) > span.start]
            if len(covered) > 1:
                merges.append((covered[0].i, covered[-1].i + 1))
        if not merges:
            return
        merges.sort()
        with doc.retokenize() as retokenizer:
            used: list[tuple[int, int]] = []
            for start, end in merges:
                if any(not (end <= a or start >= b) for a, b in used):
                    continue
                used.append((start, end))
                retokenizer.merge(doc[start:end])


def _children(tokens: list[DepToken], index: int, deps: tuple[str, ...]) -> list[DepToken]:
    return [token for token in tokens if token.head == index and token.dep in deps]


def extract_predicates(
    parse: ParseResult,
    reporting: set[str] | None = None,
) -> list[Predicate]:
    """从解析结果抽谓词：root 动词 + 引述从句动词。"""
    if parse.root_index < 0:
        return []
    tokens = parse.tokens
    root = tokens[parse.root_index]
    predicate_indices: list[int] = []
    if root.pos in _VERB_POS:
        predicate_indices.append(root.i)
    predicate_indices.extend(
        child.i for child in _children(tokens, root.i, _DESCEND_DEPS) if child.pos in _VERB_POS
    )
    if root.pos not in _VERB_POS:
        # root 是名词/形容词时（"创维推出…"根是显示器、"台积电并购后…"根是改造），
        # 从定语从句/状语从句/并列结构里补动词
        predicate_indices.extend(
            token.i for token in tokens if token.pos in _VERB_POS and token.dep in _SIDE_VERB_DEPS
        )
    else:
        predicate_indices.extend(
            token.i
            for token in tokens
            if token.pos in _VERB_POS and token.dep in ("acl", "advcl", "relcl")
        )
    predicate_indices = sorted(set(predicate_indices))
    rows: list[Predicate] = []
    for index in predicate_indices:
        verb = tokens[index]
        subjects = _children(tokens, index, _SUBJECT_DEPS)
        subjects.extend(
            [
                child
                for subject in list(subjects)
                for child in _children(tokens, subject.i, _COORD_DEPS)
            ]
        )
        objects = _children(tokens, index, _OBJECT_DEPS)
        objects.extend(
            [
                child
                for child in _children(tokens, index, _COORD_DEPS)
                if child.pos in ("NOUN", "PROPN", "NUM", "X")
            ]
        )
        attrs = _children(tokens, index, ("attr",))
        preps: list[tuple[str, int]] = []
        for child in _children(tokens, index, _PREP_DEPS):
            prep_name = ""
            for grand in _children(tokens, child.i, ("case",)):
                prep_name = grand.text
                break
            if not prep_name and child.dep == "prep":
                prep_name = child.text
            if not prep_name:
                continue
            preps.append((prep_name.lower(), child.i))
            for grand in _children(tokens, child.i, ("pobj",)):
                preps.append((prep_name.lower(), grand.i))  # noqa: PERF401
        rows.append(
            Predicate(
                verb_index=index,
                verb_text=verb.text,
                subjects=tuple(sorted({subject.i for subject in subjects})),
                objects=tuple(sorted({obj.i for obj in objects})),
                attrs=tuple(sorted({attr.i for attr in attrs})),
                preps=tuple(sorted(set(preps))),
                reported=index != parse.root_index and verb.dep in _DESCEND_DEPS,
            )
        )
    return rows


__all__ = ["DepParser", "extract_predicates"]
