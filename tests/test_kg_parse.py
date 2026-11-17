"""事实层解析：实体合并、谓词抽取。"""

from __future__ import annotations

import pytest

from ai_news_podcast.business.kg.dep import DepParser, extract_predicates
from ai_news_podcast.business.kg.fill import reporting_verbs, resolve_mention
from ai_news_podcast.business.kg.lexicon import KGlexicon


@pytest.fixture(scope="module")
def parser() -> DepParser:
    pytest.importorskip("spacy")
    lex = KGlexicon.load()
    loaded = DepParser.load(lex)
    if loaded is None:
        pytest.skip("缺少 spaCy 中文模型")
    return loaded


def test_entity_span_merged(parser: DepParser) -> None:
    parse = parser.analyze("英伟达任命 Harvey C. Jones 为董事", "zh")
    texts = [token.text for token in parse.tokens]
    assert "英伟达" in texts


def test_open_company_merged(parser: DepParser) -> None:
    parse = parser.analyze("江淮汽车发布公告称合作尚未确定", "zh")
    texts = [token.text for token in parse.tokens]
    assert any("江淮" in text for text in texts)


def test_open_subject_resolved(parser: DepParser) -> None:
    parse = parser.analyze("江淮汽车发布公告称合作尚未确定", "zh")
    mentions = [
        resolve_mention(parse, token.i)
        for token in parse.tokens
        if token.dep in ("nsubj", "nsubjpass")
    ]
    assert any(mention is not None and "江淮" in mention.text for mention in mentions)


def test_reporting_verb_descends(parser: DepParser) -> None:
    parse = parser.analyze("联想宣布推出新模型", "zh")
    predicates = extract_predicates(parse, reporting_verbs())
    assert any(pred.reported for pred in predicates)


def test_english_parse(parser: DepParser) -> None:
    parse = parser.analyze("OpenAI released a new model", "en")
    assert parse.root_index >= 0
    predicates = extract_predicates(parse, reporting_verbs())
    assert any(pred.verb_text.lower() == "released" for pred in predicates)
