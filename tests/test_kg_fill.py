"""事实层填槽与判档：关系、槽位、A/B/C。"""

from __future__ import annotations

import pytest

from ai_news_podcast.business.kg import grade as grade_mod
from ai_news_podcast.business.kg.dep import DepParser, extract_predicates
from ai_news_podcast.business.kg.fill import (
    FillContext,
    build_candidate,
    classify_relation,
    reporting_verbs,
)
from ai_news_podcast.business.kg.lexicon import KGlexicon
from ai_news_podcast.business.kg.schema import load_schemas


@pytest.fixture(scope="module")
def env():
    pytest.importorskip("spacy")
    lex = KGlexicon.load()
    parser = DepParser.load(lex)
    if parser is None:
        pytest.skip("缺少 spaCy 模型")
    return lex, parser, load_schemas(), reporting_verbs()


def _candidates(env, sentence: str, lang: str = "zh"):
    lex, parser, schemas, reporting = env
    parse = parser.analyze(sentence, lang)
    rows = []
    for pred in extract_predicates(parse, reporting):
        match = classify_relation(parse, pred, lex)
        if match is None:
            continue
        ctx = FillContext(lexicon=lex, schemas=schemas, item_id="x")
        candidate = build_candidate(parse, pred, match, ctx)
        if candidate is not None:
            rows.append((candidate, ctx))
    return rows


def test_release_filled(env) -> None:
    rows = _candidates(env, "比亚迪推出第二代刀片电池")
    assert rows
    candidate, _ = rows[0]
    assert candidate.relation == "release"
    assert "电池" in candidate.slots["object"].text


def test_personnel_role_after_verb(env) -> None:
    rows = _candidates(env, "联合创始人黄仁勋辞去公司总裁")
    assert rows
    candidate, _ = rows[0]
    assert candidate.relation == "personnel"
    assert candidate.slots["role"].text == "总裁"


def test_negation_downgrades(env) -> None:
    rows = _candidates(env, "DeepSeek并没有提供语音接口")
    assert rows
    candidate, _ctx = rows[0]
    assert candidate.polarity == "negative"
    quality, reasons = grade_mod.grade(candidate, article_subject_text="DeepSeek")
    assert quality == "B"
    assert "negated" in reasons


def test_rumor_downgrades(env) -> None:
    rows = _candidates(env, "消息称，联想正与某地方政府洽谈第二座工厂")
    if not rows:
        pytest.skip("该句未抽到候选")
    candidate, _ = rows[0]
    assert candidate.modality == "rumor"
    quality, reasons = grade_mod.grade(candidate, article_subject_text="联想")
    assert quality == "B"
    assert "rumor" in reasons


def test_a_grade_for_lexicon_subject(env) -> None:
    rows = _candidates(env, "比亚迪推出第二代刀片电池")
    candidate, _ = rows[0]
    quality, reasons = grade_mod.grade(candidate, article_subject_text="比亚迪")
    assert quality in ("A", "B")
    assert reasons


def test_b_grade_for_open_subject(env) -> None:
    rows = _candidates(env, "江淮汽车发布新车型")
    if not rows:
        pytest.skip("该句未抽到候选")
    candidate, _ = rows[0]
    quality, reasons = grade_mod.grade(candidate, article_subject_text="江淮汽车")
    assert quality == "B"
    assert "subject_open" in reasons


def test_english_release(env) -> None:
    rows = _candidates(env, "OpenAI released a new model", "en")
    assert rows
    candidate, _ = rows[0]
    assert candidate.relation == "release"


def _card(
    relation: str, subject: str, slots: dict, *, kind: str = "company", clause: str = ""
) -> object:
    from ai_news_podcast.business.kg.types import FactCandidate, SlotValue

    return FactCandidate(
        clause=clause or f"{subject} 相关事件",
        subject=SlotValue("subject", subject, "entity-1", kind),
        relation=relation,
        relation_raw="测试",
        slots=slots,
        polarity="positive",
        modality="actual",
        reasons=["dep_confirmed"],
    )


def test_a_guard_demotes_generic_object() -> None:
    from ai_news_podcast.business.kg.grade import grade
    from ai_news_podcast.business.kg.types import SlotValue

    candidate = _card("release", "联想", {"object": SlotValue("object", "模型", "", "product")})
    quality, reasons = grade(candidate, article_subject_text="联想")
    assert quality == "B"
    assert "a_guard_generic_object" in reasons


def test_a_guard_demotes_title_fragment() -> None:
    from ai_news_podcast.business.kg.grade import grade
    from ai_news_podcast.business.kg.types import SlotValue

    candidate = _card(
        "benchmark",
        "Waymo",
        {"object": SlotValue("object", "实测第一丨港科广新", "", "product")},
    )
    quality, reasons = grade(candidate, article_subject_text="Waymo")
    assert quality == "B"
    assert "a_guard_title_fragment" in reasons


def test_a_guard_demotes_personnel_without_role() -> None:
    from ai_news_podcast.business.kg.grade import grade
    from ai_news_podcast.business.kg.types import SlotValue

    candidate = _card(
        "personnel",
        "黄仁勋",
        {"person": SlotValue("person", "黄仁勋", "jensen_huang", "person")},
        kind="person",
    )
    quality, reasons = grade(candidate, article_subject_text="黄仁勋")
    assert quality == "B"
    assert "a_guard_role_missing" in reasons


def test_a_guard_demotes_identity_role() -> None:
    from ai_news_podcast.business.kg.grade import grade
    from ai_news_podcast.business.kg.types import SlotValue

    candidate = _card(
        "personnel",
        "黄仁勋",
        {
            "person": SlotValue("person", "黄仁勋", "jensen_huang", "person"),
            "role": SlotValue("role", "联合创始人", "", "role"),
        },
        kind="person",
        clause="英伟达联合创始人黄仁勋卸任后",
    )
    quality, reasons = grade(candidate, article_subject_text="黄仁勋")
    assert quality == "B"
    assert "a_guard_role_identity" in reasons


def test_a_guard_keeps_clean_card() -> None:
    from ai_news_podcast.business.kg.grade import grade
    from ai_news_podcast.business.kg.types import SlotValue

    candidate = _card(
        "release",
        "百度",
        {"object": SlotValue("object", "文心5.5", "", "product")},
        clause="百度发布文心5.5",
    )
    quality, reasons = grade(candidate, article_subject_text="百度")
    assert quality == "A"
    assert reasons == ["dep_confirmed"]
