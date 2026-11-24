"""事实层填槽与判档：关系、槽位、A/B/C。"""

from __future__ import annotations

from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parents[1]
# 词表用绝对路径：别的测试可能 monkeypatch.chdir()，相对路径会失效。
NLP_DIR = REPO_ROOT / "config" / "nlp"
SCHEMA_PATH = NLP_DIR / "event_schemas.yaml"


@pytest.fixture(scope="module")
def env():
    pytest.importorskip("spacy")
    lex = KGlexicon.load(NLP_DIR)
    parser = DepParser.load(lex)
    if parser is None:
        pytest.skip("缺少 spaCy 模型")
    return lex, parser, load_schemas(SCHEMA_PATH), reporting_verbs()


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


def test_personnel_owner_from_attributive(env) -> None:
    """「英伟达创始人黄仁勋」：公司是人名定语，也要填进 company 槽。"""
    rows = _candidates(env, "英伟达创始人黄仁勋辞去首席执行官职务")
    assert rows
    candidate, _ = rows[0]
    assert candidate.relation == "personnel"
    company = candidate.slots.get("company")
    assert company is not None
    assert company.text == "英伟达"


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


def test_evidence_from_title_demoted() -> None:
    """证据句不在正文、只在标题里 → 降 B（标题党不算事实）。"""
    from ai_news_podcast.business.kg.grade import grade
    from ai_news_podcast.business.kg.types import SlotValue

    candidate = _card(
        "partnership",
        "OpenAI",
        {"object": SlotValue("object", "Meta", "meta", "company")},
        clause="OpenAI 和 Meta 联手",
    )
    quality, reasons = grade(
        candidate,
        article_subject_text="OpenAI",
        source_text="Sierra 联合 Meta 发布 Poppy，OpenAI 推出 ChatGPT Wallet。",
        title="OpenAI 和 Meta 联手,打通了 Agent 商业「最后一公里」",
    )
    assert quality == "B"
    assert "evidence_from_title" in reasons


def test_quoted_fabrication_demoted() -> None:
    """证据句落在「作文题」引文里 → 降 B。"""
    from ai_news_podcast.business.kg.grade import grade
    from ai_news_podcast.business.kg.types import SlotValue

    subject = "英伟达创始人黄仁勋辞去 CEO 职务"
    candidate = _card(
        "personnel",
        "黄仁勋",
        {
            "person": SlotValue("person", "黄仁勋", "jensen_huang", "person"),
            "role": SlotValue("role", "CEO", "", "role"),
        },
        kind="person",
        clause=subject,
    )
    body = (
        "最绝的是野生高考出题组，连夜把 800 字材料作文题都给拟好了： "
        f"「阅读下面的材料，根据要求写作。近日，{subject}，引发广泛讨论。"
    )
    quality, reasons = grade(candidate, article_subject_text="黄仁勋", source_text=body)
    assert quality == "B"
    assert "evidence_quoted_material" in reasons


def test_normal_quote_not_demoted() -> None:
    """正常引用（如产品名「词元所」）不能误判。"""
    from ai_news_podcast.business.kg.grade import grade
    from ai_news_podcast.business.kg.types import SlotValue

    body = "百度还在此次世界大会上推出大模型交易平台「词元所」，李彦宏称之为大模型的交易所。"
    clause = "百度还在此次世界大会上推出大模型交易平台「词元所」"
    candidate = _card(
        "release",
        "百度",
        {"object": SlotValue("object", "词元所", "", "product")},
        clause=clause,
    )
    _quality, reasons = grade(candidate, article_subject_text="百度", source_text=body)
    assert "evidence_quoted_material" not in reasons


def test_personnel_owner_skips_generic_company() -> None:
    """「公司联合创始人黄仁勋」不能把泛称「公司」当成归属。"""
    from ai_news_podcast.business.kg.fill import _owner_before
    from ai_news_podcast.business.kg.types import SlotValue

    class _Tok:
        def __init__(self, text):
            self.text = text

    class _Parse:
        text = "公司联合创始人黄仁勋辞去 CEO 职务"
        tokens = tuple(_Tok(t) for t in ["公司", "联合", "创始人", "黄仁勋", "辞去", "CEO", "职务"])

    class _Pred:
        subjects = (3,)
        verb_index = 4

    person = SlotValue("person", "黄仁勋", "jensen_huang", "person")

    def fake_resolve(parse, index):
        if index == 0:
            return SlotValue("company", "公司", "", "company")
        if index == 3:
            return person
        return None

    import ai_news_podcast.business.kg.fill as fill_mod

    original = fill_mod.resolve_mention
    fill_mod.resolve_mention = fake_resolve
    try:
        assert _owner_before(_Parse(), _Pred(), person) is None
    finally:
        fill_mod.resolve_mention = original


def test_quoted_appositive_preferred(env) -> None:
    """「推出大模型交易平台「词元所」」应取引号内的专名，而不是泛称「平台」。"""
    rows = _candidates(env, "百度在此次大会上推出大模型交易平台「词元所」。")
    assert rows
    candidate, _ = rows[0]
    assert candidate.relation == "release"
    assert candidate.slots["object"].text == "词元所"


def test_quoted_appositive_ignores_long_quote(env) -> None:
    """引号里是整句（示例/引文）时，不能当专名吸附。"""
    sentence = "百度推出大模型交易平台「阅读下面的材料，根据要求写作，这是一段很长的引文内容」。"
    rows = _candidates(env, sentence)
    for candidate, _ctx in rows:
        assert (
            candidate.slots["object"].text != "阅读下面的材料，根据要求写作，这是一段很长的引文内容"
        )


def test_dash_appositive_preferred(env) -> None:
    """「高性能AI模型——Muse Ember」应取破折号后的专名。"""
    rows = _candidates(env, "Meta正式推出规模最小的高性能AI模型——Muse Ember。")
    assert rows
    candidate, _ = rows[0]
    assert candidate.slots["object"].text == "Muse Ember"


def test_english_object_kind_not_person(env) -> None:
    """回归：英文普通名词不能被当成 person（曾因正则同名覆盖）。"""
    from ai_news_podcast.business.kg.fill import FillContext, _object_value

    lex, parser, schemas, _reporting = env
    parse = parser.analyze("OpenAI released a new model", "en")
    value = _object_value(parse, 4, FillContext(lexicon=lex, schemas=schemas, item_id="x"))
    assert value.kind != "person"


def test_looks_like_product_en() -> None:
    from ai_news_podcast.business.kg.fill import _looks_like_product_en

    assert _looks_like_product_en("ChatGPT Wallet")
    assert _looks_like_product_en("Gemini Robotics 2.5")
    assert _looks_like_product_en("Claude Opus Model")
    assert not _looks_like_product_en("Alice Huang")
    assert not _looks_like_product_en("Harald Helfgott")


def test_latin_person_name_stays_person(env) -> None:
    """人名不能被产品规则误判。"""
    from ai_news_podcast.business.kg.fill import resolve_mention

    _lex, parser, _schemas, _reporting = env
    parse = parser.analyze("Alice Huang joined the board", "en")
    values = [
        value
        for index in range(len(parse.tokens))
        if (value := resolve_mention(parse, index)) is not None and value.text == "Alice Huang"
    ]
    assert values
    assert values[0].kind == "person"


def test_memory_summary_object_is_not_person(env) -> None:
    """回归：「（Memory summary）」这类英文功能名不能被判成人名而丢弃。"""
    rows = _candidates(env, "报道称谷歌正在为 Gemini 推出“记忆摘要”（Memory summary）功能")
    assert rows
    candidate, _ = rows[0]
    assert candidate.slots["object"].text == "Memory summary"
    assert candidate.slots["object"].kind != "person"


def test_versioned_product_name_is_object(env) -> None:
    """「Seedance 2.5 flash」被 spaCy 标成 nmod:range，仍应作为宾语。"""
    rows = _candidates(env, "字节推出 Seedance 2.5 flash：视频生成提速。")
    assert rows
    candidate, _ = rows[0]
    assert candidate.relation == "release"
    assert candidate.slots["object"].text == "Seedance 2.5 flash"


def _clean_candidate(*, secondary: bool) -> object:
    from ai_news_podcast.business.kg.types import FactCandidate, SlotValue

    return FactCandidate(
        clause="谷歌发布 Gemini 4 Hydrogen",
        subject=SlotValue("subject", "谷歌", "google", "company"),
        relation="release",
        relation_raw="发布",
        slots={"object": SlotValue("object", "Gemini 4 Hydrogen", "gemini4h", "product")},
        polarity="positive",
        modality="actual",
        reasons=["dep_confirmed"],
        secondary=secondary,
    )


def test_inherited_subject_gets_a2() -> None:
    """只有主语继承这一个问题时给 A2，而不是降到 B。"""
    from ai_news_podcast.business.kg.grade import grade

    quality, reasons = grade(
        _clean_candidate(secondary=True),
        article_subject_text="谷歌",
        source_text="谷歌发布 Gemini 4 Hydrogen。",
    )
    assert quality == "A2"
    assert reasons == ["subject_inherited"]


def test_inherited_plus_weak_object_stays_b() -> None:
    """A2 只覆盖「仅继承」一种情况，叠加其他问题仍然是 B。"""
    from ai_news_podcast.business.kg.grade import grade
    from ai_news_podcast.business.kg.types import SlotValue

    candidate = _clean_candidate(secondary=True)
    candidate.slots["object"] = SlotValue("object", "某个模型", "", "other")
    quality, reasons = grade(
        candidate,
        article_subject_text="谷歌",
        source_text="谷歌发布某个模型。",
    )
    assert quality == "B"
    assert "object_weak" in reasons


def test_non_inherited_clean_stays_a() -> None:
    from ai_news_podcast.business.kg.grade import grade

    quality, _reasons = grade(
        _clean_candidate(secondary=False),
        article_subject_text="谷歌",
        source_text="谷歌发布 Gemini 4 Hydrogen。",
    )
    assert quality == "A"


def test_subject_differs_from_article_not_inherited(env) -> None:
    """正文明确写了主语、但不是本文主角 → 不算继承（不进 A2）。"""
    from ai_news_podcast.business.kg.dep import extract_predicates
    from ai_news_podcast.business.kg.fill import (
        FillContext,
        build_candidate,
        classify_relation,
    )

    lex, parser, schemas, reporting = env
    parse = parser.analyze("比亚迪推出第二代刀片电池。", "zh")
    ctx = FillContext(
        lexicon=lex,
        schemas=schemas,
        item_id="x",
        article_subject=__import__(
            "ai_news_podcast.business.kg.types", fromlist=["SlotValue"]
        ).SlotValue("subject", "阿里巴巴", "alibaba", "company"),
    )
    rows = []
    for pred in extract_predicates(parse, reporting):
        match = classify_relation(parse, pred, lex)
        if match is None:
            continue
        candidate = build_candidate(parse, pred, match, ctx)
        if candidate is not None:
            rows.append(candidate)
    assert rows, "该句应抽到候选"
    # 比亚迪是正文里的专名，只是不是本文主角，不应当作「继承」
    for candidate in rows:
        assert candidate.secondary is False
        assert "subject_inherited" not in candidate.reasons
