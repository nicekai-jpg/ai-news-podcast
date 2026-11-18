"""LTP 适配层单测：转换逻辑不依赖 ltp 安装；集成用例自动 skip。"""

from __future__ import annotations

import pytest

from ai_news_podcast.business.kg.ltp_parser import convert_ltp_output


def test_convert_ltp_output_maps_labels_and_heads() -> None:
    text = "百度推出词元所"
    tokens = convert_ltp_output(
        text,
        ["百度", "推出", "词元所"],
        ["nz", "v", "n"],
        [2, 0, 2],
        ["SBV", "HED", "VOB"],
    )
    assert [t.text for t in tokens] == ["百度", "推出", "词元所"]
    assert [t.dep for t in tokens] == ["nsubj", "ROOT", "dobj"]
    assert [t.pos for t in tokens] == ["PROPN", "VERB", "NOUN"]
    assert tokens[0].head == 1
    assert tokens[1].head == 1
    assert tokens[2].head == 1
    assert tokens[0].idx == 0
    assert tokens[2].idx == 4


def test_convert_ltp_output_att_depends_on_pos() -> None:
    tokens = convert_ltp_output(
        "蓝色机器人",
        ["蓝色", "机器人"],
        ["a", "n"],
        [2, 0],
        ["ATT", "HED"],
    )
    assert tokens[0].dep == "amod"
    tokens = convert_ltp_output(
        "数据中台",
        ["数据", "中台"],
        ["n", "n"],
        [2, 0],
        ["ATT", "HED"],
    )
    assert tokens[0].dep == "compound:nn"


def test_convert_ltp_output_skips_blank_words() -> None:
    tokens = convert_ltp_output(
        "qData 发布",
        ["qData ", " ", "发布"],
        ["nz", "ws", "v"],
        [3, 3, 0],
        ["SBV", "PUN", "HED"],
    )
    assert [t.text for t in tokens] == ["qData", "发布"]
    assert tokens[0].dep == "nsubj"
    assert tokens[1].dep == "ROOT"
    assert tokens[0].idx == 0
    assert tokens[1].idx == 6


def test_ltp_parser_smoke() -> None:
    pytest.importorskip("ltp")
    from ai_news_podcast.business.kg.lexicon import KGlexicon
    from ai_news_podcast.business.kg.ltp_parser import LTPParser

    lex = KGlexicon.load("config/nlp")
    parser = LTPParser(lex)
    parse = parser.analyze("百度推出词元所")
    assert [t.text for t in parse.tokens] == ["百度", "推出", "词元所"]
    assert parse.root_index >= 0
