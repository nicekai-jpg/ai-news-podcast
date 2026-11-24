"""事实层文本层：切句、噪声、语言、文章主体。"""

from __future__ import annotations

from ai_news_podcast.business.kg.clause import split_clauses
from ai_news_podcast.business.kg.lexicon import KGlexicon
from ai_news_podcast.business.kg.noise import (
    detect_language,
    is_noise_sentence,
    pick_article_subject,
    select_news_sentences,
    split_sentences,
)
from ai_news_podcast.business.kg.types import FactsConfig


def test_split_sentences_two_sentences() -> None:
    assert split_sentences("第一句话足够长。第二句话也足够长！") == [
        "第一句话足够长。",
        "第二句话也足够长！",
    ]


def test_noise_sentence_filtered() -> None:
    text = "英伟达发布新芯片。来源：量子位"
    sentences = select_news_sentences(text, FactsConfig())
    assert sentences == ["英伟达发布新芯片。"]


def test_wechat_not_treated_as_noise() -> None:
    assert not is_noise_sentence("微信发布新的 AI 功能")


def test_long_pseudo_sentence_dropped() -> None:
    text = "【1】" + "参考文献" * 300
    assert select_news_sentences(text, FactsConfig()) == []


def test_language_detection() -> None:
    assert detect_language({"lang": "en"}) == "en"
    assert detect_language({"lang": "", "full_text": "英伟达发布新芯片"}) == "zh"
    assert detect_language({"lang": "", "full_text": "OpenAI released a new model"}) == "en"


def test_article_subject_from_title() -> None:
    lex = KGlexicon.load()
    subject = pick_article_subject("英伟达发布新一代芯片", lex)
    assert subject is not None
    assert subject.text == "英伟达"


def test_clause_split_protects_quotes() -> None:
    clauses = split_clauses("他说“A，B”，然后离开")
    assert "他说“A，B”" in clauses
