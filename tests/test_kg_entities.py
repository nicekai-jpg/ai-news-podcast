"""知识图谱：未登录实体识别。"""

from __future__ import annotations

from ai_news_podcast.business.kg.entities import detect_open_entities


def test_detects_person_name() -> None:
    entities = detect_open_entities("雷军宣布小米汽车下线")
    names = [item.text for item in entities]
    assert "雷军" in names


def test_detects_org_by_suffix() -> None:
    entities = detect_open_entities("之江实验室发布了新的模型")
    assert any(item.text.endswith("实验室") for item in entities)


def test_drops_generic_terms() -> None:
    entities = detect_open_entities("这个人工智能大模型的价格很高")
    names = [item.text for item in entities]
    assert "人工智能" not in names
    assert "大模型" not in names


def test_drops_generic_news_nouns() -> None:
    entities = detect_open_entities(
        "这家半导体材料公司称，德系车企对全球都市市场都在观望，官方与开发者也在等"
    )
    names = [item.text for item in entities]
    for word in ("半导体材料公司", "材料公司", "德系", "车企", "全球", "都市", "官方", "开发者"):
        assert word not in names


def test_english_caps_only_when_in_title_or_multiword() -> None:
    single = detect_open_entities("我们用了 Groq 的芯片")
    assert all(item.text != "Groq" for item in single) or all(
        item.kind == "org" for item in single if item.text == "Groq"
    )
    titled = detect_open_entities("Groq 发布了新芯片", title="Groq 发布新芯片")
    assert any(item.text == "Groq" for item in titled)


def test_overlaps_keep_longer_span() -> None:
    entities = detect_open_entities("清华大学人工智能学院成立")
    texts = [item.text for item in entities]
    assert any(text.endswith("学院") for text in texts)
