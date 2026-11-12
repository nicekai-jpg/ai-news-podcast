"""Tests for ocean/domain.py — 主题域分级。"""

from __future__ import annotations

from ai_news_podcast.business.ocean.domain import (
    AI_CORE,
    AI_INFRA,
    OTHER,
    UNKNOWN,
    classify_domain,
    load_domain_keywords,
    stronger_domain,
)


class TestClassifyDomain:
    def test_model_news_is_ai_core(self) -> None:
        assert classify_domain("OpenAI 发布新模型，支持 Agent")[0] == AI_CORE

    def test_autonomous_driving_is_ai_infra(self) -> None:
        assert classify_domain("小鹏发布新一代智能驾驶系统")[0] == AI_INFRA

    def test_car_discontinuation_is_other(self) -> None:
        assert classify_domain("特斯拉宣布 Cybertruck 停产")[0] == OTHER

    def test_car_toy_is_other(self) -> None:
        assert classify_domain("特斯拉上新：Cybertruck 遥控车与人形机器人摆件")[0] == OTHER

    def test_stock_news_is_other(self) -> None:
        assert classify_domain("A股午后涨幅扩大，通信板块领涨")[0] == OTHER

    def test_unknown_when_no_signal(self) -> None:
        assert classify_domain("公司发布季度公告")[0] == UNKNOWN

    def test_research_body_fallback(self) -> None:
        domain, _ = classify_domain(
            "A Novel Method for X", full_text="该方法使用大模型与 Agent 工具", category="research"
        )
        assert domain == AI_CORE


def test_stronger_domain_order() -> None:
    assert stronger_domain(UNKNOWN, OTHER) == UNKNOWN
    assert stronger_domain(OTHER, AI_INFRA) == AI_INFRA
    assert stronger_domain(AI_INFRA, AI_CORE) == AI_CORE


def test_load_domain_keywords_reads_config() -> None:
    keywords = load_domain_keywords()
    assert "汽车" in keywords["other_category"]
    assert "自动驾驶" in keywords["ai_driving"]
    assert "特斯拉" in keywords["auto_brands"]


def test_extra_keyword_from_config(tmp_path) -> None:
    path = tmp_path / "domain.yaml"
    path.write_text(
        "other_category: [演唱会]\n"
        "ai_driving: []\n"
        "ai_core_extra: []\n"
        "ai_infra_extra: []\n"
        "auto_brands: []\n",
        encoding="utf-8",
    )
    keywords = load_domain_keywords(str(path))
    assert classify_domain("某明星演唱会门票售罄", keywords=keywords)[0] == OTHER


def test_industry_fallback_maps_non_ai_healthcare() -> None:
    from ai_news_podcast.business.ocean.domain import classify_domain

    domain, evidence = classify_domain("某医院上线新的挂号系统")
    assert domain == OTHER
    assert evidence
