"""Tests for ocean/taxonomy.py — 标签体系加载/校验。"""

from __future__ import annotations

import pytest

from ai_news_podcast.business.ocean.taxonomy import (
    TaxonomyError,
    default_taxonomy,
    load_taxonomy,
)


class TestDefaultTaxonomy:
    def test_loads_enums(self) -> None:
        taxonomy = default_taxonomy()
        assert "central" in taxonomy.enum_values("ai_centrality")
        assert "other" in taxonomy.enum_values("primary_domain")

    def test_entity_alias_mapping(self) -> None:
        taxonomy = default_taxonomy()
        assert taxonomy.canonical_entity("英伟达")["id"] == "nvidia"
        assert taxonomy.canonical_entity("OpenAI")["id"] == "openai"

    def test_industry_mapping(self) -> None:
        taxonomy = default_taxonomy()
        assert taxonomy.industry_for_title("特斯拉宣布 Cybertruck 停产")["id"] == "automotive"
        assert taxonomy.industry_for_title("A股三大指数低开")["id"] == "finance"


class TestValidation:
    def _write(
        self,
        tmp_path,
        labels: str,
        entities: str = "version: 1\nentities: []",
        industries: str = "version: 1\nindustries: []",
    ):
        (tmp_path / "labels.yaml").write_text(labels, encoding="utf-8")
        (tmp_path / "entities.yaml").write_text(entities, encoding="utf-8")
        (tmp_path / "industries.yaml").write_text(industries, encoding="utf-8")
        return tmp_path

    def test_missing_dimensions_rejected(self, tmp_path) -> None:
        root = self._write(tmp_path, "version: 1\ndimensions: {}\n")
        with pytest.raises(TaxonomyError):
            load_taxonomy(root)

    def test_duplicate_entity_rejected(self, tmp_path) -> None:
        labels = "version: 1\ndimensions:\n  ai_centrality:\n    values: [central, unknown]\n"
        entities = """version: 1
entities:
  - {id: a, name: A, type: company, ai_related: true}
  - {id: a, name: B, type: company, ai_related: true}
"""
        root = self._write(tmp_path, labels, entities)
        with pytest.raises(TaxonomyError):
            load_taxonomy(root)

    def test_unknown_parent_rejected(self, tmp_path) -> None:
        labels = "version: 1\ndimensions:\n  ai_centrality:\n    values: [central, unknown]\n"
        industries = """version: 1
industries:
  - {id: ai, name: AI, ai_related: core, parent: missing, keywords: []}
"""
        root = self._write(
            tmp_path, labels, entities="version: 1\nentities: []", industries=industries
        )
        with pytest.raises(TaxonomyError):
            load_taxonomy(root)


def test_industry_match_returns_keyword_evidence() -> None:
    taxonomy = default_taxonomy()
    match = taxonomy.industry_match("某车企宣布停产")
    assert match is not None
    industry, keyword = match
    assert industry["id"] == "automotive"
    assert keyword in {"车企", "停产"}


def test_entity_industry_mapping() -> None:
    taxonomy = default_taxonomy()
    assert taxonomy.entity_industries["openai"] == "ai"
    assert taxonomy.entity_industries["nvidia"] == "semiconductors"
    assert taxonomy.entity_industries["tesla"] == "automotive"
