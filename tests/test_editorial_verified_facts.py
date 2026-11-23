"""Stage 2.5 A 档事实注入 editorial 的行为测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ai_news_podcast.business.contracts import Event, ShortlistEntry  # noqa: E402
from ai_news_podcast.business.editorial.config import load_config  # noqa: E402
from ai_news_podcast.business.editorial.prompts import build_commentary_prompt  # noqa: E402
from ai_news_podcast.business.editorial.verified_facts import (  # noqa: E402
    facts_by_item,
    load_a_cards,
    render_fact,
    verified_facts_for_event,
)


def _card(grade: str, item_id: str, subject: str, relation: str, obj: str) -> dict:
    return {
        "grade": grade,
        "subject": {"name": subject, "id": "", "kind": "company"},
        "relation": relation,
        "slots": {"object": {"text": obj, "entity_id": "", "kind": "product"}},
        "evidence": {"item_id": item_id, "sentence": "原句"},
    }


def _person_card(grade: str, item_id: str, name: str, role: str, company: str) -> dict:
    return {
        "grade": grade,
        "subject": {"name": name, "id": "", "kind": "person"},
        "relation": "personnel",
        "slots": {
            "person": {"text": name, "entity_id": "", "kind": "person"},
            "role": {"text": role, "entity_id": "", "kind": "role"},
            "company": {"text": company, "entity_id": "", "kind": "company"},
        },
        "evidence": {"item_id": item_id, "sentence": "原句"},
    }


class TestLoadACards:
    def test_only_a_grade_is_loaded(self, tmp_path):
        path = tmp_path / "facts-2026-11-16.jsonl"
        rows = [
            _card("A", "i1", "OpenAI", "release", "GPT-5"),
            _card("B", "i2", "Meta", "release", "LLaMA"),
        ]
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
        )
        cards = load_a_cards(tmp_path, "2026-11-16")
        assert len(cards) == 1
        assert cards[0]["subject"]["name"] == "OpenAI"

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_a_cards(tmp_path, "2026-01-01") == []


class TestGradeFilter:
    def test_default_only_a(self, tmp_path):
        path = tmp_path / "facts-2026-11-16.jsonl"
        rows = [
            _card("A", "i1", "OpenAI", "release", "GPT-5"),
            _card("A2", "i2", "谷歌", "release", "Gemini 4 Hydrogen"),
        ]
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
        )
        assert len(load_a_cards(tmp_path, "2026-11-16")) == 1

    def test_can_include_a2(self, tmp_path):
        path = tmp_path / "facts-2026-11-16.jsonl"
        rows = [
            _card("A", "i1", "OpenAI", "release", "GPT-5"),
            _card("A2", "i2", "谷歌", "release", "Gemini 4 Hydrogen"),
        ]
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
        )
        cards = load_a_cards(tmp_path, "2026-11-16", grades=("A", "A2"))
        assert len(cards) == 2


class TestRender:
    def test_render_triple(self):
        card = _card("A", "i1", "OpenAI", "release", "GPT-5")
        assert render_fact(card) == "OpenAI —发布→ GPT-5"

    def test_render_personnel_keeps_role_and_company(self):
        card = _person_card("A", "i1", "黄仁勋", "CEO", "英伟达")
        rendered = render_fact(card)
        assert "黄仁勋" in rendered and "CEO" in rendered and "英伟达" in rendered

    def test_render_unknown_object(self):
        card = _card("A", "i1", "OpenAI", "release", "")
        assert "未知宾语" in render_fact(card)


class TestSelection:
    def test_dedupes_and_caps(self):
        cards = [_card("A", "i1", "OpenAI", "release", "GPT-5") for _ in range(3)]
        by_item = facts_by_item(cards)
        assert by_item["i1"] == ["OpenAI —发布→ GPT-5"]

    def test_event_only_sees_its_own_items(self):
        by_item = {"i1": ["A"], "i2": ["B"]}
        assert verified_facts_for_event(["i2"], by_item) == ["B"]
        assert verified_facts_for_event(["i3"], by_item) == []

    def test_cap_applies_to_event(self):
        by_item = {f"i{n}": [f"fact{n}"] for n in range(10)}
        picked = verified_facts_for_event(list(by_item), by_item, limit=3)
        assert len(picked) == 3


class TestInjectionSwitch:
    def test_default_is_off(self):
        from ai_news_podcast.business.editorial.config import config_from_mapping

        assert config_from_mapping({}).inject_verified_facts is False

    def test_grade_selection(self):
        from ai_news_podcast.business.editorial.config import config_from_mapping

        assert config_from_mapping({}).fact_grades == ("A",)
        assert config_from_mapping({"fact_grades": ["A", "A2"]}).fact_grades == ("A", "A2")
        assert config_from_mapping({"fact_grades": ["bogus"]}).fact_grades == ("A",)

    def test_reads_yaml_switch(self):
        from ai_news_podcast.business.editorial.config import config_from_mapping

        assert config_from_mapping({"inject_verified_facts": True}).inject_verified_facts is True
        assert config_from_mapping({"inject_verified_facts": "yes"}).inject_verified_facts is True
        assert (
            config_from_mapping({"inject_verified_facts": "false"}).inject_verified_facts is False
        )

    def test_prompt_gets_nothing_when_switch_off(self):
        """开关关闭时，即便传了 verified_by_item 也不能出现在 prompt 里。"""
        from ai_news_podcast.business.contracts import Event, ShortlistEntry
        from ai_news_podcast.business.editorial.commentary import build_commentaries
        from ai_news_podcast.business.editorial.config import config_from_mapping

        event = Event(
            event_id="e1",
            representative_title="OpenAI 发布 GPT-5",
            summary="摘要",
            item_ids=["i1"],
            independent_sources=2,
        )
        prompts: list[str] = []

        def fake_llm(prompt: str) -> str:
            prompts.append(prompt)
            return '{"headline": "t", "one_liner": "o", "facts": []}'

        cfg = config_from_mapping({"inject_verified_facts": False})
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write("inject_verified_facts: false\n")
            path = handle.name
        build_commentaries(
            [ShortlistEntry(rank=1, event_id="e1", score=1.0)],
            [event],
            fake_llm,
            item_evidence={"i1": "正文"},
            config_path=path,
            verified_by_item={"i1": ["OpenAI —发布→ GPT-5"]},
        )
        assert prompts
        assert "OpenAI —发布→ GPT-5" not in prompts[0]
        assert cfg.inject_verified_facts is False


class TestPromptInjection:
    def _entry_event(self) -> tuple[ShortlistEntry, Event]:
        event = Event(
            event_id="e1",
            representative_title="OpenAI 发布 GPT-5",
            summary="摘要",
            item_ids=["i1", "i2"],
            independent_sources=2,
        )
        entry = ShortlistEntry(event_id="e1", rank=1, score=1.0)
        return entry, event

    def test_block_present_and_explicitly_secondary(self):
        entry, event = self._entry_event()
        prompt = build_commentary_prompt(
            entry,
            event,
            config=load_config(),
            evidence_ids=["i1"],
            verified_facts=["OpenAI —发布→ GPT-5"],
        )
        assert "已核验事实" in prompt
        assert "OpenAI —发布→ GPT-5" in prompt
        assert "不是新的事实来源" in prompt
        assert "以证据正文为准" in prompt

    def test_absent_when_no_facts(self):
        entry, event = self._entry_event()
        prompt = build_commentary_prompt(entry, event, config=load_config(), evidence_ids=["i1"])
        assert "本事件无机器抽取的可核验事实" in prompt
