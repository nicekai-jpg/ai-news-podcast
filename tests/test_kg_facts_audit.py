"""kg_facts audit：A 档抽查输出。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("kg_facts", ROOT / "scripts" / "kg_facts.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    grade: str,
    subject: str,
    relation: str,
    item_id: str,
    slots: dict,
    date: str = "2026-11-16",
    sentence: str = "原句",
) -> dict:
    return {
        "date": date,
        "grade": grade,
        "subject": {"name": subject, "id": "", "kind": "company"},
        "relation": relation,
        "slots": slots,
        "grade_reasons": ["dep_confirmed"],
        "evidence": {"sentence": sentence, "item_id": item_id},
        "source": {"name": "测试源"},
    }


def _write_day(directory: Path, date: str, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"facts-{date}.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


class TestRender:
    def test_release_line(self):
        module = _load_script()
        row = _row(
            grade="A",
            subject="OpenAI",
            relation="release",
            item_id="i1",
            slots={"object": {"text": "GPT-5"}},
        )
        assert module.render_fact_line(row) == "OpenAI —发布→ GPT-5"

    def test_personnel_line_keeps_role_and_company(self):
        module = _load_script()
        row = _row(
            grade="A",
            subject="黄仁勋",
            relation="personnel",
            item_id="i1",
            slots={
                "person": {"text": "黄仁勋"},
                "role": {"text": "CEO"},
                "company": {"text": "英伟达"},
            },
        )
        rendered = module.render_fact_line(row)
        assert "黄仁勋" in rendered and "CEO" in rendered and "英伟达" in rendered

    def test_missing_object_is_explicit(self):
        module = _load_script()
        row = _row(grade="A", subject="OpenAI", relation="release", item_id="i1", slots={})
        assert "缺宾语" in module.render_fact_line(row)


class TestAudit:
    def test_only_a_grade_is_listed(self, tmp_path, capsys):
        module = _load_script()
        _write_day(
            tmp_path / "facts",
            "2026-11-16",
            [
                _row(
                    grade="A",
                    subject="OpenAI",
                    relation="release",
                    item_id="i1",
                    slots={"object": {"text": "GPT-5"}},
                ),
                _row(
                    grade="B",
                    subject="Meta",
                    relation="release",
                    item_id="i2",
                    slots={"object": {"text": "LLaMA"}},
                ),
            ],
        )
        code = module.main.__globals__["main"]  # 保持模块被加载
        assert code is not None
        args = type("A", (), {})()
        args.data_dir = str(tmp_path)
        args.date = "2026-11-16"
        args.days = 1
        args.no_context = True
        args.out_md = ""
        assert module.cmd_audit(args) == 0
        out = capsys.readouterr().out
        assert "OpenAI —发布→ GPT-5" in out
        assert "LLaMA" not in out
        assert "A 档 | 1 条" in out
        assert "B 档 | 1 条" in out

    def test_prints_evidence_and_source(self, tmp_path, capsys):
        module = _load_script()
        _write_day(
            tmp_path / "facts",
            "2026-11-16",
            [
                _row(
                    grade="A",
                    subject="OpenAI",
                    relation="release",
                    item_id="i1",
                    slots={"object": {"text": "GPT-5"}},
                    sentence="OpenAI released GPT-5.",
                )
            ],
        )
        args = type("A", (), {})()
        args.data_dir = str(tmp_path)
        args.date = "2026-11-16"
        args.days = 1
        args.no_context = True
        args.out_md = ""
        module.cmd_audit(args)
        out = capsys.readouterr().out
        assert "OpenAI released GPT-5." in out
        assert "测试源" in out

    def test_writes_markdown_when_asked(self, tmp_path):
        module = _load_script()
        _write_day(
            tmp_path / "facts",
            "2026-11-16",
            [
                _row(
                    grade="A",
                    subject="OpenAI",
                    relation="release",
                    item_id="i1",
                    slots={"object": {"text": "GPT-5"}},
                )
            ],
        )
        target = tmp_path / "audit.md"
        args = type("A", (), {})()
        args.data_dir = str(tmp_path)
        args.date = "2026-11-16"
        args.days = 1
        args.no_context = True
        args.out_md = str(target)
        module.cmd_audit(args)
        assert target.is_file()
        assert "OpenAI —发布→ GPT-5" in target.read_text(encoding="utf-8")

    def test_missing_facts_returns_error(self, tmp_path):
        module = _load_script()
        args = type("A", (), {})()
        args.data_dir = str(tmp_path)
        args.date = ""
        args.days = 1
        args.no_context = True
        args.out_md = ""
        assert module.cmd_audit(args) == 2
