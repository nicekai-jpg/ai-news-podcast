"""kg_facts watch：A/A2 趋势与告警。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "kg_facts_watch", ROOT / "scripts" / "kg_facts.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["kg_facts_watch"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(grade: str, subject: str, relation: str, obj: str, *, sentence: str = "原句") -> dict:
    return {
        "grade": grade,
        "subject": {"name": subject, "id": "", "kind": "company"},
        "relation": relation,
        "slots": {"object": {"text": obj, "entity_id": "", "kind": "product"}},
        "grade_reasons": ["dep_confirmed"],
        "evidence": {"sentence": sentence, "item_id": "i1"},
        "source": {"name": "测试源"},
    }


def _write(directory: Path, date: str, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"facts-{date}.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


def _args(tmp_path: Path, **extra) -> object:
    args = type("A", (), {})()
    args.data_dir = str(tmp_path)
    args.date = ""
    args.days = 7
    args.shortlist = False
    args.top_n = 0
    args.sample = 0
    args.seed = 13
    args.min_a = 0
    args.min_a2 = 0
    args.min_days = 3
    for key, value in extra.items():
        setattr(args, key, value)
    return args


class TestTrend:
    def test_counts_a_and_a2_separately(self, tmp_path, capsys):
        module = _load_script()
        facts = tmp_path / "facts"
        _write(
            facts,
            "2026-11-16",
            [
                _row("A", "OpenAI", "release", "GPT-5"),
                _row("A2", "谷歌", "release", "Gemini 4 Hydrogen"),
                _row("B", "Meta", "release", "LLaMA"),
            ],
        )
        assert module.cmd_watch(_args(tmp_path)) == 0
        out = capsys.readouterr().out
        assert "| 2026-11-16 | 1 | 1 | 1 |" in out

    def test_no_facts_returns_error(self, tmp_path):
        module = _load_script()
        assert module.cmd_watch(_args(tmp_path)) == 2


class TestWarnings:
    def test_flags_zero_day(self, tmp_path, capsys):
        module = _load_script()
        facts = tmp_path / "facts"
        _write(facts, "2026-11-15", [_row("A", "OpenAI", "release", "GPT-5")])
        _write(facts, "2026-11-16", [_row("B", "Meta", "release", "LLaMA")])
        module.cmd_watch(_args(tmp_path))
        out = capsys.readouterr().out
        assert "A+A2 为 0 的日期：2026-11-16" in out

    def test_flags_empty_evidence(self, tmp_path, capsys):
        module = _load_script()
        facts = tmp_path / "facts"
        _write(facts, "2026-11-16", [_row("A", "OpenAI", "release", "GPT-5", sentence="")])
        module.cmd_watch(_args(tmp_path))
        assert "证据句为空" in capsys.readouterr().out

    def test_flags_threshold(self, tmp_path, capsys):
        module = _load_script()
        facts = tmp_path / "facts"
        _write(facts, "2026-11-16", [_row("A", "OpenAI", "release", "GPT-5")])
        module.cmd_watch(_args(tmp_path, min_a=3, min_a2=1, min_days=1))
        out = capsys.readouterr().out
        assert "A 档均值 1.0 < 门槛 3" in out
        assert "A2 档均值 0.0 < 门槛 1" in out


class TestSample:
    def test_sample_is_reproducible(self, tmp_path, capsys):
        module = _load_script()
        facts = tmp_path / "facts"
        rows = [_row("A", f"公司{i}", "release", f"产品{i}") for i in range(10)]
        _write(facts, "2026-11-16", rows)
        module.cmd_watch(_args(tmp_path, sample=3, seed=7))
        first = capsys.readouterr().out
        module.cmd_watch(_args(tmp_path, sample=3, seed=7))
        second = capsys.readouterr().out
        assert first == second
        assert first.count("[A] 公司") == 3


class TestCoverage:
    def _event(self, event_id: str, item_ids: list[str]) -> dict:
        return {"event_id": event_id, "item_ids": item_ids}

    def test_counts_event_level_coverage(self, tmp_path, capsys):
        module = _load_script()
        facts = tmp_path / "facts"
        _write(
            facts,
            "2026-11-16",
            [
                _row("A", "OpenAI", "release", "GPT-5"),
                _row("A2", "谷歌", "release", "Gemini 4 Hydrogen"),
                _row("B", "Meta", "release", "LLaMA"),
            ],
        )
        # 三条事实都在 i1 上，只属于 e1；e2 无事实
        for name in ("2026-11-16",):
            path = facts / f"facts-{name}.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row["evidence"]["item_id"] = "i1"
            path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
        day = tmp_path / "ocean" / "2026-11-16"
        day.mkdir(parents=True)
        (day / "events.json").write_text(
            json.dumps([self._event("e1", ["i1"]), self._event("e2", ["i2"])], ensure_ascii=False),
            encoding="utf-8",
        )
        assert module.cmd_coverage(_args(tmp_path)) == 0
        out = capsys.readouterr().out
        assert "| 2026-11-16 | 2 | 1 | 1 | 1 | 50.0% |" in out
        assert "A 档覆盖 50.0% 的事件" in out

    def test_missing_events_file_is_skipped(self, tmp_path, capsys):
        module = _load_script()
        _write(tmp_path / "facts", "2026-11-16", [_row("A", "OpenAI", "release", "GPT-5")])
        assert module.cmd_coverage(_args(tmp_path)) == 0
        out = capsys.readouterr().out
        assert "合计" not in out


class TestCoverageShortlist:
    def test_limits_denominator_to_shortlist(self, tmp_path, capsys, monkeypatch):
        """--shortlist 只算前 N 个入选事件，分母显著变小。"""
        module = _load_script()
        facts = tmp_path / "facts"
        rows = [_row("A", "OpenAI", "release", "GPT-5")]
        for row in rows:
            row["evidence"]["item_id"] = "i1"
        _write(facts, "2026-11-16", rows)
        day = tmp_path / "ocean" / "2026-11-16"
        day.mkdir(parents=True)
        events = [{"event_id": f"e{n}", "item_ids": [f"i{n}"]} for n in range(1, 6)]
        (day / "events.json").write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")

        monkeypatch.setattr(module, "_shortlist_event_ids", lambda rows, top_n: ["e1"])
        args = _args(tmp_path, shortlist=True, top_n=1)
        assert module.cmd_coverage(args) == 0
        out = capsys.readouterr().out
        assert "| 2026-11-16 | 1 | 1 | 0 | 1 | 100.0% |" in out
        assert "前 1 个入选事件" in out
