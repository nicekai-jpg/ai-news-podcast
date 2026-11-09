"""Tests for the v3 pipeline CLI entry (no network, runner mocked)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_news_podcast.business.pipeline.v3_runner_service import V3RunResult
from ai_news_podcast.presentation.cli.podcast_v3_controller import V3PipelineCommand


def test_dry_run(capsys: Any) -> None:
    assert V3PipelineCommand().run(["--date", "2026-11-09", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "2026-11-09" in out


def test_execute_calls_runner(tmp_path: Path, monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    def fake_run(date_str: str, **kwargs: Any) -> V3RunResult:
        calls["date_str"] = date_str
        calls["kwargs"] = kwargs
        return V3RunResult(
            date=date_str,
            raw_path=tmp_path / "raw.json.gz",
            ocean_dir=tmp_path / "ocean",
            events_path=tmp_path / "events.json",
            shortlist_path=tmp_path / "shortlist.json",
            commentary_path=tmp_path / "commentary.json",
            script_path=None,
        )

    monkeypatch.setattr(
        "ai_news_podcast.presentation.cli.podcast_v3_controller.run_v3_pipeline", fake_run
    )
    rc = V3PipelineCommand().run(
        ["--date", "2026-11-09", "--data-dir", str(tmp_path), "--skip-script"]
    )
    assert rc == 0
    assert calls["date_str"] == "2026-11-09"
    assert calls["kwargs"]["data_dir"] == tmp_path.resolve()
    assert calls["kwargs"]["skip_script"] is True
    assert calls["kwargs"]["llm_cfg"] is not None


def test_with_audio_and_publish(tmp_path: Path, monkeypatch: Any) -> None:
    script = tmp_path / "episode.txt"
    script.write_text("[CHAPTER] 开场\n[Host A] 你好", encoding="utf-8")
    sink: dict[str, Any] = {}

    def fake_run(date_str: str, **kwargs: Any) -> V3RunResult:
        return V3RunResult(
            date=date_str,
            raw_path=tmp_path / "raw.json.gz",
            ocean_dir=tmp_path / "ocean",
            events_path=tmp_path / "events.json",
            shortlist_path=tmp_path / "shortlist.json",
            commentary_path=tmp_path / "commentary.json",
            script_path=script,
            brief_path=tmp_path / "brief.json",
        )

    async def fake_synth(text: str, **kwargs: Any) -> None:
        sink["tts"] = {"text": text, "output": kwargs["output_path"]}

    class FakePublish:
        async def execute_async(self, args: Any, cfg: Any, root: Path) -> int:
            sink["publish"] = args.date
            return 0

    monkeypatch.setattr(
        "ai_news_podcast.presentation.cli.podcast_v3_controller.run_v3_pipeline", fake_run
    )
    monkeypatch.setattr(
        "ai_news_podcast.presentation.cli.podcast_v3_controller.synthesize", fake_synth
    )
    monkeypatch.setattr(
        "ai_news_podcast.presentation.cli.podcast_v3_controller.PublishCommand", FakePublish
    )

    rc = V3PipelineCommand().run(
        ["--date", "2026-11-09", "--data-dir", str(tmp_path), "--with-audio", "--publish"]
    )
    assert rc == 0
    assert sink["tts"]["text"].startswith("[CHAPTER]")
    assert str(sink["tts"]["output"]).endswith("2026-11-09.mp3")
    assert sink["publish"] == "2026-11-09"
