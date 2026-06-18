"""Tests for CosyVoice backend (mocked — no model required)."""

from __future__ import annotations

from pathlib import Path

from ai_news_podcast.pipeline.cosyvoice_backend import (
    CosyVoice2Engine,
    CosyVoiceConfig,
    load_cosyvoice_config,
)


def test_load_cosyvoice_config_from_yaml_dict(tmp_path: Path) -> None:
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "host_a_ref.wav").write_bytes(b"wav")
    (refs / "host_a_ref.txt").write_text("男声参考文本", encoding="utf-8")
    (refs / "host_b_ref.wav").write_bytes(b"wav")
    (refs / "host_b_ref.txt").write_text("女声参考文本", encoding="utf-8")

    cfg = {
        "tts": {
            "cosyvoice": {
                "model_dir": "/models/CosyVoice2-0.5B",
                "ref_audio": {
                    "host_a": str(refs / "host_a_ref.wav"),
                    "host_a_text": str(refs / "host_a_ref.txt"),
                    "host_b": str(refs / "host_b_ref.wav"),
                    "host_b_text": str(refs / "host_b_ref.txt"),
                },
            }
        }
    }
    result = load_cosyvoice_config(cfg, project_root=tmp_path)
    assert isinstance(result, CosyVoiceConfig)
    assert result.refs["A"]["professional"][1] == "男声参考文本"
    assert result.model_dir == Path("/models/CosyVoice2-0.5B")


def test_synthesize_chunk_dispatches_by_host(tmp_path: Path, monkeypatch) -> None:
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "host_a_ref.wav").write_bytes(b"wav")
    (refs / "host_a_ref.txt").write_text("男声", encoding="utf-8")
    (refs / "host_b_ref.wav").write_bytes(b"wav")
    (refs / "host_b_ref.txt").write_text("女声", encoding="utf-8")

    config = CosyVoiceConfig(
        model_dir=tmp_path / "model",
        refs={
            "A": {"professional": (refs / "host_a_ref.wav", "男声")},
            "B": {"professional": (refs / "host_b_ref.wav", "女声")},
        },
    )
    engine = CosyVoice2Engine(config)
    calls: list[tuple[str, str]] = []

    class FakeModel:
        def inference_zero_shot(self, text, ref_text, ref_audio, stream=False):
            calls.append((text, ref_text))
            yield {"tts_speech": "fake_tensor"}

    monkeypatch.setattr(engine, "_ensure_model", lambda: FakeModel())
    monkeypatch.setattr(engine, "_load_ref", lambda _p: "fake_audio")

    engine.synthesize_chunk(text="测试句子", host="A")
    engine.synthesize_chunk(text="另一句", host="B")
    assert calls[0] == ("测试句子", "男声")
    assert calls[1] == ("另一句", "女声")
