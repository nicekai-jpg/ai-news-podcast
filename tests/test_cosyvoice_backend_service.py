"""Tests for CosyVoice backend (mocked — no model required)."""

from __future__ import annotations

from pathlib import Path

from ai_news_podcast.business.pipeline.cosyvoice_backend_service import (
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

    monkeypatch.setattr(engine, "_ensure_model", FakeModel)
    monkeypatch.setattr(engine, "_load_ref", lambda _p: "fake_audio")

    engine.synthesize_chunk(text="测试句子", host="A")
    engine.synthesize_chunk(text="另一句", host="B")
    assert calls[0] == ("测试句子", "男声")
    assert calls[1] == ("另一句", "女声")


class TestSynthVariantsConfig:
    def test_appconfig_list_passthrough(self) -> None:
        from ai_news_podcast.data.config.models_dao import AppConfig

        cfg = AppConfig.from_dict({"tts": {"cosyvoice": {"synth_variants": ["professional"]}}})
        assert cfg.tts.cosyvoice.synth_variants == ["professional"]

    def test_appconfig_dict_passthrough(self) -> None:
        from ai_news_podcast.data.config.models_dao import AppConfig

        cfg = AppConfig.from_dict(
            {
                "tts": {
                    "cosyvoice": {
                        "synth_variants": {"host_a": ["lively"], "host_b": ["professional"]}
                    }
                }
            }
        )
        assert cfg.tts.cosyvoice.synth_variants == {
            "host_a": ["lively"],
            "host_b": ["professional"],
        }

    def test_default_is_empty_list(self) -> None:
        from ai_news_podcast.data.config.models_dao import AppConfig

        assert AppConfig.from_dict({}).tts.cosyvoice.synth_variants == []

    def test_runtime_normalizes_list_to_both_hosts(self, tmp_path: Path) -> None:
        parsed = load_cosyvoice_config(
            {"tts": {"cosyvoice": {"synth_variants": ["professional"]}}}, project_root=tmp_path
        )
        assert parsed.synth_variants == {"A": ("professional",), "B": ("professional",)}

    def test_runtime_normalizes_dict_with_aliases(self, tmp_path: Path) -> None:
        parsed = load_cosyvoice_config(
            {
                "tts": {
                    "cosyvoice": {"synth_variants": {"host_a": ["lively"], "B": ["professional"]}}
                }
            },
            project_root=tmp_path,
        )
        assert parsed.synth_variants == {"A": ("lively",), "B": ("professional",)}

    def test_runtime_unknown_host_key_ignored(self, tmp_path: Path) -> None:
        parsed = load_cosyvoice_config(
            {"tts": {"cosyvoice": {"synth_variants": {"host_c": ["lively"]}}}},
            project_root=tmp_path,
        )
        assert parsed.synth_variants == {}

    def test_runtime_default_empty(self, tmp_path: Path) -> None:
        parsed = load_cosyvoice_config({}, project_root=tmp_path)
        assert parsed.synth_variants == {"A": (), "B": ()}


class TestSelectVariants:
    def _cfg(self, synth_variants, refs_keys=("professional", "lively")) -> CosyVoiceConfig:
        return CosyVoiceConfig(
            model_dir=Path(),
            refs={
                "A": {k: (Path(), "") for k in refs_keys},
                "B": {k: (Path(), "") for k in refs_keys},
            },
            synth_variants=synth_variants,
        )

    def test_empty_config_all_variants_both_hosts(self) -> None:
        from ai_news_podcast.business.pipeline.tts_backends.cosyvoice2_service import (
            _select_variants_for_host,
        )

        cfg = self._cfg({})
        assert _select_variants_for_host(cfg, "A") == ["professional", "lively"]
        assert _select_variants_for_host(cfg, "B") == ["professional", "lively"]

    def test_global_list_applies_to_both_hosts(self) -> None:
        from ai_news_podcast.business.pipeline.tts_backends.cosyvoice2_service import (
            _select_variants_for_host,
        )

        cfg = self._cfg({"A": ("professional",), "B": ("professional",)})
        assert _select_variants_for_host(cfg, "A") == ["professional"]
        assert _select_variants_for_host(cfg, "B") == ["professional"]

    def test_per_host_selection(self) -> None:
        from ai_news_podcast.business.pipeline.tts_backends.cosyvoice2_service import (
            _select_variants_for_host,
        )

        cfg = self._cfg({"A": ("lively",), "B": ("professional",)})
        assert _select_variants_for_host(cfg, "A") == ["lively"]
        assert _select_variants_for_host(cfg, "B") == ["professional"]

    def test_no_overlap_falls_back_to_all(self) -> None:
        from ai_news_podcast.business.pipeline.tts_backends.cosyvoice2_service import (
            _select_variants_for_host,
        )

        cfg = self._cfg({"A": ("warm",), "B": ()})
        assert _select_variants_for_host(cfg, "A") == ["professional", "lively"]
        assert _select_variants_for_host(cfg, "B") == ["professional", "lively"]

    def test_empty_refs_default_pairs(self) -> None:
        from ai_news_podcast.business.pipeline.tts_backends.cosyvoice2_service import (
            _select_variants_for_host,
        )

        cfg = self._cfg({}, refs_keys=())
        assert _select_variants_for_host(cfg, "A") == ["professional", "lively"]

    def test_host_plan_defaults(self) -> None:
        from ai_news_podcast.business.pipeline.tts_backends.cosyvoice2_service import (
            _host_variant_plan,
        )

        cfg = self._cfg({"A": ("lively", "professional"), "B": ("professional",)})
        variants, defaults = _host_variant_plan(cfg)
        assert variants["A"] == ["lively", "professional"]
        assert defaults["A"] == "lively"
        assert variants["B"] == ["professional"]
        assert defaults["B"] == "professional"

    def test_host_plan_empty_config(self) -> None:
        from ai_news_podcast.business.pipeline.tts_backends.cosyvoice2_service import (
            _host_variant_plan,
        )

        _variants, defaults = _host_variant_plan(self._cfg({}))
        assert defaults == {"A": "professional", "B": "professional"}


class TestPlaylistChapterField:
    """playlist 章节字段只对有章节的 chunk 写入，历史结构（无该键）保持不变。"""

    def test_chapter_key_only_when_present(self, tmp_path: Path) -> None:
        import json

        from ai_news_podcast.business.pipeline.tts_backends.cosyvoice2_service import (
            _write_chunks_and_playlist,
        )
        from ai_news_podcast.business.pipeline.tts_types_service import DialogueChunk

        _write_chunks_and_playlist(
            chunks=[
                DialogueChunk(host="A", text="开场白。", chapter="今日头条"),
                DialogueChunk(host="B", text="回应一下。"),
            ],
            segments_by_variant={"professional": [None, None]},
            chunk_variant_names=[["professional"], ["professional"]],
            timestamps=[(0.0, 1.0), (1.0, 2.0)],
            voice_maps={"professional": {"A": "v1", "B": "v2"}},
            output_path=tmp_path / "2026-11-04.mp3",
        )

        data = json.loads((tmp_path / "2026-11-04" / "playlist.json").read_text(encoding="utf-8"))
        assert data["chunks"][0]["chapter"] == "今日头条"
        assert "chapter" not in data["chunks"][1]
        # 既有字段一个都不少
        assert data["chunks"][1] == {
            "id": 2,
            "host": "B",
            "text": "回应一下。",
            "start": 1.0,
            "duration": 2.0,
            "audios": {},
            "voices": {},
        }

    def test_legacy_chunk_without_chapter_has_no_key(self, tmp_path: Path) -> None:
        import json

        from ai_news_podcast.business.pipeline.tts_backends.cosyvoice2_service import (
            _write_chunks_and_playlist,
        )
        from ai_news_podcast.business.pipeline.tts_types_service import DialogueChunk

        _write_chunks_and_playlist(
            chunks=[DialogueChunk(host="A", text="历史脚本。")],
            segments_by_variant={"professional": [None]},
            chunk_variant_names=[["professional"]],
            timestamps=[(0.0, 1.0)],
            voice_maps={"professional": {"A": "v1"}},
            output_path=tmp_path / "2026-10-15.mp3",
        )

        data = json.loads((tmp_path / "2026-10-15" / "playlist.json").read_text(encoding="utf-8"))
        assert "chapter" not in data["chunks"][0]


def test_config_sample_rate_defaults_to_model_rate(tmp_path: Path) -> None:
    """CosyVoice2 模型是 24kHz：配置缺省时不能回落到 22050，否则整期变慢变低。"""
    result = load_cosyvoice_config({"tts": {"cosyvoice": {}}}, project_root=tmp_path)
    assert result.sample_rate == 24000
    assert CosyVoiceConfig.__dataclass_fields__["sample_rate"].default == 24000


def test_config_sample_rate_from_audio_config(tmp_path: Path) -> None:
    cfg = {"tts": {"audio": {"sample_rate": 24000}, "cosyvoice": {}}}
    assert load_cosyvoice_config(cfg, project_root=tmp_path).sample_rate == 24000

    cfg = {"tts": {"audio": {"sample_rate": 22050}, "cosyvoice": {}}}
    assert load_cosyvoice_config(cfg, project_root=tmp_path).sample_rate == 22050


def test_engine_sample_rate_falls_back_to_config(tmp_path: Path) -> None:
    cfg = load_cosyvoice_config({"tts": {"cosyvoice": {}}}, project_root=tmp_path)
    engine = CosyVoice2Engine(cfg)
    assert engine.sample_rate == 24000
