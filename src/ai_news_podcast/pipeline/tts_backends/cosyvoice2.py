"""CosyVoice 2 TTS Backend implementation."""

from __future__ import annotations

import importlib
import logging
import tempfile
from pathlib import Path
from typing import Any

from ai_news_podcast.pipeline.cosyvoice_backend import (
    CosyVoice2Engine,
    CosyVoiceConfig,
    load_cosyvoice_config,
)
from ai_news_podcast.pipeline.tts_backends.base import TTSBackend
from ai_news_podcast.pipeline.tts_parser import (
    DialogueChunk,
    parse_dialogue_chunks,
    split_text_into_sentences,
)
from ai_news_podcast.pipeline.tts_postprocess import (
    assemble_dialogue_audio,
    finalize_episode_mp3,
)
from ai_news_podcast.text_utils import strip_tts_tags

log = logging.getLogger(__name__)


def _write_clean_transcript(
    *,
    chunks: list[DialogueChunk],
    transcript_path: Path,
) -> None:
    # Save clean plain text transcript (bracketed format) to .txt file
    clean_lines = [f"[Host {chunk.host}] {strip_tts_tags(chunk.text)}" for chunk in chunks]
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text("\n\n".join(clean_lines) + "\n", encoding="utf-8")


def _write_chunks_and_playlist(
    chunks: list[DialogueChunk],
    segments_by_variant: dict[str, list[Any]],
    chunk_variant_names: list[list[str]],
    timestamps: list[tuple[float, float]],
    voice_maps: dict[str, dict[str, str]],
    output_path: Path,
) -> None:
    """将各主持人在其合成变体下的音频片段与播放清单 JSON 写入以单期 ID 命名的文件夹。"""
    import json

    chunks_dir = output_path.with_suffix("")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    playlist_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        start_sec, duration_sec = timestamps[idx - 1]

        audios = {}
        voices = {}
        for var in chunk_variant_names[idx - 1]:
            fn = f"chunk_{idx:03d}_{var}.mp3"
            segments = segments_by_variant[var][idx - 1]
            if segments is None:
                continue

            pydub = importlib.import_module("pydub")
            audio_segment_cls = pydub.AudioSegment
            silence_pad = audio_segment_cls.silent(duration=300)
            padded_seg = silence_pad + segments + silence_pad

            try:
                padded_seg.export(str(chunks_dir / fn), format="mp3", bitrate="64k")
            except TypeError:
                padded_seg.export(str(chunks_dir / fn), format="mp3")

            audios[var] = fn
            voices[var] = voice_maps[var].get(chunk.host, "unknown")

        playlist_chunks.append(
            {
                "id": idx,
                "host": chunk.host,
                "text": strip_tts_tags(chunk.text),
                "start": round(start_sec, 3),
                "duration": round(duration_sec, 3),
                "audios": audios,
                "voices": voices,
            }
        )

    playlist_data = {"episode_id": output_path.stem, "chunks": playlist_chunks}

    playlist_json_path = chunks_dir / "playlist.json"
    playlist_json_path.write_text(
        json.dumps(playlist_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _select_variants_for_host(cv_cfg: CosyVoiceConfig, host: str) -> list[str]:
    """决定某位主持人真实要合成的变体;配置与 ref_audio 无交集时回退为全部(防呆)。"""
    available = list(cv_cfg.refs.get(host, {}).keys()) or ["professional", "lively"]
    configured = [v for v in cv_cfg.synth_variants.get(host, ()) if v in available]
    if configured:
        return configured
    if cv_cfg.synth_variants.get(host):
        log.warning(
            "synth_variants[%s]=%s 与 ref_audio 无交集,回退为合成全部音色",
            host,
            list(cv_cfg.synth_variants.get(host, ())),
        )
    return available


def _host_variant_plan(cv_cfg: CosyVoiceConfig) -> tuple[dict[str, list[str]], dict[str, str]]:
    """返回 (每位主持人的合成变体列表, 每位主持人的发布默认变体,即列表首位)。"""
    variants = {host: _select_variants_for_host(cv_cfg, host) for host in ("A", "B")}
    defaults = {host: (vs[0] if vs else "professional") for host, vs in variants.items()}
    return variants, defaults


class CosyVoice2Backend(TTSBackend):
    """CosyVoice 2 TTS backend implementation."""

    def __init__(self, **config: Any) -> None:
        self.config = config

    async def synthesize(
        self,
        text: str,
        *,
        output_path: Path,
        bgm_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        chunks = parse_dialogue_chunks(text)
        if not chunks:
            raise ValueError("Input text is empty after dialogue parsing")

        cfg = kwargs.get("cfg", {})
        project_root = kwargs.get("project_root") or Path.cwd()
        transcript_path = kwargs.get("transcript_path")
        engine = kwargs.get("engine")

        torchaudio = importlib.import_module("torchaudio")
        pydub = importlib.import_module("pydub")
        audio_segment_cls = pydub.AudioSegment

        audio_cfg = (cfg or {}).get("tts", {}).get("audio", {})
        root = project_root
        cv_cfg = load_cosyvoice_config(cfg or {}, project_root=root)
        cosy_engine = engine or CosyVoice2Engine(cv_cfg)

        final_path = Path(output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)

        host_variant_map, host_default = _host_variant_plan(cv_cfg)
        all_variants = sorted({v for vs in host_variant_map.values() for v in vs})
        voice_maps = {var: {"A": f"host_a_{var}", "B": f"host_b_{var}"} for var in all_variants}

        with tempfile.TemporaryDirectory(prefix="tts-cosyvoice-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            segments_by_variant: dict[str, list[Any]] = {}
            chunk_variant_names: list[list[str]] = []

            for idx, chunk in enumerate(chunks, start=1):
                sentences = split_text_into_sentences(chunk.text, max_chars=80)
                vars_for_chunk = host_variant_map[chunk.host]
                chunk_variant_names.append(vars_for_chunk)

                for var in vars_for_chunk:
                    chunk_segments: list[Any] = []
                    for s_idx, sentence in enumerate(sentences):
                        s_text = sentence.strip()
                        if not s_text:
                            continue

                        tensor = cosy_engine.synthesize_chunk(
                            text=s_text, host=chunk.host, variant=var
                        )
                        wav_path = tmp_root / f"chunk_{idx:03d}_{var}_{s_idx:03d}.wav"
                        torchaudio.save(str(wav_path), tensor, cv_cfg.sample_rate)
                        chunk_segments.append(audio_segment_cls.from_file(str(wav_path)))

                    if chunk_segments:
                        combined_chunk = chunk_segments[0]
                        for next_seg in chunk_segments[1:]:
                            combined_chunk += audio_segment_cls.silent(duration=150) + next_seg
                    else:
                        combined_chunk = audio_segment_cls.silent(duration=100)
                    segments_by_variant.setdefault(var, [None] * len(chunks))[idx - 1] = (
                        combined_chunk
                    )

            combined, timestamps = assemble_dialogue_audio(
                chunks,
                [
                    segments_by_variant[host_default[chunk.host]][i]
                    for i, chunk in enumerate(chunks)
                ],
                chunk_silence_base=int(audio_cfg.get("chunk_silence_base", 300)),
                vocal_pad_ms=int(audio_cfg.get("vocal_pad_ms", 1000)),
                silence_min=int(audio_cfg.get("chunk_silence_min", 400)),
                silence_max=int(audio_cfg.get("chunk_silence_max", 800)),
                silence_jitter=int(audio_cfg.get("chunk_silence_jitter", 100)),
            )
            await finalize_episode_mp3(
                combined,
                final_path,
                bgm_path=bgm_path,
                audio_cfg=audio_cfg,
                tmp_dir=tmp_root,
            )

            _write_chunks_and_playlist(
                chunks=chunks,
                segments_by_variant=segments_by_variant,
                chunk_variant_names=chunk_variant_names,
                timestamps=timestamps,
                voice_maps=voice_maps,
                output_path=final_path,
            )

            if transcript_path:
                _write_clean_transcript(
                    chunks=chunks,
                    transcript_path=Path(transcript_path),
                )
