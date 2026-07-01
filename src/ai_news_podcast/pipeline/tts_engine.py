"""TTS 音频合成引擎：CosyVoice2 后端。"""

from __future__ import annotations

import importlib
import logging
import tempfile
from pathlib import Path
from typing import Any

from ai_news_podcast.pipeline.tts_parser import (
    parse_dialogue_chunks,
    split_text_into_sentences,
)
from ai_news_podcast.pipeline.tts_postprocess import (
    assemble_dialogue_audio,
    finalize_episode_mp3,
)
from ai_news_podcast.pipeline.tts_types import DialogueChunk

log = logging.getLogger(__name__)


def _write_transcript_with_timestamps(
    *,
    chunks: list[DialogueChunk],
    timestamps: list[tuple[float, float]],
    voice_map: dict[str, str],
    transcript_path: Path,
) -> None:
    xml_lines = [
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">',
    ]
    for idx, chunk in enumerate(chunks):
        start_sec, duration_sec = timestamps[idx]
        voice = chunk.voice or voice_map.get(chunk.host, voice_map.get("A", ""))
        xml_lines.append(
            f'<voice name="{voice}" start="{start_sec:.3f}" duration="{duration_sec:.3f}">'
        )
        xml_lines.append(chunk.text)
        xml_lines.append("</voice>")
    xml_lines.append("</speak>")
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text("\n".join(xml_lines), encoding="utf-8")


def _write_chunks_and_playlist(
    chunks: list[DialogueChunk],
    segments_by_variant: dict[str, list[Any]],
    timestamps: list[tuple[float, float]],
    voice_maps: dict[str, dict[str, str]],
    output_path: Path,
) -> None:
    """将各个音频片段及播放清单 JSON 写入和单期 ID 同名的文件夹中，包含所有音色版本。"""
    chunks_dir = output_path.with_suffix("")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    playlist_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        start_sec, duration_sec = timestamps[idx - 1]

        audios = {}
        voices = {}
        for var, segments in segments_by_variant.items():
            fn = f"chunk_{idx:03d}_{var}.mp3"

            import importlib

            pydub = importlib.import_module("pydub")
            audio_segment_cls = pydub.AudioSegment
            silence_pad = audio_segment_cls.silent(duration=300)
            padded_seg = silence_pad + segments[idx - 1] + silence_pad

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
                "text": chunk.text,
                "start": round(start_sec, 3),
                "duration": round(duration_sec, 3),
                "audios": audios,
                "voices": voices,
            }
        )

    playlist_data = {"episode_id": output_path.stem, "chunks": playlist_chunks}

    import json

    playlist_json_path = chunks_dir / "playlist.json"
    playlist_json_path.write_text(
        json.dumps(playlist_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def synthesize_cosyvoice2(
    chunks: list[DialogueChunk],
    output_path: Path,
    *,
    bgm_path: str | None = None,
    transcript_path: Path | None = None,
    cfg: dict | None = None,
    project_root: Path | None = None,
    engine: Any | None = None,
) -> None:
    from ai_news_podcast.pipeline.cosyvoice_backend import CosyVoice2Engine, load_cosyvoice_config

    torchaudio = importlib.import_module("torchaudio")
    pydub = importlib.import_module("pydub")
    audio_segment_cls = pydub.AudioSegment

    audio_cfg = (cfg or {}).get("tts", {}).get("audio", {})
    root = project_root or Path.cwd()
    cv_cfg = load_cosyvoice_config(cfg or {}, project_root=root)
    cosy_engine = engine or CosyVoice2Engine(cv_cfg)

    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    variants = list(cv_cfg.refs["A"].keys()) if "A" in cv_cfg.refs else ["professional", "lively"]
    voice_maps = {}
    for var in variants:
        voice_maps[var] = {"A": f"host_a_{var}", "B": f"host_b_{var}"}

    with tempfile.TemporaryDirectory(prefix="tts-cosyvoice-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        segments_by_variant = {var: [] for var in variants}

        for idx, chunk in enumerate(chunks, start=1):
            sentences = split_text_into_sentences(chunk.text, max_chars=80)

            # Generate segment for each variant
            for var in variants:
                chunk_segments: list[Any] = []
                for s_idx, sentence in enumerate(sentences):
                    s_text = sentence.strip()
                    if not s_text:
                        continue
                    tensor = cosy_engine.synthesize_chunk(text=s_text, host=chunk.host, variant=var)
                    wav_path = tmp_root / f"chunk_{idx:03d}_{var}_{s_idx:03d}.wav"
                    torchaudio.save(str(wav_path), tensor, cv_cfg.sample_rate)
                    chunk_segments.append(audio_segment_cls.from_file(str(wav_path)))

                if chunk_segments:
                    combined_chunk = chunk_segments[0]
                    for next_seg in chunk_segments[1:]:
                        combined_chunk += audio_segment_cls.silent(duration=150) + next_seg
                    segments_by_variant[var].append(combined_chunk)
                else:
                    segments_by_variant[var].append(audio_segment_cls.silent(duration=100))

        default_variant = "professional" if "professional" in variants else variants[0]
        combined, timestamps = assemble_dialogue_audio(
            chunks,
            segments_by_variant[default_variant],
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
            timestamps=timestamps,
            voice_maps=voice_maps,
            output_path=final_path,
        )

        if transcript_path:
            _write_transcript_with_timestamps(
                chunks=chunks,
                timestamps=timestamps,
                voice_map=voice_maps[default_variant],
                transcript_path=Path(transcript_path),
            )


async def synthesize(
    text: str,
    *,
    backend: str = "cosyvoice2",
    output_path: Path,
    bgm_path: str | None = None,
    **kwargs: Any,
) -> None:
    chunks = parse_dialogue_chunks(text)
    if not chunks:
        raise ValueError("Input text is empty after dialogue parsing")

    backend_name = str(backend).strip().lower()
    cosy_kwargs = {
        "bgm_path": bgm_path,
        "transcript_path": kwargs.get("transcript_path"),
        "cfg": kwargs.get("cfg"),
        "project_root": kwargs.get("project_root"),
        "engine": kwargs.get("engine"),
    }

    if backend_name in ("edge-tts", "edge"):
        raise ValueError("Edge-TTS backend has been archived and is no longer available.")

    if backend_name in ("cosyvoice2", "cosyvoice"):
        await synthesize_cosyvoice2(chunks, output_path=output_path, **cosy_kwargs)
        return

    if backend_name == "hybrid":
        # hybrid now delegates only to cosyvoice2 since edge-tts is archived
        await synthesize_cosyvoice2(chunks, output_path=output_path, **cosy_kwargs)
        return

    raise ValueError(f"Unsupported TTS backend: {backend}")
