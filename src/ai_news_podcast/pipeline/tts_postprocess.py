"""Shared TTS post-processing: silence padding, BGM mix, loudnorm, assembly."""

from __future__ import annotations

import asyncio
import importlib
import random
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai_news_podcast.pipeline.tts_types import DialogueChunk


def chunk_silence_ms(
    chunk_silence_ms: int,
    *,
    silence_min: int = 400,
    silence_max: int = 800,
    silence_jitter: int = 100,
) -> int:
    base = max(silence_min, min(silence_max, int(chunk_silence_ms)))
    low = max(silence_min, base - silence_jitter)
    high = min(silence_max, base + silence_jitter)
    return random.randint(low, high)


def mix_bgm(
    vocal_segment: Any,
    bgm_path: str | None,
    *,
    bgm_volume_db: int = -12,
    bgm_fade_in_ms: int = 2000,
    bgm_fade_out_ms: int = 3000,
    vocal_pad_ms: int = 1000,
) -> Any:
    """Mix vocal track with background music using basic ducking."""
    pydub = importlib.import_module("pydub")
    audio_segment_cls = pydub.AudioSegment

    if not bgm_path or not Path(bgm_path).exists():
        return vocal_segment

    bgm = audio_segment_cls.from_file(bgm_path)

    vocal_len = len(vocal_segment)
    fade_out_tail = bgm_fade_out_ms
    if len(bgm) < vocal_len + fade_out_tail:
        loops = (vocal_len + fade_out_tail) // len(bgm) + 1
        bgm = bgm * loops

    bgm = bgm[: vocal_len + fade_out_tail]
    bgm = bgm + bgm_volume_db
    bgm = bgm.fade_in(bgm_fade_in_ms).fade_out(bgm_fade_out_ms)

    vocal_padded = audio_segment_cls.silent(duration=vocal_pad_ms) + vocal_segment
    return bgm.overlay(vocal_padded)


async def run_loudnorm(
    input_path: Path,
    output_path: Path,
    *,
    loudnorm: str = "I=-16:LRA=11:TP=-1.5",
    sample_rate: int = 24000,
) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    ffmpeg = ffmpeg_bin if ffmpeg_bin else "ffmpeg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-af",
        f"loudnorm={loudnorm}",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        msg = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg loudnorm failed: {msg}")


def assemble_dialogue_audio(
    chunks: Sequence[DialogueChunk],
    segments: Sequence[Any],
    *,
    chunk_silence_base: int,
    vocal_pad_ms: int,
    silence_min: int,
    silence_max: int,
    silence_jitter: int,
) -> tuple[Any, list[tuple[float, float]]]:
    """Concatenate per-chunk audio with variable silence; return (combined, timestamps)."""
    if len(chunks) != len(segments):
        raise ValueError("chunks and segments length mismatch")

    pydub = importlib.import_module("pydub")
    audio_segment_cls = pydub.AudioSegment

    combined = audio_segment_cls.empty()
    timestamps: list[tuple[float, float]] = []
    current_time_ms = vocal_pad_ms

    for idx, seg in enumerate(segments):
        if idx > 0:
            silence_len = chunk_silence_ms(
                chunk_silence_base,
                silence_min=silence_min,
                silence_max=silence_max,
                silence_jitter=silence_jitter,
            )
            combined += audio_segment_cls.silent(duration=silence_len)
            current_time_ms += silence_len

        start_sec = current_time_ms / 1000.0
        duration_sec = len(seg) / 1000.0
        timestamps.append((start_sec, duration_sec))
        combined += seg
        current_time_ms += len(seg)

    return combined, timestamps


async def finalize_episode_mp3(
    combined: Any,
    output_path: Path,
    *,
    bgm_path: str | None,
    audio_cfg: dict,
    tmp_dir: Path,
) -> None:
    """BGM mix → export pre-norm MP3 → ffmpeg loudnorm → final MP3."""
    bgm_volume_db = int(audio_cfg.get("bgm_volume_db", -12))
    bgm_fade_in_ms = int(audio_cfg.get("bgm_fade_in_ms", 2000))
    bgm_fade_out_ms = int(audio_cfg.get("bgm_fade_out_ms", 3000))
    vocal_pad_ms = int(audio_cfg.get("vocal_pad_ms", 1000))
    loudnorm = str(audio_cfg.get("loudnorm", "I=-16:LRA=11:TP=-1.5"))
    sample_rate = int(audio_cfg.get("sample_rate", 24000))

    combined = mix_bgm(
        combined,
        bgm_path,
        bgm_volume_db=bgm_volume_db,
        bgm_fade_in_ms=bgm_fade_in_ms,
        bgm_fade_out_ms=bgm_fade_out_ms,
        vocal_pad_ms=vocal_pad_ms,
    )

    pre_norm = tmp_dir / "combined_prenorm.mp3"
    combined.export(str(pre_norm), format="mp3")
    await run_loudnorm(
        pre_norm,
        output_path,
        loudnorm=loudnorm,
        sample_rate=sample_rate,
    )
