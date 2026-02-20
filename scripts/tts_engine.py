import asyncio
import importlib
import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


MOOD_PRESETS: dict[str, dict[str, str]] = {
    "hook": {
        "rate": "+8%",
        "pitch": "+8Hz",
        "instruct": "用吸引注意力的悬念语气说",
    },
    "excited": {
        "rate": "+15%",
        "pitch": "+10Hz",
        "instruct": "用兴奋的语气播报",
    },
    "serious": {
        "rate": "-5%",
        "pitch": "-5Hz",
        "instruct": "用严肃的新闻主播语气说",
    },
    "calm": {
        "rate": "+0%",
        "pitch": "-2Hz",
        "instruct": "用平稳冷静的语气叙述",
    },
    "emphasis": {
        "rate": "-10%",
        "pitch": "+5Hz",
        "instruct": "放慢语速，加重语气强调",
    },
    "closing": {
        "rate": "-3%",
        "pitch": "-3Hz",
        "instruct": "用收束总结的语气说",
    },
}


@dataclass(frozen=True)
class MoodChunk:
    mood: str
    text: str


def _clean_tts_text(text: str) -> str:
    """Remove annotation markers and non-speakable elements before TTS."""
    # Remove [FACT] / [INFERENCE] / [OPINION] tags
    text = re.sub(r"\[(?:FACT|INFERENCE|OPINION)\]\s*", "", text)
    # Remove any leftover square-bracket annotations like [], [xxx]
    text = re.sub(r"\[\s*\]", "", text)
    # Remove parenthetical annotations: （doge）（狗头）（笑）etc.
    text = re.sub(
        r"[（(]\s*(?:doge|狗头|笑|手动狗头|bushi|划掉)\s*[）)]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Replace fancy quotes「」『』with nothing (TTS reads them as pauses)
    text = text.replace("「", "").replace("」", "").replace("『", "").replace("』", "")
    # Replace English parentheses/brackets used decoratively
    text = re.sub(r"[【】]", "", text)
    # Collapse multiple spaces / newlines into single space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def parse_mood_chunks(text: str) -> list[MoodChunk]:
    marker_re = re.compile(r"\[mood:([a-zA-Z0-9_-]+)\]")
    chunks: list[MoodChunk] = []
    current_mood = "calm"
    cursor = 0

    for m in marker_re.finditer(text):
        raw = text[cursor : m.start()].strip()
        if raw:
            cleaned = _clean_tts_text(raw)
            if cleaned:
                chunks.append(MoodChunk(mood=current_mood, text=cleaned))
        current_mood = m.group(1).strip().lower() or "calm"
        cursor = m.end()

    tail = text[cursor:].strip()
    if tail:
        cleaned = _clean_tts_text(tail)
        if cleaned:
            chunks.append(MoodChunk(mood=current_mood, text=cleaned))
    return chunks


def _resolve_mood_presets(
    mood_presets: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {k: dict(v) for k, v in MOOD_PRESETS.items()}
    for mood, settings in (mood_presets or {}).items():
        key = str(mood).strip().lower()
        if not key:
            continue
        base = merged.get(key, {})
        if isinstance(settings, dict):
            for field in ("rate", "pitch", "instruct"):
                value = settings.get(field)
                if value is not None:
                    base[field] = str(value)
        merged[key] = base
    return merged


def _chunk_silence_ms(chunk_silence_ms: int) -> int:
    base = max(400, min(800, int(chunk_silence_ms)))
    low = max(400, base - 100)
    high = min(800, base + 100)
    return random.randint(low, high)


async def _run_loudnorm(input_path: Path, output_path: Path) -> None:
    ffmpeg_bin = Path("/opt/homebrew/bin/ffmpeg")
    ffmpeg = str(ffmpeg_bin if ffmpeg_bin.exists() else "ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-af",
        "loudnorm=I=-16:LRA=11:TP=-1.5",
        "-ar",
        "24000",
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


async def synthesize_edge_tts(
    chunks: List[MoodChunk],
    voice: str,
    output_path: Union[str, Path],
    *,
    volume: str = "+0%",
    default_rate: str = "+0%",
    default_pitch: str = "+0Hz",
    mood_presets: Optional[Dict[str, Any]] = None,
    chunk_silence_ms: int = 500,
) -> None:
    edge_tts = importlib.import_module("edge_tts")
    pydub = importlib.import_module("pydub")
    AudioSegment = getattr(pydub, "AudioSegment")

    merged_presets = _resolve_mood_presets(mood_presets)
    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if not chunks:
        raise ValueError("No TTS chunks to synthesize")

    with tempfile.TemporaryDirectory(prefix="tts-edge-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        chunk_files: list[Path] = []

        for idx, chunk in enumerate(chunks, start=1):
            preset = merged_presets.get(chunk.mood, merged_presets.get("calm", {}))
            rate = str(preset.get("rate") or default_rate)
            pitch = str(preset.get("pitch") or default_pitch)
            tmp_chunk = tmp_root / f"chunk_{idx:03d}.mp3"

            last_err: Optional[BaseException] = None
            for attempt in range(1, 4):
                comm = edge_tts.Communicate(
                    chunk.text,
                    voice=voice,
                    rate=rate,
                    volume=volume,
                    pitch=pitch,
                )
                try:
                    await comm.save(str(tmp_chunk))
                    if not tmp_chunk.exists() or tmp_chunk.stat().st_size == 0:
                        raise RuntimeError("edge-tts produced empty chunk")
                    chunk_files.append(tmp_chunk)
                    break
                except BaseException as exc:
                    last_err = exc
                    if tmp_chunk.exists():
                        tmp_chunk.unlink()
                    if attempt < 3:
                        await asyncio.sleep(attempt)
            else:
                if last_err is not None:
                    raise last_err
                raise RuntimeError("edge-tts chunk synthesis failed")

        combined = AudioSegment.empty()
        for idx, chunk_file in enumerate(chunk_files):
            seg = AudioSegment.from_file(str(chunk_file))
            if idx > 0:
                combined += AudioSegment.silent(
                    duration=_chunk_silence_ms(chunk_silence_ms)
                )
            combined += seg

        pre_norm = tmp_root / "combined_prenorm.mp3"
        combined.export(str(pre_norm), format="mp3")
        await _run_loudnorm(pre_norm, final_path)


def _to_audio_segment_from_cosy_output(result: Any, sample_rate: int = 24000) -> Any:
    pydub = importlib.import_module("pydub")
    AudioSegment = getattr(pydub, "AudioSegment")

    if isinstance(result, AudioSegment):
        return result
    if isinstance(result, (str, Path)):
        return AudioSegment.from_file(str(result))
    if isinstance(result, bytes):
        return AudioSegment.from_file(BytesIO(result), format="wav")
    if isinstance(result, dict):
        audio = result.get("audio")
        if isinstance(audio, (str, Path, bytes)):
            return _to_audio_segment_from_cosy_output(audio, sample_rate=sample_rate)

    raise TypeError("Unsupported CosyVoice output type for audio concatenation")


async def synthesize_cosyvoice(
    chunks: List[MoodChunk],
    model_path: str,
    speaker: str,
    output_path: Union[str, Path],
    *,
    mood_presets: Optional[Dict[str, Any]] = None,
    chunk_silence_ms: int = 500,
) -> None:
    try:
        cosyvoice = importlib.import_module("cosyvoice")
    except ImportError as exc:
        raise ImportError(
            "CosyVoice backend selected but 'cosyvoice' is not installed. "
            "Install CosyVoice and related runtime dependencies first."
        ) from exc

    if not chunks:
        raise ValueError("No TTS chunks to synthesize")

    CosyVoice = getattr(cosyvoice, "CosyVoice", None)
    if CosyVoice is None:
        raise ImportError(
            "Installed 'cosyvoice' module does not expose CosyVoice class."
        )

    model = CosyVoice(model_path)
    merged_presets = _resolve_mood_presets(mood_presets)
    pydub = importlib.import_module("pydub")
    AudioSegment = getattr(pydub, "AudioSegment")

    with tempfile.TemporaryDirectory(prefix="tts-cosy-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        combined = AudioSegment.empty()
        for idx, chunk in enumerate(chunks):
            preset = merged_presets.get(chunk.mood, merged_presets.get("calm", {}))
            instruct_text = str(preset.get("instruct") or "用平稳语气叙述")

            result = model.inference_instruct(chunk.text, instruct_text, speaker)
            segment = _to_audio_segment_from_cosy_output(result)

            if idx > 0:
                combined += AudioSegment.silent(
                    duration=_chunk_silence_ms(chunk_silence_ms)
                )
            combined += segment

        pre_norm = tmp_root / "combined_prenorm.mp3"
        combined.export(str(pre_norm), format="mp3")
        await _run_loudnorm(pre_norm, Path(output_path))


async def synthesize(
    text: str,
    *,
    backend: str,
    voice: str,
    output_path: Union[str, Path],
    **kwargs: Any,
) -> None:
    chunks = parse_mood_chunks(text)
    if not chunks:
        raise ValueError("Input text is empty after mood parsing")

    backend_name = str(backend or "edge-tts").strip().lower()
    if backend_name == "edge-tts":
        await synthesize_edge_tts(
            chunks,
            voice,
            output_path,
            volume=str(kwargs.get("volume") or "+0%"),
            default_rate=str(kwargs.get("rate") or "+0%"),
            default_pitch=str(kwargs.get("pitch") or "+0Hz"),
            mood_presets=kwargs.get("mood_presets"),
            chunk_silence_ms=int(kwargs.get("chunk_silence_ms") or 500),
        )
        return

    if backend_name == "cosyvoice":
        await synthesize_cosyvoice(
            chunks,
            model_path=str(kwargs.get("model_path") or ""),
            speaker=str(kwargs.get("speaker") or ""),
            output_path=output_path,
            mood_presets=kwargs.get("mood_presets"),
            chunk_silence_ms=int(kwargs.get("chunk_silence_ms") or 500),
        )
        return

    raise ValueError(f"Unsupported TTS backend: {backend}")
