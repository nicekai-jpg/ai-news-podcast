import asyncio
import asyncio.subprocess
import importlib
import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union


@dataclass(frozen=True)
class DialogueChunk:
    host: str
    text: str


def _clean_tts_text(text: str) -> str:
    """Remove annotation markers and non-speakable elements before TTS."""
    # Remove tags produced by models
    text = re.sub(r"\[(?:FACT|INFERENCE|OPINION)\]\s*", "", text)
    # Remove mood tags left behind (if any)
    text = re.sub(r"\[mood:[^\]]+\]", "", text)
    # Remove any leftover square-bracket annotations like [], [xxx], except Host tags
    text = re.sub(r"\[(?!(Host\s*A|Host\s*B))[^\]]*\]", "", text, flags=re.IGNORECASE)
    # Remove parenthetical annotations
    text = re.sub(
        r"[（(]\s*(?:doge|狗头|笑|手动狗头|bushi|划掉)\s*[）)]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Replace fancy quotes
    text = text.replace("「", "").replace("」", "").replace("『", "").replace("』", "")
    text = re.sub(r"[【】]", "", text)
    # 规范化空格和换行
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def parse_dialogue_chunks(text: str) -> list[DialogueChunk]:
    """解析带有 [Host A] 和 [Host B] 的对话文本。"""
    marker_re = re.compile(r"\[Host\s*([AB])\]", re.IGNORECASE)
    chunks: list[DialogueChunk] = []
    current_host = "A"  # Default
    cursor: int = 0

    for m in marker_re.finditer(text):
        # Pyre string slice bypass using simple casting and indexing
        start_idx = int(m.start())
        raw = str(text[cursor:start_idx]).strip()
        if raw:
            cleaned = _clean_tts_text(raw)
            if cleaned:
                chunks.append(DialogueChunk(host=current_host, text=cleaned))
        current_host = str(m.group(1)).strip().upper()
        cursor = int(m.end())

    tail = str(text[cursor:]).strip()
    if tail:
        cleaned = _clean_tts_text(tail)
        if cleaned:
            chunks.append(DialogueChunk(host=current_host, text=cleaned))
    return chunks


def _chunk_silence_ms(chunk_silence_ms: int) -> int:
    base = max(400, min(800, int(chunk_silence_ms)))
    low = max(400, base - 100)
    high = min(800, base + 100)
    return random.randint(low, high)


def _mix_bgm(vocal_segment: Any, bgm_path: Optional[str]) -> Any:
    """Mix vocal track with background music using basic ducking."""
    pydub = importlib.import_module("pydub")
    AudioSegment = getattr(pydub, "AudioSegment")

    if not bgm_path or not Path(bgm_path).exists():
        return vocal_segment

    bgm = AudioSegment.from_file(bgm_path)

    # 调整 BGM：使其和人声一样长，甚至更长一点用于淡出
    vocal_len = len(vocal_segment)
    if len(bgm) < vocal_len + 3000:
        # Loop bgm
        loops = (vocal_len + 3000) // len(bgm) + 1
        bgm = bgm * loops

    bgm = bgm[: vocal_len + 3000]  # 给结尾留3秒尾奏

    # 基础音量调节
    bgm = bgm - 12  # 基本降低 BGM 音量，使其成为垫乐

    # 将 BGM 在两秒内淡入
    bgm = bgm.fade_in(2000).fade_out(3000)

    # Overlay vocals on to BGM (1秒后开始播报，给BGM一个前奏)
    vocal_padded = AudioSegment.silent(duration=1000) + vocal_segment
    # 对整体进行混音
    combined = bgm.overlay(vocal_padded)
    return combined


async def _run_loudnorm(input_path: Path, output_path: Path) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    ffmpeg = ffmpeg_bin if ffmpeg_bin else "ffmpeg"
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
    chunks: List[DialogueChunk],
    voices: Tuple[str, str],
    output_path: Union[str, Path],
    *,
    bgm_path: Optional[str] = None,
    volume: str = "+0%",
    rate: str = "+15%",  # News broadcast usually faster
) -> None:
    edge_tts = importlib.import_module("edge_tts")
    pydub = importlib.import_module("pydub")
    AudioSegment = getattr(pydub, "AudioSegment")

    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if not chunks:
        raise ValueError("No TTS chunks to synthesize")

    # Mapping Host A -> voices[0], Host B -> voices[1]
    voice_map = {
        "A": voices[0],
        "B": voices[1] if len(voices) > 1 else voices[0],
    }

    with tempfile.TemporaryDirectory(prefix="tts-edge-dialogue-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        chunk_files: list[Path] = []

        for idx, chunk in enumerate(chunks, start=1):
            voice = voice_map.get(chunk.host, voices[0])
            # A bit of variation in pitch implicitly separates the voices, but distinct TTS models are better.
            tmp_chunk = tmp_root / f"chunk_{idx:03d}.mp3"

            last_err: Optional[BaseException] = None
            for attempt in range(1, 4):
                comm = edge_tts.Communicate(
                    chunk.text,
                    voice=voice,
                    rate=rate,
                    volume=volume,
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
                # Add distinct silence for breathing room between conversational turns
                combined += AudioSegment.silent(duration=_chunk_silence_ms(300))
            combined += seg

        # Mixing with BGM
        combined = _mix_bgm(combined, bgm_path)

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


async def synthesize(
    text: str,
    *,
    backend: str = "edge-tts",
    voices: Tuple[str, str] = ("zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural"),
    output_path: Union[str, Path],
    bgm_path: Optional[str] = None,
    **kwargs: Any,
) -> None:
    chunks = parse_dialogue_chunks(text)
    if not chunks:
        raise ValueError("Input text is empty after dialogue parsing")

    backend_name = str(backend).strip().lower()
    if backend_name == "edge-tts":
        await synthesize_edge_tts(
            chunks,
            voices=voices,
            output_path=output_path,
            bgm_path=bgm_path,
            volume=str(kwargs.get("volume") or "+0%"),
            rate=str(kwargs.get("rate") or "+10%"),
        )
        return

    raise ValueError(
        f"Unsupported TTS backend: {backend}. Dual-voice engine currently supports only edge-tts natively."
    )
