import asyncio
import importlib
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from ai_news_podcast.pipeline.tts_postprocess import (
    assemble_dialogue_audio,
    finalize_episode_mp3,
)
from ai_news_podcast.pipeline.tts_types import DialogueChunk
from ai_news_podcast.text_utils import clean_tts_text

log = logging.getLogger(__name__)


def parse_dialogue_chunks(text: str, voices: Optional[Tuple[str, str]] = None) -> list[DialogueChunk]:
    """解析对话文本，支持标准的 SSML (XML/HTML) 格式和自定义 [Host A] / [Host B] 格式。"""
    stripped_text = text.strip()
    if stripped_text.startswith("<speak") or "<speak" in stripped_text or "<voice" in stripped_text:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")
            voice_tags = soup.find_all("voice")
            if voice_tags:
                chunks: list[DialogueChunk] = []
                for idx, voice_tag in enumerate(voice_tags):
                    voice_name = voice_tag.get("name", "").strip()
                    chunk_text = voice_tag.get_text().strip()
                    if chunk_text:
                        cleaned = clean_tts_text(chunk_text)
                        if cleaned:
                            # 默认根据 idx 交替分配
                            host = "B" if idx % 2 == 1 else "A"
                            if voice_name:
                                vn_lower = voice_name.strip().lower()
                                if voices:
                                    if vn_lower == voices[0].strip().lower():
                                        host = "A"
                                    elif vn_lower == voices[1].strip().lower():
                                        host = "B"
                                else:
                                    # 常见中文音色及标识兜底匹配
                                    if "xiaoxiao" in vn_lower or "xiaoyi" in vn_lower or "host_b" in vn_lower or "host-b" in vn_lower:
                                        host = "B"
                                    elif any(x in vn_lower for x in ("yunxi", "yunjian", "yunyang", "host_a", "host-a")):
                                        host = "A"
                            chunks.append(
                                DialogueChunk(
                                    host=host,
                                    text=cleaned,
                                    voice=voice_name if voice_name else None,
                                )
                            )
                if chunks:
                    return chunks
        except (ValueError, OSError):
            pass

    marker_re = re.compile(r"\[Host\s*([AB])\]", re.IGNORECASE)
    chunks: list[DialogueChunk] = []
    current_host = "A"  # Default
    cursor: int = 0

    for m in marker_re.finditer(text):
        # Pyre string slice bypass using simple casting and indexing
        start_idx = int(m.start())
        raw = str(text[cursor:start_idx]).strip()
        if raw:
            cleaned = clean_tts_text(raw)
            if cleaned:
                chunks.append(DialogueChunk(host=current_host, text=cleaned))
        current_host = str(m.group(1)).strip().upper()
        cursor = int(m.end())

    tail = str(text[cursor:]).strip()
    if tail:
        cleaned = clean_tts_text(tail)
        if cleaned:
            chunks.append(DialogueChunk(host=current_host, text=cleaned))
    return chunks


async def synthesize_edge_tts(
    chunks: List[DialogueChunk],
    voices: Tuple[str, str],
    output_path: Union[str, Path],
    *,
    bgm_path: Optional[str] = None,
    volume: str = "+0%",
    rate: str = "+15%",  # News broadcast usually faster
    transcript_path: Optional[Union[str, Path]] = None,
    cfg: Optional[dict] = None,
) -> None:
    edge_tts = importlib.import_module("edge_tts")
    pydub = importlib.import_module("pydub")
    AudioSegment = getattr(pydub, "AudioSegment")

    # Extract audio config with defaults matching original hardcoded values
    audio_cfg = (cfg or {}).get("tts", {}).get("audio", {})
    _vocal_pad_ms = int(audio_cfg.get("vocal_pad_ms", 1000))
    _chunk_silence_base = int(audio_cfg.get("chunk_silence_base", 300))
    _silence_min = int(audio_cfg.get("chunk_silence_min", 400))
    _silence_max = int(audio_cfg.get("chunk_silence_max", 800))
    _silence_jitter = int(audio_cfg.get("chunk_silence_jitter", 100))

    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if not chunks:
        raise ValueError("No TTS chunks to synthesize")

    # Mapping Host A -> voices[0], Host B -> voices[1]
    voice_map = {
        "A": voices[0],
        "B": voices[1] if len(voices) > 1 else voices[0],
    }

    async def _save_chunk_with_retry(
        idx: int,
        chunk: DialogueChunk,
        voice: str,
        tmp_chunk: Path,
    ) -> Path:
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
                return tmp_chunk
            except BaseException as exc:
                last_err = exc
                if tmp_chunk.exists():
                    tmp_chunk.unlink()
                if attempt < 3:
                    await asyncio.sleep(attempt)
        if last_err is not None:
            raise last_err
        raise RuntimeError("edge-tts chunk synthesis failed")

    with tempfile.TemporaryDirectory(prefix="tts-edge-dialogue-") as tmp_dir:
        tmp_root = Path(tmp_dir)

        tasks = []
        for idx, chunk in enumerate(chunks, start=1):
            voice = chunk.voice or voice_map.get(chunk.host, voices[0])
            tmp_chunk = tmp_root / f"chunk_{idx:03d}.mp3"
            tasks.append(_save_chunk_with_retry(idx, chunk, voice, tmp_chunk))

        chunk_files = await asyncio.gather(*tasks)
        segments = [AudioSegment.from_file(str(chunk_file)) for chunk_file in chunk_files]
        combined, timestamps = assemble_dialogue_audio(
            chunks,
            segments,
            chunk_silence_base=_chunk_silence_base,
            vocal_pad_ms=_vocal_pad_ms,
            silence_min=_silence_min,
            silence_max=_silence_max,
            silence_jitter=_silence_jitter,
        )
        await finalize_episode_mp3(
            combined,
            final_path,
            bgm_path=bgm_path,
            audio_cfg=audio_cfg,
            tmp_dir=tmp_root,
        )

        if transcript_path:
            _write_transcript_with_timestamps(
                chunks=chunks,
                timestamps=timestamps,
                voice_map={
                    "A": voices[0],
                    "B": voices[1] if len(voices) > 1 else voices[0],
                },
                transcript_path=Path(transcript_path),
            )


def _write_transcript_with_timestamps(
    *,
    chunks: List[DialogueChunk],
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


def split_text_into_sentences(text: str, max_chars: int = 40) -> list[str]:
    """将文本切分为较短的句子/短句，避免单次合成文本过长导致 CosyVoice 截断或语速失真。"""
    # 按照常见的标点符号进行切分，保留标点
    pattern = re.compile(r"([^，。！？；、,.!?;\s]+[，。！？；、,.!?;\s]*)")
    parts = pattern.findall(text)
    if not parts:
        return [text] if text.strip() else []

    sentences = []
    current = ""
    for part in parts:
        if len(current) + len(part) <= max_chars:
            current += part
        else:
            if current:
                sentences.append(current.strip())
            current = part
    if current:
        sentences.append(current.strip())
    return sentences


async def synthesize_cosyvoice2(
    chunks: List[DialogueChunk],
    output_path: Union[str, Path],
    *,
    bgm_path: Optional[str] = None,
    transcript_path: Optional[Union[str, Path]] = None,
    cfg: Optional[dict] = None,
    project_root: Optional[Path] = None,
    engine: Optional[Any] = None,
) -> None:
    from ai_news_podcast.pipeline.cosyvoice_backend import CosyVoice2Engine, load_cosyvoice_config

    torchaudio = importlib.import_module("torchaudio")
    pydub = importlib.import_module("pydub")
    AudioSegment = getattr(pydub, "AudioSegment")

    audio_cfg = (cfg or {}).get("tts", {}).get("audio", {})
    root = project_root or Path.cwd()
    cv_cfg = load_cosyvoice_config(cfg or {}, project_root=root)
    cosy_engine = engine or CosyVoice2Engine(cv_cfg)

    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    voice_map = {"A": "host_a", "B": "host_b"}

    with tempfile.TemporaryDirectory(prefix="tts-cosyvoice-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        segments = []
        for idx, chunk in enumerate(chunks, start=1):
            sentences = split_text_into_sentences(chunk.text, max_chars=40)
            chunk_segments = []
            for s_idx, sentence in enumerate(sentences):
                s_text = sentence.strip()
                if not s_text:
                    continue
                tensor = cosy_engine.synthesize_chunk(text=s_text, host=chunk.host)
                wav_path = tmp_root / f"chunk_{idx:03d}_{s_idx:03d}.wav"
                torchaudio.save(str(wav_path), tensor, cv_cfg.sample_rate)
                chunk_segments.append(AudioSegment.from_file(str(wav_path)))

            if chunk_segments:
                combined_chunk = chunk_segments[0]
                for next_seg in chunk_segments[1:]:
                    combined_chunk += AudioSegment.silent(duration=150) + next_seg
                segments.append(combined_chunk)
            else:
                segments.append(AudioSegment.silent(duration=100))

        combined, timestamps = assemble_dialogue_audio(
            chunks,
            segments,
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

        if transcript_path:
            _write_transcript_with_timestamps(
                chunks=chunks,
                timestamps=timestamps,
                voice_map=voice_map,
                transcript_path=Path(transcript_path),
            )


async def synthesize(
    text: str,
    *,
    backend: str = "edge-tts",
    voices: Tuple[str, str] = ("zh-CN-YunjianNeural", "zh-CN-XiaoxiaoNeural"),
    output_path: Union[str, Path],
    bgm_path: Optional[str] = None,
    **kwargs: Any,
) -> None:
    chunks = parse_dialogue_chunks(text, voices=voices)
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
    edge_kwargs = {
        "bgm_path": bgm_path,
        "volume": str(kwargs.get("volume") or "+0%"),
        "rate": str(kwargs.get("rate") or "+10%"),
        "transcript_path": kwargs.get("transcript_path"),
        "cfg": kwargs.get("cfg"),
    }

    if backend_name in ("edge-tts", "edge"):
        await synthesize_edge_tts(chunks, voices=voices, output_path=output_path, **edge_kwargs)
        return

    if backend_name in ("cosyvoice2", "cosyvoice"):
        await synthesize_cosyvoice2(chunks, output_path=output_path, **cosy_kwargs)
        return

    if backend_name == "hybrid":
        try:
            await synthesize_cosyvoice2(chunks, output_path=output_path, **cosy_kwargs)
        except Exception:
            log.warning("CosyVoice2 failed, falling back to edge-tts", exc_info=True)
            await synthesize_edge_tts(chunks, voices=voices, output_path=output_path, **edge_kwargs)
        return

    raise ValueError(f"Unsupported TTS backend: {backend}")
