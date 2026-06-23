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


def parse_dialogue_chunks(
    text: str, voices: Optional[Tuple[str, str]] = None
) -> list[DialogueChunk]:
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
                                vn_lower = voice_name.lower()
                                matched = False
                                if voices:
                                    voice_a = voices[0] if len(voices) > 0 else None
                                    voice_b = voices[1] if len(voices) > 1 else None
                                    if (
                                        isinstance(voice_a, str)
                                        and vn_lower == voice_a.strip().lower()
                                    ):
                                        host = "A"
                                        matched = True
                                    elif (
                                        isinstance(voice_b, str)
                                        and vn_lower == voice_b.strip().lower()
                                    ):
                                        host = "B"
                                        matched = True

                                if not matched:
                                    # 常见中文音色及标识兜底匹配
                                    if (
                                        "xiaoxiao" in vn_lower
                                        or "xiaoyi" in vn_lower
                                        or "host_b" in vn_lower
                                        or "host-b" in vn_lower
                                    ):
                                        host = "B"
                                    elif any(
                                        x in vn_lower
                                        for x in ("yunxi", "yunjian", "yunyang", "host_a", "host-a")
                                    ):
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


def _write_chunks_and_playlist(
    chunks: List[DialogueChunk],
    segments_by_variant: dict[str, List[Any]],  # Maps variant ID to list of AudioSegments
    timestamps: list[tuple[float, float]],
    voice_maps: dict[str, dict[str, str]],  # Maps variant ID to host voice map
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
            AudioSegment = getattr(pydub, "AudioSegment")
            silence_pad = AudioSegment.silent(duration=300)
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


def split_text_into_sentences(text: str, max_chars: int = 80) -> list[str]:
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

                def _gen_variant(var_name: str) -> Any:
                    chunk_segments = []
                    for s_idx, sentence in enumerate(sentences):
                        s_text = sentence.strip()
                        if not s_text:
                            continue
                        tensor = cosy_engine.synthesize_chunk(
                            text=s_text, host=chunk.host, variant=var_name
                        )
                        wav_path = tmp_root / f"chunk_{idx:03d}_{var_name}_{s_idx:03d}.wav"
                        torchaudio.save(str(wav_path), tensor, cv_cfg.sample_rate)
                        chunk_segments.append(AudioSegment.from_file(str(wav_path)))

                    if chunk_segments:
                        combined_chunk = chunk_segments[0]
                        for next_seg in chunk_segments[1:]:
                            combined_chunk += AudioSegment.silent(duration=150) + next_seg
                        return combined_chunk
                    else:
                        return AudioSegment.silent(duration=100)

                segments_by_variant[var].append(_gen_variant(var))

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
