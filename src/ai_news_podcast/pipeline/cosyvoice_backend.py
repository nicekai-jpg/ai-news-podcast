"""CosyVoice 2 zero-shot TTS backend (lazy import — not in main pyproject deps)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CosyVoiceConfig:
    model_dir: Path
    host_a_wav: Path
    host_a_text: str
    host_b_wav: Path
    host_b_text: str
    sample_rate: int = 22050


def load_cosyvoice_config(cfg: dict, *, project_root: Path) -> CosyVoiceConfig:
    cosy = (cfg.get("tts") or {}).get("cosyvoice") or {}
    refs = cosy.get("ref_audio") or {}

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else project_root / path

    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    host_a_wav = _resolve(str(refs.get("host_a") or "asset/refs/host_a_ref.wav"))
    host_b_wav = _resolve(str(refs.get("host_b") or "asset/refs/host_b_ref.wav"))
    host_a_text_path = _resolve(str(refs.get("host_a_text") or "asset/refs/host_a_ref.txt"))
    host_b_text_path = _resolve(str(refs.get("host_b_text") or "asset/refs/host_b_ref.txt"))

    model_dir_raw = str(cosy.get("model_dir") or os.environ.get("COSYVOICE_MODEL_DIR") or "").strip()
    model_dir = Path(model_dir_raw).expanduser() if model_dir_raw else Path()

    return CosyVoiceConfig(
        model_dir=model_dir,
        host_a_wav=host_a_wav,
        host_a_text=_read_text(host_a_text_path),
        host_b_wav=host_b_wav,
        host_b_text=_read_text(host_b_text_path),
    )


class CosyVoice2Engine:
    """Wrap CosyVoice2 model; load once, synthesize many chunks."""

    def __init__(self, config: CosyVoiceConfig):
        self._config = config
        self._model: Any = None
        self._ref_cache: dict[str, Any] = {}

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self._config.model_dir or not self._config.model_dir.exists():
            raise FileNotFoundError(
                f"CosyVoice model not found at {self._config.model_dir}. "
                "Set tts.cosyvoice.model_dir or COSYVOICE_MODEL_DIR."
            )
        from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore[import-untyped]

        self._model = CosyVoice2(
            str(self._config.model_dir),
            load_jit=False,
            load_trt=False,
        )
        return self._model

    def _load_ref(self, wav_path: Path) -> Any:
        key = str(wav_path.resolve())
        if key not in self._ref_cache:
            from cosyvoice.utils.file_utils import load_wav  # type: ignore[import-untyped]

            self._ref_cache[key] = load_wav(str(wav_path), 16000)
        return self._ref_cache[key]

    def synthesize_chunk(self, *, text: str, host: str) -> Any:
        """Return torch Tensor audio (1, T) at model sample rate."""
        if host.upper() == "B":
            ref_wav, ref_text = self._config.host_b_wav, self._config.host_b_text
        else:
            ref_wav, ref_text = self._config.host_a_wav, self._config.host_a_text

        if not ref_wav.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_wav}")

        model = self._ensure_model()
        output = next(model.inference_zero_shot(text, ref_text, str(ref_wav), stream=False))
        return output["tts_speech"]
