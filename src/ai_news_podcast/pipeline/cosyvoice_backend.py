"""CosyVoice 2 zero-shot TTS backend (lazy import — not in main pyproject deps)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CosyVoiceConfig:
    model_dir: Path
    refs: dict[str, dict[str, tuple[Path, str]]]
    sample_rate: int = 22050


def load_cosyvoice_config(cfg: dict, *, project_root: Path) -> CosyVoiceConfig:
    cosy = (cfg.get("tts") or {}).get("cosyvoice") or {}
    refs = cosy.get("ref_audio") or {}

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else project_root / path

    def _read_text(path: Path) -> str:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""

    parsed_refs = {"A": {}, "B": {}}
    for host, host_key in [("A", "host_a"), ("B", "host_b")]:
        host_data = refs.get(host_key)
        if isinstance(host_data, dict):
            for variant in ["v1", "v2"]:
                wav_path = _resolve(str(host_data.get(variant) or ""))
                text_path = _resolve(str(host_data.get(f"{variant}_text") or ""))
                parsed_refs[host][variant] = (wav_path, _read_text(text_path))
        else:
            # Fallback for single-voice setup
            wav_path = _resolve(str(host_data or f"assets/refs/host_{host.lower()}_ref.wav"))
            text_path = _resolve(str(refs.get(f"{host_key}_text") or f"assets/refs/host_{host.lower()}_ref.txt"))
            txt_content = _read_text(text_path)
            parsed_refs[host]["v1"] = (wav_path, txt_content)
            parsed_refs[host]["v2"] = (wav_path, txt_content)

    model_dir_raw = str(cosy.get("model_dir") or os.environ.get("COSYVOICE_MODEL_DIR") or "").strip()
    model_dir = Path(model_dir_raw).expanduser() if model_dir_raw else Path()

    return CosyVoiceConfig(
        model_dir=model_dir,
        refs=parsed_refs,
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

    def synthesize_chunk(self, *, text: str, host: str, variant: str = "v1") -> Any:
        """Return torch Tensor audio (1, T) at model sample rate."""
        host_key = host.upper()
        variant_data = self._config.refs.get(host_key, {}).get(variant)
        if not variant_data:
            raise ValueError(f"No reference audio configured for Host {host_key} variant {variant}")

        ref_wav, ref_text = variant_data
        if not ref_wav.exists():
            # Fallback to v1 if file for variant is missing
            fallback_data = self._config.refs.get(host_key, {}).get("v1")
            if fallback_data:
                ref_wav, ref_text = fallback_data

        if not ref_wav.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_wav}")

        model = self._ensure_model()
        output = next(model.inference_zero_shot(text, ref_text, str(ref_wav), stream=False))
        return output["tts_speech"]
