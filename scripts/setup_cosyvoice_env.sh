#!/usr/bin/env bash
# Install CosyVoice2 CPU environment (GHA + local). Not part of uv/pyproject deps.
set -euo pipefail

COSY_SRC="${COSYVOICE_SRC:-$HOME/cosyvoice_src}"
COSY_MODELS="${COSYVOICE_MODELS:-$HOME/cosyvoice_models}"
MODEL_DIR="$COSY_MODELS/CosyVoice2-0.5B"

if [ ! -d "$COSY_SRC/cosyvoice" ]; then
  git clone --recursive --depth=1 https://github.com/FunAudioLLM/CosyVoice.git "$COSY_SRC"
fi

if command -v uv >/dev/null 2>&1 && [ -d ".venv" ]; then
  PIP=(uv pip)
  PYTHON=(uv run python)
else
  PIP=(pip)
  PYTHON=(python)
fi

"${PIP[@]}" install --upgrade pip setuptools
"${PIP[@]}" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
"${PIP[@]}" install conformer==0.3.2 diffusers==0.29.0 hydra-core==1.3.2 HyperPyYAML==1.2.3 \
  inflect librosa omegaconf onnx onnxruntime openai-whisper protobuf pyworld \
  rich soundfile transformers x-transformers wetext huggingface_hub

export PYTHONPATH="$COSY_SRC:$COSY_SRC/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"

"${PYTHON[@]}" - <<EOF
from huggingface_hub import snapshot_download
import os

local_dir = os.path.expanduser("$MODEL_DIR")
if not os.path.exists(os.path.join(local_dir, "cosyvoice.yaml")):
    snapshot_download(
        "FunAudioLLM/CosyVoice2-0.5B",
        local_dir=local_dir,
        ignore_patterns=["*.bin"],
    )
    print(f"Downloaded model to {local_dir}")
else:
    print(f"Model already present at {local_dir}")
EOF

echo "COSYVOICE_MODEL_DIR=$MODEL_DIR"
