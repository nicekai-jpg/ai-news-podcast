#!/usr/bin/env bash
# Install CosyVoice2 CPU environment (GHA + local). Isolated venv — not uv/pyproject deps.
set -euo pipefail

COSY_SRC="${COSYVOICE_SRC:-$HOME/cosyvoice_src}"
COSY_MODELS="${COSYVOICE_MODELS:-$HOME/cosyvoice_models}"
COSY_VENV="${COSYVOICE_VENV:-$HOME/cosyvoice_venv}"
MODEL_DIR="$COSY_MODELS/CosyVoice2-0.5B"

if [ ! -d "$COSY_SRC/cosyvoice" ]; then
  git clone --recursive --depth=1 https://github.com/FunAudioLLM/CosyVoice.git "$COSY_SRC"
fi

if [ ! -x "$COSY_VENV/bin/python" ]; then
  python3 -m venv "$COSY_VENV"
fi

PIP=("$COSY_VENV/bin/pip")
PYTHON=("$COSY_VENV/bin/python")

"${PIP[@]}" install --upgrade "pip>=24" "setuptools>=69,<81" wheel
# Match CosyVoice upstream torch pins — avoid latest torch breaking onnxruntime wheels.
"${PIP[@]}" install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu

REQ_FILE="$COSY_SRC/requirements.txt"
if [ -f "$REQ_FILE" ]; then
  grep -vE '^(deepspeed|onnxruntime-gpu|tensorrt|openai-whisper|--extra-index-url|torch==|torchaudio==|gradio|fastapi|uvicorn|grpcio|tensorboard)' \
    "$REQ_FILE" > /tmp/cosyvoice-cpu-reqs.txt
  "${PIP[@]}" install -r /tmp/cosyvoice-cpu-reqs.txt
else
  "${PIP[@]}" install conformer==0.3.2 diffusers==0.29.0 hydra-core==1.3.2 HyperPyYAML==1.2.3 \
    inflect==7.3.1 librosa==0.10.2 lightning==2.2.4 modelscope==1.20.0 numpy==1.26.4 \
    omegaconf==2.3.0 onnx==1.16.0 onnxruntime==1.18.0 protobuf==4.25 \
    pyworld==0.3.4 rich==13.7.1 soundfile==0.12.1 transformers==4.51.3 x-transformers==2.11.24 \
    wetext==0.0.4 huggingface_hub gdown matplotlib wget pyarrow pydantic networkx
fi
# requirements.txt only lists onnxruntime for darwin/win32.
"${PIP[@]}" install "onnxruntime==1.18.0"
# frontend.py imports whisper; zero-shot path uses whisper.log_mel_spectrogram.
"${PIP[@]}" install "openai-whisper==20231117" --no-build-isolation

# Minimal project imports for gha_tts_cosyvoice.py (keep CosyVoice deps isolated from uv).
"${PIP[@]}" install pydub==0.25.1 PyYAML==6.0.1 "httpx>=0.27" "openai>=1.0" "python-dotenv>=1.2.1" "pydantic>=2.12.5" "tenacity>=8.2"
"${PIP[@]}" install --no-deps -e .

export PYTHONPATH="$COSY_SRC:$COSY_SRC/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"

"${PYTHON[@]}" - <<'EOF'
import onnxruntime
print(f"onnxruntime {onnxruntime.__version__} ok")
EOF

"${PYTHON[@]}" - <<EOF
from huggingface_hub import snapshot_download
import os

local_dir = os.path.expanduser("$MODEL_DIR")
if not os.path.exists(os.path.join(local_dir, "cosyvoice2.yaml")):
    snapshot_download(
        "FunAudioLLM/CosyVoice2-0.5B",
        local_dir=local_dir,
        ignore_patterns=["*.bin"],
    )
    print(f"Downloaded model to {local_dir}")
else:
    print(f"Model already present at {local_dir}")
EOF

echo "COSYVOICE_VENV=$COSY_VENV"
echo "COSYVOICE_MODEL_DIR=$MODEL_DIR"
