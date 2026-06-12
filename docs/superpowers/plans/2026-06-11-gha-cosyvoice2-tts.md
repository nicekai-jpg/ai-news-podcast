# CosyVoice 2 单模型 GHA TTS 替换实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 CosyVoice 2 零样本克隆替换 Edge-TTS 作为播客主线 TTS，在 GitHub Actions 上完成每日自动合成，保留 Edge-TTS 作失败降级。

**Architecture:** 将 `daily.yml` 拆为 `content → tts → deploy` 三个 Job。Job 1 用现有 uv 流水线生成剧本与站点骨架（`--no-audio`）；Job 2 在独立 Python 3.10 环境加载 CosyVoice2-0.5B，逐句推理后复用现有 BGM 混音与 loudnorm 后处理；Job 3 更新 RSS/站点并部署 gh-pages。`tts_engine.py` 新增 `cosyvoice2` backend，本地与 GHA 共用同一合成入口。

**Tech Stack:** CosyVoice2-0.5B (PyTorch CPU)、HuggingFace Hub、ffmpeg、pydub、现有 `parse_dialogue_chunks` / `_mix_bgm` / `_run_loudnorm`

**评测依据:** `docs/tts_complete_guide.md` 对比矩阵 — CosyVoice 2 韵律最佳、吞字极低；`.github/workflows/tts-benchmark.yml` 已验证 GHA `ubuntu-latest` 可跑。

**部署踩坑记录:** 见 [`docs/gha_cosyvoice2_deployment_log.md`](../gha_cosyvoice2_deployment_log.md)（GHA 实际问题、修复对照表、当前阻塞点）。

---

## 文件结构（实施前预览）

| 文件 | 职责 |
|------|------|
| `asset/refs/host_a_ref.wav` + `.txt` | Host A（博文/男）参考音频与对齐文本 |
| `asset/refs/host_b_ref.wav` + `.txt` | Host B（晓晓/女）参考音频与对齐文本 |
| `src/ai_news_podcast/pipeline/tts_postprocess.py` | 从 `tts_engine.py` 抽出的拼接/BGM/loudnorm 共用逻辑 |
| `src/ai_news_podcast/pipeline/cosyvoice_backend.py` | CosyVoice2 模型加载与单句推理 |
| `src/ai_news_podcast/pipeline/tts_engine.py` | 新增 `synthesize_cosyvoice2()`，更新 `synthesize()` 分发 |
| `src/ai_news_podcast/cli/publish_episode.py` | 从 `run_daily.py` 抽出的 Stage 5 发布逻辑 |
| `scripts/gha_tts_cosyvoice.py` | GHA Job 2 入口：安装 CosyVoice 环境 + 调用合成 |
| `scripts/setup_cosyvoice_env.sh` | CosyVoice 依赖安装（GHA 与本地复用） |
| `.github/workflows/daily.yml` | 三 Job 工作流 |
| `config/config.yaml` | `backend: cosyvoice2` 及参考音频路径 |
| `tests/test_cosyvoice_backend.py` | CosyVoice backend 单元测试（全 mock） |
| `tests/test_tts_synthesize.py` | 补充 cosyvoice2 分发测试 |

---

## Task 1: 准备参考音频资源

**Files:**
- Create: `asset/refs/host_a_ref.wav`
- Create: `asset/refs/host_a_ref.txt`
- Create: `asset/refs/host_b_ref.wav`
- Create: `asset/refs/host_b_ref.txt`
- Create: `scripts/generate_ref_audio.py`（一次性生成工具，不入 CI）

- [ ] **Step 1: 编写一次性参考音频生成脚本**

```python
#!/usr/bin/env python3
"""One-off: generate CosyVoice reference WAVs from Edge-TTS, then commit to asset/refs/."""
import asyncio
import subprocess
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "asset" / "refs"

HOST_A_TEXT = (
    "云阳，咱们今天这选题，你看啊，从底层神经网络的突破，"
    "到应用端那些草根团队基于协议和开源模型做的创新，真的非常有意思。"
)
HOST_B_TEXT = (
    "云阳老师，你又带上了技术极客的视角了啊！"
    "你能不能给咱们听众翻译翻译，这个机制被 AI 破解到底意味着什么？"
)


async def _edge_to_wav(voice: str, text: str, out_wav: Path) -> None:
    mp3 = out_wav.with_suffix(".mp3")
    await edge_tts.Communicate(text, voice=voice).save(str(mp3))
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3), "-ar", "16000", "-ac", "1", str(out_wav)],
        check=True,
    )
    mp3.unlink(missing_ok=True)


async def main() -> None:
    REFS.mkdir(parents=True, exist_ok=True)
    await _edge_to_wav("zh-CN-YunyangNeural", HOST_A_TEXT, REFS / "host_a_ref.wav")
    await _edge_to_wav("zh-CN-XiaoyiNeural", HOST_B_TEXT, REFS / "host_b_ref.wav")
    (REFS / "host_a_ref.txt").write_text(HOST_A_TEXT, encoding="utf-8")
    (REFS / "host_b_ref.txt").write_text(HOST_B_TEXT, encoding="utf-8")
    print("Reference audio written to", REFS)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 本地生成参考音频**

Run: `uv run python scripts/generate_ref_audio.py`
Expected: 四个文件写入 `asset/refs/`，每个 WAV 时长 5–8 秒、16kHz mono。

- [ ] **Step 3: 人工试听确认**

用播放器打开 `host_a_ref.wav` 和 `host_b_ref.wav`，确认无背景噪、无截断。

- [ ] **Step 4: Commit**

```bash
git add asset/refs/ scripts/generate_ref_audio.py
git commit -m "feat(tts): add CosyVoice reference audio for dual-host cloning"
```

---

## Task 2: 抽取共用后处理模块

**Files:**
- Create: `src/ai_news_podcast/pipeline/tts_postprocess.py`
- Modify: `src/ai_news_podcast/pipeline/tts_engine.py`

- [ ] **Step 1: 写失败测试 — `assemble_dialogue_audio` 拼接逻辑**

```python
# tests/test_tts_postprocess.py
from pathlib import Path
from unittest.mock import patch
import sys

import pytest

from ai_news_podcast.pipeline.tts_postprocess import assemble_dialogue_audio
from ai_news_podcast.pipeline.tts_engine import DialogueChunk


class FakeSeg:
    def __init__(self, ms: int = 1000):
        self._ms = ms

    def __len__(self):
        return self._ms

    def __add__(self, other):
        if isinstance(other, FakeSeg):
            return FakeSeg(self._ms + other._ms)
        return self

    @classmethod
    def empty(cls):
        return cls(0)

    @classmethod
    def silent(cls, *, duration: int):
        return cls(duration)


def test_assemble_dialogue_audio_returns_timestamps():
    chunks = [DialogueChunk(host="A", text="a"), DialogueChunk(host="B", text="b")]
    segments = [FakeSeg(1000), FakeSeg(2000)]
    with patch.dict(sys.modules, {"pydub": type("M", (), {"AudioSegment": FakeSeg})()}):
        combined, timestamps = assemble_dialogue_audio(
            chunks, segments, chunk_silence_base=300, vocal_pad_ms=0,
            silence_min=100, silence_max=100, silence_jitter=0,
        )
    assert len(timestamps) == 2
    assert timestamps[0] == (0.0, 1.0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_tts_postprocess.py -v`
Expected: FAIL `ModuleNotFoundError: tts_postprocess`

- [ ] **Step 3: 实现 `tts_postprocess.py`**

将 `tts_engine.py` 中的 `_chunk_silence_ms`、`_mix_bgm`、`_run_loudnorm` 移入此文件，并新增：

```python
# src/ai_news_podcast/pipeline/tts_postprocess.py
"""Shared TTS post-processing: silence padding, BGM mix, loudnorm, assembly."""

from __future__ import annotations

import asyncio
import random
import shutil
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from ai_news_podcast.pipeline.tts_engine import DialogueChunk


def chunk_silence_ms(...) -> int:  # 从 tts_engine 原样迁移
    ...


def mix_bgm(...) -> Any:  # 从 tts_engine 原样迁移，去掉前缀下划线
    ...


async def run_loudnorm(...) -> None:  # 从 tts_engine 原样迁移
    ...


def assemble_dialogue_audio(
    chunks: Sequence[DialogueChunk],
    segments: Sequence[Any],
    *,
    chunk_silence_base: int,
    vocal_pad_ms: int,
    silence_min: int,
    silence_max: int,
    silence_jitter: int,
) -> Tuple[Any, list[tuple[float, float]]]:
    """Concatenate per-chunk AudioSegments with variable silence; return (combined, timestamps)."""
    pydub = __import__("pydub", fromlist=["AudioSegment"])
    AudioSegment = pydub.AudioSegment

    combined = AudioSegment.empty()
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
            combined += AudioSegment.silent(duration=silence_len)
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
    bgm_path: Optional[str],
    audio_cfg: dict,
    tmp_dir: Path,
) -> None:
    """BGM mix → export pre-norm MP3 → ffmpeg loudnorm → final MP3."""
    combined = mix_bgm(combined, bgm_path, **{k: audio_cfg[k] for k in (
        "bgm_volume_db", "bgm_fade_in_ms", "bgm_fade_out_ms", "vocal_pad_ms"
    ) if k in audio_cfg or True})  # 使用 audio_cfg 默认值
    pre_norm = tmp_dir / "combined_prenorm.mp3"
    combined.export(str(pre_norm), format="mp3")
    await run_loudnorm(
        pre_norm,
        output_path,
        loudnorm=str(audio_cfg.get("loudnorm", "I=-16:LRA=11:TP=-1.5")),
        sample_rate=int(audio_cfg.get("sample_rate", 24000)),
    )
```

- [ ] **Step 4: 更新 `tts_engine.py` 改为从 `tts_postprocess` 导入**

在 `synthesize_edge_tts` 内用 `assemble_dialogue_audio` + `finalize_episode_mp3` 替换内联拼接逻辑；删除已迁移的私有函数。

- [ ] **Step 5: 运行全量 TTS 测试**

Run: `uv run pytest tests/test_tts_engine.py tests/test_tts_synthesize.py tests/test_tts_postprocess.py -v`
Expected: ALL PASS（行为与重构前一致）

- [ ] **Step 6: Commit**

```bash
git add src/ai_news_podcast/pipeline/tts_postprocess.py src/ai_news_podcast/pipeline/tts_engine.py tests/test_tts_postprocess.py
git commit -m "refactor(tts): extract shared postprocess for multi-backend reuse"
```

---

## Task 3: CosyVoice 2 Backend 实现

**Files:**
- Create: `src/ai_news_podcast/pipeline/cosyvoice_backend.py`
- Create: `tests/test_cosyvoice_backend.py`

- [ ] **Step 1: 写失败测试 — 参考配置解析**

```python
# tests/test_cosyvoice_backend.py
from pathlib import Path

from ai_news_podcast.pipeline.cosyvoice_backend import CosyVoiceConfig, load_cosyvoice_config


def test_load_cosyvoice_config_from_yaml_dict(tmp_path: Path):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "host_a_ref.wav").write_bytes(b"wav")
    (refs / "host_a_ref.txt").write_text("男声参考文本", encoding="utf-8")
    (refs / "host_b_ref.wav").write_bytes(b"wav")
    (refs / "host_b_ref.txt").write_text("女声参考文本", encoding="utf-8")

    cfg = {
        "tts": {
            "cosyvoice": {
                "model_dir": "/models/CosyVoice2-0.5B",
                "ref_audio": {
                    "host_a": str(refs / "host_a_ref.wav"),
                    "host_a_text": str(refs / "host_a_ref.txt"),
                    "host_b": str(refs / "host_b_ref.wav"),
                    "host_b_text": str(refs / "host_b_ref.txt"),
                },
            }
        }
    }
    result = load_cosyvoice_config(cfg, project_root=tmp_path)
    assert isinstance(result, CosyVoiceConfig)
    assert result.host_a_text == "男声参考文本"
    assert result.model_dir == Path("/models/CosyVoice2-0.5B")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_cosyvoice_backend.py::test_load_cosyvoice_config_from_yaml_dict -v`
Expected: FAIL

- [ ] **Step 3: 实现 `cosyvoice_backend.py`**

```python
# src/ai_news_podcast/pipeline/cosyvoice_backend.py
"""CosyVoice 2 zero-shot TTS backend (lazy import — not in main pyproject deps)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class CosyVoiceConfig:
    model_dir: Path
    host_a_wav: Path
    host_a_text: str
    host_b_wav: Path
    host_b_text: str
    sample_rate: int = 22050  # CosyVoice2 default output


def load_cosyvoice_config(cfg: dict, *, project_root: Path) -> CosyVoiceConfig:
    cosy = (cfg.get("tts") or {}).get("cosyvoice") or {}
    refs = cosy.get("ref_audio") or {}

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else project_root / path

    return CosyVoiceConfig(
        model_dir=Path(str(cosy.get("model_dir") or "")),
        host_a_wav=_resolve(str(refs.get("host_a") or "asset/refs/host_a_ref.wav")),
        host_a_text=_resolve(str(refs.get("host_a_text") or "asset/refs/host_a_ref.txt")).read_text(encoding="utf-8").strip(),
        host_b_wav=_resolve(str(refs.get("host_b") or "asset/refs/host_b_ref.wav")),
        host_b_text=_resolve(str(refs.get("host_b_text") or "asset/refs/host_b_ref.txt")).read_text(encoding="utf-8").strip(),
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
        from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore[import-untyped]

        self._model = CosyVoice2(
            str(self._config.model_dir),
            load_jit=False,
            load_onnx=False,
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

        model = self._ensure_model()
        ref_audio = self._load_ref(ref_wav)
        output = next(
            model.inference_zero_shot(text, ref_text, ref_audio, stream=False)
        )
        return output["tts_speech"]
```

- [ ] **Step 4: 写 mock 推理测试**

```python
def test_synthesize_chunk_dispatches_by_host(monkeypatch, tmp_path: Path):
    # ... 创建 refs ...
    config = CosyVoiceConfig(...)
    engine = CosyVoice2Engine(config)

  calls = []

    class FakeModel:
        def inference_zero_shot(self, text, ref_text, ref_audio, stream=False):
            calls.append((text, ref_text))
            yield {"tts_speech": "fake_tensor"}

    monkeypatch.setattr(engine, "_ensure_model", lambda: FakeModel())
    monkeypatch.setattr(engine, "_load_ref", lambda p: "fake_audio")

    engine.synthesize_chunk(text="测试句子", host="A")
    engine.synthesize_chunk(text="另一句", host="B")
    assert calls[0][1] == config.host_a_text
    assert calls[1][1] == config.host_b_text
```

- [ ] **Step 5: 运行测试**

Run: `uv run pytest tests/test_cosyvoice_backend.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/ai_news_podcast/pipeline/cosyvoice_backend.py tests/test_cosyvoice_backend.py
git commit -m "feat(tts): add CosyVoice2 backend with config loader"
```

---

## Task 4: 在 tts_engine 接入 cosyvoice2 backend

**Files:**
- Modify: `src/ai_news_podcast/pipeline/tts_engine.py`
- Modify: `tests/test_tts_synthesize.py`

- [ ] **Step 1: 写失败测试 — cosyvoice2 分发**

```python
@pytest.mark.asyncio
async def test_synthesize_cosyvoice2_backend(tmp_path: Path, monkeypatch):
    output = tmp_path / "out.mp3"
    called = {}

    async def fake_synth(**kwargs):
        called.update(kwargs)
        output.write_text("ok", encoding="utf-8")

    monkeypatch.setattr(
        "ai_news_podcast.pipeline.tts_engine.synthesize_cosyvoice2",
        fake_synth,
    )
    await synthesize(
        "[Host A] 你好\n[Host B] 欢迎",
        backend="cosyvoice2",
        output_path=output,
        cfg={"tts": {}},
    )
    assert output.exists()
    assert called["output_path"] == output
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_tts_synthesize.py::test_synthesize_cosyvoice2_backend -v`
Expected: FAIL

- [ ] **Step 3: 实现 `synthesize_cosyvoice2`**

```python
async def synthesize_cosyvoice2(
    chunks: List[DialogueChunk],
    output_path: Union[str, Path],
    *,
    bgm_path: Optional[str] = None,
    transcript_path: Optional[Union[str, Path]] = None,
    cfg: Optional[dict] = None,
    project_root: Optional[Path] = None,
    engine: Optional[Any] = None,  # CosyVoice2Engine, injectable for tests
) -> None:
    import torchaudio
    from ai_news_podcast.pipeline.cosyvoice_backend import CosyVoice2Engine, load_cosyvoice_config
    from ai_news_podcast.pipeline.tts_postprocess import (
        assemble_dialogue_audio,
        finalize_episode_mp3,
    )

    pydub = importlib.import_module("pydub")
    AudioSegment = pydub.AudioSegment

    audio_cfg = (cfg or {}).get("tts", {}).get("audio", {})
    root = project_root or Path.cwd()
    cv_cfg = load_cosyvoice_config(cfg or {}, project_root=root)
    cosy_engine = engine or CosyVoice2Engine(cv_cfg)

  segments = []
    with tempfile.TemporaryDirectory(prefix="tts-cosyvoice-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for idx, chunk in enumerate(chunks, start=1):
            tensor = cosy_engine.synthesize_chunk(text=chunk.text, host=chunk.host)
            wav_path = tmp_root / f"chunk_{idx:03d}.wav"
            torchaudio.save(str(wav_path), tensor, cv_cfg.sample_rate)
            segments.append(AudioSegment.from_file(str(wav_path)))

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
            Path(output_path),
            bgm_path=bgm_path,
            audio_cfg=audio_cfg,
            tmp_dir=tmp_root,
        )
        # 写 transcript 时间戳（复用 edge 逻辑，voice name 改为 host_a/host_b）
        ...
```

- [ ] **Step 4: 更新 `synthesize()` 分发**

```python
backend_name = str(backend).strip().lower()
if backend_name in ("edge-tts", "edge"):
    await synthesize_edge_tts(...)
    return
if backend_name in ("cosyvoice2", "cosyvoice"):
    await synthesize_cosyvoice2(
        chunks,
        output_path=output_path,
        bgm_path=bgm_path,
        transcript_path=kwargs.get("transcript_path"),
        cfg=kwargs.get("cfg"),
        project_root=kwargs.get("project_root"),
    )
    return
if backend_name == "hybrid":
    try:
        await synthesize_cosyvoice2(...)
    except Exception:
        log.warning("CosyVoice failed, falling back to edge-tts", exc_info=True)
        await synthesize_edge_tts(...)
    return
raise ValueError(...)
```

- [ ] **Step 5: 修复 `run_daily.py` 传参**

将 `voice=voice` 改为：

```python
voices = (
    str(tts_cfg.get("host_a_voice") or "zh-CN-YunxiNeural"),
    str(tts_cfg.get("host_b_voice") or tts_cfg.get("voice") or "zh-CN-XiaoxiaoNeural"),
)
bgm_path = str(tts_cfg.get("bgm_path") or "assets/bgm_placeholder.wav")
await synthesize(
    script_text,
    backend=backend,
    voices=voices,
    output_path=mp3_path,
    bgm_path=bgm_path if Path(root / bgm_path).exists() else None,
    ...
    project_root=root,
)
```

- [ ] **Step 6: 运行测试**

Run: `uv run pytest tests/test_tts_synthesize.py tests/test_run_daily_full.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/ai_news_podcast/pipeline/tts_engine.py src/ai_news_podcast/cli/run_daily.py tests/test_tts_synthesize.py
git commit -m "feat(tts): wire CosyVoice2 backend into synthesize dispatcher"
```

---

## Task 5: 抽取发布逻辑 + 修复 --no-audio 工作流断层

**背景:** 当前 `run_daily.py` 在 `--no-audio` 时 Stage 5 直接 `return 0`，不生成 `feed.xml` / `index.html`。三 Job 架构需要 Job 1 生成站点骨架，Job 2 合成后补全发布。

**Files:**
- Create: `src/ai_news_podcast/cli/publish_episode.py`
- Modify: `src/ai_news_podcast/cli/run_daily.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: 写失败测试 — publish_episode 更新 feed**

```python
# tests/test_publish_episode.py
def test_publish_episode_writes_feed_xml(tmp_path, monkeypatch):
    # 准备 episodes.json、mp3、brief 缓存等最小 fixture
    ...
    from ai_news_podcast.cli.publish_episode import publish_episode
    publish_episode(config_path=..., date="2026-06-11")
    assert (tmp_path / "site" / "feed.xml").exists()
```

- [ ] **Step 2: 从 `run_daily.py` 抽出 `publish_episode()` 函数**

将 L223–327（Stage 5）移入 `publish_episode.py`，接受 `episode_id`、`brief`、`cfg`、`root` 参数。

- [ ] **Step 3: 修改 `--no-audio` 行为**

```python
# run_daily.py Stage 5 调整后：
if args.no_audio:
    # 仍生成 show notes HTML，但不写 feed.xml（等 TTS 完成后再 publish）
    notes_html = generate_show_notes_html(...)
    write_text(notes_path, notes_html)
    log.info("--no-audio: show notes saved, deferring feed/site publish")
    return 0

# 正常流程（含 TTS）仍内联调用 publish_episode()
```

- [ ] **Step 4: 注册 CLI 入口**

```toml
# pyproject.toml
podcast-publish = "ai_news_podcast.cli.publish_episode:entrypoint"
```

`publish_episode` CLI 参数：`--date`、`--config`，读取已有 brief + mp3，执行 Stage 5。

- [ ] **Step 5: 运行测试**

Run: `uv run pytest tests/test_publish_episode.py tests/test_run_daily_full.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/ai_news_podcast/cli/publish_episode.py src/ai_news_podcast/cli/run_daily.py pyproject.toml tests/test_publish_episode.py
git commit -m "feat(cli): extract publish_episode for post-TTS GHA workflow"
```

---

## Task 6: GHA 合成脚本与环境安装

**Files:**
- Create: `scripts/setup_cosyvoice_env.sh`
- Create: `scripts/gha_tts_cosyvoice.py`

- [ ] **Step 1: 编写环境安装脚本（复用 benchmark.yml 逻辑）**

```bash
#!/usr/bin/env bash
# scripts/setup_cosyvoice_env.sh
set -euo pipefail

COSY_SRC="${COSYVOICE_SRC:-$HOME/cosyvoice_src}"
COSY_MODELS="${COSYVOICE_MODELS:-$HOME/cosyvoice_models}"
MODEL_DIR="$COSY_MODELS/CosyVoice2-0.5B"

if [ ! -d "$COSY_SRC/cosyvoice" ]; then
  git clone --recursive --depth=1 https://github.com/FunAudioLLM/CosyVoice.git "$COSY_SRC"
fi

pip install --upgrade pip setuptools
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r "$COSY_SRC/requirements.txt" 2>/dev/null || pip install \
  conformer==0.3.2 diffusers==0.29.0 hydra-core==1.3.2 HyperPyYAML==1.2.3 \
  inflect librosa omegaconf onnx onnxruntime openai-whisper protobuf pyworld \
  rich soundfile transformers x-transformers wetext huggingface_hub

export PYTHONPATH="$COSY_SRC:$COSY_SRC/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"

python - <<'EOF'
from huggingface_hub import snapshot_download
import os
local_dir = os.path.expanduser(""""$MODEL_DIR"""")
if not os.path.exists(os.path.join(local_dir, "cosyvoice.yaml")):
    snapshot_download("FunAudioLLM/CosyVoice2-0.5B", local_dir=local_dir, ignore_patterns=["*.bin"])
EOF

echo "COSYVOICE_MODEL_DIR=$MODEL_DIR"
```

- [ ] **Step 2: 编写 GHA 合成入口**

```python
#!/usr/bin/env python3
# scripts/gha_tts_cosyvoice.py
"""GHA Job 2 entry: synthesize episode MP3 with CosyVoice2."""
import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_news_podcast.pipeline.tts_engine import synthesize
from ai_news_podcast.utils import read_yaml


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="Path to episode .txt script")
    ap.add_argument("--output", required=True, help="Output .mp3 path")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model-dir", default=os.environ.get("COSYVOICE_MODEL_DIR", ""))
    args = ap.parse_args()

    cfg = read_yaml(ROOT / args.config)
    if args.model_dir:
        cfg.setdefault("tts", {}).setdefault("cosyvoice", {})["model_dir"] = args.model_dir

    script_text = Path(args.script).read_text(encoding="utf-8")
    await synthesize(
        script_text,
        backend="cosyvoice2",
        output_path=args.output,
        transcript_path=args.script,
        cfg=cfg,
        project_root=ROOT,
        bgm_path=str(ROOT / "assets/bgm_placeholder.wav")
        if (ROOT / "assets/bgm_placeholder.wav").exists() else None,
    )
    print(f"Audio saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 3: 本地冒烟（有 CosyVoice 环境时）**

Run:
```bash
export COSYVOICE_MODEL_DIR=~/cosyvoice_models/CosyVoice2-0.5B
bash scripts/setup_cosyvoice_env.sh
python scripts/gha_tts_cosyvoice.py \
  --script site/episodes/2026-06-04.txt \
  --output /tmp/test_cosyvoice.mp3
```
Expected: `/tmp/test_cosyvoice.mp3` 生成，时长与剧本句数匹配。

- [ ] **Step 4: Commit**

```bash
git add scripts/setup_cosyvoice_env.sh scripts/gha_tts_cosyvoice.py
git commit -m "feat(gha): add CosyVoice2 synthesis script and env setup"
```

---

## Task 7: 更新 config.yaml

**Files:**
- Modify: `config/config.yaml`

- [ ] **Step 1: 替换 TTS 配置块**

```yaml
tts:
  backend: "cosyvoice2"          # edge-tts | cosyvoice2 | hybrid
  fallback_backend: "edge-tts"     # hybrid 模式下 CosyVoice 失败时降级
  host_a_voice: "zh-CN-YunxiNeural"   # 仅 edge / hybrid 降级时使用
  host_b_voice: "zh-CN-XiaoxiaoNeural"
  bgm_path: "assets/bgm_placeholder.wav"
  cosyvoice:
    model_dir: ""                  # 空则读环境变量 COSYVOICE_MODEL_DIR
    ref_audio:
      host_a: "asset/refs/host_a_ref.wav"
      host_a_text: "asset/refs/host_a_ref.txt"
      host_b: "asset/refs/host_b_ref.wav"
      host_b_text: "asset/refs/host_b_ref.txt"
  # 以下 audio 块保持不变
  audio:
    bgm_volume_db: -12
    ...
```

- [ ] **Step 2: Commit**

```bash
git add config/config.yaml
git commit -m "config: switch default TTS backend to cosyvoice2"
```

---

## Task 8: 改造 daily.yml 为三 Job 工作流

**Files:**
- Modify: `.github/workflows/daily.yml`

- [ ] **Step 1: 替换 workflow 内容**

```yaml
name: Daily Podcast

on:
  schedule:
    - cron: "43 21 * * *"
  workflow_dispatch:

permissions:
  contents: write

jobs:
  content:
    name: Content (fetch → script)
    runs-on: ubuntu-latest
    outputs:
      episode_id: ${{ steps.episode.outputs.id }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"
      - run: sudo apt-get update && sudo apt-get install -y ffmpeg
      - run: uv sync --frozen

      - id: episode
        run: echo "id=$(TZ=Asia/Shanghai date +%F)" >> "$GITHUB_OUTPUT"

      - name: Pipeline + report + script (no audio)
        env:
          TZ: Asia/Shanghai
          SPARK_API_KEY: ${{ secrets.SPARK_API_KEY }}
        run: |
          uv run podcast-pipeline
          uv run podcast-report
          uv run podcast-daily --no-audio

      - name: Commit content artifacts
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/briefs/ data/reports/ data/episodes.json site/episodes/*.txt site/episodes/*.html
          git diff --cached --quiet || git commit -m "daily: content $(date -u +%F) [skip ci]"
          git push

  tts:
    name: TTS (CosyVoice 2)
    needs: content
    runs-on: ubuntu-latest
    timeout-minutes: 90
    steps:
      - uses: actions/checkout@v4
        with:
          ref: main
      - run: git pull   # 获取 content job 提交的剧本

      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Cache CosyVoice
        uses: actions/cache@v4
        with:
          path: |
            ~/cosyvoice_src
            ~/cosyvoice_models
          key: cosyvoice-gha-v1

      - name: Setup CosyVoice environment
        run: bash scripts/setup_cosyvoice_env.sh

      - name: Install podcast postprocess deps
        run: pip install pydub PyYAML

      - name: Synthesize episode
        env:
          COSYVOICE_MODEL_DIR: ~/cosyvoice_models/CosyVoice2-0.5B
        run: |
          EPISODE="${{ needs.content.outputs.episode_id }}"
          python scripts/gha_tts_cosyvoice.py \
            --script "site/episodes/${EPISODE}.txt" \
            --output "site/episodes/${EPISODE}.mp3"

      - name: Publish feed and site
        run: |
          pip install uv
          uv sync --frozen
          EPISODE="${{ needs.content.outputs.episode_id }}"
          uv run podcast-publish --date "$EPISODE"

      - name: Commit audio and site
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add site/ data/episodes.json
          git diff --cached --quiet || git commit -m "daily: audio $(date -u +%F) [skip ci]"
          git push

  deploy:
    name: Deploy GitHub Pages
    needs: tts
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
          keep_files: true
```

- [ ] **Step 2: 手动触发验证**

在 GitHub Actions 页面点击 **Run workflow** → 观察三个 Job 顺序执行。

Expected:
- Job `content`：~5–10 分钟，产出 `.txt` / `.html`
- Job `tts`：~25–45 分钟，产出 `.mp3`
- Job `deploy`：~1 分钟，gh-pages 更新

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/daily.yml
git commit -m "ci: split daily workflow into content/tts/deploy for CosyVoice2"
```

---

## Task 9: 文档更新

**Files:**
- Modify: `README.zh-CN.md`
- Modify: `docs/pipeline_walkthrough.md`

- [ ] **Step 1: 更新 README.zh-CN.md TTS 章节**

说明：
- 默认 TTS 已切换为 CosyVoice 2
- 本地合成需先运行 `scripts/setup_cosyvoice_env.sh`
- GHA 三 Job 架构与时序
- `backend: hybrid` 可降级 Edge-TTS

- [ ] **Step 2: 更新 pipeline_walkthrough.md Stage 4 描述**

将 "edge-tts 并行合成" 改为 "CosyVoice2 逐句零样本克隆 → 拼接 → BGM → loudnorm"。

- [ ] **Step 3: Commit**

```bash
git add README.zh-CN.md docs/pipeline_walkthrough.md
git commit -m "docs: document CosyVoice2 GHA TTS pipeline"
```

---

## Task 10: 全量验证

- [ ] **Step 1: 本地单元测试**

Run: `uv run pytest tests/ -v --ignore=tests/test_fetcher_integration.py`
Expected: ALL PASS

- [ ] **Step 2: Lint**

Run: `uv run ruff check src/ tests/ scripts/ && uv run ruff format --check src/ tests/ scripts/`
Expected: no errors

- [ ] **Step 3: GHA 端到端**

手动 `workflow_dispatch` 跑一轮，下载产出 MP3 与 benchmark 样本对比听感。

- [ ] **Step 4: 确认降级路径**

临时破坏 `COSYVOICE_MODEL_DIR`，设 `backend: hybrid`，确认 Edge-TTS 兜底生效。

---

## 风险清单

| 风险 | 缓解 |
|------|------|
| GHA 私有仓库分钟数耗尽 | 监控 Actions 用量；cache 命中后 Job 2 约 30 分钟/天 |
| CosyVoice 首次 cache miss 超 90 分钟 | 提高 `timeout-minutes: 120`；模型 cache key 保持稳定 |
| 参考音频质量差 | Task 1 人工试听；后续可换真人录音 |
| Job 1 push 后 Job 2 checkout 拿不到最新 commit | `git pull` after checkout（已在 workflow 中） |
| `run_daily` 的 `voice`/`voices` 参数不一致 | Task 4 修复 |
| CosyVoice 非 uv 依赖污染主环境 | GHA Job 2 用独立 pip 环境，不加入 `pyproject.toml` |

---

## 执行选项

计划已保存至 `docs/superpowers/plans/2026-06-11-gha-cosyvoice2-tts.md`。

**两种执行方式：**

1. **Subagent-Driven（推荐）** — 每个 Task 派发独立 subagent，任务间做代码审查，迭代快
2. **Inline Execution** — 在本会话中按 Task 顺序逐步执行，每 2–3 个 Task 设检查点

你想用哪种方式开始？
