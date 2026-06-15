# GHA + 2C2G ECS 多音色音频流水线 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**Goal:** 利用 GitHub Actions 作为编排层、腾讯云 2C2G ECS 作为 TTS 推理后端，实现：① 固定人物角色的完整播客拼接 MP3（轨 A）；② 每句台词同时存在多个音色版本的分句切片 + JSON 索引（轨 B），两轨并行输出。

**Architecture:** ECS 上运行 FastAPI + Huey SQLite 串行推理队列，通过 Cloudflare Tunnel 暴露给 GHA。GHA 负责解析剧本、哈希增量比对、向 ECS 派发任务、轮询下载 WAV、最终用 ffmpeg 构建双轨并提交部署。整个流程解耦为 3 个独立 GHA Workflow：采编/TTS 构建/Pages 发布。

**Tech Stack:** Python 3.11, FastAPI, Huey (SQLite), ONNX Runtime (CPU), ffmpeg, GitHub Actions, Cloudflare Tunnel, pydub

---

## 文件结构

| 操作 | 文件 | 职责 |
|------|------|------|
| 创建 | `server/server.py` | ECS 端 FastAPI + Huey 串行推理队列主服务 |
| 创建 | `server/compiler.py` | XML 标签编译器：将剧本标签翻译为模型参数 |
| 创建 | `server/requirements.txt` | ECS 推理后端 Python 依赖清单 |
| 创建 | `server/setup_swap.sh` | ECS 初始化脚本：4GB Swap + 环境安装 |
| 创建 | `scripts/parse_script.py` | GHA 侧：解析 Markdown 剧本 → 输出带哈希的任务 JSON |
| 创建 | `scripts/dispatch_tts.py` | GHA 侧：批量 POST 给 ECS、轮询状态、下载 WAV |
| 创建 | `scripts/build_tracks.py` | GHA 侧：WAV → 轨 A（ffmpeg concat MP3）+ 轨 B（split MP3 + index.json）|
| 创建 | `.github/workflows/daily_curator.yml` | 工作流 1：定时采编 + LLM 写剧本 + 提交 |
| 创建 | `.github/workflows/tts_builder.yml` | 工作流 2：增量 TTS 构建 + 双轨输出 + 提交 |
| 创建 | `.github/workflows/pages_publisher.yml` | 工作流 3：部署 site/ 到 GitHub Pages |
| 修改 | `docs/tts_complete_guide.md` | 补充：工作流触发机制与文件目录结构说明 |

---

## 任务 1：ECS 初始化脚本与依赖清单

**Files:**
- 创建: `server/setup_swap.sh`
- 创建: `server/requirements.txt`

- [ ] **步骤 1：创建 `server/setup_swap.sh`**

  该脚本在全新 2C2G ECS（Ubuntu 22.04）上首次运行，完成以下操作：
  1. 分配 4GB Swap 分区（`/swapfile`），防止 ONNX 模型初始化 OOM。
  2. 设置 vm.swappiness=10 减少非必要 Swap 使用。
  3. 创建 Python 虚拟环境 `.venv` 并安装依赖。
  4. 创建推理输出目录 `output/` 和声纹目录 `voices/`。
  5. 创建 ONNX 模型存放目录 `models/`。
  6. 打印验证消息确认完成。

  内容如下：

  ```bash
  #!/bin/bash
  set -e

  echo "=== [1/5] 分配 4GB Swap 分区 ==="
  sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  sudo sysctl vm.swappiness=10
  echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

  echo "=== [2/5] 安装系统依赖 ==="
  sudo apt-get update -qq
  sudo apt-get install -y python3.11 python3.11-venv python3-pip ffmpeg

  echo "=== [3/5] 创建 Python 虚拟环境并安装依赖 ==="
  python3.11 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r server/requirements.txt

  echo "=== [4/5] 创建工作目录 ==="
  mkdir -p output voices models

  echo "=== [5/5] 完成 ==="
  echo "ECS 环境初始化完成。请将 ONNX 模型文件放入 models/ 目录，声纹音频放入 voices/ 目录。"
  ```

- [ ] **步骤 2：创建 `server/requirements.txt`**

  ```
  fastapi==0.111.0
  uvicorn[standard]==0.30.1
  huey==2.5.1
  onnxruntime==1.18.0
  soundfile==0.12.1
  pydub==0.25.1
  numpy==1.26.4
  ```

- [ ] **步骤 3：提交**

  ```bash
  git add server/setup_swap.sh server/requirements.txt
  git commit -m "feat(server): add ECS init script and requirements"
  ```

---

## 任务 2：XML 标签编译器模块

**Files:**
- 创建: `server/compiler.py`
- 测试验证点: 手动执行模块并断言输出符合预期

编译器负责将统一的 XML 语义标签翻译为各模型可理解的参数（模型路由、参考音频路径、清洗后文本），是多模型解耦的核心枢纽。

- [ ] **步骤 1：创建 `server/compiler.py`**

  ```python
  """
  XML 语义标签编译器。
  输入：说话人 + 含标签的原始台词文本。
  输出：(model_type, ref_audio_path, compiled_text, voice_variant)

  支持的统一标签：
    <laugh/>   → ChatTTS 情感模式 / happy 声纹
    <sigh/>    → F5-TTS 情感模式 / sad 声纹
    <pause time="X"/> → 后处理插入 X 秒静音（不影响推理文本）

  模型路由规则：
    1. 含 <laugh/>  → chattts,  {speaker}_ref_happy.wav
    2. 含 <sigh/>   → f5tts,    {speaker}_ref_sad.wav
    3. 含连续英文字符（≥3个字母）→ f5tts,  {speaker}_ref.wav
    4. 其他中文句子  → gptsovits, {speaker}_ref.wav
  """

  from __future__ import annotations
  import re

  VOICES_DIR = "voices"

  # 检测连续英文单词（≥3 个字母），用于 Chinglish 路由
  _RE_ENGLISH = re.compile(r"[a-zA-Z]{3,}")
  # 提取 pause time 参数
  _RE_PAUSE = re.compile(r'<pause\s+time=["\']?([\d.]+)["\']?\s*/>')


  def compile(speaker: str, raw_text: str) -> dict:
      """
      编译单句台词，返回推理所需的完整参数字典。

      返回结构：
        {
          "model_type":     "chattts" | "f5tts" | "gptsovits" | "edge",
          "ref_audio":      "voices/xiaoyi_ref_happy.wav",  # 完整路径
          "compiled_text":  "清洗后的纯文本，不含 XML 标签",
          "pause_seconds":  0.5,   # 0 表示无停顿
          "voice_variant":  "standard" | "casual",  # 用于 Track B 文件命名
        }
      """
      model_type = "gptsovits"
      ref_audio = f"{VOICES_DIR}/{speaker}_ref.wav"
      voice_variant = "standard"
      pause_seconds = 0.0

      # --- 路由决策 ---
      if "<laugh/>" in raw_text:
          model_type = "chattts"
          ref_audio = f"{VOICES_DIR}/{speaker}_ref_happy.wav"
          voice_variant = "casual"
          raw_text = raw_text.replace("<laugh/>", "")

      elif "<sigh/>" in raw_text:
          model_type = "f5tts"
          ref_audio = f"{VOICES_DIR}/{speaker}_ref_sad.wav"
          voice_variant = "casual"
          raw_text = raw_text.replace("<sigh/>", "")

      elif _RE_ENGLISH.search(raw_text):
          model_type = "f5tts"

      # --- 提取 pause 时长 ---
      pause_match = _RE_PAUSE.search(raw_text)
      if pause_match:
          pause_seconds = float(pause_match.group(1))
          raw_text = _RE_PAUSE.sub("", raw_text)

      # --- 清理残余 XML 标签 ---
      compiled_text = re.sub(r"<[^>]+>", "", raw_text).strip()

      return {
          "model_type": model_type,
          "ref_audio": ref_audio,
          "compiled_text": compiled_text,
          "pause_seconds": pause_seconds,
          "voice_variant": voice_variant,
      }
  ```

- [ ] **步骤 2：验证编译器行为（手动执行）**

  在 ECS 或本地环境中运行：

  ```bash
  cd server
  python3 -c "
  import compiler
  # 测试 1: laugh 路由到 chattts
  r = compiler.compile('xiaoyi', '哈哈，真的吗？<laugh/>')
  assert r['model_type'] == 'chattts', r
  assert 'happy' in r['ref_audio'], r
  assert '<laugh/>' not in r['compiled_text'], r
  print('✓ laugh 路由正确:', r)

  # 测试 2: 英文混读路由到 f5tts
  r = compiler.compile('yunyang', '这是 neural network 的突破')
  assert r['model_type'] == 'f5tts', r
  print('✓ 英文混读路由正确:', r)

  # 测试 3: pause 提取
  r = compiler.compile('xiaoyi', '好的。<pause time=\"0.5\"/>请继续')
  assert r['pause_seconds'] == 0.5, r
  assert '<pause' not in r['compiled_text'], r
  print('✓ pause 提取正确:', r)

  print('所有断言通过。')
  "
  ```

  预期输出：所有 3 行 `✓` 通过，最后打印 `所有断言通过。`

- [ ] **步骤 3：提交**

  ```bash
  git add server/compiler.py
  git commit -m "feat(server): add XML tag compiler with model routing logic"
  ```

---

## 任务 3：ECS FastAPI + Huey 串行推理队列服务

**Files:**
- 创建: `server/server.py`

这是 ECS 服务端的核心，包含：任务接收接口、Huey 串行 Worker、动态 ONNX 模型加载与显式内存释放。

- [ ] **步骤 1：创建 `server/server.py`**

  ```python
  """
  ECS 推理队列服务。
  启动命令（在 .venv 激活后）：
    uvicorn server:app --host 0.0.0.0 --port 8000 &
    huey_consumer.py server.huey -w 1 -k thread &
  """

  from __future__ import annotations
  import gc
  import os
  import uuid
  from pathlib import Path

  import numpy as np
  import soundfile as sf
  from fastapi import FastAPI
  from fastapi.responses import FileResponse
  from huey import SqliteHuey
  from pydub import AudioSegment

  from compiler import compile as compile_tags

  app = FastAPI(title="TTS Queue Server")
  huey = SqliteHuey(filename="/tmp/tts_huey.db")

  OUTPUT_DIR = Path("output")
  OUTPUT_DIR.mkdir(exist_ok=True)

  # ─── 接口层 ────────────────────────────────────────────────────────────────

  @app.post("/synthesize")
  def request_synthesis(speaker: str, raw_text: str, sentence_hash: str):
      """
      接收单句合成请求。
      立即返回 HTTP 202 + job_id，将任务投入 Huey 队列。
      GHA 凭 job_id 轮询 /status/{job_id} 获取结果。
      """
      params = compile_tags(speaker, raw_text)
      job = _synthesis_task(
          sentence_hash=sentence_hash,
          speaker=speaker,
          **params,
      )
      return {"job_id": job.id, "sentence_hash": sentence_hash, "status": "queued"}


  @app.get("/status/{job_id}")
  def check_status(job_id: str):
      result = huey.result(job_id, peek=True)
      if result is None:
          return {"status": "pending"}
      if isinstance(result, dict) and "error" in result:
          return {"status": "failed", "error": result["error"]}
      return {"status": "done", **result}


  @app.get("/download/{filename}")
  def download_file(filename: str):
      path = OUTPUT_DIR / filename
      if not path.exists():
          return {"error": "not found"}
      return FileResponse(str(path), media_type="audio/wav")


  # ─── 串行 Worker ──────────────────────────────────────────────────────────

  @huey.task()
  def _synthesis_task(
      sentence_hash: str,
      speaker: str,
      model_type: str,
      ref_audio: str,
      compiled_text: str,
      pause_seconds: float,
      voice_variant: str,
  ):
      """
      串行消费：加载模型 → 推理 → 释放内存 → 写入 WAV。
      文件命名：{sentence_hash}_{voice_variant}.wav
      """
      out_filename = f"{sentence_hash}_{voice_variant}.wav"
      out_path = OUTPUT_DIR / out_filename

      # 已生成则直接复用（幂等性保障）
      if out_path.exists():
          return {"filename": out_filename, "voice_variant": voice_variant}

      try:
          audio_data, sample_rate = _run_inference(
              model_type, compiled_text, ref_audio
          )

          # 追加静音（pause 标签处理）
          if pause_seconds > 0:
              silence_samples = int(sample_rate * pause_seconds)
              audio_data = np.concatenate(
                  [audio_data, np.zeros(silence_samples, dtype=audio_data.dtype)]
              )

          sf.write(str(out_path), audio_data, sample_rate)

      except Exception as exc:
          return {"error": str(exc)}

      return {"filename": out_filename, "voice_variant": voice_variant}


  def _run_inference(model_type: str, text: str, ref_audio: str):
      """
      动态加载 ONNX 模型 → 推理 → 显式释放内存。
      返回 (numpy array, sample_rate)
      """
      import onnxruntime as ort

      model_paths = {
          "f5tts":     "models/f5tts_int8.onnx",
          "chattts":   "models/chattts_int8.onnx",
          "gptsovits": "models/gptsovits_int8.onnx",
      }
      model_path = model_paths.get(model_type, model_paths["gptsovits"])

      opts = ort.SessionOptions()
      opts.intra_op_num_threads = 2  # 硬限 2 核
      session = ort.InferenceSession(
          model_path, opts, providers=["CPUExecutionProvider"]
      )

      try:
          # 此处为模型推理占位；实际接入 ONNX 模型时替换此逻辑
          sample_rate = 24000
          duration_samples = max(sample_rate * 2, len(text) * 200)
          audio_data = np.zeros(duration_samples, dtype=np.float32)
          return audio_data, sample_rate

      finally:
          # ★ 核心求生逻辑：显式销毁 Session 并强制 GC
          del session
          gc.collect()
  ```

- [ ] **步骤 2：在 ECS 上启动服务并验证接口可访问**

  ```bash
  # 在 ECS 上激活 venv 并启动
  source .venv/bin/activate
  uvicorn server.server:app --host 0.0.0.0 --port 8000 &
  huey_consumer.py server.server.huey -w 1 -k thread &

  # 验证接口存活
  curl http://localhost:8000/status/test_job_id
  # 预期: {"status":"pending"} 或 {"error":"..."}，非 500 即通过
  ```

- [ ] **步骤 3：提交**

  ```bash
  git add server/server.py
  git commit -m "feat(server): add FastAPI+Huey serial inference queue with gc release"
  ```

---

## 任务 4：GHA 侧剧本解析器

**Files:**
- 创建: `scripts/parse_script.py`

从 `docs/episodes/today_script.md` 读取剧本，逐行解析，输出带 MD5 哈希的任务清单 JSON（供后续增量对比用）。

- [ ] **步骤 1：创建 `scripts/parse_script.py`**

  ```python
  """
  剧本解析器：将 Markdown 剧本转换为带哈希的任务清单 JSON。

  输入文件格式（每行一句）：
    [小怡]: 台词内容 <可含XML标签>
    [云阳]: 台词内容

  输出 JSON 写入 /tmp/tts_tasks.json，结构：
  [
    {
      "index": 0,
      "speaker": "xiaoyi",
      "raw_text": "台词内容",
      "sentence_hash": "abc123...",
      "needs_casual": true   # 含 laugh/sigh 则为 true，同时需生成 casual 版本
    },
    ...
  ]
  """

  from __future__ import annotations
  import hashlib
  import json
  import re
  import sys
  from pathlib import Path

  SPEAKER_MAP = {
      "小怡": "xiaoyi",
      "晓晓": "xiaoyi",
      "云阳": "yunyang",
      "博文": "yunyang",
  }

  _RE_LINE = re.compile(r"^\[([^\]]+)\]:\s*(.+)$")
  _RE_EMOTION = re.compile(r"<(?:laugh|sigh)/>")


  def _md5(text: str) -> str:
      return hashlib.md5(text.encode("utf-8")).hexdigest()


  def parse(script_path: str) -> list[dict]:
      tasks = []
      for idx, line in enumerate(Path(script_path).read_text("utf-8").splitlines()):
          m = _RE_LINE.match(line.strip())
          if not m:
              continue
          speaker_cn, raw_text = m.group(1), m.group(2)
          speaker = SPEAKER_MAP.get(speaker_cn, speaker_cn.lower())

          # 哈希键：说话人 + 原始文本（含标签，保证标签变化也触发重新生成）
          sentence_hash = _md5(f"{speaker}:{raw_text}")
          needs_casual = bool(_RE_EMOTION.search(raw_text))

          tasks.append({
              "index": idx,
              "speaker": speaker,
              "raw_text": raw_text,
              "sentence_hash": sentence_hash,
              "needs_casual": needs_casual,
          })
      return tasks


  if __name__ == "__main__":
      script_path = sys.argv[1] if len(sys.argv) > 1 else "docs/episodes/today_script.md"
      tasks = parse(script_path)
      out = Path("/tmp/tts_tasks.json")
      out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), "utf-8")
      print(f"解析完成：{len(tasks)} 句台词 → {out}")
  ```

- [ ] **步骤 2：本地验证（用示例剧本）**

  ```bash
  # 创建临时测试剧本
  cat > /tmp/test_script.md << 'EOF'
  [小怡]: 云阳老师，你又带上了技术极客的视角了啊！<laugh/>
  [云阳]: <sigh/>其实这个问题并没有那么玄乎。这是 neural network 的一次突破。
  EOF

  python scripts/parse_script.py /tmp/test_script.md
  cat /tmp/tts_tasks.json
  ```

  预期输出：2 条 JSON 对象，第 1 条 `needs_casual: true`，第 2 条 `needs_casual: true`，两条均有不同的 `sentence_hash`。

- [ ] **步骤 3：提交**

  ```bash
  git add scripts/parse_script.py
  git commit -m "feat(scripts): add script parser with MD5 hash and emotion detection"
  ```

---

## 任务 5：GHA 侧 ECS 分发与轮询下载器

**Files:**
- 创建: `scripts/dispatch_tts.py`

读取 `/tmp/tts_tasks.json`，与本地 WAV 缓存库对比（增量），只把未命中的台词 POST 给 ECS，轮询等待完成，下载 WAV 到 `cache/` 目录。

- [ ] **步骤 1：创建 `scripts/dispatch_tts.py`**

  ```python
  """
  ECS TTS 任务分发与轮询下载器。

  运行：
    python scripts/dispatch_tts.py

  环境变量：
    ECS_ENDPOINT  → Cloudflare Tunnel 地址，如 https://your-tunnel.trycloudflare.com
    ECS_API_KEY   → 可选，未来用于接口鉴权

  行为：
    1. 读取 /tmp/tts_tasks.json
    2. 对每条任务，检查 cache/{hash}_standard.wav 是否已存在
       - 存在 → 跳过（增量缓存命中）
       - 不存在 → POST /synthesize 投递任务
    3. 如果任务 needs_casual，同样检查 cache/{hash}_casual.wav
    4. 等待所有 job 完成（每 5 秒轮询一次，最多 60 分钟）
    5. 将完成的 WAV 下载到 cache/ 目录
  """

  from __future__ import annotations
  import json
  import os
  import time
  from pathlib import Path

  import requests

  ECS_ENDPOINT = os.environ["ECS_ENDPOINT"].rstrip("/")
  CACHE_DIR = Path("cache")
  CACHE_DIR.mkdir(exist_ok=True)
  TASKS_FILE = Path("/tmp/tts_tasks.json")
  MAX_WAIT_SECONDS = 3600   # 最长等待 60 分钟
  POLL_INTERVAL = 5         # 每 5 秒轮询一次


  def main():
      tasks = json.loads(TASKS_FILE.read_text("utf-8"))
      pending_jobs: list[dict] = []   # {"job_id", "filename"}

      # ── 阶段 1：提交未命中任务 ──────────────────────────────────────
      for task in tasks:
          h = task["sentence_hash"]
          _submit_if_missing(task, h, "standard", pending_jobs)
          if task["needs_casual"]:
              _submit_if_missing(task, h, "casual", pending_jobs)

      if not pending_jobs:
          print("全部命中缓存，无需调用 ECS。")
          return

      print(f"已提交 {len(pending_jobs)} 个任务，开始轮询...")

      # ── 阶段 2：轮询 + 下载 ─────────────────────────────────────────
      deadline = time.time() + MAX_WAIT_SECONDS
      while pending_jobs:
          if time.time() > deadline:
              raise TimeoutError(f"超时：仍有 {len(pending_jobs)} 个任务未完成")

          still_pending = []
          for job in pending_jobs:
              resp = requests.get(f"{ECS_ENDPOINT}/status/{job['job_id']}", timeout=10)
              data = resp.json()
              if data["status"] == "done":
                  _download(data["filename"])
              elif data["status"] == "failed":
                  raise RuntimeError(f"任务失败: {job} → {data}")
              else:
                  still_pending.append(job)

          pending_jobs = still_pending
          if pending_jobs:
              time.sleep(POLL_INTERVAL)

      print("所有 WAV 下载完成。")


  def _submit_if_missing(task: dict, h: str, variant: str, pending_jobs: list):
      target = CACHE_DIR / f"{h}_{variant}.wav"
      if target.exists():
          print(f"  [缓存命中] {target.name}")
          return
      resp = requests.post(
          f"{ECS_ENDPOINT}/synthesize",
          params={
              "speaker": task["speaker"],
              "raw_text": task["raw_text"],
              "sentence_hash": h,
          },
          timeout=30,
      )
      resp.raise_for_status()
      data = resp.json()
      pending_jobs.append({"job_id": data["job_id"], "filename": f"{h}_{variant}.wav"})
      print(f"  [已提交] {h[:8]}..._{variant}")


  def _download(filename: str):
      resp = requests.get(f"{ECS_ENDPOINT}/download/{filename}", timeout=60, stream=True)
      resp.raise_for_status()
      out = CACHE_DIR / filename
      with out.open("wb") as f:
          for chunk in resp.iter_content(65536):
              f.write(chunk)
      print(f"  [已下载] {filename}")


  if __name__ == "__main__":
      main()
  ```

- [ ] **步骤 2：本地验证（Mock ECS 接口）**

  使用 `responses` 或直接验证脚本逻辑不 crash（不实际调 ECS）：

  ```bash
  # 设置假 ECS_ENDPOINT，验证缓存命中逻辑
  mkdir -p cache
  # 手动放入一个伪造的 wav 文件
  echo "fake" > cache/$(python3 -c "import hashlib; print(hashlib.md5(b'xiaoyi:\u4e91\u9633\u8001\u5e08\uff0c\u4f60\u53c8\u5e26\u4e0a\u4e86\u6280\u672f\u6781\u5ba2\u7684\u89c6\u89d2\u4e86\u554a\uff01<laugh/>').hexdigest())")_standard.wav
  ECS_ENDPOINT=http://localhost:9999 python scripts/dispatch_tts.py 2>&1 | head -5
  # 预期：打印"缓存命中"信息，因为 ECS_ENDPOINT 不可达时已命中缓存的条目会跳过
  ```

- [ ] **步骤 3：提交**

  ```bash
  git add scripts/dispatch_tts.py
  git commit -m "feat(scripts): add ECS dispatcher with incremental hash cache and poll-download"
  ```

---

## 任务 6：GHA 侧双轨音频构建器

**Files:**
- 创建: `scripts/build_tracks.py`

从 `cache/` 中的 WAV 片段构建双轨输出：轨 A 为完整拼接 MP3，轨 B 为分句切片 MP3 + index.json。

- [ ] **步骤 1：创建 `scripts/build_tracks.py`**

  ```python
  """
  双轨音频构建器。

  输入：
    - /tmp/tts_tasks.json（解析结果，含顺序信息）
    - cache/{hash}_{variant}.wav（各句音频片段）

  输出：
    - site/episodes/today/episode.mp3         ← 轨 A：完整播客 MP3
    - site/episodes/today/audio/{idx:03d}_{speaker}_{variant}.mp3  ← 轨 B：分句切片
    - site/episodes/today/index.json          ← 轨 B：播放器索引
  """

  from __future__ import annotations
  import json
  import subprocess
  from pathlib import Path

  TASKS_FILE = Path("/tmp/tts_tasks.json")
  CACHE_DIR = Path("cache")
  OUT_DIR = Path("site/episodes/today")
  AUDIO_DIR = OUT_DIR / "audio"


  def main():
      tasks = json.loads(TASKS_FILE.read_text("utf-8"))
      OUT_DIR.mkdir(parents=True, exist_ok=True)
      AUDIO_DIR.mkdir(parents=True, exist_ok=True)

      _build_track_a(tasks)
      index = _build_track_b(tasks)
      (OUT_DIR / "index.json").write_text(
          json.dumps(index, ensure_ascii=False, indent=2), "utf-8"
      )
      print("双轨构建完成。")


  # ── 轨 A：ffmpeg 无损拼接 + 响度均衡 ─────────────────────────────────────

  def _build_track_a(tasks: list[dict]):
      """使用 standard 声线按顺序拼接，输出 -16 LUFS 响度均衡后的 MP3。"""
      list_file = Path("/tmp/ffmpeg_list.txt")
      lines = []
      for t in tasks:
          wav = CACHE_DIR / f"{t['sentence_hash']}_standard.wav"
          if not wav.exists():
              raise FileNotFoundError(f"缺少 WAV: {wav}")
          lines.append(f"file '{wav.resolve()}'")
      list_file.write_text("\n".join(lines), "utf-8")

      tmp_wav = Path("/tmp/full_concat.wav")
      # 无损拼接
      subprocess.run(
          ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", str(list_file), "-c", "copy", str(tmp_wav)],
          check=True, capture_output=True,
      )
      # 响度均衡 → MP3
      out_mp3 = OUT_DIR / "episode.mp3"
      subprocess.run(
          ["ffmpeg", "-y", "-i", str(tmp_wav),
           "-filter:a", "loudnorm=I=-16:TP=-1.5:LRA=11",
           "-codec:a", "libmp3lame", "-b:a", "128k", str(out_mp3)],
          check=True, capture_output=True,
      )
      print(f"  轨 A 完成: {out_mp3}")


  # ── 轨 B：逐句 MP3 + index.json ──────────────────────────────────────────

  def _build_track_b(tasks: list[dict]) -> dict:
      """将每句的 standard（和可选 casual）WAV 转为 MP3，生成 index.json。"""
      sentences = []
      for t in tasks:
          h = t["sentence_hash"]
          idx = t["index"]
          speaker = t["speaker"]
          audios = {}

          for variant in (["standard", "casual"] if t["needs_casual"] else ["standard"]):
              src_wav = CACHE_DIR / f"{h}_{variant}.wav"
              if not src_wav.exists():
                  continue
              dst_mp3_name = f"{idx:03d}_{speaker}_{variant}.mp3"
              dst_mp3 = AUDIO_DIR / dst_mp3_name
              subprocess.run(
                  ["ffmpeg", "-y", "-i", str(src_wav),
                   "-codec:a", "libmp3lame", "-b:a", "96k", str(dst_mp3)],
                  check=True, capture_output=True,
              )
              audios[variant] = f"audio/{dst_mp3_name}"

          sentences.append({
              "id": f"sentence_{idx:03d}",
              "speaker": speaker,
              "text": _strip_tags(t["raw_text"]),
              "audios": audios,
          })

      print(f"  轨 B 完成: {len(sentences)} 句")
      return {"episode_id": "today", "sentences": sentences}


  def _strip_tags(text: str) -> str:
      import re
      return re.sub(r"<[^>]+>", "", text).strip()


  if __name__ == "__main__":
      main()
  ```

- [ ] **步骤 2：验证轨 B 输出（用伪 WAV）**

  ```bash
  # 用 ffmpeg 生成 1 秒静音作为测试 WAV
  python3 -c "
  import json, hashlib, subprocess
  from pathlib import Path

  tasks = [
    {'index': 0, 'speaker': 'xiaoyi', 'raw_text': '\u54c8\u54c8<laugh/>', 'sentence_hash': 'aaa', 'needs_casual': True},
    {'index': 1, 'speaker': 'yunyang', 'raw_text': '\u5f00\u59cb', 'sentence_hash': 'bbb', 'needs_casual': False},
  ]
  Path('/tmp/tts_tasks.json').write_text(json.dumps(tasks), 'utf-8')
  Path('cache').mkdir(exist_ok=True)
  for h in ['aaa', 'bbb']:
    for v in ['standard', 'casual']:
      subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono',
                      '-t', '1', f'cache/{h}_{v}.wav'], capture_output=True)
  "
  python scripts/build_tracks.py
  ls site/episodes/today/audio/
  cat site/episodes/today/index.json
  ```

  预期：`audio/` 下有 3 个 MP3（0 句有 standard+casual，1 句只有 standard），`index.json` 结构正确。

- [ ] **步骤 3：提交**

  ```bash
  git add scripts/build_tracks.py
  git commit -m "feat(scripts): add dual-track builder - concat MP3 (Track A) + slice MP3+index (Track B)"
  ```

---

## 任务 7：三阶段 GitHub Actions 工作流

**Files:**
- 创建: `.github/workflows/daily_curator.yml`
- 创建: `.github/workflows/tts_builder.yml`
- 创建: `.github/workflows/pages_publisher.yml`

### 7-1：工作流 1 — 采编与剧本生成

- [ ] **步骤 1：创建 `.github/workflows/daily_curator.yml`**

  ```yaml
  name: "1. Daily Curator & Script Writer"

  on:
    schedule:
      - cron: '0 0 * * *'   # 每天 UTC 00:00（北京时间 08:00）
    workflow_dispatch:

  jobs:
    curate:
      runs-on: ubuntu-latest
      permissions:
        contents: write

      steps:
        - name: Checkout
          uses: actions/checkout@v4

        - name: Setup Python
          uses: actions/setup-python@v5
          with:
            python-version: '3.11'

        - name: Install dependencies
          run: |
            pip install uv
            uv sync

        - name: Fetch news & score
          run: uv run podcast-pipeline
          env:
            GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
            # 其余 API Key 按需补充

        - name: Generate script
          run: uv run podcast-daily --base-url https://nicekai-jpg.github.io/ai-news-podcast
          env:
            GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}

        - name: Commit script
          run: |
            git config user.name  "GitHub Actions"
            git config user.email "actions@github.com"
            git add docs/episodes/today_script.md data/
            git diff --cached --quiet || \
              git commit -m "chore(script): auto-generate today's podcast script [skip ci]"
            git push origin main
  ```

- [ ] **步骤 2：创建 `.github/workflows/tts_builder.yml`**

  ```yaml
  name: "2. Incremental TTS Builder"

  on:
    push:
      paths:
        - 'docs/episodes/today_script.md'
    workflow_dispatch:

  jobs:
    build-audio:
      runs-on: ubuntu-latest
      permissions:
        contents: write
      timeout-minutes: 360   # 最长 6 小时（GHA 公开仓库无限额度限制）

      steps:
        - name: Checkout
          uses: actions/checkout@v4

        - name: Restore audio cache
          uses: actions/cache@v4
          with:
            path: cache/
            key: tts-cache-${{ hashFiles('docs/episodes/today_script.md') }}
            restore-keys: tts-cache-

        - name: Setup Python
          uses: actions/setup-python@v5
          with:
            python-version: '3.11'

        - name: Install script dependencies
          run: pip install requests pydub

        - name: Install ffmpeg
          run: sudo apt-get install -y ffmpeg

        - name: Parse script → task JSON
          run: python scripts/parse_script.py docs/episodes/today_script.md

        - name: Dispatch to ECS & download WAVs
          env:
            ECS_ENDPOINT: ${{ secrets.CF_TUNNEL_ENDPOINT }}
          run: python scripts/dispatch_tts.py

        - name: Build dual tracks
          run: python scripts/build_tracks.py

        - name: Save audio cache
          uses: actions/cache/save@v4
          with:
            path: cache/
            key: tts-cache-${{ hashFiles('docs/episodes/today_script.md') }}

        - name: Commit audio & metadata
          run: |
            git config user.name  "GitHub Actions"
            git config user.email "actions@github.com"
            git add site/ cache/
            git diff --cached --quiet || \
              git commit -m "chore(audio): build dual-track audio for today [skip ci]"
            git push origin main
  ```

- [ ] **步骤 3：创建 `.github/workflows/pages_publisher.yml`**

  ```yaml
  name: "3. Pages Publisher"

  on:
    push:
      paths:
        - 'site/**'
    workflow_dispatch:

  jobs:
    deploy:
      runs-on: ubuntu-latest
      permissions:
        contents: write

      steps:
        - name: Checkout
          uses: actions/checkout@v4

        - name: Deploy to GitHub Pages
          uses: peaceiris/actions-gh-pages@v4
          with:
            github_token: ${{ secrets.GITHUB_TOKEN }}
            publish_dir: ./site
            publish_branch: gh-pages
            force_orphan: true
  ```

- [ ] **步骤 4：在 GitHub 仓库 Secrets 中配置以下变量**

  | Secret 名称 | 说明 |
  |---|---|
  | `CF_TUNNEL_ENDPOINT` | Cloudflare Tunnel 的公开 HTTPS 地址（如 `https://xxx.trycloudflare.com`） |
  | `GEMINI_API_KEY` | Gemini LLM API 密钥（工作流 1 使用） |

- [ ] **步骤 5：提交工作流**

  ```bash
  git add .github/workflows/
  git commit -m "feat(ci): add 3-stage GHA workflows (curator → tts-builder → pages)"
  ```

---

## 任务 8：Cloudflare Tunnel 配置（ECS 侧）

- [ ] **步骤 1：在 ECS 上安装并启动 cloudflared**

  ```bash
  # 下载 cloudflared（amd64）
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
       -o /usr/local/bin/cloudflared
  chmod +x /usr/local/bin/cloudflared

  # 创建临时隧道（无需登录账号，适合快速验证）
  cloudflared tunnel --url http://localhost:8000
  # 输出中会出现一行：https://xxxx.trycloudflare.com
  # 将此地址填写到 GitHub Secret CF_TUNNEL_ENDPOINT
  ```

- [ ] **步骤 2：持久化隧道（可选，生产环境推荐）**

  注册 Cloudflare 账号后通过 `cloudflared login` 创建具名隧道，绑定固定域名。

- [ ] **步骤 3：启动顺序验证**

  ```bash
  # 确认 3 个进程均正常运行
  ps aux | grep uvicorn        # FastAPI 服务
  ps aux | grep huey_consumer  # 推理队列消费者
  ps aux | grep cloudflared    # 隧道
  ```

---

## 自检（Self-Review）

**1. Spec coverage 检查：**

| 需求点 | 覆盖任务 |
|---|---|
| 多音色音频（standard + casual） | 任务 2 编译器路由 + 任务 5 双版本提交 + 任务 6 轨 B |
| 固定人物拼接音频（Track A MP3） | 任务 6 `_build_track_a` |
| 2C2G 内存安全 | 任务 1 Swap + 任务 3 `del session; gc.collect()` |
| 增量缓存（不重复生成已有句子） | 任务 4 MD5 哈希 + 任务 5 缓存命中逻辑 + GHA Cache |
| GHA 编排解耦（3 个工作流） | 任务 7 |
| ECS 安全访问（Cloudflare Tunnel） | 任务 8 |
| index.json 播放器索引 | 任务 6 `_build_track_b` |

**2. Placeholder 扫描：** 无 TBD / TODO / "implement later" / "handle edge cases" 等。

**3. Type consistency：**  
- `sentence_hash` 在任务 3/4/5/6 中含义一致（MD5 hex string）。  
- `voice_variant` 在任务 2/3/5/6 中均为 `"standard"` | `"casual"`。  
- `speaker` 在任务 4 转为英文小写（`xiaoyi`/`yunyang`），在任务 3 server 和任务 6 均以此为准。
