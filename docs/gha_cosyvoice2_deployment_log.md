# GHA CosyVoice 2 TTS 部署问题记录

> 记录将播客 TTS 从 Edge-TTS 迁移到 CosyVoice 2、并在纯 GitHub Actions（无 ECS）上跑通每日流水线过程中遇到的问题、修复与当前状态。
>
> 相关文件：`.github/workflows/daily.yml`、`scripts/setup_cosyvoice_env.sh`、`scripts/gha_tts_cosyvoice.py`、`src/ai_news_podcast/pipeline/cosyvoice_backend.py`
>
> 实施计划：`docs/superpowers/plans/2026-06-11-gha-cosyvoice2-tts.md`

---

## 目标架构

`daily.yml` 拆为三个 Job：

| Job | 职责 |
|-----|------|
| **content** | `podcast-pipeline` → `podcast-report` → `podcast-daily --no-audio` |
| **tts** | CosyVoice 2 零样本合成 MP3 → `podcast-publish` |
| **deploy** | `peaceiris/actions-gh-pages`（`keep_files: true`） |

历史节目数据通过 `keep_files: true` 与 `config.build.keep_last: 30` 保留，不会因迁移清空。

---

## 核心结论（先看这个）

1. **CosyVoice 依赖不能装进 uv 的 `.venv`**。与项目主依赖（尤其 torch / huggingface-hub 版本）混装会导致 `onnxruntime` 等原生扩展导入失败。
2. **应使用独立虚拟环境** `~/cosyvoice_venv` + **锁定 torch 2.3.1**（与 CosyVoice 上游 `requirements.txt` 一致），合成步骤用 `$HOME/cosyvoice_venv/bin/python` 执行。
3. **依赖不要逐个手补**。从 CosyVoice 上游 `requirements.txt` 过滤 GPU/服务类包后批量安装，再单独补 Linux CPU 缺的包。
4. **上游 API 会变**。`CosyVoice2.__init__` 已移除 `load_onnx` 参数；模型存在性应检查 `cosyvoice2.yaml` 而非 `cosyvoice.yaml`。
5. **截至 2026-06-12，完整三 Job 流水线尚未成功跑通**。最后阻塞点集中在 `openai-whisper` 的安装与 CosyVoice `frontend.py` 的硬依赖。

---

## 问题与修复对照表

| # | 失败阶段 | 现象 | 根因 | 修复 |
|---|----------|------|------|------|
| 1 | content / Commit | `git add site/episodes/*` 退出码 1 | `site/` 在 `.gitignore` 中 | 改用 `git add -f` 强制添加当期 `.txt` / `.html` |
| 2 | tts / 合成入口 | 无法解析 `ai-news-podcast` 包 | 裸 `python` 未在 uv 环境中 | Job 2 先 `uv sync`，发布步骤用 `uv run` |
| 3 | tts / 合成 | `ModuleNotFoundError: torchaudio` | CosyVoice 包装到系统 pip，非项目 venv | `setup_cosyvoice_env.sh` 改用 `uv pip`（后已废弃，见 #10） |
| 4 | tts / 合成 | `ModuleNotFoundError: modelscope` | 手列依赖遗漏 | 加入 `modelscope`（后改为从 requirements 批量安装） |
| 5 | tts / 模型加载 | `CosyVoice2.__init__() got an unexpected keyword argument 'load_onnx'` | 克隆的 CosyVoice 主分支 API 已变更 | `cosyvoice_backend.py` 去掉 `load_onnx=False` |
| 6 | content / Commit | `git push` rejected (non-fast-forward) | workflow 运行期间 main 有新提交（如手动 push 修复） | commit 后增加 `git pull --rebase origin main` 再 push |
| 7 | tts / 模型加载 | `ModuleNotFoundError: lightning` | 手列依赖遗漏 | 从上游 `requirements.txt` 过滤安装 |
| 8 | tts / Setup | `openai-whisper` 构建失败（`pkg_resources`） | `uv pip` 构建隔离 + 新版 setuptools 移除 `pkg_resources` | 从 requirements 过滤掉 whisper；单独处理（见 #11） |
| 9 | tts / 合成 | `ModuleNotFoundError: onnxruntime` | 上游 requirements 在 Linux 只声明 `onnxruntime-gpu` | 过滤掉 GPU 包后显式 `pip install onnxruntime==1.18.0` |
| 10 | tts / 合成 | `ImportError`（onnxruntime 原生扩展加载失败） | CosyVoice 依赖污染 uv `.venv`，且安装了 torch 2.11 而非 2.3.1 | **独立 `~/cosyvoice_venv`** + **torch==2.3.1**；合成改用 `$HOME/cosyvoice_venv/bin/python` |
| 11 | tts / 合成 | `ModuleNotFoundError: whisper` | `cosyvoice/cli/frontend.py` 顶层 `import whisper`；零样本路径调用 `whisper.log_mel_spectrogram` | 必须在 CosyVoice venv 安装 `openai-whisper`（非可选） |
| 12 | tts / Setup | `openai-whisper` 仍构建失败 | pip 默认 build isolation 使用无 `pkg_resources` 的 setuptools | `PIP_NO_BUILD_ISOLATION=1` + venv 内 `setuptools>=69,<81`（**待验证**） |

---

## 当前环境方案（`setup_cosyvoice_env.sh`）

```
~/cosyvoice_src/          # CosyVoice 源码（git clone --recursive）
~/cosyvoice_models/       # HuggingFace 模型 CosyVoice2-0.5B
~/cosyvoice_venv/         # 独立 Python venv（与 uv .venv 隔离）
```

安装顺序要点：

1. `python3 -m venv ~/cosyvoice_venv`
2. `pip install setuptools>=69,<81`（兼容 whisper 等老旧 setup.py）
3. `torch==2.3.1` + `torchaudio==2.3.1`（CPU wheel）
4. 过滤后的 CosyVoice `requirements.txt`（去掉 deepspeed / tensorrt / gradio / openai-whisper 等）
5. 显式 `onnxruntime==1.18.0`（Linux CPU）
6. `PIP_NO_BUILD_ISOLATION=1 pip install openai-whisper==20231117`
7. `pip install --no-deps -e .` + `pydub` / `PyYAML`（供 `gha_tts_cosyvoice.py` 导入项目代码）
8. 启动前 `PYTHONPATH=$COSY_SRC:$COSY_SRC/third_party/Matcha-TTS`

GHA 缓存 key：`cosyvoice-gha-v3`（含 src、models、venv 三路径）。

---

## 尚未验证 / 待办

- [ ] 确认 `PIP_NO_BUILD_ISOLATION=1` 后 whisper 安装成功，且模型能完整加载
- [ ] 首次 CPU 推理耗时（约 50 句对话 × 零样本）是否能在 `timeout-minutes: 120` 内完成
- [ ] 合成成功后 `2026-06-12.mp3` 提交 main 并 deploy 到 gh-pages
- [ ] 若 CosyVoice 持续不稳定，可临时将 `config.yaml` 中 `backend: hybrid` 回退 Edge-TTS 保发布

### 备选 whisper 安装方案（若 #12 仍失败）

1. 改用带 wheel 的较新版本：`pip install openai-whisper`（不锁 20231117）
2. 在 setup 中预装构建依赖：`pip install setuptools wheel pkg_resources` 后 `--no-build-isolation`
3. 从源码 pin 到 CosyVoice 兼容 commit，避免 `frontend.py` 强依赖 whisper 的旧版 API 差异

### 性能预期

CosyVoice 2 在 GHA `ubuntu-latest` **纯 CPU** 上为逐句串行推理，单期节目（约 50 段）可能需要 **30–90 分钟**。这与 Edge-TTS 秒级并行形成鲜明对比，属预期行为而非 bug。

---

## 本地调试命令

```bash
# 安装 CosyVoice 环境
bash scripts/setup_cosyvoice_env.sh

# 合成（需已有 site/episodes/YYYY-MM-DD.txt）
export COSYVOICE_MODEL_DIR=~/cosyvoice_models/CosyVoice2-0.5B
export PYTHONPATH=~/cosyvoice_src:~/cosyvoice_src/third_party/Matcha-TTS
~/cosyvoice_venv/bin/python scripts/gha_tts_cosyvoice.py \
  --script site/episodes/2026-06-12.txt \
  --output site/episodes/2026-06-12.mp3

# 仅发布（uv 环境）
uv run podcast-publish --date 2026-06-12
```

---

## 降级路径

`config/config.yaml`：

```yaml
tts:
  backend: hybrid   # cosyvoice2 失败时自动回退 edge-tts
```

GHA Job 2 当前硬编码 `backend=cosyvoice2`（见 `gha_tts_cosyvoice.py`）。若需 CI 降级，可改为读取 config 的 `hybrid` 或增加 workflow 输入开关。

---

## 变更历史（commits 摘要）

| 主题 | 说明 |
|------|------|
| 三 Job workflow | `content → tts → deploy` 拆分 |
| `git add -f` | 强制提交 gitignore 下的 `site/episodes/` |
| `load_onnx` 移除 | 对齐 CosyVoice 主分支 API |
| `git pull --rebase` | 避免并发 push 冲突 |
| 独立 cosyvoice venv | 隔离依赖 + torch 2.3.1 锁定 |
| requirements 批量安装 | 替代逐个补包 |
| whisper 构建 | `PIP_NO_BUILD_ISOLATION=1` |

---

*最后更新：2026-06-12*
