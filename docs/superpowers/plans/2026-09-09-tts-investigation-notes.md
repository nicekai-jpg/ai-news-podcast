# TTS 合成调查笔记（2026-09-09）

> 调查目标：daily.yml `tts` job 最近单次跑 2h27m54s（09-09 期），整条流水线逼近 240 分钟超时。
> 本文回答 4 个问题：每段合成清单 / 拼装路径 / 2.5h 去向 / 优化选项。全部结论基于代码 file:line 与
> `gh api` 实测数据（run 34291567125 等 7 次运行），未做任何代码改动。

## 结论摘要

1. **每段对白被合成了两遍，lively 版从未进入成品 MP3（约 50% 纯浪费）**。
   `cosyvoice2.py:128-133` 从 `config.yaml` 的 `ref_audio.host_*.keys()` 取出 `["professional", "lively"]`
   两个音色变体；`cosyvoice2.py:139-164` 的 chunk 循环对**每一条 Host 台词 × 每一个变体**都调用
   `synthesize_chunk()`（外层 `for var in variants` 在 142 行）。而成品只拼 `professional`：
   `cosyvoice2.py:166-169` `default_variant = "professional"`，`assemble_dialogue_audio` 只接收
   `segments_by_variant[default_variant]`。`chunk_001_lively.mp3` 和 `chunk_001_professional.mp3`
   是**同一条台词、同一主持人的两个参考音色版本**（不是 A/B 主持各自的 take）；A/B 归属由
   `playlist.json` 的 `host`/`voices` 字段记录（cosyvoice2.py:73-82 实测 2026-09-07/playlist.json：
   chunk_001 host=A voices=host_a_professional/host_a_lively，chunk_002 host=B）。
2. **成品 MP3 = professional 段落拼接 + 静音垫 + （当前关闭的）BGM + ffmpeg loudnorm**。
   `tts_postprocess.py:93-131`（pydub 拼接/静音抖动）→ `tts_postprocess.py:134-166`
   `finalize_episode_mp3`（BGM mix → 导出 pre-norm mp3 → `run_loudnorm` ffmpeg）。`bgm_path` 为空
   （config.yaml:13），`mix_bgm` 直接原样返回（tts_postprocess.py:42-43）。实测 09-09 期该阶段仅 **34 秒**
   （最后一次合成 yield 02:12:42 → "Audio saved" 02:13:16，run 34291567125 日志）；本地
   2026-09-07.mp3 ffprobe：634s / 32kbps / 24kHz。playlist.json 只决定网页播放器放哪个 chunk 文件
   （player.js:592-593），**不决定成品 MP3 内容**。
3. **2.5h 的 97% 是 CosyVoice CPU 串行推理，与缓存无关**。09-09 期 TTS job（2h27m54s）实测拆分：
   job 前置（checkout/uv/apt/cache restore/Setup CosyVoice）1m26s；Director Agent LLM 标注 3m14s；
   模型加载+wetext 下载 2m14s；**纯推理 2h20m20s（188 次调用，均 RTF 7.6，合成 1297.7s 语音，其中一半
   是被丢弃的 lively）**；后处理 34s。缓存每次全命中（`Setup CosyVoice environment` 步 7 次运行全部
   13-16 秒，daily.yml:239-249 + setup_cosyvoice_env.sh 逐条跳过已有目录），不存在 cache miss 拖慢。
4. **合成时长 ≈ 每条 Host 台词 2 分钟左右（两个变体合计），且与 runner CPU 波动强相关**。
   7 次运行：47 段→1h42m、44 段→1h55m、62 段→2h31m、44 段→1h35m、49 段→1h25m、46 段→1h36m、
   83 段→2h26m。62 段(09-05)比 83 段(09-09)还慢，说明 runner 算力噪声（ubuntu-latest 共享 CPU，
   CPU-only torch，setup_cosyvoice_env.sh:25）贡献了显著方差；剧本长度（2898-4405 字，全部超出
   config.yaml:221 的 total_chars [800,2500] 上限）决定下限。
5. **最大的单点优化 = 跳过 lively 冗余合成，可砍掉约一半推理时间（本次 run 可省约 70 分钟）**，
   成品 MP3、RSS/Apple Podcasts 内容完全不变。唯一功能损失是网页播放器的 per-host 音色切换
   "lively" 档没有专属音频（player.js:593 已有回退 professional 的逻辑，选 lively 会静默播放
   professional 文件）。此改动不触碰 ref_audio 交叉映射与 venv（两者都不必动）。
6. **超时风险目前可控**：timeout-minutes 240（daily.yml:214），近期最差 2h31m，余量约 1.5h；
   但 09-05/09-09 剧本已涨到 62/83 段（4405/4168 字），若段数继续增长、再叠加 runner 慢速噪声，
   余量会快速收缩。另注：整条流水线 wall time（2h35m）里还含 GitHub 定时触发延迟——cron 21:43 UTC，
   实际 run 创建于 23:38 UTC（约 1h55m 排队，GitHub 侧，不计入 job 超时）。

## 证据表

| # | 结论 | 证据 |
|---|------|------|
| 1 | variants 来自 config ref_audio 的 key | `src/ai_news_podcast/pipeline/tts_backends/cosyvoice2.py:128-133`；`config/config.yaml:16-26`（professional+lively 双档，host_a↔host_b 交叉映射） |
| 2 | 每条台词 × 每个变体 × 每句都合成 | `cosyvoice2.py:139-164`（139 chunk 循环、142 `for var in variants`、140 `split_text_into_sentences(max_chars=80)`、151-153 `synthesize_chunk` 调用）；句子切分实现 `tts_parser.py:50-70` |
| 3 | 成品只用 professional | `cosyvoice2.py:166-169`（`default_variant`），`cosyvoice2.py:167-175`（`assemble_dialogue_audio` + `finalize_episode_mp3`） |
| 4 | chunk 目录双变体文件（本地实测） | `site/episodes/2026-09-0{2..7}/`：每段都有 `chunk_NNN_professional.mp3` + `chunk_NNN_lively.mp3`（09-04:44/44、09-05:62/62、09-06:44/44、09-07:49/49）；`cosyvoice2.py:57-71` 写出逻辑 |
| 5 | playlist.json 记录 A/B 与双档，播放器按它选文件并回退 | `cosyvoice2.py:73-90`；`site/episodes/2026-09-07/playlist.json`（chunk_001 host=A，audios 两档，voices=host_a_*）；`site_builder/static/player.js:592-593`（`audios[variant] \|\| audios['professional']` 回退）、`player.js:652-665`（音色 pill 由 VOICES_CONFIG 渲染） |
| 6 | VOICES_CONFIG 硬编码双档 | `site_builder/html_gen.py:78-91` |
| 7 | 拼装路径与 loudnorm | `tts_postprocess.py:93-131`（拼接+静音抖动 400-800ms，config.yaml:34-37）、`tts_postprocess.py:134-166`、`tts_postprocess.py:61-90`（ffmpeg loudnorm，`-ar 24000`） |
| 8 | BGM 当前为 no-op | `config/config.yaml:13`（`bgm_path: ""`）、`tts_postprocess.py:42-43` |
| 9 | Director Agent 在合成步骤内触发（脚本无情感标签时） | `tts_engine.py:133-160`、批量 10 段 `tts_engine.py:37,157`；实测 09-07/08/09 剧本 `grep -c 'laughter\|breath…'` = 0 → 每天都触发 |
| 10 | GHA 纯 CPU | `scripts/setup_cosyvoice_env.sh:25`（`torch==2.3.1 --index-url .../whl/cpu`）、daily.yml tts job 无 GPU runner；`docs/gha_cosyvoice2_deployment_log.md:90-92` 明确"纯 CPU 逐句串行，单期 30-90 分钟"预期 |
| 11 | 缓存 key 与路径 | `daily.yml:239-246`（key `cosyvoice-gha-v7`，paths: `~/cosyvoice_src`、`~/cosyvoice_models`、`~/cosyvoice_venv`）；setup 脚本对已存在目录全部跳过（`setup_cosyvoice_env.sh:10-18,56-68`） |
| 12 | Setup 步实测 13-16s（全命中） | `gh api .../runs/<id>/jobs` 7 次："Setup CosyVoice environment" = 13s/16s/13s/14s/14s/12s/13s |
| 13 | 合成步实测时长（7 次） | 同上："Synthesize episode audio" = 1h41m54s / 1h55m13s / 2h31m12s / 1h34m55s / 1h24m44s / 1h36m8s / 2h26m24s |
| 14 | 09-09 期 job 内部时间线 | run 34291567125 job 102280504515 日志：Director 23:46:54→23:50:08；首个推理 "synthesis text" 23:52:22；最后 yield 02:12:42；"Audio saved" 02:13:16 |
| 15 | 188 次推理调用 / RTF / 语音量 | 同上日志：`yield speech len` 188 条；`rtf` 均值 7.587、最大 27.39；语音合计 1297.68s（≈2×成品 649s） |
| 16 | 双重合成的日志痕迹 | 同一 "synthesis text …" 警告成对出现（如 23:53:07 与 23:53:22 同句重复）= professional+lively 各合成一次 |
| 17 | 成品音频规格 | ffprobe `site/episodes/2026-09-07.mp3`：duration=634.0s，bit_rate=32006（≈612s professional 语音+静音垫，32kbps/24kHz） |
| 18 | wetext 每次重新下载（未缓存路径） | 日志 1375/1535 行：23:50:21 与 23:51:31 两次 "Downloading Model to directory: ~/.cache/modelscope/hub/pengzhendong/wetext"；`daily.yml:242-246` 未含 `~/.cache/modelscope` |
| 19 | 剧本规模（超出配置上限） | site/episodes/*.txt 实测字数：09-02:4121、09-03:3080、09-04:3021、09-05:4405、09-06:2898、09-07:3728、09-08:3786、09-09:4168；上限 `config.yaml:221` total_chars [800,2500] |
| 20 | artifact 传递整包 chunk 目录（含双档） | `daily.yml:277-284`；publish 只把 playlist.json 提交 main（`daily.yml:354`），mp3/chunk 均只在 gh-pages |
| 21 | 超时与并发组 | `daily.yml:214`（timeout-minutes: 240）、`daily.yml:23-25`；定时延迟：cron `43 21 * * *`（daily.yml:5）vs run createdAt 23:38:43Z（09-09 期） |

## 时间去哪了（实测数字）

### run 34291567125（09-09 期）TTS job 全拆解（总 2h27m54s）

| 阶段 | 时长 | 占比 | 证据 |
|------|------|------|------|
| job 前置：checkout/pull/uv sync/apt/Cache CosyVoice(38s)/Setup CosyVoice(13s) | ~1m26s | 1% | steps 时间戳 23:45:28→23:46:54 |
| Director Agent 情感标注（9 个 LLM batch，MiniMax-M3） | 3m14s | 2.2% | 日志 23:46:54→23:50:08 |
| CosyVoice2 模型加载 + wetext/modelscope 下载 | ~2m14s | 1.5% | 日志 23:50:08→23:52:22（首个 synthesis text） |
| **CosyVoice CPU 推理（188 次调用，串行）** | **2h20m20s** | **95%** | 日志 23:52:22→02:12:42；均 RTF 7.6 |
| 后处理：pydub 拼接导出 + ffmpeg loudnorm | 34s | 0.4% | 02:12:42→02:13:16 |
| artifact 上传 + 收尾 | ~6s | ~0% | steps 02:13:18→02:13:22 |

推理内部：188 次调用 = 83 条台词 × 2 变体（166）+ 22 次长段二次切句（>80 字，cosyvoice2.py:140）。
lively 一半（约 94-105 次调用，≈649s 语音）**不出现在成品里**，只落盘为网页播放器的预览文件。
单次调用平均 ~45s，产出约 6.9s 语音；短句调用效率极差（"行，你说。"1s 语音耗 13-16s，RTF 13.4；
"[laughter]拜拜！"1.3s 语音耗 17s）。

### 近 7 次运行横向对比（gh api steps 数据）

| run id | 期 | 合成步 | 台词段数 | 字数 | 分钟/段 |
|--------|-----|--------|---------|------|---------|
| 33695514617 | 09-03 | 1h41m54s | 47 | 3080 | 2.17 |
| 33817915607 | 09-04 | 1h55m13s | 44 | 3021 | 2.62 |
| 33928946578 | 09-05 | 2h31m12s | 62 | 4405 | 2.44 |
| 33998191453 | 09-06 | 1h34m55s | 44 | 2898 | 2.16 |
| 34066018614 | 09-07 | 1h24m44s | 49 | 3728 | 1.73 |
| 34171052437 | 09-08 | 1h36m8s | 46 | 3786 | 2.09 |
| 34291567125 | 09-09 | 2h26m24s | 83 | 4168 | 1.76 |

要点：①时长与段数大致线性（≈2 min/段，双变体合计）；②62 段(09-05)比 83 段(09-09)更慢 →
runner CPU 噪声是主要方差来源，与缓存无关（Setup 全部 13-16s）；③08-09 剧本连续 8 天超出
total_chars 配置上限 1.2-1.8 倍，是段数/时长上涨的根源。

## 优化选项清单

| # | 选项 | 改动点 | 预期收益 | 风险 | 建议 |
|---|------|--------|----------|------|------|
| a | **跳过 lively 冗余合成**（只合成默认变体） | `cosyvoice2.py:128-164` 的 variants 循环改为仅 default（或新增 `tts.synthesis_variants` 配置项，config.yaml）；`cosyvoice2.py:131-133` voice_maps、`cosyvoice2.py:57-71` 导出随之只写单档；可选同步精简 `html_gen.py:78-91` VOICES_CONFIG | **-60~-75 min/run（≈50%）**，最大单点杠杆；artifact/gh-pages 每期少 ~5.5MB | 网页播放器"lively"档无专属音频：player.js:593 会回退 professional（静默生效），pill 仍显示（html_gen.py 硬编码）；若删 VOICES_CONFIG 的 lively 则 pill 消失。**不触碰** ref_audio 交叉映射与 venv；tests 无双档断言（tests/test_tts_synthesize.py mock 引擎） | **强烈建议**。需产品决策：是否保留网页端音色预览功能。若要保留，可仅对前 5 段合成 lively 作为试听 |
| b | 缓存补漏：加入 `~/.cache/modelscope` | `daily.yml:242-246` paths 增加 `~/.cache/modelscope` | 每 run 省 1-2 min（wetext FST 重复下载，日志 1375/1535 行） | 极低；同 key 旧缓存无该路径，首跑会落空一次再回填 | 建议顺手做。SAFE |
| c | 段内并行合成（2 进程 × 2 线程） | `cosyvoice2.py:139-164` 循环改多进程/队列 | 乐观 -30%，但 torch 本就多线程吃满 4 vCPU（均 RTF 7.6），收益不确定且有 OOM（模型×2 进程，runner 7GB）风险 | 中高：结果拼接顺序、异常处理、内存 | 不建议先做；a 完成后再评估 |
| d | GPU runner / 本地合成 | daily.yml runs-on + setup 脚本 CUDA wheel | 理论 10-30x | 成本（付费 larger runner）或破坏全自动 CI；venv/torch 换 CUDA wheel 属敏感 TTS 环境改动 | 仅在 a 之后仍不够时考虑 |
| e | 控制剧本规模回配置上限 | writer/prompt 侧执行 config.yaml:221 total_chars ≤2500 | 段数 -30~40% → 约 -40~-60 min；副作用是单期时长从 10.6min 回到目标 5-10min（ffprobe 09-07=634s 已超） | 内容更短（产品取舍）；不碰 TTS | 建议与 a 二选一或叠加，先 a（收益/风险比更高） |
| f | Director Agent 移出合成步（提前到 writer 或落盘缓存） | `tts_engine.py:133-160`；或 writer 产出的脚本带标签后跳过 | -3 min/run | 低；标注质量校验逻辑要保持（tts_engine.py:57-108 的无损校验） | 可选，收益小 |
| g | 加大 timeout 至 300 / 保持 240 | `daily.yml:214` | 仅风险缓冲，不减时 | 无 | 现状 240min vs 最差 2h31m 余量足够；做了 a 之后更宽裕，**无需改** |

### 敏感性标注

- **SAFE**：b（缓存路径）、f（标注时机）、g（不改）。
- **TOUCHES-TTS-SENSITIVE**：a 与 c 都会改 `pipeline/tts_backends/cosyvoice2.py` 主循环——但
  **均不需要动** `config.yaml:16-26` 的 ref_audio 交叉映射（AGENTS.md 明示故意为之）和
  `~/cosyvoice_venv`/`cosyvoice-gha-v7` 缓存体；唯一产品影响是网页播放器 lively 预览。
- d 若换 GPU 则 venv/torch/缓存 key 全动，敏感度最高。

### 最重要的一句话

**成品 MP3（RSS/Apple Podcasts 分发的那条音轨）只用 professional 变体；lively 变体每天多花约
70 分钟纯 CPU 推理、只为网页播放器一个可回退的音色预览。把它关掉是零内容风险、约 50% 的
TTS 提速，且不触碰 ref_audio 映射与 venv。**
