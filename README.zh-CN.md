# AI 每日先锋（全自动化 AI 新闻播客与播报生成器）

English README: `README.md`

本项目每天自动抓取 AI 领域的 RSS/Atom 新闻源，进行智能筛选和摘要，并使用大语言模型（LLM）生成播客文稿与纯文本日报。同时内置文本转语音 (TTS) 以及播客 RSS Feed 生成。

## 核心功能
- **全自动新闻管家**：自动拉取 `config/sources.yaml` 中的订阅源，解析内容并根据价值进行智能去重（语义相似度度量）与打分。
- **大语言模型（LLM）写作**：基于 **MiniMax-M3**（通过 MiniMax 兼容 OpenAI 格式的接口）的多 Agent 写作框架。包含 Editor Agent 提炼今日大纲，与 Writer Agent 将大纲转化为符合 SSML 规范的双人自然对话剧本。
- **高质量语音合成 (TTS)**：以 **CosyVoice 2**（利用 `CosyVoice2-0.5B` 进行零样本/小样本声音克隆）为核心的合成引擎，对两个独立的主持人（晓晓与博文）音色进行高拟真合成，并以 **Edge-TTS** 作为兜底。
- **音频混音与后处理**：使用 **pydub** 与 **FFmpeg** 自动为每个对话分段加入适当的静音停顿（智能句读），混合背景音乐 (BGM) 并完成音量响度均衡归一化（`loudnorm`）。
- **开源免托管部署**：配置 GitHub Actions 全自动每日运行，音频与 Feed 全托管在免费的 GitHub Pages 上，零费用维护。

## 怎么“调用”（本地运行）

### 1. 配置环境变量 `.env`
在项目根目录创建或编辑 `.env` 文件，填入 MiniMax Token Plan 专属 API Key：
```env
MINIMAX_API_KEY="your-minimax-api-key"

### 2. 环境安装（推荐使用 uv）
推荐用 [uv](https://docs.astral.sh/uv/)（更快、更稳定）：
```bash
git clone https://github.com/<your-username>/ai-news-podcast.git
cd ai-news-podcast
uv sync
```

### 3. 执行功能脚本
项目已重构为标准的 Python 包结构，不同功能可以通过注册的 CLI 命令直接运行：

**A. 生成完整播客 (MP3 + RSS 页面 + 记录)**
```bash
uv run podcast-daily --base-url http://localhost
```
*提示：默认 TTS 为 CosyVoice 2 声音克隆。本地合成音频需要先运行 `bash scripts/setup_cosyvoice_env.sh` 配置环境并指定 `COSYVOICE_MODEL_DIR`。可以加上 `--no-audio` 跳过音频合成，仅生成剧本与简报。*

**C. 仅发布已合成音频（GHA Job 2 后）**
```bash
uv run podcast-publish --date 2026-06-11
```

**B. 生成今日 AI 科技新闻日报 (Markdown)**
本脚本调用 **MiniMax-M3** 大模型生成 Markdown 日报。
```bash
uv run podcast-report
```

## TTS 引擎评测与选型
我们针对双人对谈播客场景，对多种 Text-to-Speech (TTS) 模型（包括 Edge-TTS、ChatTTS、CosyVoice 2、F5-TTS、GPT-SoVITS、Kokoro、MOSS-TTS）进行了本地 CPU 推理实测与三维评估。
* 关于模型评测对比、2C2G 服务器串行队列设计、服务端代码实现、前端双缓冲播放器设计及 GHA 工作流声明，请参阅 [TTS 系统设计与技术落地全景指南](docs/tts_complete_guide.md)。
* GHA CosyVoice 2 部署问题与修复记录见 [gha_cosyvoice2_deployment_log.md](docs/gha_cosyvoice2_deployment_log.md)。

## 如何保证内容质量与专业性
你可以通过编辑 `config/` 目录下的文件来定义你的关注焦点：
- **新闻源**：编辑 `config/sources.yaml`，按需启用/禁用 RSS 来源。
- **配置规则**：编辑 `config/config.yaml` 灵活调节抓取关键字 (`selection.include_keywords`)，更换 TTS 念稿声线等。

## 启用免费播客托管 (GitHub Pages)
1. 把本项目推到 GitHub 公共仓库。
2. 仓库设置 **Settings** -> 左侧菜单栏 **Pages**。
3. **Build and deployment** -> 选择 **Deploy from a branch**。
4. Branch 下拉选 `gh-pages`，Folder 下拉选 `/(root)` 并保存。
5. 几分钟后，在各大播客 App 订阅你的地址：
   `https://<你的GitHub用户名>.github.io/<仓库名>/feed.xml`

*注意：媒体资源（MP3 音频和智能句读音频切片）仅在 `gh-pages` 分支中增量托管，不提交到主分支 `main`，从而保证您的开发分支轻巧、拉取极速。*

## 每天自动更新与音频防膨胀机制
- **配置密钥**：在 GitHub 仓库页面进入 **Settings** -> 左侧菜单栏 **Secrets and variables** -> **Actions**，点击 **New repository secret**，添加名为 `MINIMAX_API_KEY` 的密钥，填入你的 MiniMax API Key。
- **播客每日更新**：内置 GitHub Actions 工作流 `daily.yml`。每天定时运行自动构建，并将简报和播客文本提交到 `main` 分支，合成的大音频和切片则直接同步部署到 `gh-pages`。
- **定期历史清理（防膨胀）**：内置历史清理工作流 `prune_pages.yml`，设定每月 1 号执行（也支持随时手动触发）。它会备份最近 30 天有效播客的音频与切片数据，随后**彻底清空重置 `gh-pages` 分支的提交历史**（使 Commit 历史数归为 1），并把备份的 30 天数据强推上去，实现物理删除旧数据并释放 Git 大文件空间。
- **手动触发**：你可以在仓库的 Actions 页面，手动选择对应的 Workflow 并点击 **Run workflow** 强制立刻运行。
