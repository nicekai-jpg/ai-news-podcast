# AI Daily Pioneer 🎙️ (AI 每日先锋)

[![GitHub license](https://img.shields.io/github/license/nicekai-jpg/ai-news-podcast?style=flat-square)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![LLM Engine](https://img.shields.io/badge/LLM--Engine-MiniMax--M3-red?style=flat-square&logo=openai)](https://api.minimaxi.com/)
[![TTS Engine](https://img.shields.io/badge/TTS--Engine-CosyVoice2-green?style=flat-square&logo=googleplay)](https://github.com/FunAudioLLM/CosyVoice)
[![Automation](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blueviolet?style=flat-square&logo=githubactions)](.github/workflows/daily.yml)
[![Hosting](https://img.shields.io/badge/Hosting-GitHub%20Pages-orange?style=flat-square&logo=githubpages)](https://pages.github.com/)

[**English README** (英文说明文档)](README.md)

**AI Daily Pioneer (AI 每日先锋)** 是一个全自动、工业级的 AI 播客与新闻日报生成器。它每天定时抓取精心挑选的 RSS/Atom 科技新闻源，使用 NLP 算法进行智能聚类、去重与打分，编排 **多 Agent 协同写作系统** 自动撰写双人对话播客剧本，并使用 **CosyVoice 2** 进行声音克隆和拟真合成，最终将音频及播放器网页全自动发布部署至 **GitHub Pages**。

---

## 🗺️ 系统数据流水线与架构

项目被设计为一套端到端、高度自动化的线性执行流水线（Pipeline）：

```
[精选 RSS/Atom 新闻信源]
           │
           ▼  (fetcher.py)
   [httpx 异步抓取 & trafilatura 网页正文深度解析]
           │
           ▼  (processor.py)
   [智能去重：TF-IDF + 关键词 overlap 重合度 + 语义相似度度量]
           │
           ▼
   [DBSCAN 聚类分析 & 5维价值评分机制]
           │
           ▼  (scriptwriter.py)
   [主编 Agent：定调 Thesis、选定双头条与 3 条快讯大纲] ──► (MiniMax-M3 旗舰大模型)
           │
           ▼
   [撰稿 Agent：将大纲转化为符合 SSML 语法的双人对话剧本] ──► (MiniMax-M3 旗舰大模型)
           │
           ▼  (tts_engine.py)
   [TTS 语音合成：零样本 (Zero-shot) 声音克隆] ──► (CosyVoice 2 推理引擎)
           │
           ▼
   [音频后处理：句读静音增补、BGM 自动混音与 EBU R128 标准音量均衡] ──► (pydub & FFmpeg)
           │
           ▼  (site_builder/)
   [静态播放页面 & 兼容 Apple Podcasts 规范的 RSS XML 订阅源构建]
           │
           ▼  (daily.yml / prune_pages.yml)
   [Git 文本数据提交至 main 分支 / 音频与静态站点部署至 gh-pages 分支]
```

---

## 🌟 核心技术亮点

### 1. 智能去重与评分机制
* **多维语义去重**：通过结合 TF-IDF 关键词重合比对（基于 RapidFuzz 算法）与预训练的多语言语义编码模型（`paraphrase-multilingual-MiniLM-L12-v2`），自动过滤跨平台抓取到的高度重复及类似新闻。
* **DBSCAN 聚类**：将多源报导的同类科技事件进行自动归堆聚类，捕捉当天的热门风向。
* **5维打分机制**：从影响力（Impact Scope）、新颖度（Novelty）、可解释性（Explainability）、听众相关性（Relevance）以及来源权威度（Source Authority）五个维度对新闻打分，优中选优。

### 2. 多 Agent 协同双人对话文案创作
* 基于 **MiniMax-M3** 旗舰大语言模型，通过多 Agent 设定进行接力式文案创作：
  - **Editor Agent（主编）**：筛选素材，产出结构化的 Thesis、Headlines 及 Quick News JSON 大纲。
  - **Writer Agent（撰稿人）**：将大纲重写为自然、有温度的双人对谈剧本，包含两位常驻主持人：**晓晓（体验派 PM 视角，擅长比喻、发问，声线温柔）** 和 **博文（技术极客视角，提供专业解析，逻辑清晰）**。
* 通过在 Prompt 中附加严格的长度限制与 `<think>` 思考链压缩指令，保证大模型能够以完整的 XML `<speak>` 闭合输出，防止超长截断。

### 3. 高拟真 CosyVoice 2 克隆合成
* 通过 GHA 云端部署的 **CosyVoice 2** (`CosyVoice2-0.5B` 推理引擎）读取 `assets/refs/` 下的参考音频进行高品质的零样本克隆合成。
* 对话切片优化：合成前使用 `text_utils.py` 中的精细清洗规则将超长句子拆分为自然短句，避免 CosyVoice 大音频推理导致的吞音或语速失真。
* 本项目**完全针对 CosyVoice 2 声音克隆而设计**。旧版的 `Edge-TTS` 引擎已于代码中归档并禁用，调用时会抛出 ValueError 异常。

### 4. 广播级音频混音与后处理
* **智能句读**：使用 **pydub** 库在不同主持人发言以及自然段落之间插入动态毫秒级的静音段落（Pad padding），使对话节奏如同真人聊天。
* **音轨混音**：自动在人声音轨中混入低音量背景音乐 (`bgm_placeholder.wav`)，并自动对开头和结尾应用渐入渐出。
* **响度均衡**：使用 **FFmpeg EBU R128 标准响度均衡器**（`loudnorm` 参数）对最终合成的 MP3 文件进行标准化处理，保证在各种播放设备上音量饱满一致。

### 5. 独创的 Git 媒体防膨胀发布策略
为了防止 MP3 等高容量音频导致项目 Git 提交历史无限膨胀，项目采用了分支纯净的分流存储设计：
* **`main` 分支（代码与文本）**：只追踪 Python 源码、YAML 配置、Briefs 数据和剧本 `.txt`/`.html` 文本。**绝对禁止提交任何 `.mp3` 音频文件**（通过 pre-commit 钩子强制拦截）。
* **`gh-pages` 分支（媒体与部署）**：作为网站发布分支，存储网页静态文件、RSS 订阅源 XML、以及物理生成的 MP3 音频和智能句读音频片段切片文件夹。
* **定期瘦身优化机制 (`prune_pages.yml`)**：每月 1 号（或手动）触发一次。它将最近 30 天在播的有效音频备份，并在本地建立一个全新、零历史记录的 `gh-pages` 孤立分支（orphan branch），将备份移回并强推覆盖远程分支。此时 `gh-pages` 提交数会重置为 1，30天外的旧音频被彻底物理清空，极大地释放了托管空间。
---

## 🛠️ 技术栈与第三方工具

本项目的正常运行依赖于以下 Python 第三方库及系统底层工具：

### 1. 新闻数据抓取与解析
* **`feedparser`** (`v6.0.11`)：解析 RSS 及 Atom 订阅源，结构化元数据。
* **`httpx`**：异步 HTTP 请求库，用于高并发的新闻源抓取与内容请求。
* **`trafilatura`** (`v1.8`)：精细化的 HTML 正文和元数据提取引擎，能自动剥离侧边栏、导航栏与页脚广告等噪音。
* **`readability-lxml`**：作为备用的网页内容提取引擎，当正文过短时进行兜底提取。
* **`beautifulsoup4`**：对解析到的正文、自定义标签、SSML 标签进行清理与加工。

### 2. 语义处理、聚类与去重
* **`sentence-transformers`** (`v3.0.0`)：利用中文/多语言语义向量嵌入模型（`paraphrase-multilingual-MiniLM-L12-v2`）计算文章相似度。
* **`scikit-learn`**：利用 `TfidfVectorizer` 提取特征词，使用 `DBSCAN` 密度聚类算法对当日新闻进行热点聚类。
* **`jieba`**：对新闻进行中文分词与关键词提取，用于计算关键词重合比对。
* **`rapidfuzz`**：基于 C 优化的字符串模糊匹配库，快速清洗并拦截重复的标题。

### 3. LLM 接口适配
* **`openai`**：标准的大模型 SDK 客户端，用于以兼容 OpenAI 的接口规范请求 **MiniMax-M3** 旗舰大模型来生成大纲及对话剧本。

### 4. 语音合成与混音
* **`pydub`** (`v0.25.1`)：音频切片的后处理核心库，处理主持人音轨间的静音停顿和淡入淡出。
* **`FFmpeg`**：底层的系统音频处理利器，执行人声与 BGM 伴奏混音，并调用 EBU R128 标准响度均衡器 (`loudnorm`) 规范化输出音轨。
* **`CosyVoice 2`**（独立环境部署/GHA 自动构建）：零样本声音克隆模型（`CosyVoice2-0.5B`），实现播客主持人的高度拟真克隆合成。*注意：旧版 `Edge-TTS` 引擎已于代码中归档和废弃。*

### 5. 系统工具与基础设施
* **`PyYAML`** (`v6.0.1`)：加载项目全局配置（`config.yaml`）和新闻源清单（`sources.yaml`）。
* **`python-dotenv`**：加载本地环境变量。
* **`tenacity`**：对 HTTP 请求和 LLM API 调用设置带指数退避的自动重试机制。

---

## 📂 项目结构与模块映射

```
ai-news-podcast/
├── .github/workflows/          # GHA 全自动流水线工作流
│   ├── ci.yml                  # 格式化与静态检查 QA (Ruff)
│   ├── daily.yml               # 每日新闻抓取、脚本编写与语音合成主流程
│   └── prune_pages.yml         # 孤立分支瘦身与旧音频清理机制
├── assets/                     # 音频素材与音色参考
│   ├── audio_samples/          # Edge-TTS 样音
│   ├── refs/                   # CosyVoice 声音克隆参考文件（host_a_ref.wav 等）
│   └── bgm_placeholder.wav     # 播客背景混音 BGM 音轨
├── config/                     # 规则配置层
│   ├── config.yaml             # Prompt 模板、人声权重、混音参数控制
│   └── sources.yaml            # RSS/Atom 订阅源列表
├── data/                       # 本地数据与缓存
│   ├── briefs/                 # 新闻抓取归一化 JSON 数据 (brief_YYYY-MM-DD.json)
│   ├── reports/                # 大模型生成 Markdown 日报存档
│   └── episodes.json           # 历史已发布播客 Episode 元数据索引
├── docs/                       # 系统架构、开发手册及 TTS 评测报告
├── scripts/                    # CosyVoice 环境安装脚本等
├── site/                       # 静态播放器页面及 feed.xml 本地临时构建目录
├── src/ai_news_podcast/        # 项目主包源码
│   ├── cli/                    # CLI 终端命令注册接口
│   ├── pipeline/               # 流水线核心组件
│   │   ├── fetcher.py          # Stage 1: 异步网页解析抓取器
│   │   ├── processor.py        # Stage 2: 去重、聚类打分处理器
│   │   ├── scriptwriter.py     # Stage 3: Editor & Writer 双 Agent 剧本生成
│   │   ├── tts_engine.py       # Stage 4: CosyVoice/Edge-TTS 合成器
│   │   └── runner.py           # 串联流水线的整体逻辑控制器
│   ├── site_builder/           # Stage 5: RSS 生成器与静态 HTML 构建器
│   ├── prompts.py              # 大模型 Prompt 提示词合集
│   ├── text_utils.py           # 剧本特殊符号清洗与智能短句切分器
│   └── utils.py                # 常用辅助逻辑与 YAML 读写器
├── pyproject.toml              # 项目依赖项配置
└── uv.lock                     # 锁定的确定版本依赖树
```

---

## 🛠️ 本地安装与开发指南

### 1. 前置条件
* 运行环境：Python 3.11+
* 系统依赖：**FFmpeg**（必须在系统 Path 中，否则 pydub 音频混音会失败）
* 包管理器：[uv](https://docs.astral.sh/uv/)（强烈推荐，速度极快）

### 2. 克隆与安装
```bash
git clone https://github.com/<your-username>/ai-news-podcast.git
cd ai-news-podcast
uv sync
```

### 3. 配置密钥环境
在项目根目录下创建 `.env` 文件，写入你的 API 授权 Key：
```env
MINIMAX_API_KEY="你的-minimax-api-key"
```

### 4. 运行终端指令

* **A. 执行数据层抓取与打分过滤**
  ```bash
  uv run podcast-pipeline --date 2026-06-23
  ```
  *(本地生成当天的打分数据 `data/briefs/brief_2026-06-23.json`)*

* **B. 生成今日 Markdown 日报**
  ```bash
  uv run podcast-report --date 2026-06-23
  ```
  *(生成 `data/reports/daily_report_2026-06-23.md`)*

* **C. 生成对话剧本与播客音频**
  ```bash
  # 仅生成文字剧本与 HTML 详情页（跳过 TTS 语音合成）
  uv run podcast-daily --date 2026-06-23 --no-audio --base-url http://localhost
  
  # 运行完整流程（本地需要配置好 CosyVoice 音频合成服务器）
  uv run podcast-daily --date 2026-06-23 --base-url http://localhost
  ```

---

## ☁️ 云端自动化与部署托管

### 1. 配置 GitHub Actions 密钥
在你的 GitHub 仓库页面，依次点击 **Settings** -> **Secrets and variables** -> **Actions**。
点击 **New repository secret** 创建新密钥：
* **Name**: `MINIMAX_API_KEY`
* **Value**: *填入你的 MiniMax 专属 API Key*

### 2. 启用 GitHub Pages 托管播放页与 RSS
1. 将代码推送到 GitHub 仓库。
2. 前往 **Settings** -> 左侧菜单栏 **Pages**。
3. 在 **Build and deployment** 下，将 Source 设置为 **Deploy from a branch**。
4. 将分支设置为 `gh-pages`，目录设置为 `/ (root)`，点击保存。
5. 部署成功后，在各大播客 App（如小宇宙、Apple Podcasts）中填入你的 RSS 地址即可完成订阅：
   `https://<你的GitHub用户名>.github.io/<仓库名>/feed.xml`

### 3. 手动触发特定日期的重建
如果您想重新生成某一期的音频（例如：AI 在某一天念错字了，或者您手动在 main 分支上修改了 `site/episodes/日期.txt` 里的文本文档）：
您可以通过 GHA 手动运行并指定参数，或者在终端利用 GitHub CLI 执行：
```bash
gh workflow run "Daily Podcast" -f date=YYYY-MM-DD
```
*提示：由于 `daily.yml` 包含检查逻辑，若 main 分支上已存在对应日期的脚本 `.txt`，GHA 将直接使用该既有脚本合成音频，不会使用大模型重新生成覆盖，确保了您的人工修改能完美反映在最终播客中。*

## 🤝 开源协议与鸣谢

我们向开发并维护以下第三方项目和开源库的团队及开发者表示由衷的感谢，正是他们的无私奉献让 **AI Daily Pioneer** 的落地成为可能：

### 1. 引用依赖与第三方软件开源协议
* **`trafilatura`** (采用 **GNU GPL v3.0** 协议)：用于抓取新闻文章时的结构化全文提取。
* **`CosyVoice`** (采用 **Apache-2.0** 协议)：用于人声的零样本克隆及高拟真语音合成。
* **`sentence-transformers`** (采用 **Apache-2.0** 协议)：用于计算新闻内容语义嵌入向量。
* **`scikit-learn`** (采用 **BSD 3-Clause** 协议)：用于文本 TF-IDF 特征计算与 DBSCAN 算法聚类。
* **`openai`** (采用 **Apache-2.0** 协议)：用于连接兼容 OpenAI 标准接口的大模型 API。
* **`httpx`** (采用 **BSD 3-Clause** 协议)：用于异步并发执行 HTTP 数据抓取。
* **`pydub`** (采用 **MIT** 协议)：用于主持对话切片合并及音轨处理。
* **`rapidfuzz`** (采用 **MIT** 协议)：用于极速字符串模糊匹配。
* **`jieba`** (采用 **MIT** 协议)：用于中文分词及词频重合分析。
* **`feedparser`** (采用 **BSD 2-Clause** 协议)：用于解析标准 RSS/Atom 的 XML 信息流。
* **`readability-lxml`** (采用 **Apache-2.0** 协议)：作为正文提取的备用兜底解析器。
* **`beautifulsoup4`** (采用 **MIT** 协议)：用于过滤 HTML 标签及 SSML 预处理。
* **`PyYAML`** (采用 **MIT** 协议)：用于解析项目 YAML 配置文件。
* **`tenacity`** (采用 **Apache-2.0** 协议)：用于设定 HTTP 请求的指数退避重试。
* **`python-dotenv`** (采用 **BSD 3-Clause** 协议)：用于读取和加载本地环境密钥配置。

### 2. 开源协议合规性说明
* 本项目为 **AI Daily Pioneer** 编写的原创源代码部分均以 **MIT License** 协议开放。
* 需要特别说明的是，由于本项目动态导入并调用了 **`trafilatura`** 库（该库采用具有强传染性的 **GNU GPL v3.0** 协议），根据 GPLv3 协议的 Copyleft 要求，任何将本项目作为组合作品进行分发、分包或修改后的衍生版本，整体上均需遵守 **GNU GPL v3.0** 协议条款。
* 本项目所采用的 MIT 协议以及所有依赖项的 Apache/BSD/MIT/GNU GPLv3 协议在组合使用时均完全兼容，无任何许可冲突。
