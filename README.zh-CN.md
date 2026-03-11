# AI 每日先锋（全自动化 AI 新闻播客与播报生成器）

English README: `README.md`

本项目每天自动抓取 AI 领域的 RSS/Atom 新闻源，进行智能筛选和摘要，并使用大语言模型（LLM）生成播客文稿与纯文本日报。同时内置文本转语音 (TTS) 以及播客 RSS Feed 生成。

## 核心功能
- **全自动新闻管家**：自动拉取 `config/sources.yaml` 中的订阅源，解析并根据价值评分。
- **多模型灵活适配**：支持 OpenAI、Gemini，并深度适配本地部署的 Ollama 模型（断网、免流、数据不出本地也可生成）。
- **多端输出支持**：
  - 生成带背景音效的每日中文 AI 新闻播客 (MP3)。
  - 自动更新并生成支持所有主流播客客户端的 `feed.xml`。
  - 支持生成“AI 综合资讯”或垂直领域的“AI 赋能教育”纯文本日报。
- **开源免托管部署**：配置 GitHub Actions 全自动每日运行，音频与 Feed 全托管在免费的 GitHub Pages 上，零费用维护。

## 怎么“调用”（本地运行）

### 1. 配置环境变量 `.env`
在项目根目录创建或编辑 `.env` 文件，用于指定大模型 API Key（视具体调用的脚本与配置而定）：
```env
# 例如：
OPENAI_API_KEY="sk-xxxx" # 替换为原生 OpenAI、DeepSeek 或 硅基流动 的真实 Key
GEMINI_API_KEY="your-gemini-key"
LLM_API_KEY="ollama"     # 若使用本地部署的 Ollama，通常可以这样标识
```

### 2. 环境安装（推荐使用 uv）
推荐用 [uv](https://docs.astral.sh/uv/)（更快、更稳定）：
```bash
cd ai-news-podcast
uv sync
```
如果本机没有 uv，也可以用 pip：
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 执行功能脚本
项目已重构为标准的 Python 包结构 (`src/ai_news_podcast`)，不同功能可通过 `-m` 模块化运行：

**A. 生成完整播客 (MP3 + RSS 页面 + 记录)**
```bash
uv run python -m ai_news_podcast.cli.run_daily --base-url http://localhost
```
*提示：如果网络环境连不上 Edge-TTS 获取语音，可以追加 `--no-audio` 参数，仅生成文字稿与简介页。*

**B. 生成今日 AI 科技新闻日报 (Markdown)**
本脚本默认调用本地节点 (`http://192.168.7.7:11434`) 的 Ollama 流式输出。
```bash
uv run python -m ai_news_podcast.cli.daily_report
```

**C. 生成今日 AI 教育前沿日报 (Markdown)**
专门为教育从业者与关注 AI+Education 的用户定制（使用 `config/config_edu.yaml` 和 `config/sources_edu.yaml`）。
```bash
uv run python -m ai_news_podcast.cli.daily_report_edu
```

## 如何保证内容质量与专业性
你可以通过编辑 `config/` 目录下的文件来定义你的关注焦点：
- **新闻源**：编辑 `config/sources.yaml` 或 `sources_edu.yaml`，按需启用/禁用 RSS 来源。
- **配置规则**：编辑 `config/config.yaml` 或 `config_edu.yaml` 灵活调节抓取抓取关键字 (`selection.include_keywords`)，更换 TTS 念稿声线等。

## 启用免费播客托管 (GitHub Pages)
1. 把本项目推到 GitHub 公共仓库。
2. 仓库设置 **Settings** -> 左侧菜单栏 **Pages**。
3. **Build and deployment** -> 选择 **Deploy from a branch**。
4. Branch 下拉选 `main`，Folder 下拉选 `/docs`并保存。
5. 几分钟后，在各大播客 App 订阅你的地址：
   `https://<你的GitHub用户名>.github.io/<仓库名>/feed.xml`

## 每天自动更新
- 仓库内置 GitHub Actions：`.github/workflows/daily.yml`。
- 默认每天自动运行并 commit 到仓库。
- 你也可以在仓库对应 Actions 页面，手动点击 **Run workflow** 强制立刻生成一集。
