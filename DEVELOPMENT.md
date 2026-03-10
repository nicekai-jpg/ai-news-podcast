# 开发者指南

欢迎来到 `ai-news-podcast` 的开发者文档。本文档将指导您如何设置本地开发环境、运行自动化测试以及向本项目贡献代码。

## 环境要求

- **Python**：3.11 及以上版本。
- **uv**：强烈推荐使用该工具进行极速的 Python 包及环境管理（安装方式：`curl -LsSf https://astral.sh/uv/install.sh | sh`）。
- **Git**：用于版本控制。

## 搭建本地环境

首先克隆仓库，并使用 `uv` 安装所有依赖项：

```bash
git clone https://github.com/<你的用户名>/ai-news-podcast.git
cd ai-news-podcast
uv sync --dev
```

上述命令会自动在项目目录下创建一个隔离的虚拟环境 (`.venv`)，并同时安装运行依赖与开发测试所需的所有依赖包。

## 项目目录结构

```text
ai-news-podcast/
├── config/              # 运行配置文件与 RSS 新闻源列表
├── data/                # 生成的缓存 JSON 和各类 Markdown 日报数据
├── docs/                # 推送至 GitHub Pages 的 Web 根目录（HTML 与 MP3 音频）
├── src/ai_news_podcast/ # 核心的 Python 源码包
│   ├── cli/             # 供命令行执行的各个入口脚本
│   ├── pipeline/        # 核心功能流水线：抓取、处理、分发文案、TTS 语音等
│   └── site_builder/    # 用户组装静态 HTML 页面及 RSS XML
├── tests/               # 单元测试与集成测试代码
├── .env                 # 存放本地大模型的各种 API Key（请勿提交到 Git 中！）
├── pyproject.toml       # Python 项目及包依赖管理配置
└── README.md            # 项目主页的使用说明文档
```

## 运行测试集

项目采用了 `pytest` 作为测试框架。你可以通过 `uv` 运行所有测试集：

```bash
uv run pytest tests/ -v
```

### 测试指定的模块

- **测试大语言模型 (LLM) 接口连通性**：
  ```bash
  uv run pytest tests/test_llm.py
  uv run pytest tests/test_rest_llm.py  # 专门用于测试与本地 Ollama 服务的直接连接
  ```
- **测试 TTS 语音合成引擎**：
  ```bash
  uv run pytest tests/test_tts.py
  ```

## 增加新功能或源

1. **添加新的新闻源**：如果你想增加其他的新闻触角，请将新的 RSS URL 添加到 `config/sources.yaml` 当中。添加前，最好先确认该 feed 能输出全文或者结构良好的 HTML 以便解析工具抓取。
2. **接入新的 LLM 服务商**：若要集成其它的大模型 API，可在 `src/ai_news_podcast/pipeline/scriptwriter.py` 内部进行扩展。目前系统已内置支持 `google-genai`、`openai` 兼容接口，并支持向本地 `ollama` 直接发起 HTTP 调用。
3. **增加新的命令行生成任务**：可以在 `src/ai_news_podcast/cli/` 下增加你的专用脚本。运行它们时，我们推荐采用模块化的执行方法，例如：`uv run python -m ai_news_podcast.cli.your_script`。

## 代码风格与检查

目前我们对代码风格暂时不做强制拦截，但请尽量保持代码整洁，并附带必要的注释和类型提示（Type Hints）。我们推荐在提交代码前使用 `ruff` 对代码进行格式化与检查：

```bash
uvx ruff check .
uvx ruff format .
```

## GitHub Actions 持续集成与发布

核心的自动化执行流位于 `.github/workflows/daily.yml`。这套工作流依赖 GitHub Actions 每日触发自动执行。如果您提交的 Pull Request 包含了新的第三方包依赖，请确保手动运行过 `uv lock` 更新你的锁定文件，否则云端的 CI 流程在复原环境时可能会失败。
