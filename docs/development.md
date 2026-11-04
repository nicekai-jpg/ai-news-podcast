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
├── assets/              # 静态资源（背景音乐、参考音频等）
├── config/              # 运行配置文件与 RSS 新闻源列表
├── data/                # 生成的缓存 JSON、日报和 episodes 索引
├── docs/                # 项目文档
├── src/ai_news_podcast/ # 核心的 Python 源码包
│   ├── cli/             # 命令行入口脚本
│   │   ├── podcast_daily_controller.py    # 主编排器 (Stage 1 + 3-5)
│   │   ├── podcast_pipeline_controller.py # Stage 1: 数据抓取与处理
│   │   ├── podcast_writer_controller.py   # Stage 3: 播客脚本生成
│   │   ├── podcast_tts_controller.py      # Stage 4: TTS 语音合成
│   │   ├── podcast_report_controller.py   # Stage 3b: 日报生成
│   │   └── podcast_publish_controller.py  # Stage 5: 站点发布
│   ├── pipeline/        # 核心流水线：抓取、处理、文案、TTS
│   │   ├── fetcher_service.py
│   │   ├── processor*_service.py  # 已拆分为子模块 (types/dedup/cluster/context/score/thesis)
│   │   ├── podcast_writer_service.py
│   │   ├── tts_engine_service.py
│   │   ├── tts_parser_service.py
│   │   ├── tts_postprocess_service.py
│   │   └── runner_service.py
│   ├── site_builder/    # 静态 HTML 页面及 RSS XML 生成
│   ├── prompts_dao.py       # LLM Prompt 模板
│   ├── text_utils_dao.py    # 文本清洗工具
│   └── utils_dao.py         # 公共工具函数（配置加载、I/O）
├── tests/               # 测试代码
├── scripts/             # 调试与开发脚本
├── .env                 # 本地 API Key（请勿提交到 Git！）
├── pyproject.toml       # 包配置与依赖管理
└── README.md            # 项目说明文档
```

## 运行测试集

`scripts/` 目录下包含调试脚本与验证工具，可直接运行：

```bash
# 测试 LLM 接口连通性
uv run python scripts/debug_llm.py

# 测试 TTS 语音合成
uv run python scripts/debug_tts.py

# 正式 pytest 测试（如有）
uv run pytest tests/ -v
```

## 增加新功能或源

1. **添加新的新闻源**：如果你想增加其他的新闻触角，请将新的 RSS URL 添加到 `config/sources.yaml` 当中。添加前，最好先确认该 feed 能输出全文或者结构良好的 HTML 以便解析工具抓取。
2. **接入新的 LLM 服务商**：若要集成其它的大模型 API，可在 `src/ai_news_podcast/business/pipeline/podcast_writer_service.py`（或 `llm_backends/` 下新增适配器）内部进行扩展。目前系统已内置支持 `openai` 兼容接口。
3. **增加新的命令行生成任务**：在 `src/ai_news_podcast/presentation/cli/` 下增加脚本后，需在 `pyproject.toml` 的 `[project.scripts]` 段落注册入口，例如：
   ```toml
   [project.scripts]
   your-command = "ai_news_podcast.presentation.cli.your_script_controller:entrypoint"
   ```
   然后通过 `uv run your-command` 运行。

## 代码风格与检查

代码风格由 `ruff` 统一管理，提交前必须运行检查：

```bash
# 检查代码风格与潜在问题
uv run ruff check src/ tests/ scripts/

# 自动修复可修复的问题
uv run ruff check --fix src/ tests/ scripts/

# 格式化代码
uv run ruff format src/ tests/ scripts/
```

项目配置的 lint 规则包括 E/F/I/W 基础规则。

## 站点数据契约

首页是「设计稿 v23 骨架 `static/page_template.html` + 内联 `static/{style.css,app.js}` + `site-data`」，其余内容由前端按需 fetch 这些随
`site/` 一起部署的 JSON（不再内联 `data/episodes.json` 的 `description`，它只给 `feed.xml` 使用）：

- `site/data/digest_{date}.json`：每期站点数据，包含 `podcast`（标题 / MP3 / 时长 / 主持人 / 逐字稿入口）、
  `lead`（导语）、`metrics`（指标）、`chapters`（章节）、`report`（日报解析结果）、`sources`（来源聚合）、
  `radar`（项目雷达区块），以及 `schema_version` / `date` / `date_label` / `issue` / `availability` / `meta`
  等元信息。
- `site/data/calendar.json`：日期轴，`default_date` + `dates[]`，每项含 `date` / `issue_number` / `title` /
  `availability`。
- `site/episodes/{date}/playlist.json`：逐字稿分段（播放器与逐字稿抽屉读取，沿用原契约）。
  带 `chapter` 字段的 chunk 表示该段开启了新章节（新一期由 writer 的 `[CHAPTER] 标题` 标记产生）。

生成入口共 4 处，全部调用 `presentation/site_builder/digest_view.py:write_digest_outputs()`
（纯函数组装 + 落盘，单期失败只告警、绝不阻断发布）：

1. `podcast-publish`（Stage 5 发布）
2. `podcast-daily`（本地全流程）
3. `scripts/redeploy_site.py`（重新生成站点页面）
4. `scripts/rebuild_site.py`（本地预览重建，`base_url` 用 `http://localhost:8000/site`）

章节来源两档：新一期由 Stage 3 脚本里的 `[CHAPTER] 标题` 标记提供（`sources.chapters[].source="script"`，
经 playlist chunk 的 `chapter` 字段透传；parser 会剥离标记、writer 清洗用哨兵占位、Director 批重建按 chunk
继承，三者都不可丢标记）；历史剧集没有该字段时回退到转场词启发式（`cue_heuristic`，不足 2 个锚点时用固定
`fallback` 三点）。改 writer prompt / TTS 解析 / Director 链路时，必须保持这条降级路径可用。

项目雷达的候选审计（v1.5）：`radar_{date}.json` 的 `meta.candidate_audit` 记录当日候选池的
仓库 / 星数 / 日增 / 语言 / 未收录原因（`meta.audit_semantics="v5"`），digest 只透传前 12 条到
`radar.candidates.preview` 并置 `detail_available=true`；老 radar 文件没有该字段时前端只显示计数空态。

`site/` 整体在 `.gitignore` 中，但 `site/data/` 必须被 git 跟踪：CI 的 publish 步骤用
`git add -f site/data/` 把 digest/calendar 提交到 main，再随 `site/` 部署到 `gh-pages`，部署后还会校验
`data/digest_{date}.json` 存在且非空。月度 `prune_pages.yml` 会以 main 上 tracked 的 `site/` 文件为基础重建
gh-pages 孤儿分支（音频保留 30 天并先备份），因此 digest **不能挪出 `site/`**，否则下一次 prune 后就会从
gh-pages 消失。

## GitHub Actions 持续集成与发布

核心的自动化执行流位于 `.github/workflows/daily.yml`。这套工作流依赖 GitHub Actions 每日触发自动执行。如果您提交的 Pull Request 包含了新的第三方包依赖，请确保手动运行过 `uv lock` 更新你的锁定文件，否则云端的 CI 流程在复原环境时可能会失败。

### 触发方式

当前 workflow 仅支持两种触发方式（**不含 push 触发**）：

- **定时触发**：cron `30 17 * * *`（17:30 UTC / 北京 01:30）；GitHub 调度约延迟 3 小时，实际开跑≈北京时间 5:00 AM
- **手动触发**：Actions 页面点击 "Run workflow"，或 `gh workflow run "Daily Podcast"`

### 手动触发与部署

```bash
# 通过 GitHub CLI 触发
gh workflow run "Daily Podcast"

# 查看运行状态
gh run list --limit 3

# 查看某次运行的详细日志
gh run view <run-id>
```

### 本地构建

```bash
# 重新构建站点（不运行完整 pipeline）
uv run scripts/rebuild_site.py

# 重新生成 TTS 音频
uv run scripts/rebuild_tts.py
```

### 部署流程

代码推送到 main 分支不会自动触发 Daily Podcast workflow。完整部署链路为：

1. 触发 Daily Podcast workflow（定时或手动）
2. Pipeline 运行，生成 `site/` 目录
3. `peaceiris/actions-gh-pages` 将 `site/` 推送到 `gh-pages` 分支
4. GitHub Pages 检测到 `gh-pages` 更新，自动触发 `pages-build-deployment`
5. 部署到 CDN，网站更新 → https://nicekai-jpg.github.io/ai-news-podcast/

如需修改触发条件（如增加 push 触发），需编辑 `.github/workflows/daily.yml` 的 `on:` 配置。
