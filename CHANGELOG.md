# 更新日志

本项目的所有重要更新都将记录在此文件中。

该文件的格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) 规范，并且本项目遵循 [Semantic Versioning 语义化版本](https://semver.org/spec/v2.0.0.html) 规范。

## [Unreleased]

### 变更
- **Editor Agent 输出格式**: JSON → Markdown，减少约 28% token 使用量
- **代码异味清理**: 全面重构，修复 lint 违规，拆分模块

### 修复
- **日报显示**: 修复 h1 标题被隐藏、blockquote 样式缺失、h3 样式缺失
- **GitHub API**: 修复硬编码日期 `2026-06-01` → 动态 `datetime.now()`

---

## [0.2.0] - 2026-07-01

### 重构
- **模块拆分**: `processor.py` 拆分为 6 个子模块（types/dedup/cluster/context/score/thesis）
- **TTS 拆分**: `tts_engine.py` 提取 `tts_parser.py`
- **CLI 简化**: `podcast_daily.py` 提取 5 个 helper 函数
- **代码异味清理**: 修复 C901/PLR0915/N806/B009/SIM118/UP017 等 lint 违规

### 移除
- **EdgeTTS 遗留**: 移除 `voices` 参数、`host_a/b_voices`、`voice_names`、`fallback_backend`
- **SSML 死代码**: 移除 `_is_ssml()`、`preserve_ssml`、SSML 解析分支

### 修复
- **fetcher.py**: 提取 `_process_single_entry` 降低复杂度
- **dedup.py**: 拆分 `get_recent_broadcasted_texts` 和 `semantic_dedup`
- **config.yaml**: 移除死参数，修复重复键

---

## [0.1.0] - 2026-03-10

### 新增功能
- **核心流水线：** 初次发布 `ai-news-podcast` 的结构。
- **数据提取：** 支持根据配置灵活从 RSS/Atom 源拉取数据 (`config/sources.yaml`)。
- **语义处理引擎：** 新增相关性评分机制、关键词过滤以及语义去重功能。
- **大模型集成：** 播客剧本创作者支持 Google GenAI、兼容 OpenAI 的接口，也支持直接原生连接本地的 Ollama 并启用流式输出。
- **语音合成引擎：** 配置了基于 Edge-TTS 的语音合成，支持同时混音插入背景音乐。
- **站点与 RSS 生成器：** 自动化构建播客网页索引 (`index.html`)、带音频的剧集单页，以及符合 Apple Podcasts 规范的聚合 RSS 广播流 (`feed.xml`)。
- **命令行执行工具：** 
  - `podcast_daily.py`：项目的主自动化流水线执行入口。
  - `podcast_report.py`：针对通用 AI 资讯的结构化 Markdown 生成工具。
- **CI/CD 自动化：** 新增了基于 GitHub Actions 的工作流 (`daily.yml`)，支持每日准时自动执行并将生成的静态内容发布于 GitHub Pages 之中。
- **社区文档体系：** 新增了包括 `README.md`、`README.zh-CN.md`、`docs/architecture.md`、`docs/development.md` 以及 `docs/contributing.md` 在内的完善的开源生态文档。
