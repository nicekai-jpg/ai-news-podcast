# 更新日志

本项目的所有重要更新都将记录在此文件中。

该文件的格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) 规范，并且本项目遵循 [Semantic Versioning 语义化版本](https://semver.org/spec/v2.0.0.html) 规范。

## [0.1.0] - 2026-03-10

### 新增功能
- **核心流水线：** 初次发布 `ai-news-podcast` 的结构。
- **数据提取：** 支持根据配置灵活从 RSS/Atom 源拉取数据 (`config/sources.yaml`)。
- **语义处理引擎：** 新增相关性评分机制、关键词过滤以及语义去重功能。
- **大模型集成：** 播客剧本创作者支持 Google GenAI、兼容 OpenAI 的接口，也支持直接原生连接本地的 Ollama 并启用流式输出。
- **语音合成引擎：** 配置了基于 Edge-TTS 的语音合成，支持同时混音插入背景音乐。
- **站点与 RSS 生成器：** 自动化构建播客网页索引 (`index.html`)、带音频的剧集单页，以及符合 Apple Podcasts 规范的聚合 RSS 广播流 (`feed.xml`)。
- **命令行执行工具：** 
  - `run_daily.py`：项目的主自动化流水线执行入口。
  - `daily_report.py`：针对通用 AI 资讯的结构化 Markdown 生成工具。
  - `daily_report_edu.py`：针对垂直领域（AI 赋能教育）的新闻简报生成器。
- **CI/CD 自动化：** 新增了基于 GitHub Actions 的工作流 (`daily.yml`)，支持每日准时自动执行并将生成的静态内容发布于 GitHub Pages 之中。
- **社区文档体系：** 新增了包括 `README.md`、`README.zh-CN.md`、架构说明 `ARCHITECTURE.md`、开发指南 `DEVELOPMENT.md` 以及贡献指引 `CONTRIBUTING.md` 在内的完善的开源生态文档。
