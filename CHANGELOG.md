# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-10

### Added
- **Core Pipeline:** Initial release of the `ai-news-podcast` structure.
- **Data Ingestion:** Fetching from configurable RSS/Atom sources (`config/sources.yaml`).
- **Processing Engine:** Relevancy scoring, keyword filtering, and semantic deduplication.
- **LLM Integration:** Scriptwriter supports Google GenAI, OpenAI, and direct local Ollama connectivity.
- **TTS Engine:** Speech synthesis using Edge-TTS with background music mixing.
- **Site Generator:** Automated building of the podcast index (`index.html`), episode pages, and Apple Podcasts compatible RSS Feed (`feed.xml`).
- **CLI Utilities:** 
  - `run_daily.py`: the main workflow runner.
  - `daily_report.py`: generic markdown AI news report generator.
  - `daily_report_edu.py`: specialized AI-in-Education news report generator.
- **Automation:** GitHub Actions workflow (`daily.yml`) for daily automated execution and Pages deployment.
- **Documentation:** Added `README.md`, `README.zh-CN.md`, `ARCHITECTURE.md`, `DEVELOPMENT.md`, and `CONTRIBUTING.md`.
