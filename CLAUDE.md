# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Daily Pioneer (AI 每日先锋) — a fully automated daily AI news podcast generator. The pipeline fetches RSS feeds, deduplicates/clusters/scores news, generates dual-host podcast scripts via LLM (MiniMax-M3), synthesizes audio with CosyVoice 2 zero-shot cloning, and publishes to GitHub Pages with RSS feed.

## Commands

```bash
# Install dependencies
uv sync

# Lint & format
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/
uv run ruff check --fix src/ tests/ scripts/

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_processor.py -v

# Run pipeline stages individually
uv run podcast-pipeline              # Stage 1: fetch → dedup → cluster → score → brief JSON
uv run podcast-report                # Stage 1 + LLM daily markdown report (standalone)
uv run podcast-daily --base-url http://localhost  # Full pipeline: fetch → script → TTS → publish
uv run podcast-daily --no-audio      # Skip TTS, only generate text + site
uv run podcast-daily --no-audio --with-report  # Also generate daily report
uv run podcast-daily --force-refresh # Re-fetch even if brief cache exists

# Clean caches
make clean
```

## Architecture

The pipeline is a linear 5-stage flow, orchestrated by `cli/run_daily.py`:

```
Stage 1: fetcher.py + processor.py →  episode_brief (fetch → dedup → cluster → score → role assignment)
Stage 3: podcastwriter.py →  Podcast script (Editor Agent → Writer Agent)
Stage 3b: daily_report.py →  Daily tech news report (optional, via --with-report)
Stage 4: tts_engine.py →  MP3 audio (CosyVoice 2 + BGM mixing + loudnorm)
Stage 5: site_builder/ →  index.html + feed.xml + show notes
```

**Key architectural constraint**: `runner.py` is the single gateway to Stage 1. All upper-level business (podcast, daily report) must call `run_pipeline()` — never call `fetch_all()` or `process()` directly.

### Shared modules

- **`llm_client.py`**: Unified LLM caller (`call_llm()`) used by both `scriptwriter.py` and `daily_report.py`. OpenAI-compatible protocol with retry logic.
- **`material.py`**: Shared material builder (`build_material_text()`) with two strategies:
  - `score_diversity`: MMR-like entity diversity penalty for podcast scripts
  - `pure_score`: Pure score ranking for daily reports

### Data flow

- `runner.run_pipeline()` produces `data/briefs/brief_{date}.json` (episode_brief dict with `stories`, `thesis`, `metadata`)
- If a brief already exists for the date, it's reused unless `--force-refresh`
- Cross-episode dedup: `runner.py` filters URLs and semantically similar stories from the last 14 episodes before passing items to `processor.py`
- Semantic dedup uses SentenceTransformer (`paraphrase-multilingual-MiniLM-L12-v2`) with TF-IDF fallback

### Multi-Agent script generation

`scriptwriter.py` uses a two-stage LLM pipeline:
1. **Editor Agent**: selects headlines + quick news from material, outputs **Markdown** outline
2. **Writer Agent**: converts outline into `[Host A]/[Host B]` dual-host dialogue script
3. Falls back to template-based `[Host A]/[Host B]` format if LLM fails

The LLM calls go through `llm_client.call_llm()`, which handles OpenAI-compatible API with retry logic.

### TTS engine

`tts_engine.py` parses `[Host A]/[Host B]` markers into `DialogueChunk` objects. Each chunk is synthesized separately via **CosyVoice 2** zero-shot cloning, concatenated with variable silence between turns, mixed with BGM, and normalized via ffmpeg loudnorm.

**Voice mapping**:
- Host A (博文) → `host_a_professional` / `host_a_lively` (男声)
- Host B (晓晓) → `host_b_professional` / `host_b_lively` (女声)

### Site builder

- `html_gen.py`: generates a single-page app with embedded JS player, episode timeline, and daily report viewer (fetches markdown reports dynamically)
- `rss_gen.py`: generates Apple Podcasts-compliant RSS XML feed

## Configuration

- `config/config.yaml`: all pipeline parameters (LLM provider/model, TTS voices, dedup thresholds, scoring dimensions, script style/banned words, build paths)
- `config/sources.yaml`: RSS feed list with `name`, `url`, `category`, `enabled` flags
- `.env`: `MINIMAX_API_KEY` for the MiniMax Token Plan LLM (OpenAI-compatible endpoint)

The LLM config uses an OpenAI-compatible API (`api_key_env`, `base_url`, `model`). The `call_llm()` function in `llm_client.py` is shared by both podcast and report generation.

## Key data structures

- `RawItem` (fetcher.py): dataclass with `id`, `title`, `link`, `normalized_link`, `source_name`, `source_category`, `published_at`, `summary`, `full_text_snippet`, `category`, `language`
- `episode_brief` dict: `{thesis, stories: [{cluster_id, representative_title, role, role_emoji, total_score, scores, context: {factual_summary, historical_background, sources_ranked}, items}], metadata}`
- Role thresholds: main (12-15), supporting (8-11), quick (5-7), skip (<5)

## GitHub Actions

`.github/workflows/daily.yml` runs daily at 21:43 UTC (5:43 AM Shanghai time), or manually via `workflow_dispatch`. **No push trigger** — pushing to main does not re-trigger the workflow.

Steps:
1. `podcast-daily --no-audio --with-report` → brief JSON + podcast script + daily report (Stage 1 + 3)
2. Commit data/reports/briefs back to main (with `[skip ci]` to prevent future loop if push trigger is ever added)
3. TTS job synthesizes audio from the saved transcript
4. Deploy `site/` to `gh-pages` branch via `peaceiris/actions-gh-pages`

After `gh-pages` branch is updated, GitHub Pages automatically triggers `pages-build-deployment` (built-in workflow) to deploy to CDN.

Two workflows exist on GitHub:
- **Daily Podcast** — user-defined, runs the pipeline
- **pages-build-deployment** — GitHub Pages built-in, deploys static files to CDN

## Testing

Tests use `conftest.py` with a `make_raw_item()` factory and mock fixtures for feedparser/httpx/trafilatura/readability/bs4. Test files map to pipeline modules: `test_fetcher.py`, `test_processor.py`, `test_scriptwriter.py`, `test_tts_engine.py`, `test_site_builder.py`, etc. E2E tests: `test_pipeline_e2e.py`, `test_run_daily_full.py`, `test_daily_report_e2e.py`.
