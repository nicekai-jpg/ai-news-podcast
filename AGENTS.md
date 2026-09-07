# AGENTS.md

Guidance for AI agents working in this repository.

## What this is

AI Daily Pioneer (AI 每日先锋) — a fully automated daily AI-news podcast generator.
A GitHub Actions pipeline runs daily (cron `43 21 * * *` UTC = 05:43 Asia/Shanghai):
fetch RSS/Atom feeds → dedup / DBSCAN-cluster / 5-dimension score → LLM writes a
dual-host dialogue script (MiniMax-M3, two agents: editor outline → writer script) →
CosyVoice 2 zero-shot voice cloning synthesizes MP3 → static player site + Apple
Podcasts-compatible `feed.xml` deployed to GitHub Pages (`gh-pages` branch).

Python 3.11, managed by **uv** (uv.lock is committed; CI runs `uv sync --frozen`).
Src layout, single package `ai_news_podcast` under `src/`.

## Commands

```bash
uv sync                                  # install deps
make lint            # uv run ruff check src/ tests/ scripts/
make format          # uv run ruff format src/ tests/ scripts/
uv run ruff check --fix src/ tests/ scripts/
uv run lint-imports                      # verify import-linter contracts
uv run pytest tests/ -v                  # all tests; single file: pytest tests/test_processor.py -v
uv run pre-commit run --all-files        # ruff (src/tests only) + 2MB cap + audio blocker
make clean                               # remove caches
```

Stage CLIs (console scripts in pyproject):

- `podcast-pipeline --date YYYY-MM-DD` — Stage 1: fetch → brief JSON
- `podcast-writer --date YYYY-MM-DD` — Stage 3: brief → dialogue script `site/episodes/{date}.txt`
- `podcast-report --date YYYY-MM-DD` — Stage 3b: Markdown daily report
- `podcast-tts` — Stage 4: script → MP3 (needs a local CosyVoice2 env)
- `podcast-publish` — Stage 5: rebuild site/feed from `data/episodes.json`
- `podcast-daily --base-url ... [--no-audio] [--with-report] [--force-refresh]` — local full run
- `make daily` / `make report` — local shortcuts

## Architecture boundaries (enforced by `.importlinter`, check with `uv run lint-imports`)

- Layer order: `cli` → `site_builder` → `pipeline` → `utils` / `prompts` / `text_utils`.
  Higher layers may import lower ones, never the reverse.
- `pipeline/runner.py:run_pipeline()` is the **only** gateway to Stage 1. Never call
  `fetch_all()` / `process()` from upper layers.
- All LLM calls go through `pipeline/llm_client.py:call_llm()` (OpenAI-compatible API,
  tenacity retry). Backends are pluggable via `pipeline/llm_backends/` and
  `pipeline/tts_backends/` registries.
- Material selection strategies (`pipeline/strategies/`): `score_diversity` (podcast,
  MMR-like diversity penalty) vs `pure_score` (daily report); both via
  `pipeline/material.py:build_material_text()`.
- `events/` is an in-process event bus; pipeline stages emit `StageStarted` /
  `StageCompleted` / `StageFailed` — keep stages decoupled through it.
- `src/ai_news_podcast/config/` holds the pydantic `AppConfig` and YAML loader.
  All runtime knobs live in `config/config.yaml` (LLM, TTS, dedup/scoring thresholds,
  script style + banned words, audio params) and `config/sources.yaml` (~45 feeds).
- `src/ai_news_podcast/site_builder/static/*` is packaged via hatchling `artifacts`.

## Data flow and dates

- Episode date = **Asia/Shanghai** calendar date (CI sets `TZ=Asia/Shanghai`).
- Stage 1 output: `data/briefs/brief_{date}.json`; reused if present unless
  `--force-refresh`. Cross-episode dedup filters against the last 14 episodes
  (semantic model `paraphrase-multilingual-MiniLM-L12-v2`, TF-IDF fallback).
- Other artifacts: `data/reports/daily_report_{date}.md`,
  `site/episodes/{date}.txt` (script) and `.mp3` (audio),
  `data/episodes.json` (episode index, `keep_last: 30`).

## Gotchas

- **Audio must NOT be committed to `main`**. The CI TTS job passes the episode MP3 and
  chunk folder to the publish job via a workflow **artifact** (`episode-audio-{date}`);
  only the publish job's deploy step sends them to `gh-pages`. Historical commits before
  2026-09-07 still carry MP3s (main pack ≈ 600 MB) — don't resurrect the old pattern of
  `git add -f site/episodes/*.mp3`. Locally, pre-commit blocks audio outside `assets/`
  and any file over 2 MB, but those hooks don't exist in CI.
- `podcast-writer` **always regenerates and overwrites** `site/episodes/{date}.txt`.
  There is no "reuse the existing manually edited script" behavior, despite README /
  CLAUDE.md saying so. Don't hand-edit scripts expecting them to survive a rerun.
- `config.yaml` `tts.cosyvoice.ref_audio` intentionally maps host_a ↔ host_b sample
  files cross-wise; don't "fix" it without listening to the reference audio.
- TTS requires a CosyVoice2-0.5B environment: `COSYVOICE_MODEL_DIR` plus
  `PYTHONPATH` into the cloned CosyVoice repo. See `scripts/setup_cosyvoice_env.sh`,
  `scripts/gha_tts_cosyvoice.py`, and `docs/gha_cosyvoice2_deployment_log.md`.
  GHA caches the venv/models (cache key `cosyvoice-gha-v7`); workflows pin Python to
  exact **3.11.9** — keep that pin when touching CI.
- Ruff: line length 100, mccabe ≤ 15. `T201` (print) is ignored because CLIs print;
  `RUF001-003` are ignored because strings contain Chinese. `scripts/` is excluded
  from pre-commit but still linted by `make lint`.
- Tests mirror modules one-to-one. `tests/conftest.py` provides a `make_raw_item`
  factory and mock fixtures for feedparser/httpx/readability/bs4; e2e tests mock
  LLM and TTS. No pytest ini section exists — defaults apply.
- `.github/workflows/daily.yml` job chain: `stage1 → writer → report → tts → publish`;
  each job commits `daily: brief|script|report|publish {date} [skip ci]` to main.
  No push trigger, so bot commits don't re-trigger. The publish job deletes
  `.gitignore` and deploys `site/` to gh-pages via peaceiris action (`keep_files`).
  `prune_pages.yml` monthly rebuilds gh-pages as an orphan branch (30-day audio retention).

## Docs to read before touching sensitive areas

- `docs/architecture.md` and `docs/pipeline_walkthrough.md` — pipeline design
- `docs/development.md`, `docs/contributing.md` — dev workflow
- `docs/tts_complete_guide.md`, `docs/gha_cosyvoice2_deployment_log.md` — CosyVoice setup
- README and README.zh-CN are partially outdated: they list a `ci.yml` workflow that no
  longer exists (only `daily.yml` + `prune_pages.yml` remain) and describe a script-reuse
  behavior that isn't implemented. Report generation lives in `cli/podcast_report.py`
  (via `call_llm` + `build_material_text`); HTML extraction uses readability-lxml.
  Trust the code over the READMEs when they conflict.
