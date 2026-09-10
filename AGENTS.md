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
  sentence-transformers is an optional extra (`uv sync --extra semantic`, pulls torch);
  without it dedup always uses the TF-IDF fallback and logs a warning.
- Other artifacts: `data/reports/daily_report_{date}.md`,
  `site/episodes/{date}.txt` (script) and `.mp3` (audio),
  `data/episodes.json` (episode index, `keep_last: 30`).

## Gotchas

- **Network to GitHub from this machine is hostile to large HTTPS uploads** (HTTP/2
  streams get reset mid-transfer; downloads are fine). Repo config pins
  `http.version=HTTP/1.1` — keep it. For very large pushes (e.g. history rewrites),
  push in ~40-commit chunks to a temp branch, then force-push the real ref.
  `git filter-repo` (via `uvx git-filter-repo`) was used on 2026-09-07 to strip all
  audio blobs from main's history; a full backup bundle
  (`ai-news-podcast-pre-rewrite-2026-09-07.bundle`) sits next to the repo directory.

- **Audio must NOT be committed to `main`**. The CI TTS job passes the episode MP3 and
  chunk folder to the publish job via a workflow **artifact** (`episode-audio-{date}`);
  only the publish job's deploy step sends them to `gh-pages`. All historical audio was
  already stripped from main's history by `git filter-repo` (2026-09-07) — don't
  resurrect the old pattern of `git add -f site/episodes/*.mp3`. Locally, pre-commit
  blocks audio outside `assets/` and any file over 2 MB, but those hooks don't exist in CI.
- `podcast-writer` **always regenerates and overwrites** `site/episodes/{date}.txt`.
  There is no "reuse the existing manually edited script" behavior. Don't hand-edit
  scripts expecting them to survive a rerun.
- `config.yaml` `tts.cosyvoice.ref_audio` intentionally maps host_a ↔ host_b sample
  files cross-wise; don't "fix" it without listening to the reference audio.
  `tts.cosyvoice.synth_variants` controls which variants are actually synthesized
  (empty = all, the legacy behavior); it is currently `["professional"]` to halve CPU
  inference — the player falls back to professional for un-synthesized variants
  (`player.js`) and html_gen renders only the configured voice pills.
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
  LLM and TTS. `tests/test_prune_gh_pages.py` additionally covers
  `scripts/prune_gh_pages.py` (loaded by path via importlib). No pytest ini section
  exists — defaults apply.
- `prune_episodes` (cli/episode_utils.py) also sweeps "orphan" files for dates that
  are not in episodes.json (e.g. a script committed but TTS failed). It only touches
  date-shaped names and never deletes dates newer than the newest indexed episode,
  so in-flight scripts survive.
- `.github/workflows/daily.yml` job chain: main line is `stage1 → writer → tts → publish`;
  `report` is a fire-and-forget side branch off stage1 — its failure alerts but does not
  block publish. Each job commits `daily: brief|script|report|publish {date} [skip ci]`
  to main. No push trigger, so bot commits don't re-trigger. The writer job appends
  script-quality stats and the tts job a TTS duration line to the Step Summary
  (`if: always()`). The publish job has a job-level env `MIN_MP3_BYTES: '100000'`
  (mp3 truncation guard) shared by pre-publish artifact validation (mp3 exists and size
  above threshold, `feed.xml` enclosure, `episodes.json` entry) and post-deploy gh-pages
  reconciliation (mp3 size + `playlist.json`); it then deletes `.gitignore` and deploys
  `site/` to gh-pages via peaceiris action (`keep_files`), force-adding
  `site/episodes/{date}/playlist.json` — that chunk metadata is the only per-episode site
  content tracked on main, while audio, show notes HTML and full chunk dirs live only on
  gh-pages (prune_gh_pages.py backs them up on rebuild). The trailing `notify` job
  (`if: always()`, job-level `contents: read` + `issues: write` + `actions: write`)
  writes a per-job health
  report to the Step Summary; on failure it opens/comments a GitHub issue
  `⚠️ Daily pipeline failed: {date}` and auto-re-runs the workflow ONCE via
  `-f retry=true` (a failed retry only alerts, no second rerun); on success it
  auto-closes open failure issues. `daily.yml` and `prune_pages.yml` share the
  `podcast-pipeline` concurrency group so manual dispatches queue instead of racing.
  `prune_pages.yml` monthly rebuilds gh-pages as an orphan branch (30-day audio
  retention) and, on failure, alerts via a `gh-pages prune failed` issue (a
  successful prune auto-closes stale ones, mirroring the daily `notify` behavior).

## Docs to read before touching sensitive areas

- `docs/architecture.md` and `docs/pipeline_walkthrough.md` — pipeline design
- `docs/development.md`, `docs/contributing.md` — dev workflow
- `docs/tts_complete_guide.md`, `docs/gha_cosyvoice2_deployment_log.md` — CosyVoice setup
- `.github/workflows/` holds `ci.yml` (quality gate: ruff check + ruff format check
  + import contracts + pytest; bot commits carry `[skip ci]` so they don't re-trigger
  it), `daily.yml`, and `prune_pages.yml`. READMEs were corrected on 2026-09-08;
  trust the code over the docs when they conflict.
