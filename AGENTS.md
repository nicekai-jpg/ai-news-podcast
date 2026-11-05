# AGENTS.md

Guidance for AI agents working in this repository.

## What this is

AI Daily Pioneer (AI 每日先锋) — a fully automated daily AI-news podcast generator.
A GitHub Actions pipeline runs daily (cron `30 17 * * *` UTC; GitHub's scheduler runs it ~3h late, so it effectively starts ≈05:00 Asia/Shanghai):
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

- Three-tier layout (renamed 2026-09-13, order enforced one-way):
  `presentation/cli` + `presentation/site_builder` → `business/pipeline` +
  `business/events` → `data/config` / `data.utils_dao` / `data.prompts_dao` /
  `data.text_utils_dao`. Higher tiers may import lower ones, never the reverse.
- `business/pipeline/runner_service.py:run_pipeline()` is the **only** gateway to Stage 1.
  Never call `fetch_all()` / `process()` from upper tiers.
- All LLM calls go through `business/pipeline/llm_client_service.py:call_llm()`
  (OpenAI-compatible API, tenacity retry). Backends are pluggable via
  `business/pipeline/llm_backends/` and `business/pipeline/tts_backends/` registries.
- Material selection strategies (`business/pipeline/strategies/`): `score_diversity`
  (podcast, MMR-like diversity penalty) vs `pure_score` (daily report); both via
  `business/pipeline/material_service.py:build_material_text()`.
- `business/events/` is an in-process event bus; pipeline stages emit `StageStarted` /
  `StageCompleted` / `StageFailed` — keep stages decoupled through it.
- `src/ai_news_podcast/data/config/` holds the pydantic `AppConfig` and YAML loader.
  All runtime knobs live in `config/config.yaml` (LLM, TTS, dedup/scoring thresholds,
  script style + banned words, audio params) and `config/sources.yaml` (~45 feeds).
- `src/ai_news_podcast/presentation/site_builder/static/*` is packaged via hatchling
  `artifacts`.

## Data flow and dates

- Episode date = **Asia/Shanghai** calendar date (CI sets `TZ=Asia/Shanghai`).
- Stage 1 output: `data/briefs/brief_{date}.json`; reused if present unless
  `--force-refresh`. Cross-episode dedup filters against the last 14 episodes
  (semantic model `paraphrase-multilingual-MiniLM-L12-v2`, TF-IDF fallback).
  sentence-transformers is an optional extra (`uv sync --extra semantic`, pulls torch);
  without it dedup always uses the TF-IDF fallback and logs a warning.
- Stage 1 过滤（`ingest_filter_service.filter_ingested_items`）：`selection.prefer_recent_hours`
  / `fallback_recent_hours`（窗口内条目 <20 时退化到 96h）、`include_keywords` /
  `exclude_keywords`、`per_feed_cap` 现在真正生效；统计写入 brief `metadata.ingest_filter`。
  聚类后还有一次「相似标题孤立条目合并」（`clustering.merge_title_threshold`，默认 80）。
- 脚本质量：`podcast_writer_service.script_quality()` 产出 `normal|fallback|short|broken`
  （长度/轮次/主持人字符占比/未标记行/思考过程），publish 写入 `episodes.json` 的
  `quality`，digest 透传；站点/feed 目前不展示，供审计与后续降级标记使用。
- Other artifacts: `data/reports/daily_report_{date}.md`,
  `site/episodes/{date}.txt` (script) and `.mp3` (audio),
  `data/episodes.json` (episode index, `keep_last: 30`).
- 项目雷达: `data/gh_radar/radar_{date}.json`(每日一项目主推 + 备选)与
  `data/gh_snapshots/snap_{date}.json`(星数快照,供次日差分算增速)。
  两者都由 stage1 提交到 main。雷达作为独立「项目轨」与新闻管线完全隔离,
  结果挂在 brief 的 `radar` 键上,播客栏目「项目雷达」(只讲主推)与日报章节
  (主推 + 备选)均由此生成。

## Gotchas

- **Network to GitHub from this machine is hostile to large HTTPS uploads** (HTTP/2
  streams get reset mid-transfer; downloads are fine). Repo config pins
  `http.version=HTTP/1.1` — keep it. For very large pushes (e.g. history rewrites),
  push in ~40-commit chunks to a temp branch, then force-push the real ref.
  `git filter-repo` (via `uvx git-filter-repo`) was used on 2026-09-07 to strip all
  audio blobs from main's history; a full backup bundle
  (`ai-news-podcast-pre-rewrite-2026-09-07.bundle`) sits next to the repo directory.

- **Audio must NOT be committed to `main`**. The CI TTS job passes the episode MP3 +
  `playlist.json` (not the chunk folder) to the publish job via a workflow **artifact**
  (`episode-audio-{date}`); only the publish job's deploy step sends them to `gh-pages`. All historical audio was
  already stripped from main's history by `git filter-repo` (2026-09-07) — don't
  resurrect the old pattern of `git add -f site/episodes/*.mp3`. Locally, pre-commit
  blocks audio outside `assets/` and any file over 2 MB, but those hooks don't exist in CI.
- `podcast-writer` **always regenerates and overwrites** `site/episodes/{date}.txt`.
  There is no "reuse the existing manually edited script" behavior. Don't hand-edit
  scripts expecting them to survive a rerun.
- `config.yaml` `tts.cosyvoice.ref_audio` intentionally maps host_a ↔ host_b sample
  files cross-wise; don't "fix" it without listening to the reference audio.
  `tts.cosyvoice.synth_variants` 支持列表(全体统一)或字典(按 host 指定,
  键接受 host_a/A/host_b/B),空=全部合成;当前为 host_a→lively(青春女声)、
  host_b→professional(专业男声)的组合。播放器对未合成变体逐级回退
  (professional → 任一可用),html_gen_view 按 host 渲染音色按钮。发布主轨/默认
  变体 = 每位主持人**配置列表的第一项**(不是 professional 优先;professional
  优先仅是"列表为空时的回退"),别按 professional-first 误改。
- **CosyVoice2 输出是 24kHz，保存时必须用 24k**：`CosyVoiceConfig.sample_rate` 从
  `tts.audio.sample_rate` 读取（默认 24000），`CosyVoice2Engine.sample_rate` 优先用模型
  自带的采样率。历史上这里默认 22050 导致所有节目慢 8.8%、音调低约 1.5 个半音
  （2026-11 修复），改 TTS 时不要再把 sample_rate 写死成 22050。
- TTS requires a CosyVoice2-0.5B environment: `COSYVOICE_MODEL_DIR` plus
  `PYTHONPATH` into the cloned CosyVoice repo. See `scripts/setup_cosyvoice_env.sh`,
  `scripts/gha_tts_cosyvoice.py`, and `docs/gha_cosyvoice2_deployment_log.md`.
  `setup_cosyvoice_env.sh` picks its own interpreter (≥3.11 guard; `COSYVOICE_PYTHON`
  overrides) — the local venv is aligned at 3.11.9, same as CI.
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
- `prune_episodes` (presentation/cli/episode_utils_controller.py) also sweeps "orphan" files for dates that
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
  gh-pages (prune_gh_pages.py backs them up on rebuild)。**TTS artifact 只传
  `{date}.mp3` + `{date}/playlist.json`**（分块音频新播放器已不用，历史 gh-pages
  上曾囤 341MB 死文件）；prune 也只备份 playlist.json、不再保留 chunk 目录。
  **main 的 `site/` 提交在 gh-pages 部署校验成功之后**（失败不会留下 feed 指向
  不存在音频）。**重试只针对主链**（stage1/writer/tts/publish），report 失败只告警；
  `retry=true` 不会跳过已成功 stage，重跑会重新生成脚本/音频。The trailing `notify` job
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
- **项目雷达必须是可失败环节**:`runner` 用 try/except 包裹 `build_radar`,失败只发
  `StageFailed` 事件、正片照常。LLM 禁止自报 stars/增速等数字、禁止改写安装命令——
  全部由 `material_service.build_radar_text` 从结构化数据注入;日报雷达章节完全由代码生成。
  近 30 天已推荐过的仓库由 `_load_recent_picks` 排除,不要绕过。
  无 `GITHUB_TOKEN` 时走匿名限流(每日一次扫描足够),不要在雷达里加需要
  更高限流的调用。
- 站点首页 = 设计稿 v23（`今日AI看点`）骨架 `static/page_template.html` + 内联
  `static/{style.css,app.js}` + 内联 `site-data` JSON。模板里的 `{{...}}` 注入点由 app.js
  的 `hydrate()` 用 `digestToData()` 拉取的 digest 字段填充；切日期时重新 fetch
  `site/data/digest_{date}.json` + `site/episodes/{date}/playlist.json` 后重渲染。
  不要重新内联 `episodes.json.desc`（体积红线 1.5MB）。
- 深浅主题：`<html data-theme>` + localStorage（head 内联脚本先应用，避免闪烁），
  默认跟随系统 `prefers-color-scheme`；主题令牌都在 `static/style.css` 的
  `:root` / `[data-theme="dark"]`，改色只动令牌。
- 播放器用整期 `./episodes/{date}.mp3`（feed 同一份）；音频 404/被拒时自动退化为模拟时钟。
  旧的分片 `player.js` 已删除，不要再按 chunk 播放器改。
- 主持人口径统一为 **林夏（Host A）/ 陈默（Host B）**：digest `_hosts`、writer prompt
  (`prompts_dao.py` / `podcast_writer_service.py`) 与站点 roster 必须一致。
- `site/data/digest_*.json` 由 publish 生成并在 CI `git add -f` 跟踪；**不要**把它挪出
  `site/`，否则月度 prune 后会从 gh-pages 消失（prune 只从 main 的 `site/` 拷贝）。
- 章节标记清洗：`clean_tts_text` 会删掉所有非 Host 中括号，**必须**用
  `clean_tts_text_preserving_chapters`（writer 控制器/publish 都不例外），否则
  `[CHAPTER]` 全被删掉、章节导航退化为 cue 启发式。
- 章节条数据来源：writer prompt 的 `[CHAPTER] 标题` → `parse_dialogue_chunks` 的
  `DialogueChunk.chapter` → playlist chunk 的 `chapter` → digest `chapters`（`source="script"`）。
  `clean_tts_text` / `_normalize_host_tags` / Director 批重建三处都会经过脚本，改动时别把
  `[CHAPTER]` 删掉（会被念出来）或丢掉（章节退化为转场词启发式）。历史 30 期无该字段，回退
  `cue_heuristic` 是预期行为。

## Docs to read before touching sensitive areas

- `docs/architecture.md` and `docs/pipeline_walkthrough.md` — pipeline design
- `docs/development.md`, `docs/contributing.md` — dev workflow
- `docs/tts_complete_guide.md`, `docs/gha_cosyvoice2_deployment_log.md` — CosyVoice setup
- `.github/workflows/` holds `ci.yml` (quality gate: ruff check + ruff format check
  + import contracts + pytest; bot commits carry `[skip ci]` so they don't re-trigger
  it), `daily.yml`, and `prune_pages.yml`. READMEs were corrected on 2026-09-08;
  trust the code over the docs when they conflict.
