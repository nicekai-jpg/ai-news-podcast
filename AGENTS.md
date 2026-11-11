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
- Stage 1 fetchers (`fetcher_service._FETCHER_REGISTRY`): default `rss`, plus
  `github_api`, `hn_api`, and `deepseek_news`。`deepseek_news` 对应 DeepSeek 官方新闻
  （官网没有 RSS）：每次运行从 `api-docs.deepseek.com/sitemap.xml` **动态发现**
  `/news/newsYYMMDD` 页（每次发布换新地址，代码里不要写死任何一期 URL）、抓 zh-cn 正文；
  另用**稳定地址** `/zh-cn/updates` 更新日志兜底没有单独 news 页的更新
  （`changelog_url` 配置项，按 h3 条目生成、同日期已有 news 页则跳过）。
  `published_at` 取页面日期 + 12:00 UTC 近似（北京时间 20:00）。
  加其他厂商官方源时优先找 RSS，没有再用同类 sitemap/news 抓取器。
- 公众号 RSS：机器之心官网 RSS 已失效、新智元没有站内 RSS，二者改走 wechat2rss 公共服务
  （`config/sources.yaml` 里 `wechat2rss.xlab.app/feed/<id>.xml`；免费名单只覆盖部分媒体号，
  不含厂商官方号）。`_process_single_entry` 优先用 `content:encoded` 全文
  （wechat2rss 的 `summary` 只有十几字），因此不依赖被反爬的 `mp.weixin.qq.com` 页面；
  加新公众号源时确认 feed 的 entry 带 `content`。厂商官方公众号（DeepSeek Harness 等）
  需要自建 WeWe RSS / wechat2rss 私有部署桥接，公共免费服务没有收录。
- **RSSHub 快照缓存**（`rsshub_cache.yml` + `scripts/build_rsshub_cache.py` +
  `config/rsshub_routes.json`）：GHA 里临时起官方 `ghcr.io/diygod/rsshub` 容器，抓没有稳定
  原生 RSS 的少数路由（Anthropic/IT之家/36Kr/极客公园/极术 AI/AIBase/智源/DSH Releases 等），归一化到 `data/cache/*.json`
  并提交 main，供 Stage 1 消费。**不依赖公共 RSSHub 实例、不 fork RSSHub**；单源失败保留旧快照，
  手动测速用 `workflow_dispatch` 的 `dry_run`。设计见 `docs/rsshub_cache_relay.md`。
- 素材时效：`material_service.build_material_text` 给每条素材注入「发布时间（北京时间，
  约 X 小时前）」；`strategies/base_service.py` 的 `effective_score` 对 >24h/>48h 的素材
  扣 1/3 分，选材（播客与日报共用）不会让旧闻挤掉新事件。
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

## 新流水线 v3（2026-11-09 实装，旧管线保留）

设计见 `docs/pipeline_redesign_v2~v5.md`、`docs/source_pipeline_design.md`、
`docs/source_channels.md`（源清单）与 `docs/implementation_status.md`（进度/交接）。

- **入口**：`podcast-pipeline-v3 --date YYYY-MM-DD [--data-dir data] [--config-dir config]
  [--rsshub-base http://127.0.0.1:1200] [--skip-script] [--no-ledger] [--dry-run]
  [--with-audio] [--publish]`。
  `--with-audio` 复用 `tts_engine_service.synthesize`（Stage 8，需 CosyVoice 环境）；
  `--publish` 隐含音频并复用旧 `PublishCommand`（Stage 9）。旧入口 `podcast-pipeline` /
  `podcast-writer` / `podcast-report` / `podcast-tts` / `podcast-publish` 不变，新链默认不接入 GHA，可随时回滚。
- **brief 兼容（供 Stage 8/9）**：S4–S7 会额外写 `data/briefs/brief_{date}.json`（thesis/stories/metadata/radar、report_rows、report_review、podcast_review），
  字段映射自 shortlist/events/items，供旧 `podcast-publish`（依赖 brief）直接消费，无需改 publish。
- **GHA（生产）**：`.github/workflows/v3_daily.yml` 三步 job（pipeline S1–S7 + RSSHub service
  container → tts → publish）。2026-11-10 起 **cron `30 17 * * *` 切到 v3**（定时任务默认**回源全文**
  `fulltext=true`、`max_workers=12`、自动部署 gh-pages 并提交 main）。GHA 半夜跑不赶时间，
  全文用于 Stage 2 标注/Stage 3 推荐更准确，慢可以接受；手动 dispatch 可覆盖参数
  （fulltext/max_workers/deploy 等）。旧 `daily.yml` 保留 `workflow_dispatch` 仅作回滚，不再定时。
  Notify job 汇总各 job 结果，失败时开/评论 `⚠️ v3 pipeline failed: {date}` issue；
  覆盖账本在 pipeline job 结束即提交（TTS/publish 失败也不会丢跨日状态）。
  与旧链共享 `podcast-pipeline` 并发组。
- **历史 artifact（Stage 2 跨日）**：pipeline job 先用 `scripts/fetch_v3_history.py` 通过 gh CLI
  下载最近 30 天成功的 `v3-output-<date>` artifact（权限 `actions: read`），合并 `data/ocean/<date>/`
  到工作区，Stage 2 的跨日去重/连续性在 CI 里可用；artifact 只上传当天 `data/ocean/<date>/`，
  避免把 30 天历史重复打包。历史缺失/下载失败不阻断当天流水线。
- **模块**（都在 `business/` 下，只依赖 `contracts`/`data.*`，不 import `business.pipeline` 内部实现）：
  - `contracts/`：Item / Event / Labels / ShortlistEntry / Commentary + `artifact.py`（gzip JSON）；
  - `coverage/ledger.py`：跨日覆盖账本（已选中事件指纹，长期保留）；
  - `collect/`：C1 RSS、C2 Atom、C3 RSSHub、C4 网页、C5 wechat2rss（132 源（启用 127），见
    `config/sources_v3.yaml`；`contracts.artifact.write_artifact` 落盘）；
  - `ocean/`：四层去重、事件聚类、连续性、多维标注（`config/nlp/*.yaml` 词典）、证据分；
  - `recommend/`：召回→粗排（含淘汰原因）→精排（规则 0.6 + LLM rubric 0.4，只读事实标签，
    不读 lifecycle/evidence_tier/evidence_score）→重排（MMR/去重/时效）→ top 8；
  - `editorial/`：S4 日报正文（事实/观点分离、evidence_ids、客观性闸门）→ S5 日报点评（对日报）；
    S6 播客文稿（事实底稿来自 S2 全文证据，3000–3600 字、3–4 章、`[CHAPTER]`）→ S7 播客点评
    （对文稿，产出 show notes/章节导语）；
  - `pipeline/v3_runner_service.py`：S1–S7 编排、LLM 按 Stage 选模型（`LLMConfig.stage_models`）、
    覆盖账本写入、`daily_report_{date}.md` 渲染。
- **产物路径**：`data/raw/raw-{date}.json.gz`、`data/ocean/{date}/{events,items,continuity,manifest}.json`、
  `data/selection/{date}/shortlist.json`、`data/editorial/{date}/{report.json,daily_report_{date}.md,report_review.json,podcast_review.json}`、
  `site/episodes/{date}.txt`。全文只走 artifact，不进 git。
- **配置**：`config/pipeline_v3.yaml`（路径/保留期/各 Stage 模型）、`config/sources_v3.yaml`、
  `config/ocean_v3.yaml` + `config/nlp/`、`config/recommend_v3.yaml`、`config/editorial_v3.yaml`。
- **评测**：`business/eval/recommend_eval.py` + `scripts/eval_recommend.py`（推荐分阶段指标，见 `docs/recommend_evaluation.md`）；
  **测试**：`tests/test_contracts.py`、`test_coverage_ledger.py`、`test_llm_config.py`、
  `test_collect_*`、`test_ocean_*`、`test_recommend_*`、`test_editorial_*`、
  `test_v3_runner_service.py`、`test_v3_pipeline_integration.py`、`test_v3_cli_controller.py`。
- **已知取舍**：C3 路由已按 2026-11-09 GHA 实跑结果收敛（慢/失败路由改原生 RSS 或禁用，见
  `sources_v3.yaml` 头部）；C4 的智谱/MiniMax/字节 Seed/商汤/工信部/网信办走“锚点+邻近日期”
  启发式，站点改版时在 `sources_v3.yaml` 给该源加 `selectors`；Stage 2 已产出 `ai_relevance`（core/adjacent/none/unknown + 证据词）、`source_categories`（official/news/research/community）与 `best_source_tier`（1=一手/研究，2=媒体，3=社区/聚合），推荐召回把 AI 相关性作为**准入前提**（优先读标签等级，unknown 回退标题/摘要关键词），粗排同款门槛兜底；重排预留 `max_per_topic/max_per_category` 配额（默认 0=关闭，评测后再开）；全部 Stage 已切换 MiniMax-M3.1-Flash（2026-11-10 用户决定；成本约 M3 五折价的 50%，
  S6 A/B Flash 2/3 vs M3 0/3）。Flash 是推理模型：stage3 需 24k/300s，stage4 8k/240s，
  stage5 16k/300s，stage6 24k/600s，stage7 16k/300s；低 token 预算会空输出（上层有重试/降级）。

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
- `docs/rsshub_cache_relay.md` — RSSHub 快照缓存中转（GHA 临时实例）设计
- `docs/development.md`, `docs/contributing.md` — dev workflow
- `docs/tts_complete_guide.md`, `docs/gha_cosyvoice2_deployment_log.md` — CosyVoice setup
- `.github/workflows/` holds `ci.yml` (quality gate: ruff check + ruff format check
  + import contracts + pytest; bot commits carry `[skip ci]` so they don't re-trigger
  it), `daily.yml`, and `prune_pages.yml`. READMEs were corrected on 2026-09-08;
  trust the code over the docs when they conflict.
