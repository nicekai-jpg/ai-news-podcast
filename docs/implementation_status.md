# 实现进度与接口约定（Lead 维护）

> 2026-11-09 动工。4 个并行工作流 + Lead 集成。

## 工作流

| 任务 | 负责人 | 范围 | 状态 |
|---|---|---|---|
| task-1 Stage 1 采集入海 | collect-agent | `business/collect/`、`config/sources_v3.yaml`、`tests/test_collect_*` | ✅ completed（70 tests，源 132 条） |
| task-2 Stage 2 新闻潮汐 | ocean-agent | `business/ocean/`、`config/ocean_v3.yaml`、`config/nlp/`、`tests/test_ocean_*` | running（其余三个已完成） |
| task-3 Stage 3 选题推荐 | recommend-agent | `business/recommend/`、`config/recommend_v3.yaml`、`tests/test_recommend_*` | ✅ completed（63 tests） |
| task-4 Stage 4/5 点评与文稿 | editorial-agent | `business/editorial/`、`config/editorial_v3.yaml`、`tests/test_editorial_*` | ✅ completed（67 tests） |
| task-5 集成与验收 | lead | `business/contracts/`、`business/coverage/`、`business/pipeline/runner` 集成、workflows、`.importlinter`、e2e | in_progress（orchestrator + mock e2e 已就绪） |

## Lead 已完成（基础层）

- `business/contracts/__init__.py`：Item / Event / Labels / ShortlistEntry / Commentary / Fact，`canonical_link`、`make_item_id`、`make_event_id`。
- `business/contracts/artifact.py`：`write_artifact` / `read_artifact` / `latest_artifact`（gzip JSON）。
- `business/coverage/ledger.py`：跨日覆盖账本（load/save/is_covered/append_entry）。
- `.importlinter` 层序：pipeline → collect → ocean → recommend → editorial → coverage → contracts → events → data.*。
- `LLMConfig.stage_models` + `resolve_model(stage)`（支持按 Stage 指定模型，如 stage2/3 用 M3.1 Flash）。
- 测试：`tests/test_contracts.py`、`tests/test_coverage_ledger.py`、`tests/test_llm_config.py` 全绿。

## 约定接口（集成时以最终报告为准）

```python
# Stage 1
collect_and_write(config_path, date_str, output_path, *, rsshub_base=None, window_hours=72) -> Path

# Stage 2
build_ocean(raw_artifact_path, *, date_str, history_dir=None, ledger_path=None,
            config_path="config/ocean_v3.yaml") -> OceanSnapshot(events, items, continuity)

# Stage 3
recommend(events, *, config_path="config/recommend_v3.yaml", llm_judge=None,
          covered_event_ids=None) -> list[ShortlistEntry]

# Stage 4
build_commentaries(shortlist, events, llm_call, *, config_path="config/editorial_v3.yaml") -> list[Commentary]

# Stage 5
build_script(shortlist, events, commentaries, llm_call, *, config_path="config/editorial_v3.yaml") -> str
```

## 集成注意

- 各 Stage 之间只通过 contracts + JSON artifact 交换，禁止互相 import 内部模块；
- LLM 调用以可注入的 `llm_call` / `llm_judge` 传入，生产侧由 Lead 适配 `llm_client_service`（带 `resolve_model(stage)`）；
- 迁移期保留旧管线（`runner_service.run_pipeline`），新链用独立入口，默认不切换；
- 所有新模块只 import contracts/data.*，不 import `business.pipeline` 内部实现。

## 阶段交接约定（2026-11-09 确认）

| 交接 | 路径 / 结构 |
|---|---|
| Stage 1 → 2 | `data/raw/raw-{date}.json.gz`（manifest + items，contracts.Item） |
| Stage 2 → 3/4 | `data/ocean/{date}/`：`events.json`（list[Event.to_dict]）、`items.json`（item 字段 + event_id/continuity_id/labels/dup_of/dup_reason，保留 full_text）、`continuity.json`、`manifest.json` |
| Stage 3 → 4 | `data/selection/{date}/shortlist.json`（list[ShortlistEntry.to_dict]，顺序=rank） |
| Stage 4 → 5 | `data/editorial/{date}/commentary.json`（list[Commentary.to_dict]）+ `daily_report_{date}.md` |
| Stage 5 → 6 | `site/episodes/{date}.txt`（15 分钟、3000–3600 字、3–4 章节、[CHAPTER]） |

- Stage 2 的 history_dir 默认 `data/ocean/`，扫描最近 30 天、跳过当天目录；
- Stage 3 直接读 events.json，不 import ocean 包；
- LLM 一律以 `llm_call: Callable[[str], str]` / `llm_judge` 注入，生产侧用 `llm_client_service.call_llm` + `resolve_model(stage)` 适配。

## Lead 追加（2026-11-09 晚）

- `business/pipeline/v3_runner_service.py`：Stage 1→5 编排（lazy import）、item_evidence 注入、覆盖账本写入、02 日报 Markdown 渲染；`tests/test_v3_runner_service.py` mock 全链路通过。
- `config/pipeline_v3.yaml`：全局路径/保留期/各 Stage 模型配置。
- 全量回归：667 passed / 1 failed（collect-agent 自测的 URL 编码用例，已通知其修复）。
- editorial-agent 交付：`business/editorial/`（config/prompts/commentary/script/quality）+ 67 tests，全绿；Stage 4 输出 commentary.json + daily_report md（日报 md 由 Lead 编排层渲染）。

## 集成状态（2026-11-09 晚）

- collect ✅（132 源 / 70 tests）、recommend ✅（63 tests）、editorial ✅（67 tests）；
- Lead 编排：`v3_runner_service` 已接 LLM judge 适配器、daily_report md 渲染；
- 全仓回归：**743 passed**，lint-imports 4/4；
- 待 ocean-agent 完成后：真实小样本联调（Stage1 fixture → Stage2 → Stage3 → Stage4/5）。

## 最终状态（2026-11-09）

- 4 个并行工作流全部完成：collect（132 源 / 70 tests）、ocean（74 tests）、recommend（63 tests）、
  editorial（67 tests）；Lead 集成：contracts/artifact、coverage ledger、v3 编排器、LLM judge 适配、
  daily_report 渲染、CLI（podcast-pipeline-v3）、.importlinter 层序、stage_models。
- 全仓回归：**821 passed**；`ruff check` / `ruff format --check` 全绿；`lint-imports` 4/4。
- 真实网络冒烟：Stage 1 抓取机器之心/ HuggingFace 共 7 条（全文 ok）→ Stage 2 聚类/标注 → Stage 3 短名单全链路跑通。
- 集成测试：`tests/test_v3_pipeline_integration.py`（真实 ocean/recommend/editorial + 假 LLM）与
  `tests/test_v3_runner_service.py`（mock 编排）通过。
- 未提交 git；旧管线未改默认行为；GHA 尚未切到新链（下一步）。

## Stage 6/7 接入（2026-11-09 完成）

- `podcast-pipeline-v3 --with-audio` → 复用 `tts_engine_service.synthesize`（CosyVoice2），输出 `site/episodes/{date}.mp3`；
- `podcast-pipeline-v3 --publish` → 依赖音频，复用旧 `PublishCommand`（feed/site/`data/episodes.json`/digest）；
- 为兼容旧 publish，Stage 4/5 追加写 `data/briefs/brief_{date}.json`（thesis/stories/metadata/radar，
  stories 字段映射自 shortlist/events/items），publish 逻辑零改动；
- 全仓回归：**822 passed**，ruff/format/lint-imports 全绿；e2e 测试断言 brief 可被 publish 消费。

## 分支与 GHA（2026-11-09 晚）

- 分支 `feat/v3-pipeline`，提交 `79f0cd50`（822 tests 全绿后提交）。
- 新增手动工作流 `.github/workflows/v3_daily.yml`：pipeline（Stage1–5，RSSHub service container）→ tts → publish；
  默认只校验/上传 artifact，`deploy=true` 才部署 gh-pages + 提交 main；cron 未切换。
- 真实发布联调（隔离 git worktree，假 LLM + ffmpeg 合成占位 mp3）：
  `podcast-publish --date 2099-12-31` 成功产出 feed enclosure、episodes.json 条目、index.html、
  show notes、digest_{date}.json —— 证明 v3 兼容 brief 可被旧 publish 真实消费。
- 真实 TTS 未跑：本机有 `~/cosyvoice_models` + `~/cosyvoice_src`，但 `~/cosyvoice_venv` 不存在，
  需先跑 `bash scripts/setup_cosyvoice_env.sh`（较重，未擅自执行）。

## 生产切换（2026-11-10）

- 2026-11-09 GHA 实跑：Stage1–5（fulltext=false）17m37s 成功，618 items / 547 events；慢因是
  RSSHub 上游路由（bloomberg 125s、oschina 59s、geekpark 503 73s）与回源全文。
- 调整：6 个慢路由改原生 RSS（transformer-circuits/bloomberg/geekpark/ifanr/engadget/oschina），
  9 个失败且无替代的源禁用；定时任务默认 `--no-fulltext`、`max_workers=12`。
- 生产 cron 切到 `.github/workflows/v3_daily.yml`（30 17 * * *，自动 deploy + commit）；
  旧 `daily.yml` 仅保留 workflow_dispatch 回滚。

## 三项验证（2026-11-10）

### 1) C3 路由线上复核（本地 RSSHub 容器 1205）
- 首轮 53 条启用路由：**48 ok / 5 bad**（mathpix、sspai、nature、leiphone 60s 超时；gwern 404）。
- 处理：sspai/nature/leiphone/gwern 改原生 RSS（均验证 200）；mathpix 禁用。
- 二轮 48 条：**46 ok / 2 bad**（huggingface 503、nyt 60s 超时）。
- 再处理：huggingface、nyt 也改原生 RSS（均验证 200）。最终 C3 启用 46 条，全部有可用入口。

### 2) M3.1 Flash 可用性 / A/B
- 模型 ID `MiniMax-M3.1-Flash` / `-Preview` 均可用（200，中文输出正常）。
- **max_tokens 太小会空输出**：800 时 reasoning 占满 token → content 为空。
- 16384 tokens + temperature 0.5：生成 15 分钟稿 quality 通过（3554 字/4 章），但耗时 **412s**；
  8192 tokens 重复 2 次均空输出。
- 同 prompt 下 M3 2/2 quality 失败（输出像 prompt 片段回显、含禁用词、超长/章节失衡）；
  昨晚 GHA 的 M3 稿也仅 1664 字/7 章/主持失衡。
- 结论：**问题主要在 v3 文稿 prompt 与 token 预算，不在模型本身**；暂不切 Stage5 到 Flash，
  先修 prompt（精简规则、禁用词列表改为泛化约束、强制格式、max_tokens≥16k），再 A/B。

### 3) 回源全文的影响（24 条短摘要样本）
- 页面正文平均 **896 字** vs 摘要 **221 字**（约 4 倍）；数字/版本类 token 平均 **10.7 vs 4.8**；
  9/24 条页面正文 >2×摘要。
- 结论：全文对证据密度有明显价值，但全量回源太慢；
  **建议两阶段抓取**：Stage1 只抓摘要（快），Stage3/4 对 top 20 候选再回源正文（质量+时间平衡）。

## 全文策略（2026-11-10 用户定）

- 生产定时任务**保留全文回源**（fetch_fulltext=true）：GHA 半夜跑不赶时间，且 Stage 2 标注
  （labels.py 已把 full_text 拼入标注文本，上限 4000 字）与 Stage 3 推荐需要全文更准确。
- 两阶段抓取方案不采用；`--no-fulltext` 仅保留给人工快速验证。

## Stage5 模型 A/B 与切换（2026-11-10）

- 方法：同一份 2026-11-09 数据（events/shortlist/commentary），本地直连 MiniMax API，
  同一套 quality 闸门；每模型 3 次采样（temp 0.5 / 16k token）。
- 结果：M3 **0/3 通过**（章节 6–15、超长 25k–32k 字、主持失衡、回显要求）；
  M3.1-Flash **2/3 通过**（3191/3314 字，3–4 章，无 issue），第 3 次空输出。
- 24k token 复测：1/2 通过、1 次空输出（疑似超模型输出上限），故 stage5 固定 16k。
- 结论/改动：Stage5 模型切换为 `MiniMax-M3.1-Flash`，temperature 0.4，max_output_tokens 16384，
  `max_rewrites=2`；prompt 去掉禁用词列表回显、末尾强制“直接输出文稿、不复述要求”。
- 剩余风险：偶发空输出（约 1/3），靠重写兜底；上线后需观察前几期质量。

## 全量切 M3.1-Flash（2026-11-10，用户决定）

- 定价：Flash 输入 ¥1.00/M、输出 ¥4.00/M；M3 五折 ¥2.10/¥8.40 → Flash 约为 M3 五折价的 50%
  （估算一期 ≈¥0.11 vs ¥0.23）。
- 适配：Flash 为推理模型，长 prompt 必须给足 token/超时，否则空输出：
  - stage3（top20 事件 rubric）：24k tokens、timeout 300s（实测 169s，解析 19/20）；
  - stage4 点评：8k tokens、timeout 240s；
  - stage5 文稿：16k tokens、timeout 600s（实测 232–430s）。
- 失败兜底：stage3 judge 异常自动降级纯规则分；stage4 单条失败跳过；stage5 最多 2 次重写。
- stage_models：stage2/3/4/5 全部 `MiniMax-M3.1-Flash`。

## 复盘发现与修复（2026-11-10）

已修：
- 覆盖账本持久化：pipeline artifact 增加 data/coverage/，publish job 提交 `data/coverage/`；
- 覆盖语义：账本只记录进入 Commentary（真正播报）的事件，不再把整个 shortlist 标为已覆盖；
- pipeline job timeout 150→240 分钟（全文跑实测 104 分钟，留余量）；
- 新增 GHA 脚本检查：文稿缺失/为空直接失败，质量不达标给 warning；
- 旧 rsshub_cache workflow 去掉定时（v3 已用实时 RSSHub，且旧任务连续失败）。

未修（P0/P1，待决策）：
- 历史事件未持久化：GHA 每次 checkout 没有 `data/ocean` 历史，Stage2 的连续性/30 天跨日去重
  目前只依赖覆盖账本；events.json ~1.5–1.8MB/天，30 天约 50MB，不适合直接入 git。
  建议每日写 compact history 快照（~100–200KB/天）入 git，或按 run-id 下载历史 artifact。
- Stage2 实际不调用 LLM（规则标注），“全 Stage Flash”对 stage2 无实际影响。
- 无 token/费用统计（可在 llm 客户端记录 usage 落 manifest）。

## 历史 artifact 方案（2026-11-10，采用 B；A 取消）

- 采用设计稿原方案 B：pipeline job 用 `scripts/fetch_v3_history.py`（gh CLI）下载最近 14 天
  成功的 `v3-output-<date>` artifact，把 `data/ocean/<date>/` 合并进工作区，Stage 2 历史扫描即可用。
- workflow 增加 `actions: read`；artifact 上传只含当天 `data/ocean/<date>/`，避免历史重复打包。
- 本地验证：下载 2026-11-09/11-10 两天历史 → build_ocean history_events=547、continuity_updates=33、
  reappeared_events=325，跨日链路生效。
- 取消 A（compact history 入 git）：不再需要，artifact 已提供统一规格文章 + 推荐属性 + 全文。
- Stage 4 已正式合并“点评 + 日报”：`editorial/commentary.py` 产出点评，
  `editorial/report.py` 渲染 `daily_report_{date}.md`；Stage 5 复用同一份点评写播客文稿。

## Stage 拆分：日报点评 / 播客点评（2026-11-10 设计确认）

用户确认的语义：**先有结果，再点评**；点评对象是已生成的日报和播客文稿。

| Stage | 产物 | 输入 |
|---|---|---|
| S1 采集入海 | `raw-{date}.json.gz` | 132 源 + 全文 |
| S2 新闻潮汐 | `ocean/{date}/{events,items,continuity}.json` | 原始 items + 历史 artifact |
| S3 选题推荐 | `selection/{date}/shortlist.json` | S2 属性 |
| S4 日报生成 | `editorial/{date}/report.json` + `daily_report_{date}.md` | S3 + **S2 全文证据底稿** |
| S5 日报点评 | `editorial/{date}/report_review.json` | **已生成的 S4 日报** + 证据 |
| S6 播客文稿生成 | `site/episodes/{date}.txt` | S3 + **S2 全文证据底稿** |
| S7 播客点评 | `editorial/{date}/podcast_review.json` | **已生成的 S6 文稿** + 日报要点（含 show notes/章节导语） |
| S8 音频 / S9 发布 | mp3 / feed+site | 文稿 / brief |

- 页面数据：`brief` 与 `digest` 已透传 `report_rows`、`report_review`、`podcast_review`；
  02 页面用日报+日报点评，01 页面用播客+播客点评（前端渲染下一步接）。
- 生成与点评分离后，S4/S6 不再消费点评，事实约束由 Stage 2 全文证据底稿提供。
- 配置：`pipeline_v3.yaml` stage4（日报生成）/stage5（日报点评）/stage6（文稿）/stage7（播客点评），全部 M3.1-Flash。

## S6 空输出修复（2026-11-10）

- 现象：新版 S6（事实底稿 prompt）首调偶发空内容，pipeline 报 LLM 失败。
- 抓包：16k max_tokens 时 reasoning 13.6k、completion 15,635，接近上限；24k 时 reasoning 19.4k、内容正常。
- 修复：stage6 `max_output_tokens` 16384 → 24576（timeout 600s 不变）；LLM 后端在空 content 时打
  warning（含 finish_reason），便于以后定位。
- 复测：同数据走 `make_llm_call(stage6)` 真实 API 通过（正文/质量按 quality 闸门判断）。

## 本地可验证项结果（2026-11-10 收盘前）

- 真实 API 跑通 S4→S7 全段（真实 2026-11-09 数据，676s）：
  - 生成 `report.json`、`daily_report_2026-11-09.md`、`report_review.json`、文稿、`podcast_review.json`；
  - 4 条日报正文 / 4 条日报点评 / 3 章播客点评；文稿 quality ok（3271 字、3 章、无 issue）。
- brief 兼容：21.3KB，含 `report_rows`/`report_review`/`podcast_review`；旧 `show_notes` 渲染正常。
- digest 页面映射：`lead.source=podcast_review`（01 编辑导语）、`report.editor_verdict=日报点评`
  （02 编辑短评），与 app.js 既有槽位对齐。
- CLI `--dry-run` 正常；GHA artifact 已包含 `data/editorial/`、`data/coverage/`、当天 ocean。
- 尚未验证：GHA 全链路（采集 + TTS + publish + 自动提交/部署），等今晚定时首跑。

## 全链路审计补充（2026-11-11）

- 修复 02 页面 report 块：v3 日报 md 不兼容旧 parser，改为从 brief.report_rows/report_review 结构化构造。
- 覆盖账本改为 pipeline job 结束即提交（continue-on-error），不依赖 TTS/publish 成功。
- v3 增加 Notify job：汇总各 job 结果，失败时开/评论 issue（此前 v3 失败无告警）。
- 全 Stage Flash 的 token/超时预算、历史 artifact 下载、watchdog 监控 v3 均已在 GHA 验证或修正。

## 缺口修复（2026-11-11）

- 历史 artifact 窗口 14→30 天（artifact 保留期正好 30 天）。
- S5 日报点评逐条点评不再截断（编辑短评展示全部）。
- S7 章节导语/highlights 接入 show notes 页面（episodes/{date}.html）。
- 新增 LLM token 用量与费用统计：`data/editorial/{date}/llm_usage.json`（价格表在 pipeline_v3.yaml llm.pricing）。
- C3 本地 RSSHub 复测：重新启用 6 个源（deeplearning/mathpix/pixelstech/ofweek/synced/36kr），
  启用数 121→127；剩余 5 个禁用（tmtpost/scientificamerican/csia/benedictevans 路由 404，venturebeat 429）。

## 时长分析与采集提速（2026-11-11）

首期成功跑（run 42945166054）总耗时 3h23m，拆解：

```
采集 S1        1h36m   (47%)
TTS S8         1h15m   (37%)
LLM S3-S7      16m     (8%)
其他/发布      16m     (8%)
```

- 采集：stage1 原 12 workers、同域间隔 1.0s；改为 24 workers、间隔 0.5s（config/pipeline_v3.yaml stage1），
  目标降到约 45–60 分钟；站点限流会导致部分 full_text failed，非致命。
- TTS：CosyVoice2 CPU 推理 56 个 chunk 约 75 分钟（约 1.3 分钟/chunk）。可选优化：GPU runner、
  分块并行合成、减少音色/句级切分；需单独评估质量与成本，暂未改。

## 非 AI 新闻过滤（2026-11-11）

用户反馈页面出现汽车类新闻。排查：`玛莎拉蒂与尊界联合造车` 因主体「华为」命中主体召回通道，
且规则标签给的是 business/capital/technology，没有 AI 信号，却进入 shortlist 第 5 位。

修复：
- 推荐粗排新增硬门槛 `not_ai`：事件标题+摘要必须命中 AI 关键词（config/recommend_v3.yaml
  `coarse.require_ai_relevance: true`，关键词表在 recommend/config.py `DEFAULT_AI_TEXT_KEYWORDS`）。
- 用 2026-11-11 真实数据回归：718 events 淘汰 192 条（汽车/股票/消费/社会新闻），
  原 shortlist 里唯一的汽车事件被淘汰，其余 6 条 AI 事件保留。
- 原始采集仍保留全部源（设计上采集不做重要性判断），非 AI 内容在推荐阶段被挡掉。

## 召回准入标准修正（2026-11-11）

问题不是技术路线，而是准入标准：原召回是六通道 **OR**（时效/主题/技术栈/主体/证据/连续性），
汽车新闻可凭「时效+主体+连续性」拿到候选资格，再靠通用新闻价值分进 shortlist。

修正：
- 召回改为 `AI 相关性前提 AND（六通道任一）`：`recall.require_ai_relevance` 默认 true，
  标题/摘要命中 AI 关键词才允许进入召回；主题/技术栈不再承担“过滤”职责。
- 粗排 `coarse.require_ai_relevance` 保留为兜底。
- 关键词表统一在 `recommend/config.py: DEFAULT_AI_TEXT_KEYWORDS`；两处均可配置关闭。
- 测试：846 passed。

## 推荐评测基础设施（2026-11-11）

- 新增 `business/eval/recommend_eval.py`：事件标题 ↔ 标注 JSONL 匹配，跑完整推荐链路，
  输出召回/粗排/top8 分阶段指标与漏召回、漏放清单。
- 新增 `scripts/build_eval_template.py`（等步长抽样生成 200 条标注模板）、
  `scripts/eval_recommend.py`（评测 CLI）；说明见 `docs/recommend_evaluation.md`。
- 种子标注 16 条（assistant_seed，未复核）冒烟：召回正样本 8/8，负样本漏入 2/8，
  粗排负样本 0/2，top8 精确率 80%；人工 200 条标注后替换。

## 推荐评测集扩容（2026-11-11）

用户指出 16 条种子太单薄。已改为多天分层：
- `build_eval_template.py` 支持多 events 文件、按标题去重、按 core/adjacent/none/unknown
  配额抽样，稀有负样本全部纳入；
- 生成 `data/eval/recommend/review_queue_2026-11.jsonl` 600 条（3 天 artifact）：
  core 320、adjacent 82、none 48、unknown 150；
- `eval_recommend.py` 支持多天聚合，新增 `aggregate_reports` 与分天明细；
- 现有 v3 artifact 仅 3 天，后续每天自动积累，队列可持续扩容。

## 标注工具（2026-11-11）

- `scripts/split_review_queue.py`：按天拆分 600 条队列（11-09 198 / 11-10 167 / 11-11 235）；
- `scripts/label_queue.py`：终端交互标注，回车接受建议、逐条写盘、可断点续标；
- 队列/天文件在 `data/eval/recommend/days/`，标注产物 `labels_<date>.jsonl`。

## ai_relevance 分级（2026-11-11）

- `business/ocean/relevance.py`：规则版分级（core/adjacent/none/unknown + 证据词 + 置信度）；
- `Labels` 新增 `ai_relevance` / `ai_evidence`；`label_text` 逐条打标，`merge_labels` 按
  core > adjacent > none > unknown 合并；
- 召回/粗排 AI 门槛优先读标签等级：core/adjacent 放行，none 淘汰，unknown 回退文本关键词；
- 测试：`tests/test_ocean_relevance.py`、召回/粗排标签优先用例；871 passed。

## 推荐重排配额（2026-11-11）

- `Event` 新增 `source_categories`（official/news/research/community），由事件内 items 汇总；
- 重排新增 `max_per_topic` / `max_per_category` 配额，默认 0（关闭）；
  开启后同一主题/来源类别超过上限的候选不再入选，用于后续“防偏科”策略；
- 默认关闭的原因：主题标签较宽（技术/产品/商业），未评测前开启可能把 top8 压到 3 条；
- 测试：`tests/test_recommend_rerank.py` 覆盖主题/类别上限；873 passed。

## 来源可信度分级（2026-11-11）

- `business/ocean/source_tier.py`：按 Item.category 分级（official/research=1，news=2，
  community/video/podcast=3），事件取最可信来源 `best_source_tier`；
- 重排同分时优先更可信来源（tie-break，不改分数）；字段可供后续页面/审计展示；
- 测试：`tests/test_ocean_source_tier.py`、重排 tie-break；879 passed。

## 采集源健康度（2026-11-11）

- `business/collect/health.py`：聚合多天 raw artifact 的 source_stats + items 全文状态，
  输出每源天数/条目/失败天数/空源天数/全文成功率/平均耗时/健康标记；
- `scripts/source_health.py`：读 `data/raw/raw-*.json.gz` 生成 Markdown/JSON 报告；
- 阈值：失败过半或连续多天 0 条=bad；有失败/空源过半/全文率<50%=warn；
- 测试：`tests/test_source_health.py`；884 passed。

## LLM rubric 多次采样（2026-11-11）

- `FineSettings.llm_samples`（默认 1）：>1 时对精排前 `llm_top_k` 条重复调用 LLM，
  各维度取中位数再融合，降低单次打分波动；
- 成本随采样次数线性增加，默认关闭；打开前先用评测集确认收益；
- 测试：`test_llm_samples_use_median_scores`；885 passed。

## AI 信号收紧（2026-11-12）

验证跑发现：汽车新闻正文提到“自动驾驶/辅助驾驶”会被判 core/adjacent，Cybertruck 停产进了 top8。
修正：
- Stage 2 规则分级：正文兜底只对 official/research 来源开放，新闻/社区只看标题+摘要；
- 召回/粗排门槛：core/adjacent 标签也必须**标题**带 AI 信号（一手/研究来源例外）；
- 用验证 artifact（763 事件）回归：Cybertruck 停产被挡，7 条 shortlist 中 6 条 AI 事件保留；
- 边界：标题含“机器人”的玩具新闻仍会通过，属可接受误放，后续可用主题/对象标签继续收紧。

## 主题域分级（2026-11-12）

用户提醒“今天汽车，明天可能是别的类”。新增 `business/ocean/domain.py`：
- 标题判定 `primary_domain`：ai_core / ai_infra / other / unknown；
- 汽车、金融、消费、体育、娱乐、社会、政治等统一归 other；
- 召回/粗排遇到 other 直接淘汰，不再依赖“AI 关键词在不在”；
- 验证 artifact（763 事件）回归：31 条 other 全部挡住；原 top8 保留 6/7；
- 测试：898 passed。

## 标签体系骨架（2026-11-12）

用户要求先定骨架/技术路线/标签类型，再填词。已落地：
- `config/taxonomy/labels.yaml`：决策标签受控枚举（ai_centrality/primary_domain/object_type/primary_action/lifecycle/evidence_tier）；
- `config/taxonomy/entities.yaml`：公司/实验室/人物/产品 id + 别名（首批种子）；
- `config/taxonomy/industries.yaml`：有限行业清单 + ai_related + 关键词；
- `business/ocean/taxonomy.py`：加载/校验/别名映射/行业查询；`tests/test_taxonomy.py`；
- 文档：`docs/taxonomy.md`（含规则+LLM 标注路线、补词流程）。

## 标签覆盖审计（2026-11-12）

用户指出同一类“词表覆盖不足”问题可能遍布其他环节。新增：
- `business/eval/label_coverage.py` + `scripts/label_coverage.py`：统计 topics/tech_stack/
  subjects/actions/narrative_angle/audience_persona/sensitivity 的空白率并告警；
- 2026-11-11 实测：subjects 48% 空白、narrative 51%、tech 29%、actions 25%、topics 16%；
- docs/taxonomy.md 增补“其他位置的同类风险”表（impact/novelty/risk/来源分级/证据/聚类/全文/账本）；
- 结论：每个字段都要有覆盖度量，不再假设词典完整。

## P1 标签元数据 + P0 队列扩容（2026-11-12）

- `Labels` 新增 `taxonomy_version`、`label_evidence`、`label_confidence`、`label_method`
  （field → 证据/置信度/rule|llm|hybrid）；`label_text` 与 `merge_labels` 已填充/合并；
- 旧 artifact 读入时默认 unknown/rule，向后兼容；
- 生成 1000 条分层评测队列 `data/eval/recommend/review_queue_2026-11_1000.jsonl`
  （core 584 / unknown 300 / adjacent 76 / none 40，负样本池有限）；
- 测试：914 → 917 passed。

## P2 规则接入 taxonomy（2026-11-12）

- `Labels` 新增 `industry` / `industry_evidence`（taxonomy 规范 id + 命中词）；
- `label_text` 用 `taxonomy.industry_match` 打行业标签，并写入证据/置信度/方法；
- `classify_domain` 关键词判不出时用行业表兜底（core→ai_core、infra→ai_infra、
  application→ai_application、none→other）；
- 测试：行业匹配、行业兜底、标签行业/证据；916 passed。

## P2 作用域与词表扩展（2026-11-12）

- `label_text` 改为“标题+摘要优先、正文兜底”：每个维度先在标题/摘要匹配，命中为空才看正文，
  减少正文背景词污染；
- 词典扩展：topics 8→12、actions 16→20、narrative 7→10、persona 5→7、tech 17→22；
  新增 hardware/open_source/infrastructure/application、personnel/deployment/award/recall、
  howto/crisis/trend、enterprise/creator、mcp/world_model/reinforcement/video/cloud；
- 所有词典字段补 `label_method` / `label_confidence`；
- 测试：924 passed。

## P2 验证结果（2026-11-12 本地重建）

用 11-12 同一份 raw artifact，对比“改造前 artifact”与“当前代码重建”的标签空白率：

| 字段 | 改造前 | 改造后 |
|---|---|---|
| topics | 17% | 16% |
| tech_stack | 27% | 25% |
| subjects | 49% | 49% |
| actions | 25% | 24% |
| narrative_angle | 51% | 39% |
| audience_persona | 11% | 10% |

- 行业未知 55% → 38%（关键词 + 主体行业兜底）；
- 叙述角度明显改善；
- subjects 空白率没变，实体词典仍是最大缺口（P3/词表继续补）；
- GHA 已加 `Label coverage report` 步骤（continue-on-error，报告写入 Step Summary 并随 artifact 上传）。

## P3 LLM 语义标注（2026-11-12，默认关闭）

- `business/ocean/semantic.py`：批量 prompt（枚举 + 证据 + 置信度）→ 校验枚举/证据必须出现在
  标题或摘要 → `SemanticLabeler.apply` 与规则标签合并（method=hybrid）；
- `build_ocean(..., semantic_labeler=...)` 与 `v3_runner_service` 已接入；
- `config/ocean_v3.yaml` 的 `labels.semantic.enabled` 默认 false，开启后 Stage 2 才会调用 LLM；
- 证据不匹配、枚举非法、置信度低于 0.6 一律丢弃/保留规则值；
- 测试：931 passed。

## P4 来源可信度与同源识别（2026-11-12）

- 新增 `config/source_registry.yaml`：按具体源覆盖 tier（1 一手/研究、2 媒体、3 社区），
  category 只做兜底；
- `source_tier.tier_for_item/best_tier` 支持 registry，证据分与同分排序可解释；
- `evidence.dedup_syndicated_titles` 开关（默认关）：同标题多源转载只算一个独立来源；
- 测试：934 passed。

## P5 全文降级与源健康度（2026-11-12）

- `extract_article_text` 增加 JSON-LD `articleBody` 降级：readability 失败时不再直接丢正文；
- GHA 增加 `Source health report` 步骤：最近 7 天 raw artifact 聚合，写 Step Summary 并随 artifact 上传；
- 测试：937 passed。

## P6/P7 点评质量与运维快照（2026-11-12）

- `business/editorial/review_quality.py`：日报/播客点评的结构评分（overall/items/show_notes/chapters/
  comment），低分进入人工复核；
- `scripts/editorial_quality.py`：命令行检查当天两份点评；
- `scripts/ops_snapshot.py`：每日运维快照（事件数、推荐数、点评质量、覆盖率告警、源健康度），
  写入 GHA Step Summary 并随 artifact 上传；
- GHA 已加 Editorial quality / Ops snapshot 两步（continue-on-error，不阻断主链）。

## P0 预标注队列（2026-11-12）

- `scripts/prelabel_queue.py`：对评测队列逐条跑规则标注，输出
  `prelabel_ai_relevance/primary_domain/industry/topics/tech_stack/actions/narrative/confidence`；
- 生成 `data/eval/recommend/review_queue_2026-11_1000_prelabel.jsonl`，人工只需确认/修正；
- 下一步：标 300 条 → 跑规则基线。

## 机器预标注冒烟（2026-11-12）

- `scripts/build_machine_labels.py`：预标注队列 → 评测标注文件 + 分层抽查样本；
- 生成 300 条机器标注（`labels_2026-11_machine_smoke.jsonl`）和 50 条抽查样本；
- 在 11-11 artifact 上跑评测：匹配 157 条，top8 精确率 80.0%、召回 67.3%、粗排保留 5.94%；
- **注意：这是循环冒烟，不是金标准**，只能验证工具链跑通，不能作为上线依据；
- 人工标注 300 条后用同一脚本重跑，才是真基线。

## 预标注自审与 relevance 收紧（2026-11-12）

- 逐条看 50 条抽查：剩余错误约 5–8 条（多数是 adjacent 边界，少数漏判）；
- 修正：移除过宽词（arxiv/论文/模型/算法/图像/视频/开源/learning/generation 等），
  补充 AI 专用词（world model、MoE、agentic、chatbot、NeurIPS/ICLR/CVPR、VLA 等），
  新增“本图片由 AI 生成”这类配图说明的过滤；
- 重新生成预标注队列、300 条机器标注、50 条抽查；
- 冒烟指标：召回 74.05%，top8 精确率 75.0%，粗排保留 6.19%；
- **决定：机器标注只用于降低人工成本，不当作金标准；P3/P4 继续默认关闭。**

## 改造后验证跑结果（2026-11-12，GHA 43173216866）

- 跑完 S1–S7（未发布），耗时约 1h41m，成功；
- 事件数 589；subjects 空白率 38%（改造前 49%），narrative 36%（改造前 51%），
  actions 24%，topics 16%；
- 新顶点：`ai_relevance` core 373 / none 178 / adjacent 35 / unknown 3；
  `primary_domain` ai_core 299 / ai_infra 39 / unknown 251（旧 artifact 无该字段）；
- top8 全部为 AI 主题，无汽车/消费混入；
- 新报告已生成：`label_coverage.md`（告警 2 项：subjects、narrative）、`source_health.md`；
- 日报/播客点评质量均为 1.0；
- 发现 GHA 的 Editorial quality / Ops snapshot 两步漏在 main 外，已补；
- 结论：标签覆盖改善、无类别污染；但 subjects 仍是最大缺口，P3/P4 继续关闭。

## 指标口径修正（2026-11-13）

用户指出“点评质量 1.0”和“空白率下降”不可信，修正如下：

- `review_quality` 改名口径为 **structure_completeness（结构完整度）**：只检查字段/长度，
  不代表内容质量；输出字段改为 `structure_score`，GHA/运维快照同步改名；
- 覆盖率报告新增 **证据覆盖率**，并明确标注“空白率是词表覆盖诊断，不是准确率”；
- 修正对比口径：49%→38% 是两次不同 raw（777 vs 589 事件），不能算改善；
  同一份 11-12 raw 本地重建的真实变化是 narrative 51%→39%，subjects 49%→49%；
- 下一步要补的是**人工金标准上的 precision/recall/校准**，不是继续看空白率。
