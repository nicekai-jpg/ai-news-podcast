# 03 数据供给矩阵与新 per-date 数据契约

> 任务：回答"新设计需要的每块内容，今天的数据管线能提供什么、缺什么、新增什么 JSON 契约"。
> 只读分析，未修改 `src/`、`tests/`、`site/`、`data/`、`config/`。样本日期 `2026-10-15`，结论均带 `file:line`。

## 0. 结论摘要

1. 设计稿 A（`design/2026-10-redesign/directions/A/index.html`）拆成 26 块内容（§2）：**绝大多数已有真实数据源或可确定性推导**（逐字稿/时间戳、音色、日报全文、来源、雷达主推/备选/空态、日期轴、播放器 URL、时长、指标）；**真正缺数据源的只有 5 类**：期号、播客副标题、章节编辑标题、编辑数字（10,000/36+/76x）、雷达候选明细（可用快照重建降级，见 §5.1）。
2. 推荐新增**单个 per-date 文件** `digest_{date}.json`（schema v1），由新模块 `presentation/site_builder/digest_view.py` 在 publish 时生成，同时落 `data/digests/`（真源、随 main 提交）与 `site/data/`（部署物，需 `git add -f`）。
3. **期号无解**：设计稿 `2026-10-15 = 第 286 期` 只能拟合成"2026-01-03 起每日 +1"（`directions/A/index.html:1655-1660` 六连样例严格线性），但仓库可见历史最早只到 `2026-02-18`（初始提交）、现存 brief 从 `2026-06-04` 起，按有提交的日期最多 134 期（日历天最多 136 期）；286 比仓库历史至少多 150 期，**无法验证 → 必须新增持久化期号或产品接受去期号**（§3.1）。
4. **雷达候选明细是硬缺口**：`radar_{date}.json` 的 `meta` 只有计数（`gh_radar_service.py:949-972`），12 个候选的仓库名/星数/排除原因**不落盘**；但可用 `snap_{date-1}.json` + `snap_{date}.json` + 近 30 天 radar 历史**重建本样本 12 个候选的名单/星数/日增/重复收录原因**（12/12 命中），主语言与编辑原因无法复原（§5.1）。推荐 v1 用重建降级表、v2 再改管线落盘审计。
5. 时长两种口径并存：`末段 start+duration = 618.245s（10:18）` 是真实时间轴，`Σduration = 592.821s（09:53）` 是设计稿口径（`index.html:1687,2147`）。契约两个都存，UI 统一用时间轴口径（§3.3）。
6. 部署：`site/` 被 `.gitignore` 整体忽略；新增 `site/data/*.json` 必须在 CI `Commit site` 步骤 `git add -f`（`daily.yml:355-357`），否则月度 prune 的 orphan 重建会把它从 gh-pages 抹掉（`prune_gh_pages.py:122-153`）；`prune_episodes` 的 sweep 只扫 `episodes/` 下日期形状文件，不会误删（`episode_utils_controller.py:52-80`）。
7. 兼容性：`feed.xml`/音频 URL 依赖字段保持不变即可（`rss_gen_view.py:26-61`）；`episodes.json` 只**新增** `issue_no`，不影响现有消费方（§6）。
8. 工作量：v1 契约+推导 ≈1.5–2 人日；雷达审计管线 ≈1 人日；编辑字段 LLM 结构化 ≈1 人日（须 fail-soft）。

## 1. 样本与验证环境

| 样本 | 规模/关键事实 | 用途 |
| --- | --- | --- |
| `data/briefs/brief_2026-10-15.json` | 8450 行；`stories=123`（main 33 / supporting 33 / quick 57），`items=123`；`metadata.total_raw=125,total_deduped=123,total_clusters=123`；`radar.projects=[]` | 新闻侧真值 |
| `data/reports/daily_report_2026-10-15.md` | 1680 字符（CJK 944）；`**导语**` + `## 🚀 重磅解读`(2×三段式) + `## ⚡ 前沿情报`(3 条) + `> **AI 小编深度点评**`；**无雷达章节**（空雷达时 `podcast_report_controller.py:58-59` 返回空串） | 导读/日报契约 |
| `data/gh_radar/radar_2026-10-15.json` | 13 行；`projects=[]`；`meta` 仅 6 键：`pick_repo/runner_up_repos/degraded/reason/candidates=12/excluded_recent=12`（v4 前代码生成；HEAD `build_radar` 已扩到 15 键但无历史样本，`gh_radar_service.py:949-972`） | 雷达空态与硬缺口 |
| `site/episodes/2026-10-15/playlist.json` 与 `.txt` | playlist 55 chunks（host A 28 / B 27；`start` 1.0→616.46，末段 end 618.245，`Σduration=592.821`，段间空档 0.401–0.5s）；`.txt` 55 段 `[Host A]/[Host B]`，文本与 playlist 逐段一致（已校验） | 逐字稿/时长/音色；txt 为旧前端 fallback |
| `data/episodes.json` | 30 期（`keep_last:30`，`config.yaml:8`），`2026-09-16..10-15`；字段 `id/title/description/pubDate/guid/link/enclosure_url/enclosure_length/published_at_iso` | 期号载体、feed 输入 |
| `config/config.yaml` | `podcast`(1-9)、`tts.cosyvoice.synth_variants/ref_audio`(16-29)、`gh_radar`(241-268)、`scoring.role_thresholds`(185-189) | 元信息真值 |
| 历史覆盖 | brief 134 天（`2026-06-04..10-15` 连续）；report 133 个（缺 `2026-08-13`）；radar 34 个（`2026-09-12` 起，22 个有项目）；episodes 30 个 | 期号反推、日期轴可用性 |

## 2. 供给矩阵（设计块 → 今天的数据）

| # | 设计内容（设计稿行号） | 今天的数据源（file:line → 字段） | 抽取方式 | 可靠度 | v1 决策/工作量 |
| --- | --- | --- | --- | --- | --- |
| 1 | 品牌徽章"2026-10-15 · 第 286 期"（`:1612`） | 日期：`data/episodes.json` 的 `id`；期号：**无** | JSON + 推导 | 日期高/期号无 | 新增 `issue_no`（§3.1） |
| 2 | 日期轴 6 pills、前/后日、历史归档（`:1655-1660`） | `player.js:1` 注入的 `DATES`（来自 `episodes.json`，30 期）、`html_gen_view.py:100-128` | JSON | 高 | 复用；pill 期号随 #1 |
| 3 | 播客标题（`:1689`） | `episodes.json[].title`（`podcast_publish_controller.py:54` 生成格式） | JSON | 高 | 直接用 |
| 4 | 副标题"万级智能体协同长跑…"（`:1690`） | **无**；最接近的是 `brief.thesis`（`processor_thesis_service.py:16-22,46-66`，模板腔，10-15 实际=「当DevFest is back遇上实际落地，我们看到了什么」） | 推导 | 低 | v1 用 thesis 标 `fallback` 或隐藏 |
| 5 | 时长+段数"09:53 · 55 段"（`:1687`） | `playlist.json` 的 `chunks[].start/duration`（写入 `cosyvoice2_service.py:42-98`） | JSON+计算 | 高 | 两口径都存（§3.3） |
| 6 | 55 段逐字稿+speaker+时间戳（`:1735,2139`） | `playlist.json`（host+text+start+duration）或 `.txt`（仅 host+text） | JSON | 高 | 前端直接复用 playlist |
| 7 | 双主持音色按钮×2（`:1704-1725`） | `playlist.voices/audios`；`config.yaml:16-18` 每 host 仅 1 个变体；标签 `html_gen_view.py:20-23` | JSON+配置 | 中 | 第二按钮禁用/隐藏（§3.5） |
| 8 | 播放器时钟/进度/±15/倍速/音量/句同步（`:2045-2076`） | `player.js` 已有（`:838,930-1003,1215-1240`） | JSON | 高 | 字段不变（§6） |
| 9 | 今日导读/日报导语（`:1747-1749`） | report md 第 3 行 `**导语**：`（`daily_report_2026-10-15.md:3`）；无 report（如 `2026-08-13`）回退 `brief.thesis` | md 正则 | 高（30/30） | 解析进 digest（§3.6） |
| 10 | 4 格指标"10,000 个/36+ 小时/76 倍/30 条"（`:1752-1768`） | 前三是 main 标题里的编辑数字，无结构字段；"30 条"=main 前 30 | 计算 | 编辑项无 | 换真实指标（§3.7） |
| 11 | 四章节锚点（`:1772-1786`） | **无**；playlist 只有 55 段起点；脚本无章节标记 | 启发式 | 低 | 转场词锚点+截断标题（§3.9） |
| 12 | 日报标题/元信息"全文 1,850 字/约 4 分钟"（`:1799,1808`） | report md h1 + 字符数 + 标题列表 | md 正则+计算 | 高 | 规则见 §3.6 |
| 13 | 重磅解读 2 篇×三段式（`:1821-1862`） | `## 🚀 重磅解读` + `###` + `**事件背景**/**核心突破点**/**对行业与用户的深远影响**` | md 正则 | 高（30/30；老格式 49 篇缺章节，06-04..07-22） | 解析，缺失降空态 |
| 14 | 前沿情报 3 条（`:1864-1890`） | `## ⚡ 前沿情报` + `- **标题**：正文` | md 正则 | 高 | 同上 |
| 15 | AI 小编深度点评（`:1892-1902`） | 末尾 `> **AI 小编深度点评**：…` blockquote（`player.js:337-352` 已抽取） | md 正则 | 高（132/133） | 解析 |
| 16 | 日报内雷达章节 | 代码追加（`podcast_report_controller.py:53-79,146`）；空雷达不追加（`:58-59`） | md 正则 | 高 | 忽略 md 版，用 radar JSON |
| 17 | 来源计数"30 条"与 role 标签（`:1914,1922,2205`） | `brief.stories` 中 `role=="main"` 前 30（main 实际 33）；全量非 skip=123；`role_emoji` | JSON+过滤 | 高 | 规则见 §3.2/3.8 |
| 18 | publisher 分组+徽章（`:2187-2205`） | item 的 `source_name`/`source_category`（`processor_service.py:145-161`；`config/sources.yaml`）；旧前端按 host 归并（`player.js:599-604`） | JSON | 高 | Python 侧归一（§3.8） |
| 19 | 筛选 pill 计数与标题/来源搜索（`:1922-1934`） | 可计算；真实值见 §3.8（设计稿计数差 ±1~3） | 计算/前端 | 高 | 计数必须由数据生成 |
| 20 | 雷达主推卡/备选（`:1988-2020`；旧站 `player.js:750-760`） | `radar.projects[0]` 与 `projects[1:]`（`RadarProject`，`gh_radar_service.py:212-233`；`runner_up_count:2`，`config.yaml:248`），含 `repo/url/description/language/stars/delta_stars/license/readme_intro/readme_excerpt/score/score_parts` | JSON | 高（22/34 天有项目） | 直接用 |
| 21 | 雷达空态文案+扫描数（`:1963-1986`） | `meta.candidates/excluded_recent`；规则在 `config.yaml:241-252` | JSON | 高（计数） | 文案由数据生成（§5.2） |
| 22 | 12 候选折叠表 5 列（`:2022-2038,2142`） | **无明细**（meta 只有计数）；`CANDIDATES_DATA` 是手写 mock | — | 无 | 见 §5.1 两条路 |
| 23 | 昨日项目+delta | HEAD 代码 `meta.yesterday_pick/yesterday_pick_delta`（`gh_radar_service.py:944,969-970`；`_load_yesterday_pick:307-331`），10-15 样本无此键；10-14 radar 为空 → 10-15 真实值=无 | JSON+快照 | 中 | 读前一日 radar+快照；本样本空态 |
| 24 | 雷达规则文案"近 7 日/300 星/天"（`:1974`） | 真实门槛：`min_stars=500`、`created_window_days=30`、`recent_push_days=21`、`repeat_window_days=30`、`readme_probe_limit=12`；**无 300/天阈值** | 配置注入 | 高 | 用 config 真值渲染 |
| 25 | 空/加载/错误态 | 现有 `setEmpty/setLoading` 散落（`player.js:149-155,354-359`）；数据侧无可用性字段 | 计算 | — | digest 增加 `availability`+`warnings` |
| 26 | 封面/LOGO 与 RSS/Apple 入口（`:1608,1628-1633,2051`） | `site/logo.png`（feed `itunes:image` 同源，`rss_gen_view.py:34`）；`feed.xml`+`base_url`（`episode_utils_controller.py:15-37`）；无每期封面 | 静态 | 高 | v1 用 logo；URL 不改（生成封面需 Pillow 新依赖，不建议） |

## 3. 确定性推导规则（逐条用 2026-10-15 验证）

### 3.1 R1 期号：无现有数据源，公式只能拟合设计稿

设计稿已知样例（`directions/A/index.html:1655-1660`）：`10-10=281, 10-11=282, 10-12=283, 10-13=284, 10-14=285, 10-15=286`，斜率严格 1/天。反解锚点得：

`issue(date) = (date − 2026-01-03).days + 1` → `2026-01-03→1`、`2026-02-18→47`、`2026-06-02→151`、`2026-06-04→153`、`2026-10-10→281`、`2026-10-15→286`（6 个设计样例全中）。与留存数据交叉：brief 134 天连续，`06-04=153`、`153+133=286`，**同义反复地相容**（留存窗口恰好 134 天，不构成独立证据）。

反向证据：仓库最早提交 `2026-02-18`（`git log --reverse` 首条，initial commit）、最早 daily 提交 `2026-02-18`、现存最早 report `2026-06-04`；`data/briefs` 在 `2026-06-05` 的清理提交 `7199cc4` 中被裁掉 06-03 及更早的数据（只留 06-04 之后）。仓库可见的 daily 提交共 134 个日期（`2026-02-18` 一天 + `2026-06-02..10-15` 中 133 天）；即使按 06-02 起每日不间断计算也只有 136 期，与 286 差 150 期以上；仓库里没有任何 2026-01 前后的产物。代码侧也没有计数器：`grep "第.*期"` 在 `src/` 无命中；标题硬编码 `f"AI 新闻快报 | {ep_id}"`（`podcast_publish_controller.py:54`）；`episodes.json` 无 `issue_no`；feed 无 `itunes:episode`（`rss_gen_view.py:45-61`）。

**结论：无现有数据源、无唯一可验证解。** 286 只在"2026-01-03 起每日一期、从未断档"假设下自洽，该假设不可证。落地方式：
- **推荐 A（持久化）**：`data/episodes.json` 每条新增 `issue_no`（写入口 `podcast_publish_controller.py:122-134`、`podcast_daily_controller.py:178-190`），规则 = 前一期 `issue_no+1`（同日重跑取原值）；首次迁移按 `286 − (2026-10-15 − date).days` 回填 30 期。风险低：`prune_episodes` 保留整 dict（`episode_utils_controller.py:83-122`）、feed 按名取键（`rss_gen_view.py:45-61`）。风险：若 286 本身是设计误写，错误被固化。
- **备选 B（锚点）**：`config.yaml` 增 `podcast.issue_anchor: {date: 2026-01-03, number: 1}` 运行时计算。风险：断更/一天两期会永久漂移，历史无法修正。
- 产品无法确认真值时：**v1 去掉期号**（`issue.number=null, issue.source="unavailable"`）。

### 3.2 R2 role 映射与分布

实际取值只有 `main/supporting/quick`，**全仓 11287 条 story 无 `skip`**（统计：main 2557、supporting 4234、quick 4496）；`skip` 只在阈值定义与防御分支（`config.yaml:185-189`，`publish:97`、`show_notes_view.py:26`）。阈值：main 12-15、supporting 8-11、quick 5-7（实测 main 12-14、supporting 8-11、quick 5-7，与 `processor_service.py:96-124` 一致）。2026-10-15：main 33/supporting 33/quick 57。设计稿"来源 30 条"= `role=="main"` 的**前 30 条**（main 33 条），顺序与 brief 一致（已逐条比对 `design/.../content/sources_sample.json`）。展示映射：`main→核心信源`、`supporting/quick→前沿参考`（同 `index.html:2205`），`role_emoji` 直接透传。

### 3.3 R3 时长

playlist 时间轴由 TTS 装配产生：`current_time_ms` 从 `vocal_pad_ms=1000` 起算、段间插 400–800ms 静音（`tts_postprocess_service.py:93-131`，`config.yaml:30-40`）。因此：
- **时间轴时长** = `chunks[-1].start+duration = 616.46+1.785 = 618.245s → 10:18`（无 BGM 时 `mix_bgm` 不加尾垫，`tts_postprocess_service.py:37-58`）；
- **净语音时长** = `Σduration = 592.821s → 09:53`（设计稿 `index.html:1687,2147` 用的口径；`player.js:1215-1220` 优先 `audio.duration`、缺失回退 sum）。
- 契约两个字段都存：`duration_seconds=618.245/"10:18"`、`speech_seconds=592.821/"09:53"`；UI 显示与进度条统一用 `duration_seconds`。**验收注意：按时间轴口径设计稿 09:53 会变 10:18，这是口径修正。**

### 3.4 R4 逐字稿与 speaker

`playlist.json` 单文件已含 `id/host/text/start/duration/audios/voices`；`.txt` 只有 `[Host A]/[Host B]`+文本（55 段逐段相同）。设计需要"逐字稿+speaker+时间戳"→**直接用 playlist，digest 不复制全文**，只声明 `transcript.segments_url/text_url/segment_count`，避免双份真值漂移；旧前端解析 `.txt`（`player.js:266,361-445`）保留为 fallback。

### 3.5 R5 host 与音色变体

- host：A=苏晴（女）、B=周航（男），定义 `data/prompts_dao.py:107-112,190-191`，前端写死（`html_gen_view.py:277-284`）。
- 合成变体：`config.yaml:16-18` 当前 `host_a:[lively]`、`host_b:[professional]`，**每 host 仅 1 个真实变体**；`playlist.voices` 每段只有 1 键（已统计 55 段）。
- 设计稿每 host 2 按钮（`index.html:1710-1722`），第二个按钮（专业女声/温暖男声）**今天没有音频**：`warm` 连 `ref_audio` 都没有（`config.yaml:19-29` 只有 professional/lively，且 host 交叉映射，见 AGENTS.md 警告）；标签也不同（真实为 `青春女声/专业男声`，`html_gen_view.py:20-23`）。
- v1 只渲染真实存在的变体（现有代码已是此行为：`html_gen_view.py:35-67`），第二按钮置灰并 title 注明"当前未合成"。若要求双变体：改 `config.yaml:16-18`，TTS 成本约 ×2（`cosyvoice2_service.py:167-194`），host_a 需补女声参考音频。

### 3.6 R6 日报 Markdown 解析（含格式变异证据）

2026-10-15 结构：h1 标题→`**导语**：`一段→`## 🚀 重磅解读`（`### 一、`/`### 二、`，各 3 个加粗小标题 `**事件背景**`/`**核心突破点**`/`**对行业与用户的深远影响**`）→`## ⚡ 前沿情报`（`- **标题**：正文`×3）→末尾 blockquote 点评。全量 133 篇变异（脚本统计）：`导语` 130 篇有、3 篇无（`2026-06-22/07-14/07-22`）；`重磅解读`+`前沿情报` 各 83 篇有、49 篇无（集中在 `06-04..07-22` 老格式）；blockquote 132 篇有、1 篇无（`2026-07-02`）；第二/三段小标题写法极不稳定（`核心突破` 19 与 `核心突破点` 131 并存；影响段 20+ 种：`对行业生态与用户的深刻影响` 52、`深刻影响` 28、`对行业生态的深刻影响` 16、`深层影响` 14、`行业影响`/`行业生态与用户影响` 各 13、`对行业与用户的深远影响` 仅 2）。**当前 30 期窗口全部是标准新格式**（逐篇校验），v1 只需覆盖新格式+缺失降级，不必兼容 6-7 月老格式。导出字段：`lead`、`deep_dives[{heading,background,breakthrough,impact}]`（按三个标签宽松匹配，未匹配段落进 `raw_sections`）、`frontier[{title,summary}]`、`editor_verdict`、`toc`、`char_count=len(md)`、`reading_minutes=max(1,round(char_count/400))`。10-15 实测 `char_count=1680`、`round(1680/400)=4` → **"约 4 分钟"与设计稿一致**；"全文 1,850 字"是手填（`index.html:1799`），真值 1680。雷达章节由代码追加、空雷达不存在（`podcast_report_controller.py:58-79,146`），解析时不要当新闻；前端雷达统一读 radar JSON。实现位置建议 `presentation/site_builder/report_digest_view.py`（纯函数+单测），不进 `business/`。

### 3.7 R7 指标：可计算真值 vs 编辑数字

10-15 可计算真值：`total_raw=125`、`total_deduped=123`、`total_clusters=123`（`processor_service.py:137-142`）、`stories=123`、`main/supporting/quick=33/33/57`、`sources_shown=30`、`segments=55`、`duration/speech=618.245/592.821`、`radar.candidates=12/excluded_recent=12`。设计稿"10,000 个/36+ 小时/76 倍"来自两条 main 标题（`How Third Moment Research runs 10,000 agents for 36+ hours`、`Asana cuts model costs 76x…`），**无结构化字段**，正则抽取会把 arXiv 编号/融资额误当指标。
- v1 建议 3–4 格真实指标：`今日入库 123 条`/`核心研判 33 条`/`双人对谈 55 段`/`精选信源 30 条`。
- 若必须保留编辑数字：v2 新增一次"digest 编辑字段"结构化 LLM 调用（输入 brief+report，输出 metrics/subtitle，数字必须能在输入原文中找到，校验失败整块丢弃），publish 前 fail-soft，约 60 行+1 测试；不要改 writer/report 的 prompt（破坏其契约）。

### 3.8 R8 publisher 分组与筛选分类

数据基础：每条 item 自带 `source_name`（如 `OpenAI Blog`/`arXiv · DeepSeek`/`量子位`）与 `source_category`（`research/official/news/community/tools/product/analysis`，来自 `config/sources.yaml` 的 feed 配置，`processor_service.py:145-161`）。归一规则（与旧前端 `player.js:599-604` 语义一致，由 Python 固定）：
1. link host 以 `arxiv.org` 结尾 → `publisher="arXiv 论文"`；2. 否则 `publisher=item.source_name`（空则用 host）；3. `group`：`arXiv 论文`（同上）/`OpenAI`（source_name 前缀 OpenAI 或 host `openai.com`）/`Google`（`DeepMind Blog`、`Google AI Blog` 或 host `blog.google`）/`NVIDIA`（source_name 含 NVIDIA 或 host `blogs.nvidia.com`）/`Hacker News`（source_name=`Hacker News` 或 host `news.ycombinator.com`）/其余 `科技媒体`。

10-15 实测（验收用）：main 前 30 = arXiv 11/科技媒体 7/OpenAI 6/HN 3/Google 2/NVIDIA 1；全部 123 条 = arXiv 67/HN 25/科技媒体 22/OpenAI 6/Google 2/NVIDIA 1。**设计稿 pill 计数（OpenAI 7、arXiv 12、HN 4、媒体 4）与真实值不符**（`index.html:1922-1928`），移植时必须由数据生成。

### 3.9 R9 章节锚点（无真值，只能启发式）

脚本有稳定转场词可确定性定位（10-15 实测）：chunk21 `start=228.346`「说到这个，NVIDIA今天还有个动作」、chunk31/33 `start=347.985`「还有几条快讯/第一条是…」、chunk39 `start=420.297`「第二条…」、chunk47 `start=519.828`「最后一条，Google DevFest…」；开场固定 `chunks[0].start=1.0`。规则：按 `开场/说到这个|接下来/第[一二三四五]条/最后一条` 取锚点，`title` 取该段前 12 字（**不是编辑标题**），`source="cue_heuristic"`；不足 2 个锚点退回 `[开场,中段,收尾]` 固定三点。fallback 脚本措辞不同（`podcast_writer_service.py:65-121`），匹配不到即退固定锚点。设计稿四章节编辑标题（导语开场/万级智能体长跑/…）**无任何数据源**；要真标题必须改 writer prompt 让 LLM 输出显式章节标记（`data/prompts_dao.py:158+`），属 v2 且只对新一期生效（历史 30 期无法回填）。

## 4. 新产物契约：`digest_{date}.json`（schema v1）

### 4.1 文件、生成位置与调用点

- 真源（随 main 提交）：`data/digests/digest_{date}.json`；部署物：`site/data/digest_{date}.json` + 聚合 `site/data/calendar.json`。
- 新模块 `src/ai_news_podcast/presentation/site_builder/digest_view.py` 公开三个纯函数：`build_digest(root, date, cfg, base_url) -> dict`、`build_calendar(root, dates, issues) -> dict`、`write_digest_outputs(root, dates, cfg, base_url) -> list[Path]`。
- 调用点（三处）：① `podcast_publish_controller.py:158`（`build_index_html` 之后、`_sync_tree` `:160-162` 之前），对 `pruned` 每期生成；② `podcast_daily_controller.py:216`（本地 full run）；③ `scripts/redeploy_site.py:48-55`（Redeploy Site 只重建站点不跑管线，是历史回填/前端改版主力路径）。
- 失败策略：整体 `try/except Exception: logger.warning(...)`，**绝不能让 digest 失败阻断 publish**（同"雷达可失败"原则）；缺数据板块写 `null`+`availability=false`+`warnings`。`availability` 实测：episodes 30 天、report 133 天（缺 `2026-08-13`）、radar 34 天（22 天有项目），日期轴扩到 134 天归档时前端据此渲染空态，不再"fetch 失败即空态"。
- 单文件 vs 按板块拆：推荐单文件（一次 fetch 拿全本期元信息，现有前端已按日期懒加载 reports/radar/playlist 三请求）；逐字稿仍走 playlist、不进 digest。内容不写 `now()`（否则每次 publish 30 文件全 diff）。

### 4.2 JSON Schema（draft 2020-12）

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "DailyDigest",
  "type": "object", "additionalProperties": false,
  "required": ["schema_version","date","date_label","issue","availability","podcast","lead","metrics","chapters","report","sources","radar","meta"],
  "properties": {
    "schema_version": {"const": 1},
    "date": {"type": "string", "format": "date"}, "date_label": {"type": "string"},
    "issue": {"type": "object", "required": ["number","label","source"], "properties": {
      "number": {"type": ["integer","null"], "minimum": 1}, "label": {"type": ["string","null"]},
      "source": {"enum": ["persisted","config_anchor","unavailable"]}}},
    "availability": {"type": "object", "required": ["podcast","transcript","playlist","report","sources","radar"],
      "properties": {"podcast": {"type": "boolean"}, "transcript": {"type": "boolean"}, "playlist": {"type": "boolean"},
        "report": {"type": "boolean"}, "sources": {"type": "boolean"}, "radar": {"type": "boolean"}}},
    "podcast": {"type": "object",
      "required": ["title","subtitle","mp3_url","duration_seconds","duration_label","speech_seconds","segment_count","hosts","transcript"],
      "properties": {
        "title": {"type": "string"}, "subtitle": {"type": ["string","null"]}, "mp3_url": {"type": "string"},
        "duration_seconds": {"type": "number", "minimum": 0, "description": "末段 start+duration"},
        "duration_label": {"type": "string", "pattern": "^[0-9]{2}:[0-9]{2}$"},
        "speech_seconds": {"type": "number", "minimum": 0, "description": "Σ chunk duration"},
        "segment_count": {"type": "integer", "minimum": 0},
        "hosts": {"type": "array", "items": {"type": "object", "required": ["key","name","variants"], "properties": {
          "key": {"enum": ["A","B"]}, "name": {"type": "string"}, "gender": {"enum": ["female","male"]},
          "variants": {"type": "array", "items": {"type": "object", "required": ["id","label","default"], "properties": {
            "id": {"type": "string"}, "label": {"type": "string"}, "default": {"type": "boolean"}}}}}}},
        "transcript": {"type": "object", "required": ["format","segment_count","segments_url","text_url"], "properties": {
          "format": {"const": "playlist"}, "segment_count": {"type": "integer"},
          "segments_url": {"type": "string"}, "text_url": {"type": ["string","null"]}}}}},
    "lead": {"type": ["object","null"], "required": ["text","source"], "properties": {
      "text": {"type": "string"}, "source": {"enum": ["report_lead","brief_thesis"]}, "source_file": {"type": "string"}}},
    "metrics": {"type": "array", "minItems": 0, "maxItems": 5, "items": {"type": "object",
      "required": ["key","label","value","unit"], "properties": {"key": {"type": "string"}, "label": {"type": "string"},
        "value": {"type": ["number","string"]}, "unit": {"type": "string"}, "source": {"type": "string"}}}},
    "chapters": {"type": "array", "items": {"type": "object", "required": ["title","start","label","source"], "properties": {
      "title": {"type": "string"}, "start": {"type": "number", "minimum": 0}, "label": {"type": "string"},
      "source": {"enum": ["cue_heuristic","fallback"]}}}},
    "report": {"type": ["object","null"],
      "required": ["title","char_count","reading_minutes","lead","deep_dives","frontier","editor_verdict","toc"],
      "properties": {
        "title": {"type": "string"}, "char_count": {"type": "integer"}, "reading_minutes": {"type": "integer"},
        "lead": {"type": ["string","null"]},
        "deep_dives": {"type": "array", "items": {"type": "object", "required": ["heading","background","breakthrough","impact"],
          "properties": {"heading": {"type": "string"}, "background": {"type": ["string","null"]},
            "breakthrough": {"type": ["string","null"]}, "impact": {"type": ["string","null"]},
            "raw_sections": {"type": "array", "items": {"type": "string"}}}}},
        "frontier": {"type": "array", "items": {"type": "object", "required": ["title","summary"], "properties": {
          "title": {"type": "string"}, "summary": {"type": "string"}}}},
        "editor_verdict": {"type": ["string","null"]},
        "toc": {"type": "array", "items": {"type": "object", "required": ["anchor","title"], "properties": {
          "anchor": {"type": "string"}, "title": {"type": "string"}}}}, "source_file": {"type": "string"}}},
    "sources": {"type": "object", "required": ["total_all","total_main","shown","role_filter","groups","items"], "properties": {
      "total_all": {"type": "integer"}, "total_main": {"type": "integer"}, "shown": {"type": "integer"},
      "role_filter": {"enum": ["main","all"]},
      "groups": {"type": "array", "items": {"type": "object", "required": ["publisher","count"], "properties": {
        "publisher": {"type": "string"}, "count": {"type": "integer"}, "group": {"type": "string"}}}},
      "items": {"type": "array", "items": {"type": "object",
        "required": ["rank","role","role_label","publisher","group","title","link"], "properties": {
          "rank": {"type": "integer"}, "role": {"enum": ["main","supporting","quick"]}, "role_label": {"type": "string"},
          "publisher": {"type": "string"}, "group": {"enum": ["OpenAI","Google","NVIDIA","arXiv 论文","Hacker News","科技媒体"]},
          "source_name": {"type": "string"}, "category": {"type": "string"}, "title": {"type": "string"},
          "link": {"type": "string"}, "published_at": {"type": ["string","null"]}}}}}},
    "radar": {"type": "object", "required": ["available","status","rules","pick","runner_ups","yesterday","candidates","message"],
      "properties": {
        "available": {"type": "boolean"}, "status": {"enum": ["pick","empty","degraded","unavailable"]},
        "rules": {"type": "object", "properties": {"min_stars": {"type": "integer"}, "created_window_days": {"type": "integer"},
          "recent_push_days": {"type": "integer"}, "repeat_window_days": {"type": "integer"}, "probe_limit": {"type": "integer"},
          "pick_count": {"type": "integer"}, "runner_up_count": {"type": "integer"}}},
        "pick": {"type": ["object","null"]}, "runner_ups": {"type": "array"},
        "yesterday": {"type": ["object","null"], "required": ["repo","delta_stars","date"], "properties": {
          "repo": {"type": "string"}, "delta_stars": {"type": ["integer","null"]}, "date": {"type": "string"}}},
        "candidates": {"type": "object", "required": ["total","excluded_recent","fresh","detail_available"], "properties": {
          "total": {"type": "integer"}, "excluded_recent": {"type": "integer"}, "fresh": {"type": ["integer","null"]},
          "probe_limit": {"type": "integer"}, "semantics": {"type": "string", "description": "pre-v4=README 探测数; v4=去重后池内数量"},
          "detail_available": {"type": "boolean"},
          "preview": {"type": ["array","null"], "items": {"type": "object", "properties": {
            "repo": {"type": "string"}, "stars": {"type": "integer"}, "delta_stars": {"type": ["integer","null"]},
            "excluded_reason": {"type": "string"}}}},
          "reconstruction_note": {"type": ["string","null"]}}},
        "message": {"type": "string"}}},
    "meta": {"type": "object", "required": ["generated_from","warnings"], "properties": {
      "generated_from": {"type": "object", "additionalProperties": {"type": "string"}},
      "warnings": {"type": "array", "items": {"type": "string"}}}}
  }
}
```

### 4.3 用 2026-10-15 真实值填充的示例（节选，数值均见 §1）

```json
{
  "schema_version": 1, "date": "2026-10-15", "date_label": "2026年10月15日",
  "issue": {"number": 286, "label": "第 286 期", "source": "config_anchor"},
  "availability": {"podcast": true, "transcript": true, "playlist": true, "report": true, "sources": true, "radar": true},
  "podcast": {
    "title": "AI 新闻快报 | 2026-10-15", "subtitle": "当DevFest is back遇上实际落地，我们看到了什么",
    "mp3_url": "https://nicekai-jpg.github.io/ai-news-podcast/episodes/2026-10-15.mp3",
    "duration_seconds": 618.245, "duration_label": "10:18", "speech_seconds": 592.821, "segment_count": 55,
    "hosts": [{"key": "A", "name": "苏晴", "gender": "female", "variants": [{"id": "lively", "label": "青春女声", "default": true}]},
              {"key": "B", "name": "周航", "gender": "male", "variants": [{"id": "professional", "label": "专业男声", "default": true}]}],
    "transcript": {"format": "playlist", "segment_count": 55,
      "segments_url": "episodes/2026-10-15/playlist.json", "text_url": "episodes/2026-10-15.txt"}
  },
  "lead": {"text": "今日全球科技圈呈现出\"智能体规模化落地\"的清晰主线——NVIDIA 重金押注法国 AI 生态……",
           "source": "report_lead", "source_file": "data/reports/daily_report_2026-10-15.md"},
  "metrics": [
    {"key": "deduped_items", "label": "今日入库信源", "value": 123, "unit": "条", "source": "brief.metadata.total_deduped"},
    {"key": "main_stories", "label": "核心研判", "value": 33, "unit": "条", "source": "brief.stories[role=main]"},
    {"key": "podcast_segments", "label": "双人对谈", "value": 55, "unit": "段", "source": "playlist.chunks"},
    {"key": "sources_shown", "label": "精选信源", "value": 30, "unit": "条", "source": "brief.stories[role=main][:30]"}
  ],
  "chapters": [
    {"title": "导语开场", "start": 1.0, "label": "00:01", "source": "cue_heuristic"},
    {"title": "说到这个，NVIDIA今天还有个动作", "start": 228.346, "label": "03:48", "source": "cue_heuristic"},
    {"title": "还有几条快讯也挺有意思", "start": 347.985, "label": "05:48", "source": "cue_heuristic"},
    {"title": "最后一条，Google DevFest…", "start": 519.828, "label": "08:40", "source": "cue_heuristic"}
  ],
  "report": {
    "title": "✨ 科技新闻日报 | 2026年10月15日", "char_count": 1680, "reading_minutes": 4,
    "lead": "今日全球科技圈呈现出\"智能体规模化落地\"的清晰主线——……",
    "deep_dives": [
      {"heading": "一、NVIDIA 押注法国，智能体计算挺进欧洲纵深",
       "background": "在欧盟加速推进 AI 主权战略的关键节点，NVIDIA 正式宣布对法国初创企业集群的新一轮重大投资……",
       "breakthrough": "本次投资最具信号意义的关键词是\"agentic computing\"的产业级落地……",
       "impact": "这一布局意味深长——欧洲不再甘于仅做 AI 监管规则的制定者……"},
      {"heading": "二、Third Moment Research：1 万智能体、36 小时长时任务的工业范本",
       "background": "AI 安全研究机构 Third Moment Research 公开披露……",
       "breakthrough": "本次突破同时跨越\"规模\"与\"时长\"两条临界线……",
       "impact": "这标志着 AI 正式从\"对话工具\"过渡为\"可托管的虚拟员工\"……"}
    ],
    "frontier": [{"title": "DevFest 2026 强势回归", "summary": "Google 官方宣布年度开发者盛会 DevFest 2026 启动全球报名……"},
                 {"title": "SentinelOne × OpenAI Daybreak 提速千倍", "summary": "头部网络安全厂商 SentinelOne 接入 OpenAI Daybreak 模型后……"}],
    "editor_verdict": "今日新闻看似分散，实则共同勾勒出 AI 产业的\"三段式跃迁\"图景——……",
    "toc": [{"anchor": "deep-dive-1", "title": "一、NVIDIA 押注法国，智能体计算挺进欧洲纵深"},
            {"anchor": "deep-dive-2", "title": "二、Third Moment Research：1 万智能体、36 小时长时任务的工业范本"}],
    "source_file": "data/reports/daily_report_2026-10-15.md"
  },
  "sources": {
    "total_all": 123, "total_main": 33, "shown": 30, "role_filter": "main",
    "groups": [{"publisher": "arXiv 论文", "group": "arXiv 论文", "count": 11},
               {"publisher": "量子位", "group": "科技媒体", "count": 3},
               {"publisher": "OpenAI Blog", "group": "OpenAI", "count": 6}],
    "items": [{"rank": 1, "role": "main", "role_label": "核心信源", "publisher": "Google AI Blog", "group": "Google",
               "source_name": "Google AI Blog", "category": "official", "title": "DevFest is back",
               "link": "https://blog.google/innovation-and-ai/technology/developers-tools/devfest2026/",
               "published_at": "2026-09-14T16:00:00+00:00"}]
  },
  "radar": {
    "available": true, "status": "empty",
    "rules": {"min_stars": 500, "created_window_days": 30, "recent_push_days": 21, "repeat_window_days": 30,
              "probe_limit": 12, "pick_count": 1, "runner_up_count": 2},
    "pick": null, "runner_ups": [], "yesterday": null,
    "candidates": {"total": 12, "excluded_recent": 12, "fresh": 0, "detail_available": false,
      "semantics": "pre-v4: candidates=README 探测数; v4: candidates=去重后池内数量",
      "preview": [{"repo": "storytold/photocraft", "stars": 60783, "delta_stars": 3176, "excluded_reason": "recent_pick"},
                  {"repo": "Ebony-Vinyl/dsh-our-free-model", "stars": 16083, "delta_stars": 2049, "excluded_reason": "recent_pick"},
                  {"repo": "storytold/wordcraft", "stars": 12260, "delta_stars": 1292, "excluded_reason": "recent_pick"}],
      "reconstruction_note": "由 snap_2026-10-15 + snap_2026-10-14 + 近30天 radar 历史重建；缺 language/编辑原因"},
    "message": "今日候选池经多轮筛查，未检出达标项目"
  },
  "meta": {"generated_from": {"brief": "data/briefs/brief_2026-10-15.json", "report": "data/reports/daily_report_2026-10-15.md",
      "radar": "data/gh_radar/radar_2026-10-15.json", "playlist": "site/episodes/2026-10-15/playlist.json",
      "episodes": "data/episodes.json", "config": "config/config.yaml"},
    "warnings": ["issue_number_unverified", "design_duration_label_uses_speech_sum", "radar_candidate_audit_reconstructed"]}
}
```

`calendar.json`（聚合，日期轴一次拿到期号/可用性）：`{"schema_version":1,"default_date":"2026-10-15","dates":[{"date":"2026-10-15","issue_number":286,"availability":{...},"title":"AI 新闻快报 | 2026-10-15"}]}`（30 条≈4KB，整页只多 1 个请求；拿不到时前端退回现有 `DATES`）。

### 4.4 部署链路（对照现有 site/reports、site/radar）

- `_sync_tree(src,dst)` 在 publish 里把 `data/reports→site/reports`、`data/gh_radar→site/radar` 整体复制（`podcast_publish_controller.py:26-32,160-162`）；`site/` 被 `.gitignore` 整体忽略，gh-pages 由 peaceiris `keep_files:true` 部署（`daily.yml:369-375`）。**必须 `git add -f`**：`Commit site` 只有 `git add -u site/ data/episodes.json`（`daily.yml:355`），新文件不会入 main；需追加 `git add -f site/data/`（或精确到新日期文件）。否则文件只在 gh-pages 活到下一次月度 prune——`prune_gh_pages.py` 先 `git rm -rf .` 重建 orphan（`:122-137`），再从 main 的 `site/` 复制顶层目录（`:139-153`，仅跳过 `episodes`），未跟踪的 `site/data` 在 main checkout 不存在→被抹掉。现存同款隐患：`site/radar/` 在 main 上 0 个 tracked 文件，prune 后 radar 会 404 直到下一次 daily publish。
- **`prune_episodes` 不会误删**：orphan sweep 只在 `episodes_dir` 内匹配 `^(\d{4}-\d{2}-\d{2})(\.(mp3|html|txt))?$`（`episode_utils_controller.py:52-80`）；`site/data/digest_YYYY-MM-DD.json` 既不在该目录也不匹配正则。
- 建议同时提交真源 `data/digests/`（`git add data/digests/`，与 `data/reports/` 同层，便于 redeploy 重建与测试），`site/data` 只是部署副本；想彻底免 `-f` 可改 `prune_gh_pages.copy_site_assets` 顺带同步 `data/reports|gh_radar|digests`（约 15 行+`tests/test_prune_gh_pages.py` 用例），但这是删站脚本，风险高于 v1 的 `git add -f`，**建议 v1 先不改 prune**。单期 digest≈6–8KB，30 期≈240KB，pre-commit 2MB 单文件上限与禁音频钩子均不受影响。

## 5. 硬缺口与两条路

### 5.1 雷达候选明细（本任务最硬的缺口）

今天只有计数：10-15 的 `meta={pick_repo:"",runner_up_repos:[],degraded:false,reason:"",candidates:12,excluded_recent:12}`（v4 前写入逻辑，见 `daily: brief 2026-10-15` commit）；12 个候选的 repo/stars/排除原因**都没有落盘**。注意 **v4 后语义变了**：`candidates=len(seen)`（全部去重池），新增 `fresh_candidates/excluded_recent/pool_by_source/direction_fallback/yesterday_pick/yesterday_pick_delta`（`gh_radar_service.py:949-972`）——同名字段跨版本含义不同，消费方必须看 `semantics`。

**路 A：改管线落盘候选审计（推荐 v2）**
- 改动点：`_hard_filter` 改为返回 `(ok, reason)`（`gh_radar_service.py:193-209`）；`_merge_hard_filtered` 记录被硬过滤项及原因（`:415-430`）；`_select_probe_list` 记录"被 recent_pick 排除/被 probe_limit 截断/回选冷却不足"（`:585-609`）；`_probe_and_score` 保留全部已评分项而非只留 top keep（`:763-792`）；`build_radar` 组装 `candidate_audit:[{repo,stars,delta_stars,language,license,source,affinity,score,exclusion_reason}]` 写入 `meta`（`:871-973`）。
- 工作量：约 80–120 行 + `tests/test_gh_radar_service.py`（现 41 用例）新增 5–8 断言；JSON 每期 +3–8KB。风险中：雷达是"可失败环节"（`runner_service` try/except），审计组装必须同理 try/except，失败只丢审计不丢雷达；`_hard_filter` 签名变更影响所有调用点/测试；`exclusion_reason` 枚举需产品确认（hard_filtered/repeat_window/probe_limit_truncated/direction_gate/score_rank）。收益：设计稿 5 列表格 100% 可填。

**路 B：v1 降级——从快照与历史重建（推荐 v1）**
- 依据：v4 前 `snap_{date}.json` 在"排重与 README 截断之前"写入全部候选且按星数降序（HEAD 版 `gh_radar_service.py:880-891` 注释同义）；`snap_2026-10-15.json` 前 12 个 key 即当日 12 候选，`snap_2026-10-14.json` 提供前一日星数→日增；近 30 天 `radar_*.json` 的 `projects`/`meta.pick_repo`/`runner_up_repos` 给出"已推荐"集合（与 `_load_recent_pick_dates:286-304` 同规则）。
- 实测：前 12 候选 = `storytold/photocraft(60783,+3176)`、`NandhaKishorM/laya(32903,+143)`、`Niko1221/Strata(24524,+862)`、`browser-use/jev-ultrafast(22911,+93)`、`robbietilton/Compositor(16898,+198)`、`Ebony-Vinyl/dsh-our-free-model(16083,+2049)`、`storytold/pdfcraft(13608,+942)`、`storytold/filmcraft(13203,+747)`、`yetone/magpie(12684,+1214)`、`storytold/lightcraft(12524,+568)`、`storytold/wordcraft(12260,+1292)`、`shihabal3amri/DiPlay(10596,+319)`；12/12 命中近 30 天已推荐集合，与 `excluded_recent=12` 完全一致。
- 局限：没有 `language`（设计稿第 4 列）与编辑原因（只能给"近 30 天已推荐/重复收录"），且只对 **v4 前**历史成立；v4 后 `probe_list` 不是星数 top-12（`_select_probe_list` 新品优先），重建会失真 → 契约用 `detail_available=false`+`preview`+`reconstruction_note` 显式标注。**推荐 v1 路 B**（零管线风险、覆盖 2026-10-15 表格），文案如实写"参与评估的 12 个候选（由星数快照重建）"，主语言列显示"—"；路 A 排 v2。

### 5.2 其他硬缺口

| 缺口 | 现状 | v1 处理 | 需改管线时（file:line） |
| --- | --- | --- | --- |
| 期号 | 无持久化（§3.1） | `issue_no` 持久化或去期号 | `podcast_publish_controller.py:122-134`、`podcast_daily_controller.py:178-190` |
| 编辑数字 10,000/36+/76x | 只在 main 标题，无结构 | 换真实指标（§3.7） | 新增结构化 LLM 调用（fail-soft） |
| 副标题/章节编辑标题 | thesis 模板腔；脚本无章节标记 | thesis 标 fallback 或转场词锚点（§3.9） | `processor_thesis_service.py:46-66`、`podcast_writer_service.py`+`prompts_dao.py:158+`（v2，仅新期） |
| 每期封面 | 只有 `site/logo.png` | 用 logo（feed 已引用 `rss_gen_view.py:34`） | 需 Pillow 新依赖，不建议 |
| 雷达"300/天"门槛文案 | config 无此参数（真实 500 星/30 天/21 天活跃） | 用 config 真值渲染（§2 #24） | 无需改 |
| 栏目元信息（全文 X 字/阅读时间/扫描数） | 可计算（§3.6/3.7） | digest 计算字段 | 无需改 |

## 6. 兼容性：feed.xml 与音频 URL 不能动的字段

- `feed.xml` 频道：`title/link/description/language/lastBuildDate/itunes:author/itunes:explicit/itunes:category/itunes:image/atom:link`（`rss_gen_view.py:26-43`，值来自 `config.yaml:1-7`）；条目：`title/description/pubDate/guid(@isPermaLink=false)/link/enclosure(url,length,type=audio/mpeg)`（`:45-61`，来自 `data/episodes.json` 七个键）。**改版不得改**：`guid`（`{base_url}/episodes/{date}`）、`link`（`…/{date}.html`）、`enclosure.url`（`…/{date}.mp3`）、`enclosure.length`、`pubDate`；标题格式 `AI 新闻快报 | {date}` 建议保持。
- 音频 URL：整期 `{base_url}/episodes/{date}.mp3`（`podcast_publish_controller.py:112`）、切片 `./episodes/{date}/{chunk.audios[variant]}`（`player.js:829-838`）、show notes `…/{date}.html`（`:113`）。digest/calendar 只新增 `./data/digest_{date}.json`，不涉及上述 URL。
- `episodes.json` 只**新增** `issue_no` 安全：feed 按名取键（`rss_gen_view.py:45-61`）、index 注入只取 `title/mp3/desc`（`html_gen_view.py:114-123`）、`prune_episodes` 保留整 dict（`episode_utils_controller.py:83-122`）。回归：`uv run pytest tests/test_site_builder_view.py tests/test_html_gen_view.py tests/test_cli_controller.py tests/test_run_daily_full_controller.py -q`，并用真实样本比对 `site/feed.xml` 改动前后 XML 等值（除 `lastBuildDate`）。

## 7. 工作量、风险与验收

| 编号 | 工作 | 改动面 | 量级 | 风险 |
| --- | --- | --- | --- | --- |
| W1 | `digest_view.py`+report parser+sources 归一+schema/单测 | 新模块 ~260 行+`tests/test_digest_view.py` | 1–1.5 人日 | 低（不碰 feed） |
| W2 | 三处调用点接线+失败隔离 | 3 个 controller+30 行 | 0.3 人日 | 低 |
| W3 | 期号持久化（`issue_no`+回填 30 期+config 锚点） | publish/daily controller+迁移脚本 | 0.5 人日 | 中（值本身可能错，需产品确认） |
| W4 | `calendar.json`+前端 fetch 降级+CI 落盘（`git add -f site/data/`、`data/digests/` 提交） | `player.js`+digest_view+`daily.yml` | 0.5 人日 | 低（漏加则 prune 丢文件） |
| W5 | 雷达候选审计（路 A） | `gh_radar_service.py`+41→50 测试 | 1 人日 | 中（可失败原则、meta 语义） |
| W6 | 编辑字段 LLM 结构化（副标题/指标/章节标题，可选） | 新 LLM 调用+校验 | 1 人日 | 中（幻觉需校验、成本） |

**v1 数据层验收**：① 本地起服务后 `./data/digest_2026-10-15.json` 返回 200 且通过 Schema 校验；② 用 2026-10-15 断言 `duration_seconds=618.245`、`segment_count=55`、`sources.total_main=33/shown=30`、`report.reading_minutes=4`、`radar.status="empty"`、`radar.candidates.total=12/excluded_recent=12`；③ 同日重跑 publish 后 digest 字节级不变（确定性）；④ 缺 report 的日期（`2026-08-13`，若日期轴扩展）返回 `report=null` 且空态；⑤ `site/feed.xml` 与改版前逐字段等值（除 `lastBuildDate`）；⑥ prune 演练：`git ls-files site/data` 非空，prune 后 gh-pages 上 `data/*.json` 仍 200。
