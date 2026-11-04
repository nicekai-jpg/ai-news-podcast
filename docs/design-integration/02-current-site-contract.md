# 02 · 现状站点契约盘点 — 改版必须保持什么

> 目的：为「站点改版移植」提供现状边界。改版可换骨架、换 class、换布局，但下列生成链路、
> 数据注入、运行时 fetch、外部产物（feed / Apple Podcasts / 音频 URL）与部署留存约束不能无声破坏。
>
> 方法：只读静态分析，行号对应当前 main 工作区；生成物以 `site/`（2026-10-15 期）为准。
> 配套：`01-prototype-inventory.md`（设计稿）、`04-test-deploy-risk.md`（测试/部署风险）。

## 0. 摘要：改版不可动的 9 条红线

| # | 红线 | 证据 |
|---|---|---|
| 1 | `index.html` 是**单文件内联**产物：`style.css` + `player.js` 拼进同一 HTML，无外链 CSS/JS、无构建步骤 | `html_gen_view.py:133-148,503-509` |
| 2 | 数据靠**字符串占位符替换**注入：`{dates_json}` / `{episodes_map_json}` / `{voices_config_json}` | `player.js:1-3`；`html_gen_view.py:512-519` |
| 3 | 运行时按日期 fetch 四类相对路径：`reports/*.md`、`radar/*.json`、`episodes/*.txt`、`episodes/*/playlist.json` | `player.js:203,236,266,287` |
| 4 | 来源列表从 `episodes.json` 的 `desc` HTML 用 DOM 解析（`li`/`a`/`small`）得来，不是独立数据文件 | `player.js:496-572`；`podcast_publish_controller.py:95-110` |
| 5 | RSS/Apple feed 结构与音频 URL 绝对路径不变 | `rss_gen_view.py:7-64`；`podcast_publish_controller.py:112-134` |
| 6 | `playlist.json` chunk 顺序必须与 `.txt` 逐段顺序**一一对齐**（`#trans-row-{i}` 是唯一锚点） | `player.js:293-307`；`cosyvoice2_service.py:56-98` |
| 7 | gh-pages 只持久化「main 跟踪的 site 文件 + keep_files 残留」；音频/切片/备注 HTML/雷达只在 gh-pages，靠 keep_files 与月度 prune 备份 | `.gitignore:32`；`daily.yml:315-319,366-374`；`prune_gh_pages.py:92-119` |
| 8 | 无前端测试设施：DOM 断言都是「字符串出现在 763KB HTML 里」，死 CSS 也能撑绿 | `tests/test_html_gen_view.py:76-90,104-119` |
| 9 | 现状有**两个未声明变量**（`bgmAudio`、`showToast`）的 ReferenceError，不属于「必须保持的行为」 | `player.js:784,926` 等 |

## 1. 生成链路

### 1.1 `index.html` 如何拼出（`html_gen_view.py:93-521`）

由 `episodes` 算 `dates_list`（降序去重）与 `episodes_map`（`:100-111`；ep id 取 `id`，缺失回退
`guid` 末段去 `.mp3`，`:104`）→ 生成三段 JSON（`:114-128`）→ 读静态文件**整段内联**（`:133-134`；
CSS 进 `<style>` `:144-148`，JS 进 `<script>` `:503-507`）→ 一次性 `str.replace` 六个占位符
（`:512-519`）→ 写 `site/index.html`（`:521`）。唯一外部资源是 marked.js CDN（`:143`）。

实测 763,731 B = 脚本块 653,763 B（其中 `EPISODES` JSON 602,909 B，≈ 全文件 79%）+ CSS 块 88,295 B +
模板 ≈ 21.7 KB（`style.css` 88,897 B / `player.js` 52,038 B）→ 体积由「30 期 × 平均 16KB 的 `desc` 全量内联」
驱动，换来日期切换零请求。

### 1.2 一次发布写/同步了哪些文件

| 产物 | 生成 | 落点 | 进 main |
|---|---|---|---|
| `index.html` / `feed.xml` | `html_gen_view.py:521` / `rss_gen_view.py:7-64` | `site/` | 是（`daily.yml:355`） |
| 每期 show notes HTML | `show_notes_view.py:77-124` | `site/episodes/{date}.html` | **否** |
| 日报 md | report job → `data/reports/` → `site/reports/`（`podcast_publish_controller.py:160`） | `site/reports/` | 是（`daily.yml:356`） |
| 雷达 json | stage1 → `data/gh_radar/` → `site/radar/`（`:162`） | `site/radar/` | **否** |
| 脚本 txt + `playlist.json` | writer（`daily.yml:123`）/ TTS（`cosyvoice2_service.py:95-98`） | `site/episodes/{date}.txt` / `{date}/` | 是（`git add -f`，`:357`） |
| mp3 / 切片 | TTS，经 artifact 传递（`daily.yml:315-319`） | `site/episodes/` | **否**（`daily.yml:279`） |
| `episodes.json` | publish（`podcast_publish_controller.py:120-137`） | `data/episodes.json` | 是 |

`_sync_tree` 是「先 rmtree 再 copytree」（`:26-32`）→ `site/reports`、`site/radar` 每次 publish 整体重建。

### 1.3 四条生成入口的差异（最容易漏）

| 入口 | 写 index/feed | 同步 reports/radar | 备注 |
|---|---|---|---|
| CI publish（`daily.yml:321-325`） | ✅ | ✅（`podcast_publish_controller.py:160-162`） | 完整链路，含 notes/索引/prune |
| 本地 `podcast-daily`（`podcast_daily_controller.py:145-216`） | ✅（`:215-216`） | ❌ **不同步** | 本地看日报/雷达 Tab 会 404 |
| `scripts/rebuild_site.py:39-49` | ✅ | ❌ | base_url 硬编码 `http://localhost:8000/site`，同步 `data/_preview/`（`:51-71`） |
| `Redeploy Site` → `scripts/redeploy_site.py:23-58` | ✅ | ✅（`:49-55`） | 不跑音频，`keep_files` 保音频（`redeploy_site.yml:42-47`） |

**改版含义**：新生成函数必须同时接进 publish 与 redeploy，否则上线后第二天修 CSS 会把线上打回旧结构。

## 2. 注入契约

### 2.1 三个占位符（唯一 Python→JS 数据通道）

```js
const DATES = {dates_json};                 // player.js:1
const EPISODES = {episodes_map_json};       // player.js:2
const VOICES_CONFIG = {voices_config_json}; // player.js:3
```

替换点 `html_gen_view.py:512-519`。`{base_url}` 也在替换列表（`:514`），但模板/JS/CSS 中**没有任何
`{base_url}` 占位**（grep 计数 0）→ no-op，可删；`base_url` 只用于 Python 侧拼 mp3 回退地址（`:118`）与 feed。
`DATES` 是降序 `list[str]`（`:111`）；`EPISODES` 是 `{id:{title,mp3,desc}}`（`:114-123`），`mp3` 取
`enclosure_url`，缺失拼 `{base_url}/episodes/{id}.mp3`（`:118`）——**绝对 URL**；`VOICES_CONFIG` 由
`_voice_labels` 按 `tts.cosyvoice.synth_variants` 收敛（`:35-67`，默认标签表 `:20-23`，未配置变体不渲染按钮
`:64-67`），当前 `host_a:["lively"]`/`host_b:["professional"]`（`config/config.yaml:16-18`），实测注入
`{"host_a":{"lively":"青春女声"},"host_b":{"professional":"专业男声"}}`。

### 2.2 `episodes.json` 条目字段

publish 写入（`podcast_publish_controller.py:122-134`）：`id / title / description / pubDate / guid /
link / enclosure_url / enclosure_length / published_at_iso`。消费方：

| 字段 | 消费方 |
|---|---|
| `id` | `DATES` 键、日期轴、fetch 文件名、prune 白名单（`episode_utils_controller.py:97`） |
| `title` / `description` | 播放器/B 版标题（`player.js:255-258`）；来源列表、日报/逐字稿兜底（`player.js:186,222,276`）与 feed `<description>` |
| `enclosure_url/length`、`pubDate/guid/link` | 音频 URL（`player.js:331,816`）与 feed item 全字段（`rss_gen_view.py:49-61`） |
| `published_at_iso` | 排序键（`podcast_publish_controller.py:139-145`），缺失按 1970 |

索引形状必须是**顶层数组**（`coerce_episode_list` 只接受 list，`episode_utils_controller.py:44-47`）。

### 2.3 `desc` HTML 生成规则（隐式接口，最易忽略）

两处实现结构一致、**转义不一致**：publish 不转义标题/来源（`podcast_publish_controller.py:95-110`），
local daily 用 `_safe_html` 转义（`podcast_daily_controller.py:26-43`）。输出恒为
`<p>{description}</p><ol><li>{role_emoji} <a href="{link}">{title}</a> <small>({source_name})</small></li>…</ol>`：
跳过 `role=="skip"`，只取每个 story 的 `items[0]`。实测 2026-10-15：21,596 B / 123 个 `<li>`；
30 期平均 16.3 KB（6.0–28.6 KB）。

前端依赖（`player.js`）：

`parseSourceItems`（`:551-572`）与 `parseSourcesHtml`（`:496-549`）都以 `li` 为单元、无 `li` 退化 `a`
（`:556-557`）；`a.textContent` 去开头 emoji 白名单（`:562`、`:519`）；来源名优先 `<small>` 去括号
（`:566-567`），否则取 host；两者都无时 `parseSourcesHtml` 把整段塞 `.sources-raw-body`（`:502-508`）。
来源聚合把 `host==="arxiv.org"` 统一归到「arXiv 论文」卡（`:601`）。

**含义**：`desc` 同时是「RSS 摘要 + 来源数据源 + 日报/剧本兜底 HTML」。改成独立
`sources/{date}.json` 要动 publish/daily/prune/redeploy 四链路，且历史日期只有 `desc` 可用。

### 2.4 `playlist.json` 与逐字稿对齐

`{episode_id, chunks[]}`；chunk 字段 `id(1-based)/host("A"|"B")/text/start/duration/audios{变体:文件名}/voices`
（`cosyvoice2_service.py:81-91`），前端只用 `host/start/duration/audios`（`player.js:296-303,832-836`）。
对齐假设：**chunk 顺序 == `.txt` 中 `[Host A]/[Host B]` 段落顺序**。服务端 `parse_dialogue_chunks` 用该正则切段
（`tts_parser_service.py:34-57`）；前端 `parseTranscript` 优先 `<voice>` 标签（`player.js:370-388`），失败退回同款
正则（`:405-412`），再退回「姓名：」（`:416-430`）与交替兜底（`:433-444`）；`loadDate` 用 `currentPlaylist[idx]`
填 `scriptParagraphs[idx]` 与 `#trans-row-{idx}` 的 `data-start/data-duration`（`:293-307`）。
实测当前 `.txt` 全是 `[Host A]/[Host B]`（31 个文件 `<voice` 计数 0），SSML 分支仅历史兼容。`audios` 只含该 host
实际合成的变体；前端回退顺序 `selectedVoices[host] → "professional" → 首个可用键 → chunk.audio → chunk_{NNN}.mp3`
（`:836`）——**不要写死 professional 优先**（配置优先才是语义）。

## 3. 运行时 fetch 契约

路径都是相对站点根（gh-pages 上为 `/ai-news-podcast/...`），保持相对路径即可。

| # | 路径 | 触发 | 成功 | 失败/空态 | 行号 |
|---|---|---|---|---|---|
| 1 | `./reports/daily_report_{date}.md` | report Tab / B 版 | `marked.parse()` 塞 `#report-panel-body`，再抽首个 `<blockquote>` 作「小编锐评」 | 有 `desc` 用 desc 兜底，否则 `📰 该日期暂无日报`，隐藏锐评卡 | `player.js:197-218,337-352` |
| 2 | `./radar/radar_{date}.json` | radar Tab | `renderRadar()`；badge `主推 {repo}`/`该期无雷达` | badge `—` + `📡 该日期暂无项目雷达`（`projects:[]` 也走此路） | `player.js:232-247,693-762` |
| 3 | `./episodes/{date}.txt` | podcast Tab / B 版 | `parseTranscript()` 渲染逐字稿 | 有 desc 用 desc，否则 `🎙️ 该日期暂无剧本` | `player.js:264-283` |
| 4 | `./episodes/{date}/playlist.json` | 紧跟 #3 | 设 `currentPlaylist`、映射时间戳、显示句读按钮 | `!ok`/异常 → 隐藏 `#playback-btn-sentence`（该按钮不存在）→ 回退整轨 | `player.js:285-318` |
| 5 | `./episodes/{date}/{chunk_file}` | 句子模式 `loadChunk` | `audio.src` 播放 | **无错误态**（静默失败） | `player.js:829-865` |
| 6 | `{enclosure_url}`（绝对） | 整轨 `loadAudio` | 同上 | **无错误态** | `player.js:930-943` |
| 7 | marked.js CDN | 页面加载 | `window.marked` 可用 | 未定义时 fetch #1 在 try 内抛错 → desc 兜底 | `html_gen_view.py:143`；`player.js:203-217` |
| 8 | `assets/bgm_placeholder.wav` | 句子模式 | 文件未部署 | 变量未声明，先抛 ReferenceError（§4.8） | `player.js:327,806,1251` |

加载/空态唯一注入点：`setLoading` / `setEmpty`（`player.js:354-359`），DOM 为 `.state-placeholder` /
`.loading-spinner`（`style.css:1083-1085`）。**未覆盖**：音频 404/解码失败、离线、localStorage 被禁。

## 4. DOM id / class 契约

> 判据：`player.js` 的 `getElementById`/`querySelector`、内联 `onclick`、`style.css` 选择器。
> `H:`=`html_gen_view.py`，`J:`=`player.js`，`C:`=`style.css`。

### 4.1 播放器（A 版）

`#main-audio`（`H:497`，`J:11-12` 设音量 0.8）、`#side-podcast-title`（`H:232`/`J:255-257`）、
`#podcast-date-tag`（`H:233`/`J:250-252`）、`#current-time`+`#total-time`（`H:238,243`/`J:1268-1280`）、
`#console-progress-track`（`H:239`；`onclick="seekAudio(event)"`，`J:1011,1031` 用其 `getBoundingClientRect`）、
`#console-progress-fill`（`H:240`/`J:1263-1266`）、`#console-btn-play`（`H:251`；`J:1199,1206` 写 ▶/⏸）、
`#console-speed-btn`（`H:255`；`onclick="cycleSpeed()"`）、`#volume-icon`+`#volume-slider`（`H:257-258`；
`J:1046-1074` 换 SVG path）、`#vinyl-disc`（`H:221`；元素 class 是 `vinyl-disc-lg`，`J:1200,1207` 加/删
`spinning`，`C:256`）。±15s 按钮**无 id**，只有 `onclick="skipAudio(∓15)"`（`H:248,252`），改版不能丢绑定。

### 4.2 逐字稿 / 滚动同步

`#cast-panel-body`+`#cast-panel-body-b`（`H:310,365`，`onscroll="handleTranscriptScroll()"`；`J:856,1165`
作滚动容器）、`#back-to-sync-btn`+`.show`（`H:311`；`J:1090,1101`；`C:857,881`）、JS 生成
`#trans-row-{index}`（`J:462`，索引=chunk 索引）与 `data-offset/data-len/data-start/data-duration`
（`J:462-465,302-303`；无时间戳按字符比例估算，`J:956-963,1136-1153`）、`.transcript-row`+
`.host-a-bubble`/`.host-b-bubble`（`J:456,462`；`C:794,806`）、`.speaking`（`J:849-851,1156-1160`；`C:822`）、
`onclick="seekToParagraph(index)"`（`J:466`→`:945-972`）。`#episode-insight-banner`/`#insight-text-content`
（`H:304,306`）是**静态文案，无 JS 写入** → 每期不变。

### 4.3 音色

`#host-a-voice-pills`/`#host-b-voice-pills`（`H:279,286`）与 B 版 `#host-a-voice-pills-b`/`#host-b-voice-pills-b`
（`H:401,402`）由 `buildVoiceSelectors` 注入（`J:867-913`）；生成 `.voice-pill-btn[data-host][data-variant]` +
`.pill-variant-{variantId}`（`J:898-909`）。点击依赖 `data-host`/`data-variant`（`J:919-921`）。
`selectedVoices` 的键是单字母 `A`/`B`（`J:19`，与 `chunk.host` 一致），**不是** `host_a`。

### 4.4 日期轴（四条并列）

`#date-pills`（`H:202`）、`#report-date-pills`（`H:416`）、`#sources-date-pills`（`H:454`）、
`#radar-date-pills`（`H:479`）；`buildDatePills` 取这四个 id（`J:90-93`），生成 `.date-pill[data-date]`
（`J:108-121`）；`setActiveDate` 用全局 `.date-pill` 同时高亮四组（`J:160-165`）。四组数据都是 `DATES` 全集
（`J:99-102`），podcast 轴的「有 mp3」过滤实际恒真（§2.1）。`#date-pills-b`（`H:343`）注释称「JS Populated」
但从未被填充 → 死 id。

### 4.5 日报 / 来源 / 雷达

- 日报：`#panel-report`（`H:409`）、`#report-date-tag`（`H:422`）、`#report-panel-body`（`H:428`）、
  `#editor-verdict-wrapper`+`#editor-quote-body`（`H:430,439`）。JS 注入 `.report-content`（`J:206`），
  但 CSS 无该规则（正文样式实际由 `.magazine-body` 与 markdown 默认样式承担）。`#magazine-main-title`
  （`H:423`）无引用 → 死 id。
- 来源：`#panel-sources`（`H:447`）、`#sources-board-badge`（`H:465`）、`#sources-board-body`（`H:467`）；
  `renderSourcesBoard` 注入 `.pub-board/.pub-group/.pub-articles/.pub-article/.pub-more`（`J:611-634`，`C:505-570`），
  「展开全部」靠 `onclick="expandPubGroup(this)"`（`J:629,638-643`）；A 版引证卡 `#sources-card`+`#sources-list-body`
  （`H:321,330`），B 版 `#sources-list-body-b`（`H:376`）。
- 雷达：`#panel-radar`（`H:472`）、`#radar-badge`（`H:490`）、`#radar-body`（`H:492`）；`renderRadar` 注入
  `.radar-pick/.radar-chip/.radar-repo/.radar-lead/.radar-intro/.ri-*/.radar-others/.radar-other`（`J:713-761`，
  `C:588-656`）。消费 `projects[].repo/url/stars/delta_stars/language/license/description/readme_intro/
  readme_excerpt` 与 `meta.pick_repo`（`J:695-760`），空数组走空态（`J:696`）。

### 4.6 四 Tab 与 A/B 变体（两套并行骨架）

四 Tab：`#btn-mode-{podcast,report,sources,radar}`（`H:161,165,169,173`，`onclick="switchMode('…')"`）+
`#panel-{…}`（`H:191,409,447,472`）；`switchMode` 切 `body.theme-report/sources/radar`（`J:70-72`，`C:23,31,49`）
并重建日期轴（`J:86`）。A/B：`#btn-variant-a`/`#btn-variant-b`（`H:182,183`）、`#variant-container-a/-b`
（`H:206,336`）；`switchVariant` 切 `body.theme-variant-b`（`J:26`）、持久化 `localStorage['ab_variant']`
（`J:20,24`）、支持 `?variant=A|B`（`J:1287-1291`），并**重跑 `loadDate`**（`J:38-41`）。
B 版是 A 版的**平行 DOM 副本**（标题/日期/进度/逐字稿各一套 `-b` id），且 CSS 在 B 下直接隐藏来源/雷达 panel
（`C:659-660`）而顶部按钮仍在 → B 版点「来源速览」无反应。brief 已判定 A/B 开关不应存在于产品
（`brief.md:20`）；删除需同步清 `switchVariant`、`body.theme-variant-b` 全部分支（`C:1327-1956`）、localStorage
键、`?variant` 参数与两套 `-b` DOM。
**状态类/数据属性契约**：`.active`（tab/pill/voice-pill）、`.show`（toast、back-to-sync）、`.speaking`、
`.spinning`（唱片）、`.dimmed`（`J:780`，CSS 无规则）、`data-date`、`data-host`/`data-variant`、`data-more`、
`data-offset/data-len/data-start/data-duration`。

### 4.7 已死 / 可删清单（改版可顺手清）
**死函数**：`setPlaybackMode`（`J:764-827`）零调用，配套 `pendingPlaybackMode` 块（`J:1225-1258`）因此永不执行
（只在 `:772,1243` 置 null）；`showToast` **被调用但未定义**（`J:926`）→ `#toast`（`H:499`）与 `.toast`
（`C:1088-1089`）是死 UI；`generate_show_notes`（`show_notes_view.py:17-74`）与 `format_friendly_date`
（`html_gen_view.py:70-85`）生产零引用、仅测试用；`MODE_DESC`（`J:49-54`）只写给不存在的 `#hero-desc`。

**死 DOM id（JS 引用、HTML 无，均有 `if` 保护）**：`#hero-title`（`J:173`）、`#hero-desc`（`J:74`）、
`#sources-card-b`（`J:180`）、`#playback-btn-full`/`#playback-btn-sentence`（`J:309-318,773-776`）、
`#voice-selector-row`（`J:778`）。**死 DOM id（HTML 有、JS/CSS 不用）**：`#bgm-audio`（`H:498`）、
`#btn-variant-a-b`（`H:348`）、`#console-btn-play-hero-b`（`H:359`）、`#date-pills-b`（`H:343`）、
`#episode-insight-banner`/`#insight-text-content`（`H:304,306`）、`#host-card-a`/`#host-card-b`（`H:274,281`）、
`#magazine-main-title`（`H:423`）、`#player-title-b`（`H:385`）、`#vinyl-disc-b`（`H:358`；CSS 只匹配
class `.vinyl-disc-b`）、`#visualizer-waves`（`H:225`）。**B 版半死骨架**：`#console-btn-play-b`/
`#console-btn-play-hero-b` 只绑 `toggleAudio()`，而 ▶/⏸ 只写 `#console-btn-play`（`J:1199,1206`）→ 状态永不同步；
`#console-progress-track-b`（`H:395`）的 `onclick="seekAudio(event)"` 内部却取 `#console-progress-track`
（`J:1011,1031`）→ 按隐藏 A 版几何计算，是现存 bug。
**死 CSS**：旧 A 控制台层 `C:349-380,726-744,1191-1210`（`.console-player`、`.console-upper-deck`、
`.console-time-row`、`.console-ctrl-bar`、`.console-lower-player`、`.top-studio-console`、`.podcast-pane` 等 18 个）；
旧唱片层 `C:678-707`（`.vinyl-disc` 与现行 `.vinyl-disc-lg` 重复 → `.visualizer-waves` 的 idle/bounce 动画永不触发，
A 版声波条是静的）；旧 B 层 `C:1170-1240,1477-1660`（`.unified-podcast-card`、`.player-upper`、`.player-top-row`、
`.vinyl-display-b`、`.progress-*-b`、`.b-lower-grid` 等，`.transcript-para` 另见 `C:1354,1631`）；无 DOM 的 `.playback-mode-switcher`
（重复定义 3 次，`C:298,371,1094`）与 `.voice-selector-row`（`C:1126`）；`.sources-card`/`.card-header`
（`C:755-756`，现行 `.sources-card-full`）；`.hero-title`/`.hero-desc`（`C:94-95`）；`.podcast-workspace`（`C:385`）。
另有约 80 个零引用 class 候选（含 `.stagger-3/4`、`.is-visible` 等无 JS 写入的 reveal 类），可用 §9 复跑确认。

### 4.8 已知运行时缺陷（不属于「必须保持的行为」）

`bgmAudio` 在 `player.js` 被引用 14 次（`J:327,784,806,808,842,1041,1049,1063,1067,1193,1202,1209,1251,1252`）
但**从未声明**（`src/` 全仓无 `bgmAudio =`；生成的 `site/index.html` 同样 14 处引用 / 0 处声明）。非严格模式下
进入分支即 ReferenceError：`cycleSpeed` 倍速生效但 `:1043` 抛错 → 按钮文字永停 `1.0x`；`toggleMute`/
`changeVolume` 音频生效但图标/slider 不更新；`loadChunk` 自动播放时 `:840-842` 抛错 → `:849-864` 的高亮/滚动被跳过，
靠 `timeupdate` 每 250ms 补（`J:1283`）；BGM 本身不存在（`assets/bgm_placeholder.wav` 不进 gh-pages）。
改版若复用旧 JS 先清这两处，别把「倍速标签/静音图标不变」当设计意图。

## 5. 功能对齐清单（brief 硬约束 → 现状实现）

| 功能 | 现状实现 | 行号 |
|---|---|---|
| 播放/暂停 + ±15 秒 | `toggleAudio` + audio `play/pause` 事件写按钮/唱片；`skipAudio` 在句子模式按虚拟总时长跨 chunk 换算 | `J:974-981,1187-1211,983-1005` |
| 倍速 + 音量/静音 | `SPEEDS=[1,1.25,1.5,1.75,2]`+`cycleSpeed`；range `0..1`+`changeVolume`/`toggleMute`（有 §4.8 缺陷） | `J:5,1038-1044,1046-1074`；`H:256-259` |
| 双主持音色 | 按 host 渲染变体按钮；句子模式切 chunk 时按 `selectedVoices` 选音频 | `J:867-928,829-865` |
| 逐字稿滚动同步 + 章节跳转 | 时间戳优先、字符比例兜底，用户滚动 6s 后恢复；点句 `seekToParagraph` 换 chunk | `J:1110-1185,1083-1104,945-972` |
| 日期切换 | 四条 `.date-pill` + `loadDate` 加载当日全部数据 | `J:89-165,167-335` |
| 日报渲染 | marked 解析 md + 抽 blockquote 作「小编锐评」 | `J:197-218,337-352` |
| 来源渲染 | desc → `parseSourceItems` → 按媒体分组瀑布流（12 条折叠） | `J:219-230,551-643` |
| 雷达渲染 | radar JSON → 主推卡 + 备选 +「无雷达」空态 | `J:232-247,645-762` |
| RSS 入口 / Apple feed / 音频 URL | 头部 `<a href="./feed.xml">`；`rss_gen_view` 全字段；音频 = `episodes/{date}.mp3`（整轨）+ `episodes/{date}/chunk_NNN_*.mp3`（句子） | `H:185`；`rss_gen_view.py:23-61`；`J:836,838` |

## 6. 部署与留存约束

### 6.1 main 上有什么、gh-pages 上有什么

`.gitignore:32` 忽略整个 `site/`，但 main 实际跟踪 178 个 `site/` 文件：`index.html`、`feed.xml`、`logo.png`、
`.nojekyll`、`reports/*.md`（114）、`episodes/*.txt`（30）、`episodes/*/playlist.json`（30）；`episodes/*.html`、
`*.mp3`、切片音频、`radar/*.json` **不在 main**。部署时 publish 先 `rm -f .gitignore`（`daily.yml:366-367`），
再用 peaceiris 推 `./site` 至 gh-pages，`keep_files: true`（`:369-374`）——该次没有的旧文件不删；
`Redeploy Site` 同样 `keep_files`（`redeploy_site.yml:42-47`）。因此 show notes、音频、切片、`radar/` 都只是
keep_files 残留，不存在于任何分支的最终可见状态（gh-pages 除外）。

### 6.2 30 天留存与月度 prune 假设

`keep_last: 30`（`config/config.yaml:8`）→ `prune_episodes` 删索引外日期并清扫孤儿文件
（`episode_utils_controller.py:83-121`；只动日期形状，且不删比最新期更新的「在途」文件，`:52-80`）。
月度 `prune_pages.yml`（`:4-6`，与 daily 共用并发组 `:15-17`）跑 `scripts/prune_gh_pages.py`：克隆 gh-pages →
按 `episodes.json` 白名单备份 `{id}.mp3/.html/.txt` 与 `{id}/`（`:92-119`）→ 建 orphan（`:122-137`）→
**把 main 的 `site/` 静态文件原样拷回**（`:139-153`，跳过 `episodes/`）→ 还原备份（`:163-201`）→ force-push（`:204-221`）。
关键推论：**只有 main `site/` 里存在的目录能在 prune 后幸存**；`site/radar/` 不在 main → prune 后 gh-pages
**丢掉 radar 目录**，直到下次 daily publish 用 `_sync_tree` 补齐（`podcast_publish_controller.py:162`）。
这是「月度 prune 后雷达 Tab 短暂 404」的根因。改版新增 `site/` 子目录（如 per-date digest JSON）时必须决定：
进 main 跟踪 / 写进 prune 备份白名单 / 接受 404 窗口。

### 6.3 feed / Apple Podcasts 稳定性

`feed.xml` = `rss@2.0` + `channel`（title/link/description/language/lastBuildDate/itunes:author/itunes:explicit/
itunes:category/itunes:image/atom:link rel=self）+ 每期 `item`（title/description/pubDate/guid@isPermaLink=false/
link/enclosure@url,length,type=audio/mpeg）（`rss_gen_view.py:23-61`）。必须稳定三条 URL：`guid =
{base_url}/episodes/{id}`、`enclosure = {base_url}/episodes/{id}.mp3`、`link = {base_url}/episodes/{id}.html`
（`podcast_publish_controller.py:112-118`）——改形状会让 Apple 当新节目/丢订阅。`base_url` 顺序：CLI →
`PODCAST_BASE_URL` → `cfg.podcast.base_url` → 由 `GITHUB_REPOSITORY(_OWNER)` 拼 `https://{owner}.github.io/{repo}`
→ `http://localhost`（`episode_utils_controller.py:15-37`）；config 未设 `podcast.base_url`，线上实际
`https://nicekai-jpg.github.io/ai-news-podcast`。体积 642,014 B / 30 期，`description` 全量内联（单期 6–29 KB）；
改版若往 description 塞更富 HTML，feed 会继续膨胀，建议富内容留在站点页。

### 6.4 其他部署硬约束

无 push 触发（`daily.yml:3-15`、`redeploy_site.yml:7-8` 仅 schedule/dispatch）→ 合并代码**不会**上线；两链共用
并发组 `podcast-pipeline`（`daily.yml:24-25`、`prune_pages.yml:15-17`）；音频禁入 main（`.pre-commit-config.yaml:26-31`，
CI 走 artifact `daily.yml:279-287`）；`static/` 靠 hatchling `artifacts` 打包（`pyproject.toml:52`），改目录要同步。
publish 四道闸门不可绕：mp3 存在且 > `MIN_MP3_BYTES`(100000)、feed 有 enclosure、episodes.json 有条目
（`daily.yml:327-349`），部署后回查 gh-pages 的 mp3 大小与 `playlist.json`（`:376-404`）。

## 7. 测试耦合面（KEEP / UPDATE / DELETE）

无 JS 测试设施：DOM 断言都是「字符串出现在 763KB HTML 里」，且因 CSS 内联，**死 class 也能满足断言**；
更细的测试改造见 `04` 文档 §2。

| 测试 | file:line | 判定 | 说明 |
|---|---|---|---|
| 日期工具 / index 存在 / 含标题 | `test_html_gen_view.py:10-53` | KEEP | 路径契约 `html_gen_view.py:521`；日期函数生产零引用（`:70-85`），删函数则删测试 |
| 含 episodes map / 空列表 `DATES = []` | `:55-74` | **UPDATE** | 数据注入契约（§2.1）；`:69-74` 硬绑 `player.js:1` 字面量，JS 改名必破 |
| key elements / base_url | `:76-102` | **UPDATE** | `layout-grid`、`player-bar` 全仓不存在，只靠 `or` 另一半撑住——`podcast-workspace`(`C:385`)、`console-player`(`C:731`) 是**死 CSS**，删死代码会误报回归；`./feed.xml` 必须保留 |
| transcript parser | `:104-119` | **UPDATE** | `dialogue-line` 全仓不存在；`speaker-label` 只在死 CSS `C:1641,1648,1653`（`.transcript-para` 无元素） |
| TestVoiceLabels 7 例 | `:122-196` | **KEEP（红线）** | 音色注入契约（`html_gen_view.py:35-67`），与 DOM 无关 |
| 来源 Tab / 雷达 Tab | `:199-254` | UPDATE | 全是现状 id/函数名；保留 id 则 KEEP，建议保留空态文案断言 |
| RSS/Apple feed | `test_site_builder_view.py:8-54` | **KEEP（红线）** | `rss_gen_view.py:7-64` 不应触碰 |
| show notes md | `test_show_notes_view.py:10-110` | KEEP | 但被测函数生产零引用（§4.7），可改测 HTML 版 |
| publish 命令 / 产物存在性 | `test_cli_coverage_controller.py:99-130`、`test_run_daily_full_controller.py:176-182`、`test_pipeline_e2e_controller.py:218-225` | KEEP + ADD | 渲染被 mock（`:118-127`），**不会**因改版失败，是「假保护」；ADD 不 mock 的契约用例 |

**建议改版首个 PR 先修** `test_html_gen_view.py:69-74` 与 `:76-102`，否则删死 CSS 时会被误报为回归。

## 8. 改版硬约束与风险表

| # | 约束/风险 | 证据 | 影响 / 建议 |
|---|---|---|---|
| R1 | 763KB 单文件内联，`desc` 占 79% | `site/index.html` 实测；`html_gen_view.py:114-123,133-148` | 首屏/移动端内存；保留「零请求切日期」或改按需 fetch，别两套并存 |
| R2 | 无构建步骤；marked.js 依赖 CDN | `brief.md:40`；`html_gen_view.py:133-134,143`；降级 `player.js:203-217` | 设计稿带工具链无法落地，要求单 `style.css`+`player.js` 内联；断网时日报退化为 desc，保留降级或改服务端渲染 |
| R3 | `localStorage['ab_variant']` + `?variant=`；`theme-*` 5 套调色板 | `player.js:20,24,26,70-72,1287-1291`；`style.css:23-65,1327-1956` | 删 A/B 后残留无害但**别复用**键名；主题与「一个日期四个板块」冲突，改单一 token 层并同步删 JS class 写入 |
| R4 | 两套平行 DOM（A/B 各一套 id）+ 日期轴 ×4 | `H:206-332` vs `H:336-405`；`player.js:89-158` | 改一个控件改两处，B 版已有 3 处状态 bug（§4.7）；建议收敛为一套 + 统一日期轴，保留 `data-date` 语义 |
| R5 | 音频 URL / chunk 文件名硬编码 | `player.js:836,838`；`cosyvoice2_service.py:63` | 改 naming 同时破 feed/CI 校验/prune/播放器；四链路一起改并加迁移测试 |
| R6 | 单一 `#main-audio` 跨 Tab 共享 | `H:497`；`J:11` | brief 的「常驻播放器」现状已具备；新版保持「单一 audio + 切板块不断播」 |
| R7 | `site/` 被 gitignore，PR 看不到渲染 diff | `.gitignore:32` | review 盲区；PR 附 `scripts/rebuild_site.py:51-71` 生成的 `data/_preview/` 截图 |
| R8 | 新增 `site/` 子目录会踩 prune | `prune_gh_pages.py:139-153` | 月度 prune 后目录消失（如 radar）；进 main 跟踪或写进 prune 白名单 |
| R9 | 无前端测试设施 | §7；`04` §3 | 靠人工清单 + 字符串断言；建议新增「数据注入块解析」断言 + 冒烟清单 |
| R10 | `bgmAudio`/`showToast` 未声明 | `player.js:784,926` 等 | 复用旧 JS 会把 bug 带进新版；重写时删 BGM/toast 或补声明 |
| R11 | `desc` 双实现、publish 不转义 | `podcast_publish_controller.py:105-108`；`podcast_daily_controller.py:38-41` | 标题含 `<`/`&` 时可能注入/截断；收敛为单一 `build_description_html` 并统一转义 |
| R12 | 日期轴不校验日报/雷达文件存在 | `J:99-102` vs `J:203,236` | 任意历史日期都可能 404 后走空态——空态是既有契约，必须保持 |

## 9. 复核命令（只读，可复跑）

```bash
wc -c site/index.html site/feed.xml                       # 763731 / 642014
git ls-files site | awk -F/ '{print $2}' | sort | uniq -c  # 178 个（无 radar）
grep -n "fetch(" src/ai_news_podcast/presentation/site_builder/static/player.js  # 4 处
grep -n "{dates_json}\|{episodes_map_json}\|{voices_config_json}\|{base_url}" \
  src/ai_news_podcast/presentation/site_builder/html_gen_view.py
grep -n "bgmAudio\s*=\|showToast\s*=\|function showToast" \
  src/ai_news_podcast/presentation/site_builder/static/player.js  # 预期无输出
uv run pytest tests/test_html_gen_view.py tests/test_site_builder_view.py \
  tests/test_show_notes_view.py tests/test_cli_coverage_controller.py -q
```
