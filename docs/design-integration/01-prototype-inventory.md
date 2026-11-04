# 方向 A（每日刊）原型盘点与移植前评估

> 盘点对象：`design/2026-10-redesign/directions/A/{index.html,mobile.html,DESIGN.md}`，输入约束见
> `design/2026-10-redesign/brief.md`、`design/2026-10-redesign/README.md`。
> 对照生产实现：`src/ai_news_podcast/presentation/site_builder/`、`data/`、`site/`。
> 本文所有行号均指目标文件内的物理行，格式 `file:line`。盘点只读，未改任何生产代码。

## 0. 结论摘要（给计划的第一屏）

1. **形态**：桌面稿是单文件自包含原型，`design/.../A/index.html` 共 2599 行；`<style>` 7–1600（1593 行 / 40.2KB），
   DOM 1602–2135，内联 JS 2137–2597（460 行 / 36.1KB）。移动稿 `mobile.html` 共 1036 行：CSS 7–641（634 行 / 15.5KB），
   DOM 643–851，JS 853–1034（181 行 / 24.6KB）。两文件均零外部依赖（无 CDN、无网络字体，字体栈自托管于 `index.html:47-49`）。
2. **不是响应式**：两个文件里 `@media` 出现次数均为 **0**；移动稿把 390/480px 布局写死（`mobile.html:70-77`，
   `.mobile-app{max-width:480px}`），并用 `m-` 前缀整套重命名 DOM/类。它是**独立 fork**，不是桌面稿的窄屏分支。
3. **但可统一**：移动稿 205 条唯一 CSS 声明中 **157 条（76.6%）**与桌面稿逐字相同，`:root` 令牌 22 个里 20 个重合；
   类名只有 8 个共享（`active/arxiv/google/openai/open/host-a/host-b/show`）。重复的是"设计系统 + 组件外观"，
   不重复的是"布局意图"。因此“一套响应式构建”成立，**移动稿应作为断点证据而不是交付物**（详见 §3.4）。
4. **模拟数据真伪已核对**（Python 逐值比对，见 §8）：
   - `PLAYLIST_DATA`（`index.html:2139`）与 `site/episodes/2026-10-15/playlist.json` **完全相等**（55 段，A 28/B 27）；
   - `RADAR_DATA`（`:2141`）与 `data/gh_radar/radar_2026-10-15.json` **完全相等**（`projects:[]`，`meta.candidates=12`）；
   - `SOURCES_DATA`（`:2140`）= `data/briefs/brief_2026-10-15.json` 中 `role=="main"` 的**前 30 条**（33 条 main 截断为 30，
     `role/title/source/link` 30/30 与 brief 一致）；
   - `CANDIDATES_DATA`（`:2142`）**没有生产来源**：radar JSON 的 `meta` 只有计数字段（`gh_radar_service.py:956-965`），
     12 行候选（repo / stars / 日增速 / 排除原因）是手写 mock。
5. **日报是真实内容的静态化**：`index.html:1804-1904` 的正文与 `data/reports/daily_report_2026-10-15.md` 逐节对应，
   `:1748` 的导语就是该 md 的“导语”段（仅引号字符差异）。生产在运行时 fetch md 并 `marked.parse`（`player.js:203-206`）。
6. **播放模型不一致（最大的移植工作量）**：原型是“单个整集 MP3 + setInterval 假时钟”
   （`index.html:2177` 硬编码 gh-pages URL、`:2360-2371` 定时器兜底）；生产是“逐 chunk 音频 + playlist.json 时间轴”
   （`player.js:287` 拉 playlist、`:829/838` 按 `chunk.audios[voice]` 换源）。原型的双音色胶囊（`:1710-1722`）
   只改文字和 toast（`switchVoice` `:2433-2440`），从不换音频。
7. **音色数据现实**：生产 `config/config.yaml:16-18` 当前 `host_a:["lively"]`、`host_b:["professional"]`，
   2026-10-15 的 playlist 每段只有一个 `audios` 变体；“温暖男声(warm)”在生产 `_DEFAULT_VOICE_NAMES`（`html_gen_view.py:20-25`）
   与 config 中都不存在。原型四音色两两可切是 mock-only。
8. **关键数据缺口**：期号“第 286 期”（`index.html:1612,1655-1660`）、hero 指标四项（`:1752-1769`）、
   hero 副标题（`:1690`）、章节 4 个时间点（`:1774-1785`）、候选 12 行明细、项目日增速、`duration=592.8`（`:2147`）
   都没有生产数据源；`episodes.json` 只留 30 期（`config.yaml:7 keep_last: 30`），无法还原累计期号。
9. **移植后收益**：生产 `site/index.html` 当前 763KB，其中约 490KB 是 30 期 `description` 内嵌（`data/episodes.json` 549KB）；
   原型整页 110KB。统一响应式后预计可省 ~450 行 CSS 与 18.2KB 重复数据常量（移动稿 JS 24.6KB 中 18.2KB 是数据）。

## 1. 页面骨架（桌面 `index.html`）

```text
body                                                  1602
├─ header.site-header (sticky, z=50, backdrop-blur)   1605–1639
│   ├─ .brand-group > img.brand-logo-img + .brand-title/.brand-badge  1607–1616
│   ├─ nav.header-nav (4 个锚点)                       1619–1624
│   └─ .header-actions (RSS / Apple Podcasts / 390px 预览按钮)        1627–1637
├─ main.app-layout (max-width 1400, padding 0 32)     1641
│   ├─ section.shared-date-axis                       1644–1666
│   │   ├─ .date-nav-controls（前/后一日）              1645–1652
│   │   ├─ .date-pills-carousel（6 个 .date-pill）      1654–1661
│   │   └─ .date-archive-link（历史归档）               1663–1665
│   ├─ section#section-hero.content-section           1669–1790
│   │   └─ .hero-chronicle-grid（grid 500px 1fr）       1670–1789
│   │       ├─ .podcast-hero-card#podcast-card         1673–1738
│   │       │   ├─ .podcast-cover-meta（黑胶+波形+标题） 1674–1692
│   │       │   ├─ .host-control-box（双主持音色）       1695–1726
│   │       │   └─ .podcast-cta-row（收听/逐字稿）      1729–1737
│   │       └─ .editorial-lead-card                    1741–1787
│   │           ├─ .editorial-header-stamp             1742–1745
│   │           ├─ .lead-quote-text（导语）             1747–1749
│   │           ├─ .metrics-summary-strip（4 指标）     1752–1769
│   │           └─ .chapters-row（4 章节跳转）          1772–1786
│   ├─ section#section-daily.content-section           1793–1907
│   │   └─ .daily-reading-layout（max-width 760）       1794
│   │       └─ article.daily-report-wrapper            1804–1904
│   │           ├─ .daily-meta-bar（pill/阅读时长/TOC）  1805–1816
│   │           ├─ h1.daily-article-title              1818
│   │           ├─ #daily-sec-1 重磅解读（2 张 .deep-dive-card） 1821–1862
│   │           ├─ #daily-sec-3 前沿情报（3 条 .frontier-item）  1865–1890
│   │           └─ #daily-sec-4 .editor-quote-card（点评）       1893–1902
│   ├─ section#section-sources.content-section         1910–1948
│   │   ├─ .sources-controls-row（分类 pills + 搜索）    1919–1936
│   │   ├─ #sources-grid（JS 渲染，桌面 3 列）           1939–1941
│   │   └─ .source-expand-bar（展开全部）               1943–1947
│   └─ section#section-radar.content-section           1951–2041
│       └─ .radar-card-container                       1960–2040
│           ├─ #radar-empty-box（canvas + 空态动作）     1963–1986
│           ├─ #radar-hit-box（模拟命中态）              1989–2020
│           └─ #candidates-table-wrap（12 行手风琴）     2023–2038
├─ footer.docked-audio-player#docked-player (fixed, 72px)  2046–2094
│   ├─ .player-left（封面 + 当前句）                     2050–2056
│   ├─ .player-center（±15s/播放/进度条）                2059–2077
│   └─ .player-right（倍速/音量/逐字稿开关）             2080–2091
├─ aside.transcript-drawer#transcript-drawer (fixed 440px) 2097–2118
├─ #toast-pill                                         2121
├─ #mobile-modal（iframe 预览 mobile.html，移植时删除）  2124–2132
└─ audio#main-audio                                    2135
```

移动稿骨架同理但为单列：`mobile.html:646-657` header → `662-668` 横滑日期条 → `671-708` 播客卡 →
`711-735` 导语/指标 → `738-781` 日报（纯静态 prose，无 TOC/无前沿情报结构）→ `784-799` 来源（最多 15 条、无搜索）→
`802-818` 雷达（仅空态，无候选表/无命中态）→ `823-835` 迷你播放器 → `838-848` 全屏 bottom sheet。

## 2. 组件清单（DOM / 职责 / JS / 数据 / 绑定方式）

| 组件 | DOM id / class | 职责 | 相关 JS（行号） | 用到的 mock 常量 | 事件绑定 |
| --- | --- | --- | --- | --- | --- |
| 共享日期轴 | `.shared-date-axis`、`.date-pill`、`.date-step-btn` | 切日期、前后日、历史归档 | `simulateDateSwitch`（`:2509-2514`）只切 active + toast | 日期文本/期号硬编码 `:1655-1660` | 内联 `onclick` |
| 播客头条卡 | `#podcast-card`、`.podcast-*` | 标题、总时长、段数、黑胶波形动画 | `togglePlay/playAudio/pauseAudio`（`:2338-2382`）；`.is-playing` 驱动 `barPulse`（`:423-432`） | 标题/副标题/`09:53 · 55 段` 静态 `:1687-1690` | 内联 `onclick` |
| 双主持音色 | `.host-control-box`、`#live-speaker-badge`、`.voice-btn` | 两组音色胶囊 + 当前发言人 | `switchVoice`（`:2433-2440`）仅记录 `selectedVoice` + toast；`updatePlayerUI` 切 badge（`:2464-2470`） | 4 个音色名静态 `:1710-1722` | 内联 `onclick` |
| 导语/指标/章节 | `.editorial-lead-card`、`.metrics-summary-strip`、`.chapters-row` | 编辑部导语、4 个定量指标、章节跳转 | `seekToTimestamp`（`:2391-2396`） | 导语文本 `:1748`、指标 `:1752-1769`、章节时间 `:1774-1785` | 内联 `onclick` |
| 日报阅读 | `.daily-report-wrapper`、`#daily-sec-*` | 结构化长文、TOC、金标点评 | 无（纯静态 DOM） | 正文全部静态 `:1804-1904` | 无（锚点 `<a href>`） |
| 来源网格 | `#sources-grid`、`#source-filters`、`#source-search`、`#btn-expand-sources` | 12 条起、分类过滤、搜索、展开全部 | `renderSources`（`:2188-2229`）、`filterSources`（`:2232-2245`）、`handleSourceSearch`（`:2248-2254`）、`toggleExpandSources`（`:2256-2261`） | `SOURCES_DATA`（`:2140`，30 条） | 内联 `onclick` + `oninput` |
| 项目雷达 | `.radar-card-container`、`#radarCanvas`、`#radar-empty-box`、`#radar-hit-box`、`#candidates-table-wrap` | 空态扫描动画、命中态预览、候选手风琴、mock 切换 | `renderCandidatesTable`（`:2303-2316`）、`toggleCandidatesTable`（`:2318-2321`）、`toggleRadarMockState`（`:2323-2335`）、`initRadarSweep/draw`（`:2523-2587`） | `RADAR_DATA`（`:2141`）、`CANDIDATES_DATA`（`:2142`） | 内联 `onclick` |
| 吸底播放器 | `#docked-player`、`#dock-play-btn`、`#scrubber-track/#scrubber-fill`、`#speed-btn`、`#mute-btn/#vol-slider`、`#btn-drawer-toggle` | 播放控制、进度、倍速、音量、抽屉入口 | `togglePlay/playAudio/pauseAudio/skipAudio/handleScrubberClick/cycleSpeed/handleVolume/toggleMute/updatePlayerUI`（`:2338-2483`） | `duration=592.8`、`SPEEDS`（`:2147-2149`）、`PLAYLIST_DATA`；音频 URL `:2177` | 内联 `onclick`/`oninput` + `audio` 元素 |
| 逐字稿抽屉 | `#transcript-drawer`、`#drawer-chunk-list`、`.drawer-filter-group` | 55 段列表、按句跳播、高亮跟随、只看某主持 | `renderTranscriptChunks`（`:2264-2294`）、`filterDrawerChunks`（`:2296-2300`）、`toggleTranscriptDrawer`（`:2486-2494`）、`updatePlayerUI` 高亮（`:2458-2482`） | `PLAYLIST_DATA.chunks` | 行级 `addEventListener('click')`（`:2287`） |
| Toast | `#toast-pill` | 操作反馈 | `showToast`（`:2498-2506`） | — | 内联/函数调用 |
| 移动预览 | `#mobile-modal` | iframe 内嵌 `mobile.html`（评审用） | `toggleMobileModal`（`:2517-2520`） | — | 内联 `onclick` |
| 移动迷你播放器 | `.m-mini-player`、`#m-mini-play-btn`、`#m-player-line`（`mobile.html:823-835`） | 封面+当前句+播放/暂停 | `togglePlay`（`:872-898`）、`updateUI`（`:900-921`） | `PLAYLIST`（`:854`） | 内联 `onclick` |
| 移动 Bottom Sheet | `#m-sheet`、`#m-sheet-body`（`mobile.html:838-848`） | 全屏逐字稿、按句跳播 | `renderMChunks`（`:956-983`）、`toggleSheet`（`:985-987`） | `PLAYLIST` | 内联 + 行级 `addEventListener`（`:975`） |
| 移动来源/雷达 | `#m-source-list`、`#mRadarCanvas`（`mobile.html:789-818`） | 15 条列表、分类筛选、小结雷达动画 | `renderMSources/filterMSources`（`:923-954`）、`initMRadar`（`:1003-1027`） | `SOURCES`（`:855`） | 内联 `onclick` |

绑定方式统计（Python 计数）：桌面 DOM 内联属性 `onclick` 43 处、`oninput` 2 处，`addEventListener` 仅 2 处（都在逐字稿行）；
移动端 `onclick` 20 处、`addEventListener` 2 处。即**几乎所有行为都挂在全局函数上**，函数作用域就是 `<script>` 顶层。

## 3. CSS 体系与“桌面/移动能否统一”

### 3.1 Token 块

`index.html:8-50` 共 34 个自定义属性：背景/表面 6、边框 3、文字 4、强调色 6、host 色 4、圆角 5、
阴影 3、字体栈 3（`:47-49`）。移动稿 `mobile.html:8-38` 只有 22 个，缺 `--accent-glow/--accent-rose/--font-serif/--radius-xl(24→20)/三个 shadow`
与 4 个 host/inset 色。**token 层是超集关系**，统一构建只需以桌面 `:root` 为准，移动稿无需另立一套。

### 3.2 CSS 分区（桌面）

| 区块 | 行号 | 布局手法 |
| --- | --- | --- |
| reset + body + 滚动条 | 52–97 | `body{overflow-x:hidden;padding-bottom:96px}` 给吸底播放器留位（`:67`） |
| 页面容器 | 99–104 | `.app-layout{max-width:1400px;padding:0 32px}` |
| 顶栏 | 106–237 | `position:sticky;top:0;z-index:50;backdrop-filter:blur(16px)`（`:107-116`） |
| 日期轴 | 238–324 | flex 三段；`.date-pills-carousel{overflow-x:auto;scrollbar-width:none}`（`:278-288`） |
| 版块头 | 325–359 | flex + 下边框；`.content-section{scroll-margin-top:86px}`（`:326-329`） |
| Hero 双栏 | 360–367 | `grid-template-columns:500px 1fr` |
| 播客卡 | 368–468 | 卡片 + `::before` 径向光斑（`:381-390`）+ 5 根波形 `barPulse`（`:423-432`） |
| 双主持控件 | 469–577 | 行内 flex + 分段胶囊 `.voice-segmented`（`:546-552`） |
| CTA / 导语 / 指标 / 章节 | 578–739 | 指标 `grid repeat(4,1fr)`（`:668-676`），章节 flex-wrap |
| 日报 | 740–912 | 外层 760px 居中（`:741-744`），内卡 `padding:36px 40px`（`:745-754`），卡片化论点与金标引言 |
| 来源 | 913–1060 | 控制行 flex-wrap + `grid repeat(3,1fr)`（`:975-979`），卡片 hover 位移 |
| 雷达 | 1061–1206 | 状态卡片；空态 flex；候选表格默认 `max-height:0`，`.open` 展开（`:1152-1153`） |
| 吸底播放器 | 1207–1406 | `position:fixed;bottom:0;height:72px;z-index:90`（`:1208-1220`）；内层 `grid-template-columns:320px 1fr 280px`（`:1223-1232`）；自定义进度条（`:1323-1350`） |
| 逐字稿抽屉 | 1407–1534 | `fixed;right:0;width:440px;height:100vh;transform:translateX(100%)`，`.open` 归零，`cubic-bezier(.16,1,.3,1)`（`:1408-1424`） |
| Toast | 1535–1558 | `fixed;bottom:84px;left:50%`，`.show` 上移淡入 |
| 移动预览 Modal | 1559–1600 | `fixed;inset:0;z-index:300`（评审用，删） |

### 3.3 移动稿差异与重复度量（回应架构问题）

移动稿 634 行 CSS 的“独有意图”只有 5 块：横滑日期条（`mobile.html:133-160`）、播客卡窄屏变体（`161-205`）、
导语+2×2 指标（`275-323`）、单列来源列表（`401-467`）、迷你播放器（`495-564`）与 bottom sheet（`565-621`）。
其余（header、卡片、徽章、字体、颜色、圆角、hover 态）是桌面样式的改名复制。

量化结果（Python 解析两文件 `<style>` 后比对）：

| 指标 | 桌面 index.html | 移动 mobile.html | 结论 |
| --- | --- | --- | --- |
| CSS 行数 / 字符 | 1593 / 40.2KB | 634 / 15.5KB | 移动 40% |
| 规则数 | 231 | 87 | 移动 38% |
| 唯一声明数 | 398 | 205 | — |
| 移动声明在桌面中逐字存在 | — | **157/205 = 76.6%** | 重复率高 |
| 声明 Jaccard | — | **0.352** | 结构差异主要在布局 |
| 类名集合 | 172 | 75 | 共享仅 8 个（`active/arxiv/google/openai/open/host-a/host-b/show`） |
| `:root` 令牌 | 34 | 22 | 20 个完全相同 |
| `@media` | **0** | **0** | 无任何断点 |
| `@keyframes` | `barPulse` | 无 | 移动端丢掉了播放动画 |

JS 侧：移动稿 181 行 / 24.6KB 中，两条数据常量 `PLAYLIST`/`SOURCES`（`mobile.html:854-855`）占 18.2KB，
与桌面 `:2139-2140` 是同值复制（仅变量名从 `PLAYLIST_DATA` 改成 `PLAYLIST`）；剩余 180 行代码里 133 行非空，
与桌面逐字相同的只有 31 行（23.3%），但 `formatTime/togglePlay/updateUI/renderSources→renderMSources/
initRadarSweep→initMRadar` 是**语义平行实现**，两套 `isPlaying/currentTime/audio` 状态互不相通。

**建议（会影响计划架构，建议直接采纳）**：以 `index.html` 为唯一模板，用 3 个断点收敛移动稿的独有意图，删除 `mobile.html`：

1. `@media (max-width: 1024px)`：`.hero-chronicle-grid{grid-template-columns:1fr}`、`.metrics-summary-strip{repeat(2,1fr)}`、
   `.sources-grid-layout{repeat(2,1fr)}`、`.player-inner{grid-template-columns:1fr auto}`。
2. `@media (max-width: 720px)`：`.app-layout{padding:0 16px}`、header 只留品牌 + RSS（隐藏锚点导航）、
   `.sources-grid-layout{1fr}`、`.shared-date-axis` 横滑、日报内边距降到 `20px 18px`。
3. `@media (max-width: 720px)` 下把 `.transcript-drawer` 改写为 bottom sheet（`inset:0;width:auto;transform:translateY(100%)`），
   把 `.docked-audio-player` 降级为 64px 迷你条（隐藏 scrubber/倍速/音量，保留封面+当前句+播放），
   即用**同一个节点 + 同一份播放状态**覆盖移动稿 `495-621` 的两块。（移动稿的 `m-*` 组件与 180 行 JS 全部不需要移植。）
4. 收益/代价：预计新增约 150–180 行 `@media`，替换掉 634 行独立 CSS，净减约 450 行；删除 18.2KB 重复数据常量；
   消灭两套播放器状态机。代价是需在断点里覆盖 5 个移动专有块，且要补回 `.is-playing` 波形动画（移动稿丢失）。

### 3.4 外部依赖

原型零 CDN、零字体请求；图标全部用 emoji；图片仅相对路径 `logo.png`（`index.html:1608,2051`；`mobile.html:648,825`）。
生产反而有外部依赖：`html_gen_view.py:143` 的 jsDelivr `marked.min.js`（日报 Markdown 渲染），
这是重设计后**需要决策**的点：要么继续 CDN（国内网络/离线不稳），要么构建期渲染 md，要么把 marked 内联进 static。

## 4. 模拟数据 → 真实数据映射

| 页面内容（原型行号） | 真实来源（生产） | 缺口 / 改造点 |
| --- | --- | --- |
| 期号“第 286 期”、日期 pill 文案（`:1612,1655-1660`） | `data/episodes.json` 只有 `id/title/pubDate/...`（30 期）；`player.js:1-2` 注入 `DATES/EPISODES` | **无期号来源**（历史已 prune 到 30 期，`config.yaml:7`）。需新增计数（如 brief/metadata 或独立计数器），否则去掉期号 |
| 日期轴前后日/归档（`:1645-1665`） | `episodes.json` → `DATES`（`player.js:1`，`html_gen_view.py:113`） | `simulateDateSwitch` 必须升级为真正的换期加载（playlist/report/radar/sources），见 §5 |
| 标题/副标题/总时长（`:1687-1690`） | 标题 `episodes.json[].title`；时长 `playlist.json` 末段 `start+duration` 或 `audio.duration` | 副标题“万级智能体协同长跑与欧洲算力主权”**无来源**；`09:53` 与真实 618.2s（10:18）不符 |
| 导语（`:1748`） | `data/reports/daily_report_{date}.md` 的“导语”段（运行时 `player.js:203` fetch、`:206` marked） | 需 md→HTML 管道；生产现有面板已有，新页面复用即可 |
| 4 个指标（`:1752-1769`） | **无结构化来源**（日报正文里是自然语言） | 要么删，要么在 report/brief 增加结构化摘要字段（LLM 或代码生成） |
| 章节 4 个时间点（`:1774-1785`） | `playlist.json` 的 `chunk.start` | 原型 `105.0/200.0` 并非 chunk 边界（最近边界 102.463/193.696），章节标签也无来源；建议用 chunk 索引 + 摘要生成 |
| 日报正文（`:1804-1904`） | `data/reports/daily_report_{date}.md`（发布到 `site/reports/`，`podcast_publish_controller.py:160`） | 原型是静态 HTML；生产用 marked 渲染。需要保留 `article-section-block/deep-dive-card` 语义类，md 结构（`##/###/*强调*`）→ 组件映射要写清 |
| “全文 1,850 字 / 约 4 分钟”（`:1799,1808`） | 可由 md 字数计算 | 原型数字与真实 md（1680 字符）不符，需改为构建期计算 |
| 来源 30 条（`:2140`、`:1922-1928`） | `brief_{date}.json` 的 `stories`（`role=="main"` 33 条）或 `episodes.json[].description` 的 `<ol><li>`（123 条，`player.js:551` `parseSourceItems`） | 新页面需要 `role/title/source/link` 结构化源；建议 build 期从 brief 注入（publish 已有 `brief`，`podcast_publish_controller.py:65`），而不是运行时解析 description。分类计数（OpenAI 7/arXiv 12…）需动态计算，原型硬编码 `:1922-1928` |
| 来源筛选/搜索/展开（`:2232-2261`） | 同上一行 | 筛选类别应由数据生成；展开阈值需支持 120+ 条（brief 123 条），原型只到 30 |
| 雷达状态/候选数（`:1955,1974`） | `radar_{date}.json` 的 `meta.candidates=12`、`meta.pick_repo=""`（`player.js:236` fetch，`:693` renderRadar） | 空/命中应由 `projects.length` 驱动，删除 `toggleRadarMockState`（`:2323`）与假项目卡（`:1989-2020`） |
| 雷达命中态字段（`:2002-2017`） | `radar.projects[0]`：`repo/url/description/language/stars/delta_stars/license`（`gh_radar_service.py:956` 附近写盘） | `★5,280(+820/d)` 的“日增速”**未落盘**（服务内 `_vel_per_day` 仅评分用）；“首发版本 v0.8.2”无字段。需加字段或改文案 |
| 候选 12 行明细（`:2142`，`:2023-2038`） | **无**（`meta` 只有 `candidates/excluded_recent` 计数） | 需在 `gh_radar_service` 输出候选明细（repo/stars/日均/语言/排除原因），否则删除该表；这是唯一“全新数据契约” |
| 音色按钮（`:1710-1722`） | `playlist.chunks[].audios` + `VOICES_CONFIG`（`player.js:3,867,894`） | 当前每 host 只 1 个变体（`config.yaml:16-18`）；`warm` 无后端；可见按钮必须按实际 `audios` keys 生成 |
| 逐字稿 55 段（`:2139`） | `site/episodes/{date}/playlist.json`（含 `text/start/audios`）或 `episodes/{date}.txt`（`player.js:266,287`） | 段数随日期变化（2026-09-24 有 259 段），标题“55 句”需动态 |
| 音频 URL（`:2177`） | `episodes.json[].enclosure_url` / 相对 `./episodes/{date}.mp3`；生产逐 chunk `./episodes/{date}/{file}`（`player.js:838`） | 原型的 gh-pages 绝对 URL 与“整集播放”需改成相对路径 + chunk 播放 |
| Apple Podcasts 链接（`:1631`） | `config.yaml` 无该字段 | 占位链接，需产品决定是否保留 |
| RSS 链接（`:1628`） | `site/feed.xml`（`html_gen_view.py` 模板 `./feed.xml`） | 保留 `./feed.xml` |

## 5. 行为清单（含“只在 mock 下成立”标注）

| 行为 | 原型实现 | 真实数据下的问题 / 需改函数 |
| --- | --- | --- |
| 播放/暂停 | `togglePlay/playAudio/pauseAudio`（`:2338-2382`），`setInterval(200ms)` 更新 | **mock-only**：`:2360-2371` 在 `audio` 不可用时用假时钟推进 `currentTime`。真实必须改为 `audio` 事件（`timeupdate/ended/loadedmetadata`）+ chunk 模式（`player.js:829`） |
| ±15 秒 | `skipAudio`（`:2384-2389`）改 `audio.currentTime` | 整集 mp3 成立；chunk 模式需换算到全局时间（生产用 `chunk.start` 偏移） |
| 倍速 | `cycleSpeed`（`:2406-2413`）循环 `SPEEDS`（`:2149`） | 成立，但需持久化/回填到 UI |
| 音量/静音 | `handleVolume/toggleMute`（`:2415-2431`） | 成立；生产默认 0.8（`player.js:12`） |
| 双主持音色 | `switchVoice`（`:2433-2440`）只改 `selectedVoice` 与 UI | **mock-only**：从不换音源；真实需按 `chunk.audios` 在段边界换 `audio.src`（生产 `loadChunk`），当前数据每 host 仅 1 变体 |
| 逐字稿同步高亮 | `updatePlayerUI`（`:2443-2483`）每 tick 线性扫描并 `querySelectorAll` 全部行 | 55 段可接受；259 段（`site/episodes/2026-09-24/playlist.json`）时每 200ms 全量查询 + 滚动会成为性能点 |
| 点句跳转 | 行级 `addEventListener`（`:2287-2290`）→ `seekToTimestamp` + 自动播放 | 成立；chunk 模式下等价 |
| 抽屉开关 | `toggleTranscriptDrawer`（`:2486-2494`）加 `.open`、滚到当前行 | 成立；缺 focus 管理/Esc 关闭 |
| 日期轴切换 | `simulateDateSwitch`（`:2509-2514`）只切 pill + toast | **mock-only**：不换内容；真实要实现换期数据编排（playlist/report/radar/sources 全量刷新 + 播放器重置） |
| 来源筛选/搜索/展开 | `filterSources`（`:2232`）/`handleSourceSearch`（`:2248`）/`toggleExpandSources`（`:2256`） | 逻辑可用，但类别/计数硬编码；搜索只在当前 `SOURCES_DATA`（30 条）上生效 |
| 雷达空/命中切换 | `toggleRadarMockState`（`:2323-2335`）+ 假项目卡 | **mock-only**：评审开关，生产必须删；状态由 `radar.projects.length` 决定 |
| 移动迷你播放器 / bottom sheet | `mobile.html` 独立 `togglePlay/updateUI/renderMChunks`（`:872-987`） | 与桌面播放器状态互不通；统一响应式后应共用一套状态机 |
| Canvas 雷达动画 | `initRadarSweep`（`:2523-2587`）`requestAnimationFrame` 常驻；移动 `initMRadar`（`:1003-1027`）另一份 | 不可持续：无可见性暂停/reduced-motion；两处重复实现 |
| Toast / 移动预览弹层 | `showToast`（`:2498`）、`toggleMobileModal`（`:2517`） | Toast 可留；移动预览是评审工具，删 |

## 6. 移植风险

1. **内联事件 + 全局函数**：桌面 43 个 `onclick`、移动 20 个，函数全在 `<script>` 顶层。移植到生产后与
   `player.js` 现有全局（`toggleAudio/cycleSpeed/seekAudio/switchMode/switchVariant` 等）**命名冲突**风险高，
   且 `main-audio`、`toast`、`volume-slider` 等 id 与生产模板（`html_gen_view.py:232,239,279,497-499`）重复。
   建议新 JS 用 IIFE/模块包裹，或延用生产 id 白名单（brief 已点名 `side-podcast-title`、`console-progress-fill`、
   `host-a-voice-pills`、`variant-container-a`）。
2. **生产播放器深度耦合旧 DOM**：`static/player.js` 1294 行，依赖 52 个唯一旧 id（`player.js:11,28-34,90-93,149-155,250-261,868-871`）
   与 A/B 变体 `theme-variant-b`。重设计若换 DOM，**player.js 必须重写/大改**，不能只换模板；
   现有测试也钉住了旧结构：`tests/test_html_gen_view.py:87-90` 断言 `layout-grid|podcast-workspace`、`player-bar|console-player`、
   `parseTranscript`、`DATES = []`，模板重写时须同步改测试。
3. **播放模型迁移**：整集 mp3 ↔ 逐 chunk 音频（`player.js:829/838`）是行为差异最大的部分；
   双音色/逐句跳播/进度条都建立在 chunk 时间轴上。计划里应把“播放器状态机 + chunk 播放”列为独立任务，
   并明确：是否保留 chunk 文件（gh-pages 上每期数十个文件）还是退化为整集播放。
4. **DOM id 命名**：原型 id 有 `hero-play-btn/dock-play-btn` 双播放按钮；生产是 `console-btn-play` 等。
   建议以原型语义为准重命名，并在迁移清单里逐项列出“旧 id → 新 id”，避免 `player.js` 静默失效（`getElementById` 返回 null 不报错，
   现有代码多处 `if (el)` 保护会掩盖问题）。
5. **绝对路径/CDN**：`index.html:2177-2178` 与 `mobile.html:864` 硬编码 `https://nicekai-jpg.github.io/...`；
   生产必须走 `base_url`/相对路径。原型零 CDN，但生产 `html_gen_view.py:143` 依赖 jsDelivr 的 marked；
   若新页面也运行时渲染 md，要么沿用 CDN，要么构建期预渲染。
6. **动画性能**：两个 rAF 雷达循环（`:2523`、`mobile.html:1003`）无 `visibilitychange` 暂停、无 `prefers-reduced-motion`；
   55 段（极端 259 段）逐字稿每 200ms 全量 DOM 查询（`:2473`）；`scrollIntoView({behavior:'smooth'})` 在句切换时可能抖动。
   建议：单 rAF + IntersectionObserver 暂停、`reduce` 时不启动、逐字稿用 `data-index` 定位或虚拟列表。
7. **a11y 缺口**：两文件 `aria-*` 计数均为 0，抽屉/弹层不是 `dialog`，无焦点陷阱、无 Esc 关闭、
   图标按钮无 `aria-label`（`:2062-2065`、`:2084`）、进度条不可键盘操作（`:2070` 只挂 `onclick`）、
   无 `prefers-reduced-motion`。生产现有页面同样缺（`style.css`/`player.js` grep 结果为 0），
   建议作为验收项而不是“顺手补”。
8. **静态文案与真实数据漂移**：期号 286、`09:53`、`55 句`、`全文 1,850 字`、雷达门槛“300/天”、
   音色“温暖男声”、指标数字（`:1687,1799,1955,1710-1722,1752-1769`）。必须逐项改为数据驱动或明确标注为编辑文案，
   否则上线即与真实数据矛盾。
9. **体积**：生产 `site/index.html` 763KB（30 期 description 约 490KB 内嵌）；原型 110KB。
   新方案应改为“按日期懒加载”（playlist/report/radar 均已是独立文件），避免把 30 期 description 再内嵌一次。
10. **空/错误态**：brief 要求覆盖加载态/错误态/空态；原型只有雷达空态，无 loading/error（日报 markdown 缺失、
    playlist 404、radar JSON 不存在时都没有降级 UI；生产现有实现有 `setLoading/setEmpty`，`player.js:149-155` 可复用）。

## 7. 建议的任务拆分（供计划引用）

| # | 任务 | 依赖 | 写入范围（建议） |
| --- | --- | --- | --- |
| T1 | 数据契约层：`build_index_html` 增参（brief/radar/sources），或新增 `episode_data_{date}.json`；`podcast_publish_controller` 注入 | — | `site_builder/*.py`、`presentation/cli/podcast_publish_controller.py`、tests |
| T2 | 设计系统落地：以 `index.html:8-50` 的 token 替换 `static/style.css` 的默认主题与三套页签主题变量（`style.css:1-22` 及 `23,31,49`） | T1 可并行 | `static/style.css` |
| T3 | 页面骨架模板：header/日期轴/区块顺序/吸底播放器/抽屉（桌面 1440 优先） | T2 | `site_builder/html_gen_view.py` |
| T4 | 播放器重写：单状态机 + chunk 播放 + 双音色（按实际 audios）+ 逐字稿同步 | T1、T3 | `static/player.js`、template |
| T5 | 组件数据化：来源（动态筛选/计数/搜索/展开）、日报（md→HTML/TOC）、雷达（空/命中/候选缺失降级） | T1、T3 | `static/player.js`、template |
| T6 | 响应式统一：3 个断点 + 迷你播放器/bottom sheet（删 `mobile.html`） | T3、T4 | `static/style.css`、template |
| T7 | 数据缺口决策：期号、hero 指标、候选明细、日增速、音色变体（删 UI 还是补数据契约） | 需产品/数据 owner 决策 | 视决策 |
| T8 | 质量门：a11y（focus/Esc/aria/reduced-motion）、性能（259 段）、`tests/test_html_gen_view.py` 迁移、1440/390 截图核对 | T4–T6 | tests、docs |

关键顺序建议：T1 先定数据形状，T4/T5 才能并行；T6 必须在 T3 模板稳定后做，否则断点反复返工；
`player.js` 重写（T4）是风险最高、最值得单独排期与回归的项。

## 8. 盘点口径与可复现证据

- 结构统计：Python 读取两文件，剥离 `<style>`/`<script>` 后统计行数、规则、声明、类名、`@media`、`@keyframes`（见 §3.3 表）。
- 数据核对：对 `PLAYLIST_DATA/SOURCES_DATA/RADAR_DATA` 做 `json.loads` 后与生产文件逐值比较：
  `PLAYLIST_DATA == site/episodes/2026-10-15/playlist.json` → True；`RADAR_DATA == data/gh_radar/radar_2026-10-15.json` → True；
  `SOURCES_DATA == brief.stories(role=main)[:30]` → True（title/link 30/30）。
- 播放时长：playlist 末段 `start=616.46 + duration=1.785 = 618.245s`（10:18），而原型 `duration=592.8`（`:2147`）与文案 `09:53` 均不一致。
- 内容对照：日报正文/导语与 `data/reports/daily_report_2026-10-15.md` 逐段比对，除引号字符外一致。
- 未验证项：`site/episodes/2026-10-15.mp3` 本地不存在（音频只在 gh-pages），音频真实时长未能直接测量；
  `site/radar/` 目录本地缺失，但发布逻辑会同步（`podcast_publish_controller.py:161-162`），需在部署后验证运行时 fetch 路径。
