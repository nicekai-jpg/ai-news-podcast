# 今日AI看点 · Direction A 每日刊 — 开发交付规范

> 本文件是 `directions/A/index.html` 的配套开发文档。所有视觉令牌、组件 DOM/class、状态清单、后端注入字段均以本文件为准。  
> 生成日期：2026-11-03 · 版本：v23（编辑短评卡随主题切换：浅色主题改为纸面浅底深字）

---

## 1. 项目信息

| 项 | 值 |
|---|---|
| 项目 | 今日AI看点（AI Daily Chronicle） |
| 方向 | A · 每日刊 |
| 入口 | `directions/A/index.html` |
| 状态页 | `directions/A/states/`（6 个状态页 + `_shared.css` / `_shared.js`） |
| 资产 | `directions/A/logo.svg`（矢量，64×64 画布，会意「明」：日实月虚） |
| RSS | `./feed.xml`（RSS 2.0，含第 286 期示例条目；链接为相对路径，部署后应替换为绝对地址） |
| 目标视口 | 桌面 1440 · 平板 1024 · 移动 390 |

---

## 2. 色板（最终）

### 2.1 表面

| Token | 值 | 用途 |
|---|---|---|
| `--bg-canvas` | `#fafaf7` | 页面底（近白纸感） |
| `--surface-card` | `#ffffff` | 内容卡 |
| `--surface-card-hover` | `#f8f6f2` | 卡 hover |
| `--surface-inset` | `#f0ede6` | 凹陷面板 |

### 2.2 文字

| Token | 值 | 用途 |
|---|---|---|
| `--text-primary` | `#1a1714` | 主文（深墨） |
| `--text-secondary` | `#4a4540` | 次文 |
| `--text-muted` | `#6b665f`（深色 `#a39a8d`） | 弱文，对比度 ≥ 5.4:1 |
| `--text-faint` | `#767169`（深色 `#978e83`） | 极弱，对比度 ≥ 4.6:1 |

### 2.3 强调与状态

| Token | 值 | 用途 |
|---|---|---|
| `--accent-primary` | `#d94a2b` | 朱砂红 · 主强调（CTA、活跃态、编号） |
| `--accent-primary-solid` | `#b93d1e` | 实心填充（播放按钮 / 选中日期胶囊，白字 5.6:1） |
| `--accent-primary-solid-hover` | `#a53518` | 实心填充 hover |
| `--editor-card-accent` | `#b93d1e`（深色 `#d4a574`） | 编辑短评卡标题色（浅底 / 深底各自适配） |
| `--accent-secondary` | `#1e3a5f` | 深海蓝 · 辅强调（编辑短评、章节） |
| `--accent-teal` | `#2a7a72` | 青绿 · 冷调对比（链接、交互、数据强调） |
| `--color-success` | `#5fa87a` | 成功态（雷达主推、状态页） |

### 2.4 主持人配色

| Token | 值 |
|---|---|
| `--host-a-color` | `#c45a7a`（林夏，深色主题 `#e89bb5`） |
| `--host-b-color` | `#3a6a8a`（陈默，深色主题 `#8fb8d4`） |

### 2.5 边框

| Token | 值 |
|---|---|
| `--border-subtle` | `rgba(242,236,224,0.08)` |
| `--border-medium` | `rgba(242,236,224,0.16)` |
| `--border-strong` | `rgba(242,236,224,0.28)` |

---

## 3. 字号与排版

### 3.1 字号阶梯

| 角色 | 大小 | 行高 | 字重 |
|---|---|---|---|
| Display（板块大标题） | 1.25–1.4rem | 1.3 | 900 |
| H2（板块标题） | 1.25rem | 1.3 | 900 |
| H3（卡片标题） | 1.0–1.15rem | 1.35 | 700 |
| Body（正文） | 0.85–0.92rem | 1.7–1.75 | 400 |
| Small（辅助） | 0.72–0.78rem | 1.6 | 600 |
| Caption（元信息） | 0.66–0.7rem | 1.5 | 400–700 |

### 3.2 CJK 排版规则（强制）

- 标题行高 ≥ 1.3（CJK 填满 em 框，Latin 的 1.0 会导致行碰撞）
- 正文行高 1.7–1.75
- CJK 显示文字字距 0（不应用负字距）
- 拉丁大写必须加 `letter-spacing: 0.06em–0.1em`
- 正文 `max-width: 65ch`

### 3.3 字体族

| 用途 | 字体栈 |
|---|---|
| 标题（衬线） | `"Noto Serif SC", "Source Han Serif SC", "Songti SC", Georgia, serif` |
| 正文（无衬线） | `"Noto Sans SC", -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif` |
| 等宽（元信息/编号） | `"SF Mono", "JetBrains Mono", Consolas, monospace` |

### 3.4 间距阶梯

4px 基准：`4 / 8 / 12 / 16 / 20 / 24 / 32 / 40 / 48`（对应 `--space-1` … `--space-12`）。禁止任意值。

---

## 4. 断点

| 断点 | 宽度 | 布局变化 |
|---|---|---|
| Desktop | ≥ 1181px | 双栏 hero（480 + 1fr）、双栏日报（1fr + 380）、3 列信源 |
| Laptop | 1025–1180px | 双栏 hero（380 + 1fr）、双栏日报、2 列信源 |
| Tablet | 901–1024px | 双栏 hero（340 + 1fr）、日报上下排列（正文在前）、2 列信源 |
| Mobile | 641–900px | 单栏 hero、隐藏导航、2 列指标、单列信源、冷暖底色全出血 |
| Small | 521–640px | 隐藏副标题、顶栏只留图标 |
| XSmall | 391–520px | 日期轴：胶囊独占一行，上一期/下一期在第二行两端 |
| Tiny | ≤ 390px | 播放按钮全宽、播放器单列、抽屉全屏 |

---

## 5. 组件 DOM / Class 清单

### 5.1 页头

| Class | 元素 | 说明 |
|---|---|---|
| `.site-header` | `<header>` | 粘性顶栏 |
| `.header-inner` | `<div>` | 内层 flex |
| `.brand-group` | `<div>` | Logo + 字标 |
| `.brand-logo-img` | `<img>` | 34×34 logo（`logo.svg`，矢量，会意「明」标志） |
| `.brand-title` | `<div>` | 字标「今日AI看点」（6 字，1.12rem / 0.02em） |
| `.brand-badge` | `<span>` | 「DAILY」徽章（品牌名已含 AI，徽章不再重复） |
| `.brand-subtitle` | `<div>` | `{{issue_tagline}}` |
| `.header-nav` | `<nav>` | 章节锚点导航（4 项） |
| `.nav-anchor-link` | `<a>` | 单个锚点 |
| `.nav-icon` | `<svg>` | 14×14 描边图标 |
| `.header-actions` | `<div>` | RSS 按钮容器 |
| `.btn-header-pill` | `<a>` | RSS 按钮 |
| `.btn-header-rss` | `<a>` | RSS hover 变体 |

### 5.2 日期轴

| Class | 元素 | 说明 |
|---|---|---|
| `.shared-date-axis` | `<div>` | 日期轴容器 |
| `.date-nav-controls` | `<div>` | 上/下期按钮组 |
| `.date-step-btn` | `<button>` | 上一期 / 下一期 |
| `.date-pills-carousel` | `<div>` | 期号胶囊滚动区 |
| `.date-pill` | `<button>` | 单个期号 |
| `.date-pill.active` | `<button>` | 当前期 |

期号胶囊由期号列表 `ISSUE_ORDER` 生成（标签取 `issue_date` 的 `MM-DD`），点击调用 `switchIssue()`。

### 5.3 板块

| Class | 元素 | 说明 |
|---|---|---|
| `.content-section` | `<section>` | 板块容器（纵向间距 40px；scroll-margin-top: 72px） |
| `.content-section.tint-cool` | `<section>` | 信源板块冷调底色（桌面左右外扩 20px，≤900 / ≤390 收为 16px / 12px，纵向内边距 24px） |
| `.content-section.tint-warm` | `<section>` | 雷达板块暖调底色（同上） |
| `.section-header` | `<div>` | 板块表头 |
| `.section-title-group` | `<div>` | 编号 + 标题组 |
| `.section-number` | `<span>` | 编号（01/02/03/04） |
| `.section-title` | `<h2>` | 板块标题 |
| `.section-desc` | `<p>` | 板块描述 |
| `.section-meta` | `<span>` | 元信息（计数/字数） |

### 5.4 Hero 区

| Class | 元素 | 说明 |
|---|---|---|
| `.hero-grid` | `<div>` | 480px + 1fr 网格（≤1180 为 380px + 1fr，≤1024 为 340px + 1fr，均保持两栏；≤900 单栏） |
| `.podcast-card` | `<article>` | 播客主推卡 |
| `.podcast-cover-row` | `<div>` | 封面 + 标题行 |
| `.podcast-disc` | `<div>` | 唱片封面（含波形） |
| `.disc-bars` / `.disc-bar` | `<div>` | 波形动画 |
| `.podcast-title-area` | `<div>` | 标题区 |
| `.podcast-heading` | `<h3>` | 播客标题 |
| `.podcast-subhead` | `<p>` | 副标题 |
| `.host-roster` | `<div>` | 本期主持（静态展示） |
| `.host-roster-label` | `<span>` | 「本期主持」标签 |
| `.host-roster-list` | `<div>` | 主持人列表（`#host-roster-list`，由 `host_a_name` / `host_b_name` 渲染） |
| `.host-identity` | `<span>` | 头像 + 名字 |
| `.host-avatar` | `<span>` | 圆形头像 |
| `.podcast-cta-row` | `<div>` | CTA 行 |
| `.btn-play-primary` | `<button>` | 播放按钮 |
| `.btn-transcript-cta` | `<button>` | 逐字稿按钮 |
| `.editorial-card` | `<article>` | 编辑导语卡 |
| `.editorial-stamp` | `<div>` | 导语头部 |
| `.stamp-label` | `<div>` | 「编辑导语」标签 |
| `.stamp-date` | `<span>` | 期号日期 |
| `.lead-quote` | `<p>` | 导语正文 |
| `.metrics-strip` | `<div>` | 指标条（4 列） |
| `.metric-item` | `<div>` | 单个指标 |
| `.metric-value` | `<div>` | 数值 |
| `.metric-unit` | `<span>` | 单位 |
| `.metric-label` | `<div>` | 指标名 |

### 5.5 日报

| Class | 元素 | 说明 |
|---|---|---|
| `.broadside-grid` | `<div>` | 1fr + 380px 网格 |
| `.broadside-main` | `<div>` | 主列 |
| `.broadside-header` | `<div>` | 日报头 |
| `.broadside-meta` | `<div>` | 元信息行 |
| `.broadside-pill` | `<span>` | 「深度研判」药丸 |
| `.broadside-toc` | `<div>` | 日报章节目录，标签取自 `chapter_1~4_tag` |
| `.broadside-toc-link` | `<a>` | 单个跳转 |
| `.broadside-title` | `<h3>` | 日报标题 |
| `.deep-dive` | `<div>` | 深度卡片 |
| `.deep-dive-header` | `<div>` | 卡头 |
| `.deep-dive-tag` | `<span>` | 标签 |
| `.deep-dive-topic` | `<span>` | 话题 |
| `.deep-dive-title` | `<h4>` | 卡标题 |
| `.deep-dive-point` | `<div>` | 要点 |
| `.point-label` | `<div>` | 要点标签 |
| `.point-label .dot` | `<span>` | 圆点 |
| `.point-text` | `<p>` | 要点正文 |
| `.deep-dive-point.highlight` | `<div>` | 高亮要点 |
| `.broadside-sidebar` | `<aside>` | 侧栏 |
| `.editor-quote-card` | `<div>` | 编辑短评卡（浅色主题：纸面浅底 + 深墨字；深色主题：暖色渐变深卡 + 反白） |
| `.editor-quote-header` | `<div>` | 短评头 |
| `.editor-quote-text` | `<p>` | 短评正文 |
| `.editor-quote-author` | `<div>` | 作者 |

### 5.6 信源

| Class | 元素 | 说明 |
|---|---|---|
| `.sources-grid` | `<div>` | 信源网格 |
| `.sources-filter` | `<div>` | 来源分类筛选条 |
| `.source-filter-chip` | `<button>` | 分类胶囊（全部 / OpenAI / Google / 行业媒体 …） |
| `.chip-count` | `<span>` | 胶囊内的数量 |
| `.source-card` | `<a>` / `<article>` | 单个信源；有外链时渲染为 `<a target="_blank" rel="noopener">`，无链接时退回 `<article>` |
| `.source-card-header` | `<div>` | 头部 |
| `.source-badge` | `<span>` | 来源徽章（openai/anthropic/google/meta/arxiv/nvidia/industry） |
| `.source-meta` | `<span>` | 元信息 |
| `.source-title` | `<div>` | 标题 |
| `.source-summary` | `<div>` | 摘要（2 行 clamp） |
| `.sources-expand-bar` | `<div>` | 展开按钮容器 |
| `.btn-expand-sources` | `<button>` | 展开/收起 |

### 5.7 雷达

| Class | 元素 | 说明 |
|---|---|---|
| `.radar-card` | `<div>` | 雷达卡 |
| `.radar-empty-state` | `<div>` | 空态容器（`#radar-empty`，有主推时加 `hidden`） |
| `.radar-pick-state` | `<div>` | 主推态容器（`#radar-pick`，无主推时加 `hidden`） |
| `.radar-pick-head` | `<div>` | 仓库名 + Star 增速同行（状态只在板块头出现，卡内不重复） |
| `.radar-pick-name` | `<h3>` | 仓库名（`owner/repo`） |
| `.radar-pick-meta` | `<div>` | 语言 · Star 数 · 许可证 |
| `.radar-pick-desc` | `<p>` | 项目说明 |
| `.radar-pick-action` | `<div>` | 外链区 |
| `.radar-visual` | `<div>` | 雷达可视化 |
| `.radar-canvas` | `<canvas>` | 扫描动画 |
| `.radar-headline` | `<h3>` | 空态标题 |
| `.radar-paragraph` | `<p>` | 空态段落 |
| `.radar-meta-line` | `<div>` | 元信息行 |
| `.radar-meta-line .sep` | `<span>` | 分隔符 |

### 5.8 底部播放器

| Class | 元素 | 说明 |
|---|---|---|
| `.docked-player` | `<div>` | 固定底部播放器 |
| `.player-inner` | `<div>` | 内层 grid |
| `.player-info` | `<div>` | 信息区 |
| `.player-disc` | `<div>` | 小唱片 |
| `.player-text` | `<div>` | 文字区 |
| `.player-title` | `<div>` | 标题 |
| `.player-subtitle` | `<div>` | 副标题 |
| `.player-time` | `<div>` | 时间码（当前 / 总时长） |
| `#player-current` | `<span>` | 当前时间，`mm:ss` |
| `#player-total` | `<span>` | 总时长，取自 `podcast_duration` |
| `.player-controls` | `<div>` | 控制按钮组 |
| `.player-btn` | `<button>` | 按钮 |
| `.player-btn.play` | `<button>` | 播放按钮 |
| `.player-transcript-btn` | `<button>` | 逐字稿按钮 |

### 5.9 逐字稿抽屉

| Class | 元素 | 说明 |
|---|---|---|
| `.transcript-drawer` | `<aside>` | 抽屉容器 |
| `.transcript-drawer.open` | `<aside>` | 打开态 |
| `.drawer-header` | `<div>` | 抽屉头 |
| `.drawer-title` | `<h3>` | 标题 |
| `.drawer-close` | `<button>` | 关闭 |
| `.drawer-body` | `<div>` | 内容区 |
| `.transcript-line` | `<div>` | 单行 |
| `.transcript-speaker` | `<span>` | 说话人 |
| `.transcript-speaker.host-a/b` | `<span>` | 主持人颜色 |
| `.transcript-text` | `<div>` | 文本 |
| `.transcript-text .timestamp` | `<span>` | 时间戳 |
| `.drawer-overlay` | `<div>` | 遮罩 |
| `.drawer-overlay.visible` | `<div>` | 可见态 |

---

## 6. 状态清单

| 文件 | 场景 | 说明 |
|---|---|---|
| `states/_shared.css` | 共享令牌 | 色板/字号/间距与主页面 `:root` 一致 |
| `states/player-error.html` | 播放器错误 | 音频加载失败 / 网络错误 |
| `states/empty-report.html` | 日报为空 | 无日报内容 |
| `states/empty-sources.html` | 信源为空 | 无信源数据 |
| `states/empty-radar.html` | 雷达为空 | 默认空态（主页面同款） |
| `states/radar-pick.html` | 雷达有主推 | 成功态；与主页面共用 `.radar-pick-*` DOM 与同一份字段 |
| `states/transcript-open.html` | 逐字稿打开 | 桌面抽屉展开态 |

### 状态页约定

- 每个状态页是**独立可打开的 HTML 文件**，共享 `_shared.css`
- 状态页使用与主页面相同的 `:root` 令牌；`_shared.css` 同时提供 `[data-theme="dark"]`，`_shared.js` 沿用 `localStorage` 主题并提供右上角切换按钮，保证深浅两套都能核对
- 状态页不包含导航/页头/页脚，仅展示该状态的内容区域
- 状态页只做「视觉 + 文案」示意，不放置任何按钮或可点控件；交互行为以主页面为准
- 状态页的 `{{...}}` 占位符与主页面字段名一致；`states/_shared.js` 只保留状态页用到的字段，不复制主页面整份 DATA

---

## 7. 后端注入字段清单

### 7.1 期号与元信息

| 字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `issue_number` | string | `286` | 期号 |
| `issue_date` | string | `2026-10-15` | 日期 |
| `issue_tagline` | string | `AI 前沿 · 每日一刊` | 副标题 |
| `demo_note` | string | `演示样例` | 仅演示期填写；01 板块头显示标记，为空时整块隐藏 |

### 7.2 播客

| 字段 | 类型 | 示例 |
|---|---|---|
| `podcast_duration` | string | `36:42` |
| `podcast_audio_url` | string | `""` | 音频地址；非空时播放器走真实音频，为空时使用模拟时钟 |
| `podcast_chapter_count` | string | `4` |
| `podcast_title` | string | `Third Moment 与 Agent Kernel` | 播客卡与底部播放器共用这一份标题 |
| `podcast_subtitle` | string | `双主持对谈 · 今日焦点解读` |
| `host_a_name` | string | `林夏` | 主持人 A；roster 与逐字稿头像共用 |
| `host_b_name` | string | `陈默` | 主持人 B |
| `lead_quote` | string | `今日全球科技圈呈现出…` |

### 7.3 指标（4 组，每组 value/unit/label）

| 字段 | 类型 | 示例 |
|---|---|---|
| `metric_sources_value` | string | `10,000` |
| `metric_sources_unit` | string | `个` |
| `metric_sources_label` | string | `扫描信源` |
| `metric_hours_value` | string | `36+` |
| `metric_hours_unit` | string | `小时` |
| `metric_hours_label` | string | `研究时长` |
| `metric_speedup_value` | string | `76` |
| `metric_speedup_unit` | string | `倍` |
| `metric_speedup_label` | string | `效率提升` |
| `metric_items_value` | string | `30` |
| `metric_items_unit` | string | `条` |
| `metric_items_label` | string | `精选条目` |

**显示规则**：有数据时显示 value + unit + label；无数据时整块隐藏（`display: none`），不显示空占位。

**口径约束**：指标条是「这一期怎么产出来的」生产口径，不是新闻数字的复述。`metric_*` 的数值不得与 `lead_quote` 里的新闻数字混用同一含义（例如「扫描信源 10,000」与「上万智能体」必须能区分）。

### 7.4 日报

| 字段 | 类型 | 示例 |
|---|---|---|
| `report_word_count` | string | `1850` |
| `report_read_minutes` | string | `4` |
| `report_pill` | string | `深度研判` |
| `report_title` | string | `Third Moment：长任务调度的第三时刻` |
| `chapter_1_tag` … `chapter_4_tag` | string | `Third Moment` |
| `chapter_1_topic` … `chapter_4_topic` | string | `长任务调度` |
| `chapter_1_title` … `chapter_4_title` | string | 章节标题 |
| `chapter_N_point_M_label` | string | 要点标签（M = 1–3） |
| `chapter_N_point_M_text` | string | 要点正文 |

> 每章支持 1–3 个要点；第 3 点带 `data-requires`，字段为空时整块隐藏，不会残留占位符。每章第 2 点统一加 `.highlight` 强调。
| `editor_quote` | string | 编辑短评 |
| `editor_name` | string | `林夏` |
| `editor_role` | string | `主编` |

### 7.5 信源

| 字段 | 类型 | 示例 |
|---|---|---|
| `sources_list` | JSON string | 信源数组（见下方 schema）；条数由数组长度自动统计 |
**sources_list schema**：
```json
[
  {
    "badge": "openai",
    "badge_label": "OpenAI",
    "meta": "2026-10-15",
    "title": "标题",
    "summary": "摘要（可选；与标题相同时不渲染，避免重复一行）",
    "url": "https://…（可选，缺省时按 badge 回落到来源站点首页）"
  }
]
```
badge 可选值：`openai` / `anthropic` / `google` / `meta` / `arxiv` / `nvidia` / `industry`。

> 分类计数由前端按 `badge` 实时统计，不再接受后端注入的 `filter_*_count` 字段（已移除，避免与列表口径不一致）。

**外链规则**：优先使用条目自带的 `url`；没有时按 `badge_label`（来源名）回落到该来源站点首页，其次按 `badge`；不会拼接或编造文章地址。两者都没有时渲染为不可点击的 `<article>`。

### 7.6 雷达

| 字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `candidate_count` | string | `12` | 候选仓库数 |
| `excluded_recent` | string | `12` | 近期已收录排除数 |
| `radar_status_label` | string | `无主推项目` | 状态标签（只出现在板块头，卡内不再重复） |
| `radar_headline` | string | `今日候选池经多轮筛查，未检出达标项目` | 空态标题 |
| `radar_paragraph` | string | `...` | 空态段落；**不得写死候选数字**，数量只由 `candidate_count` 渲染 |
| `radar_reason` | string | `宁缺毋滥、轮空主推` | 原因 |
| `radar_pick_repo` | string | `microsoft/autogen` | 有主推时的仓库名；为空则显示轮空态 |
| `radar_pick_lang` | string | `Python` | 项目语言 |
| `radar_pick_stars` | string | `42.3k` | Star 数 |
| `radar_pick_growth` | string | `312` | 近 7 日 Star 日均增速 |
| `radar_pick_license` | string | `MIT` | 许可证 |
| `radar_pick_desc` | string | `...` | 项目说明 |
| `radar_pick_url` | string | `https://github.com/...` | 仓库地址 |

**显示规则**：`radar_pick_repo` 非空时显示主推卡、隐藏轮空态；为空时只显示轮空态（候选数 / 排除数 / 原因）。`radar_status_label` 随状态取值「有主推项目」或「无主推项目」。

### 7.7 播放器与逐字稿

| 字段 | 类型 | 示例 |
|---|---|---|
| `player_subtitle` | string | `今日AI看点 · 第 286 期` |
| `transcript_list` | JSON string | 逐字稿数组（见下方 schema）；发言人姓名取 `host_a_name` / `host_b_name` |

**播放器规则**：`podcast_audio_url` 非空时创建 `<audio>`，播放/暂停、时间码、前后 15 秒均绑定真实音频；音频 `loadedmetadata` 后用真实总时长覆盖显示。为空时维持模拟时钟（每 1 秒推进）。切期会释放上一期音频并清零时间。播放器标题复用 `podcast_title`，不单独提供字段。

**transcript_list schema**：
```json
[
  { "speaker": "A", "time": "00:12", "text": "文本内容" },
  { "speaker": "B", "time": "00:45", "text": "文本内容" }
]
```
speaker 可选值：`A` / `B`

### 7.8 日期轴

期号胶囊不再单独提供数组：前端按期号列表生成，标签取该期 `issue_date` 的 `MM-DD`。

| 数据 | 来源 | 说明 |
|---|---|---|
| 待展示的期号列表 | `ISSUE_ORDER` | 顺序即展示顺序，当前期由 `DATA.issue_number` 匹配 |
| 每期内容 | `ISSUE_PATCHES[id]` | 与 `ISSUE_286` 合并成该期完整数据 |

后端接入时，把「期号列表 + 每期数据」换成接口返回即可，胶囊会自动重建。

---

## 8. 交互说明

| 交互 | 触发 | 行为 |
|---|---|---|
| 播放/暂停 | 点击「播放本期」或底部播放键 | 两处按钮图标同步切换，唱片波形开始/停止；播放时时间码每秒递增 |
| 前进/后退 15 秒 | 点击底部 ⏪ / ⏩ | 时间码 ±15 秒，边界钳制在 0 与总时长之间 |
| 逐字稿打开 | 点击「逐字稿」 | 右侧抽屉滑入 + 遮罩 |
| 逐字稿关闭 | 点击关闭 / 遮罩 | 抽屉滑出 |
| 信源展开 | 点击「展开全部」 | 显示全部信源，按钮变为「收起」 |
| 信源筛选 | 点击分类胶囊 | 只显示该来源的信源，按钮与计数随之更新 |
| 信源外链 | 点击来源卡 | 新标签页打开该来源（`url` 或来源站点首页） |
| 期号切换 | 点击期号胶囊 / 上一期 / 下一期 | 切换期号并重新渲染全页，播放进度归零。上一期 = 更早一期，下一期 = 更新一期；到达首/末一期时对应按钮禁用 |
| 主题切换 | 点击顶栏主题按钮 | 切换深浅主题并写入 localStorage；图标随目标主题切换（深色显示太阳、浅色显示月亮） |
| 逐字稿关闭 | Esc 键 | 与点击关闭等效，焦点回到打开前的元素 |
| 雷达动画 | 自动 | canvas 扫描动画循环；主推态或 `prefers-reduced-motion` 下停止/只绘制单帧 |

---

## 9. 可访问性

- 所有交互元素有 `aria-label`
- 颜色对比度 ≥ 4.5:1（正文）/ 3:1（大文字）
- 键盘可操作：Tab 导航、Enter/Space 激活
- `prefers-reduced-motion`：全局动画与过渡降级，雷达只绘制静态单帧
- 逐字稿抽屉使用 `role="dialog"` / `aria-modal`，打开时聚焦关闭按钮，关闭后焦点归位
- 装饰性 radar canvas 标注 `aria-hidden="true"`；logo 为装饰图（`alt=""`），品牌名由相邻文字承担
- 不依赖颜色传达信息（徽章附带文字）

---

## 10. 已知限制

1. **占位符为字符串形式**：`{{...}}` 写在 JS 字面量中需带引号，保证文件可直接打开。后端注入时替换整个数据对象。
2. **音频需后端提供**：当前示例数据 `podcast_audio_url` 为空，播放器走模拟时钟；填入音频地址后即切换为真实播放。
3. **多期与雷达为演示样例**：第 285–282 期为演示用示例内容（含标题、正文、信源、逐字稿），不是真实新闻；第 286 期为原有样本。第 286 期雷达主推卡里的仓库名（microsoft/autogen）、Star 数与增速同为演示数据，不是实时抓取结果。真实环境由接口返回。
4. **来源外链为站点首页**：示例信源没有逐篇 URL，点击按来源名回落到该来源站点首页；不编造文章地址。
5. **渲染验证范围**：已在 Chrome 中核对浅色/深色、1440/1024/640 以及 390（iframe 固定视口）；390 实测 `clientWidth=390 / scrollWidth=390`，无横向溢出。
6. **字体依赖网络**：Noto Serif SC / Noto Sans SC 从 Google Fonts 加载，离线时回退到系统字体。
7. **雷达 canvas**：扫描动画为装饰性，不影响功能。

---

## 11. 变更记录

| 版本 | 日期 | 说明 |
|---|---|---|
| v1 | 2026-10-26 | 初版（深蓝黑科技风） |
| v2 | 2026-10-27 | 重做（重新来） |
| v3 | 2026-11-03 | P0 可开发交付修订（模板化、状态文件） |
| v4 | 2026-11-03 | 深色编辑报刊重做（最终版） |
| v5 | 2026-11-03 | 板块间距统一 40px；移除编辑导语卡内重复的「章节跳转」及其 `playlist` 字段 |
| v6 | 2026-11-03 | 移除「全部归档」空链接；首屏与底部播放按钮状态同步 |
| v7 | 2026-11-03 | 补齐期号切换（5 期演示数据）、播放器模拟时间码与前后 15 秒、来源卡外链 |
| v8 | 2026-11-03 | 日报目录改为随章节标签生成；信源板块新增来源分类统计与筛选 |
| v9 | 2026-11-03 | 雷达改为数据驱动：`radar_pick_repo` 非空时渲染主推卡，否则显示轮空态 |
| v10 | 2026-11-04 | 更换品牌标志为矢量 `logo.svg`（先锋 A 字形） |
| v11 | 2026-11-04 | 标志改为「信号柱」造型（音频柱状 + 品牌三色）；新增 `logo-options.html` 候选对比页 |
| v12 | 2026-11-04 | 标志改为「旌旗」造型（旗杆 + 三角旗 + 杆顶信标）；对比页改为旌旗 / 笔锋 / 套印 A 三个高辨识度方向，并加入顶栏锁版预览 |
| v13 | 2026-11-04 | 标志改为象形方向：主用「目」（甲骨文眼形 + 朱砂瞳孔），对比页改为目 / 山 / 井三个象形字方案 |
| v14 | 2026-11-04 | 标志定稿为会意「明」：朱砂实心日压在象牙空心月环之上；对比页改为单一方案说明页（含几何参数） |
| v15 | 2026-11-04 | 品牌更名为「观象」（观象授时）；字标字号 1.05rem → 1.28rem、字距 -0.01em → 0.05em（两字字标需要更大字号与正字距）；徽标 `DAILY` → `AI DAILY` |
| v16 | 2026-11-04 | 品牌更名为「明说」（原名「观象」过于书面、不易理解）；取自「明说」= 把话说明白，与标志「明」同源 |
| v17 | 2026-11-04 | 品牌更名为「剖面」（「明说」只说了态度、没有分量）；取自剖面图——把一件事切开看内部结构，直接对应「深度」 |
| v18 | 2026-11-04 | 品牌定名为「今日AI看点」（简单易懂，直接说明产品内容）；字标 2 字 → 6 字，字号 1.28rem → 1.12rem、字距 0.05em → 0.02em，徽章 `AI DAILY` → `DAILY` |
| v19 | 2026-11-04 | 修复浅色主题变量自引用（编辑短评卡白底白字、唱片无渐变、冷暖底色丢失）；上一期/下一期方向纠正并加边界禁用；小字对比度提到 4.5:1 以上；1024 下 01 保持两栏、02 改为正文在前；信源分类修正 NVIDIA/arXiv 归属；章节第 3 点可缺省；补充 reduced-motion、抽屉 dialog 语义与焦点管理 |
| v20 | 2026-11-04 | 完成 390px 实测并修正日期轴换行；浅色短评卡补抬升阴影；演示期（285–282）增加「演示样例」标记；新增 `podcast_audio_url` 支持真实音频；新增 `feed.xml`（RSS 2.0，示例条目）；更新断点表与已知限制 |
| v21 | 2026-11-04 | 设计定稿收尾：移除主持人声线切换（无真实音轨的无效控件）与「发言中」假状态，改为静态「本期主持」roster；状态页补齐深色令牌与主题切换；删除未引用的旧 `logo.png` |
| v22 | 2026-11-04 | 冗余清理：信源卡标题与摘要相同时不再重复渲染；01 板块头去掉重复时长、只留章节数；「展开全部」不再重复总条数；雷达删除卡内状态徽章，状态只在板块头出现；删除派生字段 `report_edition` / `player_title` / `transcript_episode` / `sources_total` / `podcast_tag` / `radar_status_badge`；主持人姓名抽为 `host_a_name` / `host_b_name`，roster 与逐字稿共用并修正示例稿姓名；雷达段落不再写死候选数；`metric_items_label` 改为「精选条目」并增加指标口径约束；状态页删除全部无处理函数的按钮（`state-actions` / `state-btn`）；`states/radar-pick.html` 改用主页面同款 `.radar-pick-*` DOM，移除自造字段 `radar_pick_stars_delta` / `radar_pick_version` 与 `href="#"` 假链接；修正文档字段名 `sources_list` / `transcript_list`；删除 7 个未使用令牌与移动端残留样式 |
| v23 | 2026-11-04 | 编辑短评卡随主题切换：浅色主题由「深海军蓝底 + 反白」改为纸面浅底 + 深墨字（标题 #b93d1e，作者 --text-muted），深色主题保持暖色深卡反白；补充浅色专用边框、阴影与作者分隔线配色 |
