# 站点改版对接实施计划（Open Design 方向 A → 生产站点）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development（推荐）或 executing-plans 逐任务执行。步骤使用 `- [ ]` 复选框跟踪。

**Goal:** 把已完成的方向 A 原型（`design/2026-10-redesign/directions/A/`，单页长滚动「每日刊」）移植进生产站点生成器 `presentation/site_builder/`：一个日期、四个板块（播客 / 科技日报 / 来源速览 / 项目雷达）、一个常驻播放器；纯 HTML/CSS/vanilla JS，保持 Apple Podcasts feed 与音频 URL 不变。

**Architecture:** 服务端生成「骨架 + 内联数据块 + 内联静态资源」，每期内容由 publish 阶段从 brief / report md / radar json / playlist 聚合为一个 `digest_{date}.json`（新增 `digest_view` 模块，纯函数 + 失败隔离）；前端按日期懒加载 digest + playlist，用单一响应式布局覆盖桌面 1440 与移动 390。删除现有「四 Tab + A/B 双布局」两套平行骨架、marked.js CDN 与 BGM 死代码。

**Tech Stack:** Python 3.11 / uv / pytest / ruff / import-linter；`presentation/site_builder`（纯字符串/JSON 生成）+ `static/{style.css,app.js,player.js}`（无构建步骤、无外部依赖）；GitHub Actions + gh-pages（peaceiris，`keep_files: true`）。

**范围边界（不做）：** 不改 `rss_gen_view.py`、不改 `show_notes_view.py`、不改音频 URL/切片命名、不改 gh-pages 部署动作、不改 `prune_episodes` 的日期清扫语义；不引入 Tailwind/React/Vue/构建步骤/新 Python 依赖（除既有 `jieba` 以外不需要新依赖）。

**证据文档（本计划的事实来源，执行前必读）：**
- `docs/design-integration/01-prototype-inventory.md` — 原型逐组件行号、mock 真伪、CSS 度量
- `docs/design-integration/02-current-site-contract.md` — 现状注入 / fetch / DOM / 部署契约与死代码清单
- `docs/design-integration/03-data-contract-gaps.md` — 数据供给矩阵、digest schema、推导规则
- `docs/design-integration/04-test-deploy-risk.md` — 测试判定、CI/回滚/验收阶梯

---

## 0. 决策记录（执行时不要重新讨论，除非用户推翻）

| # | 决策 | 理由（证据） | 被否方案 |
|---|---|---|---|
| D1 | 单页长滚动 + 一个统一日期轴；删除四 Tab、A/B 变体、`theme-*`、`?variant=` | brief 明说「一个日期、四个板块、一个常驻播放器」「A/B 开关不应出现在产品」（`design/2026-10-redesign/brief.md:20-36`）；现状双骨架有 3 处已知状态 bug（`02 §4.7`） | 保留四 Tab 仅换皮（播客孤岛问题不解决） |
| D2 | 数据通道 = 内联 `site-data` JSON 块（**唯一**注入点）+ 运行时 `fetch('./data/digest_{date}.json')` + `./episodes/{date}/playlist.json` | 现有字符串占位符注入不可解析、测试脆弱（`02 §2.1`）；改为 JSON script 块后 Python 可稳定断言（`04 §3.2`） | 继续用 `{dates_json}` 三处 `str.replace`（无法解析、易碎） |
| D3 | 新增 `site/data/digest_{date}.json`，**随 main 跟踪**（CI `git add -f site/data/`） | 月度 prune 是 orphan 重建，只从 main 的 `site/` 工作区拷贝（`scripts/prune_gh_pages.py:139-153`）；不跟踪的数据在 prune 后消失（现成同款 bug：`site/radar` tracked=0，`04 §4.2`）；跟踪后 prune 无需改动 | 不落 main、每天全量重建 + 改 prune 备份（多改一个删站脚本，风险更高）；`data/digests/` + `site/data/` 双份真源（冗余 240KB/天） |
| D4 | **不显示期号**（已拍板）：设计稿「第 286 期」不真实，v1 只显示日期；digest `issue.number=null / source="unavailable"`，UI 不渲染期号徽章；不持久化 `issue_no`。未来若 Podcasts Connect 上有真实累计数，再回头补 | 用户 2026-11-04 确认 286 不真实；`03 §3.1` 已证明无唯一解 | 硬编码 286（假数据）；运行时公式拟合（断更即漂移） |
| D5 | 日报在 publish 阶段解析成结构化 JSON，前端不再渲染 Markdown；移除 marked.js CDN | 现依赖 jsDelivr，断网/内网时日报空白（`02 §8 R3`）；设计稿日报是结构化卡片（`01 §4`）；当前 30 期窗口全部标准新格式（`03 §3.6`） | 继续 CDN marked（外部依赖 + 客户端二次解析 + 无法空态断言） |
| D6 | 来源数据从 `brief.stories` 结构化生成（不再解析 `episodes.json.desc` HTML）；`role_filter="all"`，默认折叠 12 条、按组筛选、可展开全部 | desc 同时承担 RSS 摘要 + 来源数据 + 兜底，双实现且 publish 不转义（`02 §2.3, R14`）；brief 有 role/source_name/link 结构化字段（`03 §3.2,126-133`）；生产有 123 条来源，设计只截了 main 30（`01 §4`） | 保留 desc 解析（脆弱、需改四链路） |
| D7 | 指标卡换成真实管线指标：今日入库 / 核心研判 / 双人对谈段数 / 覆盖媒体数 | 设计稿「10,000 个 / 36+ 小时 / 76 倍」是 main 标题里的编辑数字，无结构字段且正则抽取会误伤（`03 §3.7`）；真实值可从 brief/playlist 算 | 新增一次 LLM 结构化调用产出编辑数字（v2 再议，见 Task 11 可选项） |
| D8 | 播客副标题用 `brief.thesis`（标 `fallback`），为空则隐藏 | 无副标题数据源，thesis 是最接近的编辑性一句话（`03 §2`） | 硬编码/留空（留空视觉塌陷） |
| D9 | **章节条 v1 用「转场词锚点」启发式**（用户 2026-11-04 确认）：`开场 / 说到这个|接下来 / 第[一二三四五]条 / 最后一条`，标题取该段前 12 字；不足 2 个锚点则整行隐藏 | 脚本无章节标记（`03 §3.9`）；10-15 实测可稳定命中 chunk 21/31/39/47（`start=228.346/347.985/420.297/519.828`） | 用日报标题/故事标题做编辑标题（与脚本顺序不一致，实测误配到收尾段）；直接删章节（brief 硬约束要求章节跳转，逐字稿点句跳转可兜底但体验退化） |
| D10 | **雷达 v1 只做真实主推/备选/空态 + `meta` 计数 + config 规则文案**（用户 2026-11-04 确认）：**候选明细表删除**，候选审计落盘放 v1.5 | `meta` 只有计数，无 12 个候选明细（`gh_radar_service.py:949-972`）；重建方案只对 v4 前历史成立、新日期失真（`03 §5.1`） | 用 mock 候选表（违反 brief「不要假数据」）；v1 就改雷达管线（触及可失败环节，风险与本计划目标不匹配） |
| D11 | 播放器保留「逐 chunk 音频 + 虚拟全局时间轴 + 按 host 音色变体切源」引擎，重写为单状态机；不改音频文件命名 | 双音色/逐句跳播/进度条都建立在 chunk 时间轴（`01 §6`）；改名会同时破 feed/CI 校验/prune/播放器（`02 §8 R8`） | 退回整集 MP3（丢失双音色与逐句同步，等于功能回退） |
| D12 | **移动端用单一响应式构建**（用户 2026-11-04 确认无独立移动端需求；3 个断点 + 同节点降级为迷你播放器/bottom sheet），删除 `mobile.html` 独立稿 | 移动稿 76.6% CSS 声明与桌面逐字相同、token 20/22 重合、无任何 `@media`，是 fork 不是分支（`01 §3.3`）；两套播放状态机互不通 | 维护两份 HTML/JS（双倍维护 + 18.2KB 重复数据 + 状态不同步） |
| D13 | 静态资源拆 3 个文件：`static/style.css`、`static/app.js`（页面状态/渲染）、`static/player.js`（音频引擎）；`html_gen_view` 按序内联 | 现 `player.js` 1294 行混合「引擎 + 四 Tab + 日报/来源/雷达 + A/B」，重写必须解耦（`01 §6`）；仍保持单文件产物、无构建步骤 | 单文件 JS（1300+ 行继续膨胀）；外链 asset（brief 要求纯 HTML/无构建，且 gh-pages 路径风险） |
| D14 | 清删死代码：`bgmAudio`（未声明）、`showToast` 缺失声明、`setPlaybackMode`、`format_friendly_date`（生产零引用）、`generate_show_notes`（Markdown 版零引用）、旧 `-b` DOM、死 CSS | `02 §4.7-§4.8` 已列全清单 | 保留（新代码继续被旧 id 污染） |
| D15 | **Apple Podcasts 入口移出 v1**（已拍板）：头部只保留 RSS 入口，不新增 `apple_podcasts_url` 配置；feed.xml 本身不动（Apple 兼容性不受影响）。等真正在 Podcasts Connect 上架后再加 | 用户 2026-11-04：「从来没搞过，放到最后」 | v1 放一个空/无效入口（误导用户） |

---

## 1. 新数据契约（落地前先冻结）

### 1.1 `site/data/digest_{date}.json`（schema_version=1）

完整 JSON Schema 与真实值示例见 `docs/design-integration/03-data-contract-gaps.md` §4.2/§4.3。本计划的实现以该 schema 为准，**字段名不得在实现中改写**。最小必需结构：

```jsonc
{
  "schema_version": 1,
  "date": "2026-10-15",
  "date_label": "2026年10月15日",
  "issue": { "number": null, "label": null, "source": "unavailable" },
  "availability": { "podcast": true, "transcript": true, "playlist": true,
                    "report": true, "sources": true, "radar": true },
  "podcast": {
    "title": "AI 新闻快报 | 2026-10-15", "subtitle": "当…（brief.thesis）",
    "mp3_url": "https://…/episodes/2026-10-15.mp3",
    "duration_seconds": 618.245, "duration_label": "10:18",
    "speech_seconds": 592.821, "segment_count": 55,
    "hosts": [{ "key": "A", "name": "苏晴", "gender": "female",
                "variants": [{ "id": "lively", "label": "青春女声", "default": true }] }],
    "transcript": { "format": "playlist", "segment_count": 55,
                    "segments_url": "episodes/2026-10-15/playlist.json",
                    "text_url": "episodes/2026-10-15.txt" }
  },
  "lead": { "text": "…", "source": "report_lead|brief_thesis", "source_file": "data/reports/…" },
  "metrics": [{ "key": "deduped_items", "label": "今日入库", "value": 123, "unit": "条", "source": "brief.metadata.total_deduped" }],
  "chapters": [{ "title": "说到这个，NVIDIA今天…", "start": 228.346, "label": "03 §2", "source": "cue_heuristic|fallback" }],
  "report": { "title": "…", "char_count": 1680, "reading_minutes": 4, "lead": "…",
              "deep_dives": [{ "heading": "…", "background": "…", "breakthrough": "…", "impact": "…",
                               "raw_sections": [] }],
              "frontier": [{ "title": "…", "summary": "…" }],
              "editor_verdict": "…", "toc": [{ "anchor": "deep-dive-1", "title": "…" }],
              "raw_paragraphs": [], "source_file": "…" },
  "sources": { "total_all": 123, "total_main": 33, "shown": 123, "role_filter": "all",
               "groups": [{ "publisher": "arXiv 论文", "group": "arXiv 论文", "count": 67 }],
               "items": [{ "rank": 1, "role": "main", "role_label": "核心信源",
                           "publisher": "Google AI Blog", "group": "Google",
                           "source_name": "Google AI Blog", "category": "official",
                           "title": "…", "link": "https://…", "published_at": null }] },
  "radar": { "available": true, "status": "pick|empty|degraded|unavailable",
             "rules": { "min_stars": 500, "created_window_days": 30, "recent_push_days": 21,
                        "repeat_window_days": 30, "probe_limit": 12, "pick_count": 1, "runner_up_count": 2 },
             "pick": null, "runner_ups": [], "yesterday": null,
             "candidates": { "total": 12, "excluded_recent": 12, "fresh": 0,
                             "semantics": "pre-v4|v4", "detail_available": false, "preview": null },
             "message": "今日候选池经多轮筛查，未检出达标项目" },
  "meta": { "generated_from": { "brief": "data/briefs/brief_2026-10-15.json" },
            "warnings": ["issue_number_unverified", "radar_candidate_detail_unavailable"] }
}
```

硬性规则（实现 + 测试都要守）：
1. **不写 `now()`**：digest 内容只由源文件决定，保证同日重跑零 diff（`03 §4.1`）。
2. **失败隔离**：任何板块构造失败只降级该板块（`availability.X=false` + `warnings` 追加），绝不抛到 publish。
3. **`sources.items` 上限**：单期最多 150 条（brief 历史最大 123），超出截断并写 warning。
4. `podcast.mp3_url` 只在 `enclosure_url` 缺失时由 `base_url` 拼，别改写既有 URL。

### 1.2 `site/data/calendar.json`

```json
{ "schema_version": 1, "default_date": "2026-10-15",
  "dates": [{ "date": "2026-10-15", "issue_number": null, "title": "AI 新闻快报 | 2026-10-15",
              "availability": { "podcast": true, "report": true, "sources": true, "radar": true } }] }
```

前端启动时 fetch 它拿到每期标题与可用性（`issue_number` 恒为 null，D4）；拿不到时退回内联 `site-data.dates`（同现有降级思路）。
`issue_number` 为 `null` 时 pill 只显示日期。

### 1.3 内联 `site-data`（唯一 Python→JS 通道）

```html
<script type="application/json" id="site-data">{"dates":["2026-10-15"],
"episodes":{"2026-10-15":{"title":"AI 新闻快报 | 2026-10-15","mp3":"https://…/2026-10-15.mp3"}},
"voices":{"host_a":{"lively":"青春女声"},"host_b":{"professional":"专业男声"}}}</script>
```

- 生成：`json.dumps(..., ensure_ascii=False).replace("</", "<\\/")`
- 前端：`JSON.parse(document.getElementById('site-data').textContent)`
- 内联只放「首屏立刻要」的最小数据（约 3-5KB）；`desc` 不再内联（现状 490KB 的 79% 体积来源，`02 §1.1`）。

---

## 2. 文件结构（先锁定边界再动手）

| 文件 | 动作 | 职责 |
|---|---|---|
| `src/ai_news_podcast/presentation/site_builder/report_digest_view.py` | 新建 | 日报 Markdown → 结构化 report 块（纯函数，无 I/O） |
| `src/ai_news_podcast/presentation/site_builder/sources_digest_view.py` | 新建 | brief.stories → sources 块（publisher/group/role 归一，纯函数） |
| `src/ai_news_podcast/presentation/site_builder/digest_view.py` | 新建 | 组装 digest / calendar；时长、章节、指标、radar 归一（期号字段恒为 null）；`write_digest_outputs()` 落盘 + 清理过期 digest |
| `src/ai_news_podcast/presentation/site_builder/html_gen_view.py` | 重写 | 新页面骨架 + 内联 `site-data` + 按序内联 `style.css/app.js/player.js`；删 marked CDN、A/B、四 Tab、旧模板 |
| `src/ai_news_podcast/presentation/site_builder/static/style.css` | 重写 | 方向 A 设计令牌 + 桌面长滚动布局 + 3 个响应式断点 |
| `src/ai_news_podcast/presentation/site_builder/static/app.js` | 新建 | 日期轴、digest 加载与四板块渲染、来源筛选/搜索、雷达空态/命中、抽屉、移动端交互 |
| `src/ai_news_podcast/presentation/site_builder/static/player.js` | 重写 | 纯音频引擎（chunk 播放/±15s/倍速/音量/双音色/句同步回调），不碰 DOM 渲染 |
| `src/ai_news_podcast/presentation/cli/podcast_publish_controller.py` | 修改 | 调用 `write_digest_outputs`（fail-soft） |
| `src/ai_news_podcast/presentation/cli/podcast_daily_controller.py` | 修改 | 同上（本地 full run 链路） |
| `scripts/redeploy_site.py` | 修改 | 重建时刷新 digest（否则上线后修 CSS 会打回无数据页面） |
| `scripts/rebuild_site.py` | 修改 | 本地预览同样生成 digest |
| `.github/workflows/daily.yml` | 修改 | `git add -f site/data/`；Validate 增 digest 断言；部署后对账增 digest |
| `tests/test_report_digest_view.py` | 新建 | md 解析单测（标准格式/缺章节/老格式/空文件） |
| `tests/test_sources_digest_view.py` | 新建 | publisher/group/role/截断单测 |
| `tests/test_digest_view.py` | 新建 | 时长双口径、章节锚点、指标、radar 两代 meta、日历、期号缺省、确定性（同输入同输出） |
| `tests/test_html_gen_view.py` | 重写 | 新骨架结构断言、`site-data` 可解析、无原型 mock 残留、DOM id 契约、体积预算 |
| `tests/test_cli_controller.py` / `test_run_daily_full_controller.py` | 修改 | 增 digest 落盘断言 |
| `docs/development.md`、`AGENTS.md` | 修改 | 新生成链路、digest 契约、prune 注意点 |

---

## 3. 任务（按依赖顺序执行；每个 Task 独立可提交）

### Task 1: 日报 Markdown 解析器

**Files:**
- Create: `src/ai_news_podcast/presentation/site_builder/report_digest_view.py`
- Test: `tests/test_report_digest_view.py`

- [ ] **Step 1: 写失败测试（覆盖标准格式 + 变异 + 空输入）**

```python
"""report_digest_view 单测：日报 md → 结构化 report 块。"""
from ai_news_podcast.presentation.site_builder.report_digest_view import parse_report_markdown

STANDARD_MD = """# ✨ 科技新闻日报 | 2026年10月15日

**导语**：今日全球科技圈呈现"智能体规模化落地"的清晰主线。

---

## 🚀 重磅解读

### 一、NVIDIA 押注法国，智能体计算挺进欧洲纵深

**事件背景**：在欧盟加速推进 AI 主权战略的关键节点。

**核心突破点**：agentic computing 的产业级落地。

**对行业与用户的深远影响**：欧洲不再甘于仅做监管者。

### 二、万级智能体长时任务

**事件背景**：Third Moment 披露。

**核心突破点**：首个万级长时案例。

**行业影响**：从对话工具到虚拟员工。

---

## ⚡ 前沿情报

- **DevFest 2026 强势回归**：Google 宣布启动全球报名。
- **SentinelOne 提速千倍**：恶意软件分析缩至小时级。

---

> **AI 小编深度点评**：今日新闻共同勾勒出三段式跃迁。
"""


class TestParseStandard:
    def test_title_and_lead(self) -> None:
        r = parse_report_markdown(STANDARD_MD)
        assert r["title"].startswith("✨ 科技新闻日报")
        assert "智能体规模化落地" in r["lead"]

    def test_deep_dives_three_points(self) -> None:
        r = parse_report_markdown(STANDARD_MD)
        assert len(r["deep_dives"]) == 2
        first = r["deep_dives"][0]
        assert first["heading"].startswith("一、NVIDIA")
        assert first["background"].startswith("在欧盟")
        assert first["breakthrough"].startswith("agentic computing")
        assert first["impact"].startswith("欧洲不再")

    def test_loose_impact_label_variants(self) -> None:
        md = STANDARD_MD.replace("**对行业与用户的深远影响**", "**行业影响**")
        assert parse_report_markdown(md)["deep_dives"][0]["impact"].startswith("欧洲不再")

    def test_frontier_items(self) -> None:
        r = parse_report_markdown(STANDARD_MD)
        assert [i["title"] for i in r["frontier"]] == ["DevFest 2026 强势回归", "SentinelOne 提速千倍"]

    def test_editor_verdict(self) -> None:
        assert parse_report_markdown(STANDARD_MD)["editor_verdict"].startswith("今日新闻共同")

    def test_char_count_and_minutes(self) -> None:
        r = parse_report_markdown(STANDARD_MD)
        assert r["char_count"] == len(STANDARD_MD)
        assert r["reading_minutes"] >= 1


class TestParseVariants:
    def test_missing_sections_fall_back_to_raw_paragraphs(self) -> None:
        md = "# 日报\n\n**导语**：一句话。\n\n## 别的栏目\n\n正文一段。\n"
        r = parse_report_markdown(md)
        assert r["deep_dives"] == [] and r["frontier"] == []
        assert any("正文一段" in p for p in r["raw_paragraphs"])

    def test_empty_returns_none(self) -> None:
        assert parse_report_markdown("") is None

    def test_radar_section_is_not_news(self) -> None:
        md = STANDARD_MD + "\n## 📡 项目雷达 | 2026年10月15日\n\n### 🥇 主推 [o/r](https://x)\n"
        r = parse_report_markdown(md)
        assert all("项目雷达" not in d["heading"] for d in r["deep_dives"])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_report_digest_view.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现解析器**

```python
"""日报 Markdown → 结构化 report 块（纯函数，无 I/O）。

只覆盖 2026-09 之后的标准格式；更老的格式（06-04..07-22）没有章节结构，
解析结果会退化为 raw_paragraphs，前端按纯段落渲染。
"""

from __future__ import annotations

import re

_LEAD_RE = re.compile(r"\*\*导语\*\*[：:]\s*(.+)")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
_BULLET_RE = re.compile(r"^[-*]\s+\*\*(.+?)\*\*[：:]\s*(.+)$")
_QUOTE_RE = re.compile(r"^>\s*(.+)$")
_BOLD_POINT_RE = re.compile(r"^\*\*(.+?)\*\*[：:]\s*(.+)$")
_QUOTE_LABEL_RE = re.compile(r"^\*\*.+?\*\*[：:]\s*")

_DEEP_DIVE_H2 = "重磅解读"
_FRONTIER_H2 = "前沿情报"
_RADAR_H2 = "项目雷达"

# 第三段小标题在历史样本里有 9 种写法，统一按关键词宽松匹配。
_IMPACT_KEYWORDS = ("影响", "意义", "冲击")
_BREAKTHROUGH_KEYWORDS = ("突破", "核心", "进展", "创新")


def _classify_point(label: str) -> str:
    if any(keyword in label for keyword in _BREAKTHROUGH_KEYWORDS):
        return "breakthrough"
    if any(keyword in label for keyword in _IMPACT_KEYWORDS):
        return "impact"
    return "background"


class _ReportParser:
    """按行推进的日报解析状态机；每个方法只做一件事，便于单测与维护。"""

    def __init__(self, char_count: int) -> None:
        self.char_count = char_count
        self.title = ""
        self.lead: str | None = None
        self.section = ""
        self.current_dive: dict | None = None
        self.deep_dives: list[dict] = []
        self.frontier: list[dict] = []
        self.raw_paragraphs: list[str] = []
        self.quote_lines: list[str] = []
        self.toc: list[dict] = []

    def feed(self, line: str) -> None:
        if not line or line in ("---", "***"):
            return
        heading = _HEADING_RE.match(line)
        if heading:
            self._feed_heading(len(heading.group(1)), heading.group(2).strip())
            return
        if self.section == _RADAR_H2:
            return
        if self._feed_lead(line) or self._feed_bullet(line):
            return
        if line.startswith(">"):
            self._feed_quote(line)
            return
        if self._feed_point(line):
            return
        self.raw_paragraphs.append(line)

    def _feed_heading(self, level: int, text: str) -> None:
        if level == 1:
            self.title = text
            return
        if level == 2:
            self.current_dive = None
            self.section = self._section_kind(text)
            return
        if self.section == _DEEP_DIVE_H2:
            self.current_dive = {
                "heading": text,
                "background": None,
                "breakthrough": None,
                "impact": None,
                "raw_sections": [],
            }
            self.deep_dives.append(self.current_dive)
            self.toc.append({"anchor": f"deep-dive-{len(self.deep_dives)}", "title": text})

    @staticmethod
    def _section_kind(text: str) -> str:
        if _DEEP_DIVE_H2 in text:
            return _DEEP_DIVE_H2
        if _FRONTIER_H2 in text:
            return _FRONTIER_H2
        if _RADAR_H2 in text:
            return _RADAR_H2
        return "other"

    def _feed_lead(self, line: str) -> bool:
        if self.lead is not None:
            return False
        match = _LEAD_RE.search(line)
        if not match:
            return False
        self.lead = match.group(1).strip()
        return True

    def _feed_bullet(self, line: str) -> bool:
        if self.section != _FRONTIER_H2:
            return False
        match = _BULLET_RE.match(line)
        if match:
            self.frontier.append(
                {"title": match.group(1).strip(), "summary": match.group(2).strip()}
            )
            return True
        if line.startswith(("-", "*")):
            self.frontier.append({"title": line[1:].strip(), "summary": ""})
            return True
        return False

    def _feed_quote(self, line: str) -> None:
        match = _QUOTE_RE.match(line)
        text = match.group(1).strip() if match else ""
        text = _QUOTE_LABEL_RE.sub("", text)
        if text:
            self.quote_lines.append(text)

    def _feed_point(self, line: str) -> bool:
        match = _BOLD_POINT_RE.match(line)
        if not match or self.current_dive is None:
            return False
        key = _classify_point(match.group(1))
        if self.current_dive[key] is None:
            self.current_dive[key] = match.group(2).strip()
        else:
            self.current_dive["raw_sections"].append(f"{match.group(1)}：{match.group(2)}")
        return True

    def result(self) -> dict:
        return {
            "title": self.title,
            "char_count": self.char_count,
            "reading_minutes": max(1, round(self.char_count / 400)),
            "lead": self.lead,
            "deep_dives": self.deep_dives,
            "frontier": self.frontier,
            "editor_verdict": " ".join(self.quote_lines).strip() or None,
            "toc": self.toc,
            "raw_paragraphs": self.raw_paragraphs,
        }


def parse_report_markdown(md: str) -> dict | None:
    """把日报 md 解析为 digest 的 report 块；空内容返回 None。"""
    if not md or not md.strip():
        return None
    parser = _ReportParser(char_count=len(md))
    for raw_line in md.splitlines():
        parser.feed(raw_line.strip())
    return parser.result()
```

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_report_digest_view.py -q`
Expected: PASS（≥ 9 passed）

- [ ] **Step 5: 用真实样本回归**

Run:
```bash
uv run python - <<'PY'
from pathlib import Path
from ai_news_podcast.presentation.site_builder.report_digest_view import parse_report_markdown
md = Path("data/reports/daily_report_2026-10-15.md").read_text(encoding="utf-8")
r = parse_report_markdown(md)
assert len(r["deep_dives"]) == 2, r["deep_dives"]
assert len(r["frontier"]) == 3
assert r["editor_verdict"], "missing verdict"
assert r["reading_minutes"] == 4, r["reading_minutes"]
print("OK", r["char_count"], r["deep_dives"][0]["heading"], len(r["frontier"]))
PY
```
Expected: `OK 1680 一、NVIDIA… 3`

- [ ] **Step 6: 提交**

```bash
git add src/ai_news_podcast/presentation/site_builder/report_digest_view.py tests/test_report_digest_view.py
git commit -m "feat(site): parse daily report markdown into digest blocks"
```

---

### Task 2: 来源归一（publisher / group / role）

**Files:**
- Create: `src/ai_news_podcast/presentation/site_builder/sources_digest_view.py`
- Test: `tests/test_sources_digest_view.py`

- [ ] **Step 1: 写失败测试**

```python
"""sources_digest_view 单测：brief.stories → sources 块。"""
from ai_news_podcast.presentation.site_builder.sources_digest_view import (
    build_sources_block,
    classify_group,
)


def _story(title: str, source: str, link: str, role: str = "main", category: str = "official"):
    return {
        "representative_title": title,
        "role": role,
        "role_emoji": "🔴",
        "items": [{"source_name": source, "link": link, "source_category": category}],
    }


class TestClassifyGroup:
    def test_arxiv_by_host(self) -> None:
        assert classify_group("arXiv cs.AI", "https://arxiv.org/abs/2610.1") == "arXiv 论文"

    def test_openai_by_source_name(self) -> None:
        assert classify_group("OpenAI Blog", "https://openai.com/index/x") == "OpenAI"

    def test_google_by_host(self) -> None:
        assert classify_group("DeepMind Blog", "https://blog.google/x") == "Google"

    def test_nvidia_by_source_name(self) -> None:
        assert classify_group("NVIDIA 官方博客", "https://blogs.nvidia.com/x") == "NVIDIA"

    def test_hn(self) -> None:
        assert classify_group("Hacker News", "https://news.ycombinator.com/x") == "Hacker News"

    def test_media_fallback(self) -> None:
        assert classify_group("量子位", "https://qbitai.com/x") == "科技媒体"


class TestBuildSourcesBlock:
    def test_counts_and_order(self) -> None:
        stories = [
            _story("A", "OpenAI Blog", "https://openai.com/a"),
            _story("B", "量子位", "https://qbitai.com/b", role="supporting"),
            {"representative_title": "skip", "role": "skip", "items": []},
        ]
        block = build_sources_block(stories)
        assert block["total_all"] == 2
        assert block["total_main"] == 1
        assert block["items"][0]["role_label"] == "核心信源"
        assert block["items"][1]["role_label"] == "前沿参考"
        assert {g["group"] for g in block["groups"]} == {"OpenAI", "科技媒体"}

    def test_story_without_items_is_dropped(self) -> None:
        assert build_sources_block([{"representative_title": "x", "role": "main", "items": []}])["items"] == []

    def test_hard_cap_150(self) -> None:
        stories = [_story(f"t{i}", "量子位", f"https://qbitai.com/{i}") for i in range(200)]
        block = build_sources_block(stories)
        assert len(block["items"]) == 150
        assert "sources_truncated" in block["warnings"]

    def test_role_filter_all_includes_everything(self) -> None:
        stories = [
            _story("A", "量子位", "https://x/a"),
            _story("B", "量子位", "https://x/b", role="quick"),
        ]
        assert build_sources_block(stories)["shown"] == 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_sources_digest_view.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现**

```python
"""brief.stories → digest 的 sources 块（纯函数，无 I/O）。"""

from __future__ import annotations

from collections import Counter
from urllib.parse import urlparse

ROLE_LABELS = {"main": "核心信源", "supporting": "前沿参考", "quick": "快讯"}
_MAX_ITEMS = 150


def _host(link: str) -> str:
    try:
        return (urlparse(link).hostname or "").lower()
    except ValueError:
        return ""


def classify_group(source_name: str, link: str) -> str:
    """按 link host 优先、source_name 兜底，归到设计稿的 6 个筛选组。"""
    host = _host(link)
    name = source_name or ""
    if host.endswith("arxiv.org") or name.startswith("arXiv"):
        return "arXiv 论文"
    if host == "openai.com" or host.endswith(".openai.com") or name.startswith("OpenAI"):
        return "OpenAI"
    if host == "blog.google" or name in ("Google AI Blog", "DeepMind Blog"):
        return "Google"
    if host == "blogs.nvidia.com" or "NVIDIA" in name:
        return "NVIDIA"
    if host == "news.ycombinator.com" or name == "Hacker News":
        return "Hacker News"
    return "科技媒体"


def build_sources_block(
    stories: list[dict], role_filter: str = "all"
) -> dict:
    """把非 skip 的 story 转成来源条目；role_filter=all 时保留三类角色。"""
    items: list[dict] = []
    warnings: list[str] = []
    for story in stories:
        role = str(story.get("role") or "")
        if role == "skip":
            continue
        if role_filter not in ("all", role):
            continue
        first = (story.get("items") or [{}])[0]
        title = str(story.get("representative_title") or first.get("title") or "").strip()
        link = str(first.get("link") or "").strip()
        if not title or not link:
            continue
        source_name = str(first.get("source_name") or _host(link)).strip()
        group = classify_group(source_name, link)
        items.append(
            {
                "rank": len(items) + 1,
                "role": role if role in ROLE_LABELS else "quick",
                "role_label": ROLE_LABELS.get(role, "快讯"),
                "publisher": "arXiv 论文" if group == "arXiv 论文" else source_name,
                "group": group,
                "source_name": source_name,
                "category": str(first.get("source_category") or ""),
                "title": title,
                "link": link,
                "published_at": first.get("published_at"),
            }
        )

    if len(items) > _MAX_ITEMS:
        items = items[:_MAX_ITEMS]
        warnings.append("sources_truncated")

    counts = Counter(item["group"] for item in items)
    groups = [
        {"publisher": publisher, "group": publisher, "count": count}
        for publisher, count in counts.most_common()
    ]
    return {
        "total_all": len(items),
        "total_main": sum(1 for item in items if item["role"] == "main"),
        "shown": len(items),
        "role_filter": role_filter,
        "groups": groups,
        "items": items,
        "warnings": warnings,
    }
```

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_sources_digest_view.py -q`
Expected: PASS（10 passed）

- [ ] **Step 5: 真实样本抽查**

Run:
```bash
uv run python - <<'PY'
import json
from pathlib import Path
from ai_news_podcast.presentation.site_builder.sources_digest_view import build_sources_block
brief = json.loads(Path("data/briefs/brief_2026-10-15.json").read_text(encoding="utf-8"))
block = build_sources_block(brief["stories"])
assert block["total_all"] == 123, block["total_all"]
assert block["total_main"] == 33, block["total_main"]
print("OK", block["total_all"], [(g["group"], g["count"]) for g in block["groups"]])
PY
```
Expected: `OK 123 [('arXiv 论文', 67), ('Hacker News', 25), ('科技媒体', 22), ('OpenAI', 6), ('Google', 2), ('NVIDIA', 1)]`（组顺序按计数降序）

- [ ] **Step 6: 提交**

```bash
git add src/ai_news_podcast/presentation/site_builder/sources_digest_view.py tests/test_sources_digest_view.py
git commit -m "feat(site): build structured sources block from brief stories"
```

---

### Task 3: digest 组装器（时长/章节/指标/radar/日历；期号缺省）

**Files:**
- Create: `src/ai_news_podcast/presentation/site_builder/digest_view.py`
- Test: `tests/test_digest_view.py`

- [ ] **Step 1: 写失败测试（全部用字面量样本，不读生产数据）**

```python
"""digest_view 单测：组装规则与确定性。"""
import json

from ai_news_podcast.presentation.site_builder.digest_view import (
    build_calendar,
    build_digest,
    detect_chapters,
    duration_stats,
    resolve_issue,
)

CHUNKS = [
    {"id": 1, "host": "A", "text": "听众朋友大家好。", "start": 1.0, "duration": 8.0,
     "audios": {"lively": "chunk_001_lively.mp3"}},
    {"id": 21, "host": "A", "text": "说到这个，NVIDIA今天还有个动作。", "start": 228.346, "duration": 4.18,
     "audios": {"lively": "chunk_021_lively.mp3"}},
    {"id": 31, "host": "A", "text": "还有几条快讯也挺有意思。", "start": 347.985, "duration": 7.88,
     "audios": {"lively": "chunk_031_lively.mp3"}},
    {"id": 47, "host": "A", "text": "最后一条，Google DevFest。", "start": 519.828, "duration": 5.18,
     "audios": {"lively": "chunk_047_lively.mp3"}},
]

BRIEF = {
    "thesis": "当 DevFest 遇上落地",
    "metadata": {"total_raw": 125, "total_deduped": 123, "total_clusters": 123},
    "stories": [
        {"representative_title": "A", "role": "main",
         "items": [{"source_name": "OpenAI Blog", "link": "https://openai.com/a"}]},
        {"representative_title": "B", "role": "supporting",
         "items": [{"source_name": "量子位", "link": "https://qbitai.com/b"}]},
    ],
}

CFG = {
    "podcast": {"title": "AI 每日先锋"},
    "gh_radar": {"min_stars": 500, "created_window_days": 30, "recent_push_days": 21,
                 "repeat_window_days": 30, "readme_probe_limit": 12,
                 "pick_count": 1, "runner_up_count": 2},
    "tts": {"voice_names": {"host_a": {"lively": "青春女声"},
                            "host_b": {"professional": "专业男声"}},
            "cosyvoice": {"synth_variants": {"host_a": ["lively"], "host_b": ["professional"]}}},
}

RADAR_EMPTY = {"date": "2026-10-15", "projects": [],
               "meta": {"pick_repo": "", "runner_up_repos": [], "degraded": False,
                        "reason": "", "candidates": 12, "excluded_recent": 12}}


class TestIssue:
    def test_missing_issue_is_unavailable(self) -> None:
        issue = resolve_issue(None)
        assert issue["number"] is None and issue["label"] is None
        assert issue["source"] == "unavailable"

    def test_persisted_issue_is_passed_through(self) -> None:
        issue = resolve_issue(42)
        assert issue["number"] == 42 and issue["label"] == "第 42 期"
        assert issue["source"] == "persisted"


class TestDuration:
    def test_timeline_and_speech(self) -> None:
        timeline, speech = duration_stats(CHUNKS)
        assert round(timeline, 3) == 525.008
        assert round(speech, 3) == 25.24

    def test_empty(self) -> None:
        assert duration_stats([]) == (0.0, 0.0)


class TestChapters:
    def test_cue_heuristic_matches(self) -> None:
        chapters = detect_chapters(CHUNKS)
        starts = [c["start"] for c in chapters]
        assert starts[0] == 1.0
        assert 228.346 in starts and 347.985 in starts and 519.828 in starts
        assert all(c["source"] == "cue_heuristic" for c in chapters)

    def test_fallback_when_no_cues(self) -> None:
        quiet = [{"id": i, "host": "A", "text": f"普通段落{i}", "start": i * 10.0,
                  "duration": 9.0, "audios": {}} for i in range(1, 6)]
        chapters = detect_chapters(quiet)
        assert 2 <= len(chapters) <= 3 and all(c["source"] == "fallback" for c in chapters)


class TestBuildDigest:
    def _build(self, **overrides):
        kwargs = dict(
            date_str="2026-10-15", brief=BRIEF, report_md="", radar=RADAR_EMPTY,
            playlist={"chunks": CHUNKS}, episode={"title": "AI 新闻快报 | 2026-10-15"},
            cfg=CFG, base_url="https://example.com",
        )
        kwargs.update(overrides)
        return build_digest(**kwargs)

    def test_schema_core(self) -> None:
        digest = self._build()
        assert digest["schema_version"] == 1
        assert digest["date"] == "2026-10-15"
        assert digest["issue"]["number"] is None
        assert digest["issue"]["source"] == "unavailable"
        assert digest["availability"]["podcast"] is True
        assert digest["availability"]["report"] is False
        assert digest["report"] is None

    def test_metrics_are_real_numbers(self) -> None:
        metrics = {m["key"]: m["value"] for m in self._build()["metrics"]}
        assert metrics["deduped_items"] == 123
        assert metrics["main_stories"] == 1
        assert metrics["podcast_segments"] == 4
        assert metrics["media_outlets"] == 2

    def test_radar_empty_state(self) -> None:
        radar = self._build()["radar"]
        assert radar["status"] == "empty" and radar["pick"] is None
        assert radar["candidates"]["total"] == 12
        assert radar["rules"]["min_stars"] == 500

    def test_radar_pick(self) -> None:
        radar = {"date": "2026-10-15",
                 "projects": [{"repo": "a/b", "url": "https://github.com/a/b", "description": "d",
                               "language": "Python", "stars": 5000, "delta_stars": 800,
                               "license": "MIT", "readme_intro": "intro", "readme_excerpt": "x",
                               "score": 1.0, "score_parts": {}, "affinity": 0.5,
                               "topics": [], "is_new": True, "created_at": "", "pushed_at": "",
                               "has_install_docs": True, "mentions": 0}],
                 "meta": {"pick_repo": "a/b", "runner_up_repos": [], "candidates": 30,
                          "excluded_recent": 3, "fresh_candidates": 27, "pool_by_source": {}}}
        out = self._build(radar=radar)["radar"]
        assert out["status"] == "pick" and out["pick"]["repo"] == "a/b"
        assert out["candidates"]["semantics"] == "v4"

    def test_deterministic_no_now(self) -> None:
        assert json.dumps(self._build(), sort_keys=True) == json.dumps(self._build(), sort_keys=True)
        assert "generated_at" not in self._build()["meta"]

    def test_calendar(self) -> None:
        cal = build_calendar(
            episodes=[{"id": "2026-10-15", "title": "t", "issue_no": None},
                      {"id": "2026-10-14", "title": "t2", "issue_no": None}]
        )
        assert cal["default_date"] == "2026-10-15"
        # D4：v1 不显示期号，字段保留但为 null
        assert [d["issue_number"] for d in cal["dates"]] == [None, None]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_digest_view.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 digest_view.py（组装 + 落盘）**

```python
"""每期站点数据聚合：digest_{date}.json / calendar.json。

设计原则：
- 纯函数不读文件（输入全部由调用方传入），方便单测与回放；
- 绝不抛异常到 publish：单板块失败只降级 + warnings；
- 不写时间戳，保证同源重跑零 diff。
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any

from ai_news_podcast.presentation.site_builder.report_digest_view import parse_report_markdown
from ai_news_podcast.presentation.site_builder.sources_digest_view import build_sources_block

log = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_CUE_PATTERNS = (
    re.compile(r"说到这个|接下来|另外|再说"),
    re.compile(r"第[一二三四五六七八九十]条"),
    re.compile(r"还有几条|快速过一下"),
    re.compile(r"最后一条"),
)
_CHAPTER_GAP_SECONDS = 60.0


def _zh_date(date_str: str) -> str:
    dt = datetime.date.fromisoformat(date_str)
    return f"{dt.year}年{dt.month:02d}月{dt.day:02d}日"


def _fmt_duration(seconds: float) -> str:
    total = round(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def resolve_issue(issue_no: int | None) -> dict:
    """期号：只有真实持久化值才显示；v1 不持久化，UI 因此只显示日期（D4）。"""
    if issue_no:
        return {"number": int(issue_no), "label": f"第 {int(issue_no)} 期", "source": "persisted"}
    return {"number": None, "label": None, "source": "unavailable"}


def duration_stats(chunks: list[dict]) -> tuple[float, float]:
    """返回 (时间轴时长, 净语音时长)。"""
    if not chunks:
        return (0.0, 0.0)
    last = chunks[-1]
    timeline = float(last.get("start") or 0) + float(last.get("duration") or 0)
    speech = sum(float(c.get("duration") or 0) for c in chunks)
    return (timeline, speech)


def detect_chapters(chunks: list[dict]) -> list[dict]:
    """按转场词定位章节锚点；不足 2 个锚点时退化为固定三点。"""
    if not chunks:
        return []
    chapters: list[dict] = [{"title": _short_title(chunks[0]), "start": float(chunks[0]["start"]),
                             "label": _fmt_duration(float(chunks[0]["start"])),
                             "source": "cue_heuristic"}]
    for chunk in chunks[1:]:
        text = str(chunk.get("text") or "")
        if not any(pattern.search(text) for pattern in _CUE_PATTERNS):
            continue
        start = float(chunk.get("start") or 0)
        if start - chapters[-1]["start"] < _CHAPTER_GAP_SECONDS:
            continue
        chapters.append({"title": _short_title(chunk), "start": start,
                         "label": _fmt_duration(start), "source": "cue_heuristic"})
        if len(chapters) == 4:
            break
    if len(chapters) >= 2:
        return chapters
    last = chunks[-1]
    mid = chunks[len(chunks) // 2]
    return [
        {"title": "开场", "start": float(chunks[0]["start"]),
         "label": _fmt_duration(float(chunks[0]["start"])), "source": "fallback"},
        {"title": "中段", "start": float(mid["start"]),
         "label": _fmt_duration(float(mid["start"])), "source": "fallback"},
        {"title": "收尾", "start": float(last["start"]),
         "label": _fmt_duration(float(last["start"])), "source": "fallback"},
    ]


def _short_title(chunk: dict, limit: int = 12) -> str:
    text = re.sub(r"\s+", " ", str(chunk.get("text") or "")).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _section_cfg(cfg: Any, name: str) -> dict:
    """同时支持 dict 与 AppConfig 数据类；返回浅拷贝，避免调用方误改配置。"""
    if isinstance(cfg, dict):
        return dict(cfg.get(name) or {})
    model = getattr(cfg, name, None)
    return dict(vars(model)) if model is not None else {}



def _voice_labels(cfg: Any) -> dict[str, dict[str, str]]:
    from ai_news_podcast.presentation.site_builder.html_gen_view import _voice_labels as labels

    return labels(cfg)


def _hosts(cfg: Any) -> list[dict]:
    names = {"host_a": ("A", "苏晴", "female"), "host_b": ("B", "周航", "male")}
    voice_labels = _voice_labels(cfg)
    hosts = []
    for config_key, (key, name, gender) in names.items():
        variants = [{"id": variant, "label": label, "default": index == 0}
                    for index, (variant, label) in enumerate(voice_labels.get(config_key, {}).items())]
        hosts.append({"key": key, "name": name, "gender": gender, "variants": variants})
    return hosts


def _metrics(brief: dict, sources: dict, chunks: list[dict]) -> list[dict]:
    metadata = brief.get("metadata") or {}
    publishers = {item["source_name"] for item in sources["items"]}
    return [
        {"key": "deduped_items", "label": "今日入库", "value": int(metadata.get("total_deduped") or 0),
         "unit": "条", "source": "brief.metadata.total_deduped"},
        {"key": "main_stories", "label": "核心研判", "value": sources["total_main"],
         "unit": "条", "source": "brief.stories[role=main]"},
        {"key": "podcast_segments", "label": "双人对谈", "value": len(chunks),
         "unit": "段", "source": "playlist.chunks"},
        {"key": "media_outlets", "label": "覆盖媒体", "value": len(publishers),
         "unit": "家", "source": "brief.stories[].items[0].source_name"},
    ]


def _radar_block(radar: dict | None, date_str: str, cfg: Any) -> dict:
    rules_cfg = _section_cfg(cfg, "gh_radar")
    rules = {
        "min_stars": int(rules_cfg.get("min_stars", 500)),
        "created_window_days": int(rules_cfg.get("created_window_days", 30)),
        "recent_push_days": int(rules_cfg.get("recent_push_days", 21)),
        "repeat_window_days": int(rules_cfg.get("repeat_window_days", 30)),
        "probe_limit": int(rules_cfg.get("readme_probe_limit", rules_cfg.get("probe_limit", 12))),
        "pick_count": int(rules_cfg.get("pick_count", 1)),
        "runner_up_count": int(rules_cfg.get("runner_up_count", 2)),
    }
    if radar is None:
        return {"available": False, "status": "unavailable", "rules": rules, "pick": None,
                "runner_ups": [], "yesterday": None,
                "candidates": {"total": 0, "excluded_recent": 0, "fresh": None,
                               "semantics": "unknown", "detail_available": False, "preview": None},
                "message": "该日期暂无项目雷达"}
    meta = radar.get("meta") or {}
    projects = radar.get("projects") or []
    meta_keys = set(meta)
    semantics = "v4" if meta_keys & {"fresh_candidates", "pool_by_source"} else "pre-v4"
    yesterday = None
    if meta.get("yesterday_pick"):
        yesterday = {"repo": str(meta["yesterday_pick"]),
                     "delta_stars": meta.get("yesterday_pick_delta"),
                     "date": ""}
    status = "pick" if projects else ("degraded" if meta.get("degraded") else "empty")
    message = "今日候选池经多轮筛查，未检出达标项目" if not projects else "今日主推项目"
    return {
        "available": True,
        "status": status,
        "rules": rules,
        "pick": projects[0] if projects else None,
        "runner_ups": projects[1:],
        "yesterday": yesterday,
        "candidates": {
            "total": int(meta.get("candidates") or 0),
            "excluded_recent": int(meta.get("excluded_recent") or 0),
            "fresh": int(meta["fresh_candidates"]) if "fresh_candidates" in meta else None,
            "semantics": semantics,
            "detail_available": False,
            "preview": None,
        },
        "message": message,
    }


def build_digest(
    *,
    date_str: str,
    brief: dict | None,
    report_md: str | None,
    radar: dict | None,
    playlist: dict | None,
    episode: dict | None,
    cfg: Any,
    base_url: str,
) -> dict:
    """组装单期 digest；任何板块失败只降级，不抛异常。"""
    warnings: list[str] = []
    brief = brief or {}
    chunks = list((playlist or {}).get("chunks") or [])
    timeline, speech = duration_stats(chunks)

    try:
        sources = build_sources_block(brief.get("stories") or [])
        warnings.extend(sources.get("warnings") or [])
    except ValueError:
        log.warning("sources block failed for %s", date_str, exc_info=True)
        sources = {"total_all": 0, "total_main": 0, "shown": 0, "role_filter": "all",
                   "groups": [], "items": [], "warnings": ["sources_build_failed"]}
        warnings.append("sources_build_failed")

    try:
        report = parse_report_markdown(report_md or "")
        if report is not None:
            report["source_file"] = f"data/reports/daily_report_{date_str}.md"
    except ValueError:
        log.warning("report parse failed for %s", date_str, exc_info=True)
        report = None
        warnings.append("report_parse_failed")

    radar_block = _radar_block(radar, date_str, cfg)
    if not radar_block["candidates"]["detail_available"]:
        warnings.append("radar_candidate_detail_unavailable")

    episode = episode or {}
    mp3_url = str(episode.get("enclosure_url") or f"{base_url}/episodes/{date_str}.mp3")
    lead_source = "report_lead" if report and report.get("lead") else "brief_thesis"
    lead_text = (report or {}).get("lead") or brief.get("thesis") or None
    issue = resolve_issue(episode.get("issue_no"))

    hosts = _hosts(cfg)
    variants_ok = all(host["variants"] for host in hosts)
    if not variants_ok:
        warnings.append("voice_variants_incomplete")

    return {
        "schema_version": _SCHEMA_VERSION,
        "date": date_str,
        "date_label": _zh_date(date_str),
        "issue": issue,
        "availability": {
            "podcast": bool(chunks),
            "transcript": bool(chunks),
            "playlist": bool(chunks),
            "report": report is not None,
            "sources": bool(sources["items"]),
            "radar": radar_block["available"],
        },
        "podcast": {
            "title": str(episode.get("title") or f"AI 新闻快报 | {date_str}"),
            "subtitle": brief.get("thesis") or None,
            "mp3_url": mp3_url,
            "duration_seconds": round(timeline, 3),
            "duration_label": _fmt_duration(timeline),
            "speech_seconds": round(speech, 3),
            "segment_count": len(chunks),
            "hosts": hosts,
            "transcript": {
                "format": "playlist",
                "segment_count": len(chunks),
                "segments_url": f"episodes/{date_str}/playlist.json",
                "text_url": f"episodes/{date_str}.txt",
            },
        },
        "lead": ({"text": lead_text, "source": lead_source,
                  "source_file": f"data/reports/daily_report_{date_str}.md"}
                 if lead_text else None),
        "metrics": _metrics(brief, sources, chunks),
        "chapters": detect_chapters(chunks),
        "report": report,
        "sources": sources,
        "radar": radar_block,
        "meta": {
            "generated_from": {
                "brief": f"data/briefs/brief_{date_str}.json",
                "report": f"data/reports/daily_report_{date_str}.md",
                "radar": f"data/gh_radar/radar_{date_str}.json",
                "playlist": f"site/episodes/{date_str}/playlist.json",
                "episodes": "data/episodes.json",
            },
            "warnings": warnings,
        },
    }


def build_calendar(episodes: list[dict]) -> dict:
    """日期轴聚合文件：期号 + 标题 + 可用性（可用性由 digest 是否存在推断）。"""
    dates = []
    for episode in episodes:
        date_str = str(episode.get("id") or "")
        if not date_str:
            continue
        dates.append(
            {
                "date": date_str,
                "issue_number": None,  # D4：v1 不显示期号；字段保留给未来真实刊号
                "title": str(episode.get("title") or f"AI 新闻快报 | {date_str}"),
                "availability": {"podcast": True, "report": True, "sources": True, "radar": True},
            }
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "default_date": dates[0]["date"] if dates else None,
        "dates": dates,
    }


def write_digest_outputs(
    root: Path,
    episodes: list[dict],
    cfg: Any,
    base_url: str,
    *,
    site_dir: Path,
    episodes_dir: Path,
) -> list[Path]:
    """为 episodes 里每个日期生成 digest + calendar；单期失败跳过，绝不抛异常。"""
    data_dir = site_dir / "data"
    written: list[Path] = []
    kept: set[str] = set()
    for episode in episodes:
        date_str = str(episode.get("id") or "")
        if not date_str:
            continue
        kept.add(f"digest_{date_str}.json")
        try:
            brief = _read_json_or_none(root / "data" / "briefs" / f"brief_{date_str}.json")
            report_md = _read_text_or_none(root / "data" / "reports" / f"daily_report_{date_str}.md")
            radar = _read_json_or_none(root / "data" / "gh_radar" / f"radar_{date_str}.json")
            playlist = _read_json_or_none(episodes_dir / date_str / "playlist.json")
            digest = build_digest(
                date_str=date_str, brief=brief, report_md=report_md, radar=radar,
                playlist=playlist, episode=episode, cfg=cfg, base_url=base_url,
            )
            target = data_dir / f"digest_{date_str}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_dump(digest), encoding="utf-8")
            written.append(target)
        except (OSError, ValueError, TypeError):
            log.warning("digest generation failed for %s", date_str, exc_info=True)

    calendar_path = data_dir / "calendar.json"
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    calendar_path.write_text(_dump(build_calendar(episodes)), encoding="utf-8")
    for stale in data_dir.glob("digest_*.json"):
        if stale.name not in kept:
            stale.unlink(missing_ok=True)
    return written


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _read_json_or_none(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text_or_none(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None
```

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_digest_view.py -q`
Expected: PASS（约 15 passed）。若 `media_outlets` 断言失败，检查 `sources` 统计口径（OpenAI Blog + 量子位 = 2 家）。

- [ ] **Step 5: 真实数据端到端抽查（不写文件）**

Run:
```bash
uv run python - <<'PY'
import json
from pathlib import Path
from ai_news_podcast.data.config.loader_dao import load_config
from ai_news_podcast.presentation.site_builder.digest_view import build_digest
root = Path(".")
d = "2026-10-15"
cfg = load_config(Path("config/config.yaml"))
digest = build_digest(
    date_str=d,
    brief=json.loads((root/"data/briefs"/f"brief_{d}.json").read_text(encoding="utf-8")),
    report_md=(root/"data/reports"/f"daily_report_{d}.md").read_text(encoding="utf-8"),
    radar=json.loads((root/"data/gh_radar"/f"radar_{d}.json").read_text(encoding="utf-8")),
    playlist=json.loads((root/"site/episodes"/d/"playlist.json").read_text(encoding="utf-8")),
    episode={"title": f"AI 新闻快报 | {d}"}, cfg=cfg, base_url="https://example.com",
)
print(json.dumps({k: digest[k] for k in ("date","issue","metrics","chapters")}, ensure_ascii=False, indent=1)[:1500])
assert digest["issue"]["number"] is None  # D4：286 不真实，v1 不显示期号
assert digest["radar"]["status"] == "empty"
assert digest["sources"]["total_all"] == 123
assert digest["report"]["deep_dives"][0]["heading"].startswith("一、NVIDIA")
assert digest["podcast"]["duration_label"] == "10:18"
PY
```
Expected: `issue` 为 null、雷达 empty、来源 123、日报 2 篇、时长 10:18（与设计稿 09:53 的差异是口径修正，见 `03 §3.3`）。

- [ ] **Step 6: 提交**

```bash
git add src/ai_news_podcast/presentation/site_builder/digest_view.py tests/test_digest_view.py
git commit -m "feat(site): add per-date digest builder (issue/duration/chapters/metrics/radar)"
```

---

### Task 4: 接线四条生成链路（publish / daily / redeploy / rebuild）

**Files:**
- Modify: `src/ai_news_podcast/presentation/cli/podcast_publish_controller.py:136-165`
- Modify: `src/ai_news_podcast/presentation/cli/podcast_daily_controller.py:176-216`
- Modify: `scripts/redeploy_site.py:27-58`
- Modify: `scripts/rebuild_site.py:39-71`
- Test: `tests/test_cli_coverage_controller.py`（追加落盘断言）

- [ ] **Step 1: 写失败测试（不 mock 渲染，验证 digest 真的落盘）**

```python
# tests/test_cli_coverage_controller.py 追加
def test_write_digest_outputs_creates_files(tmp_path) -> None:
    """write_digest_outputs 必须在 site/data/ 下写出 digest_{date}.json 与 calendar.json。"""
    import json

    root = tmp_path
    (root / "data/briefs").mkdir(parents=True)
    (root / "data/reports").mkdir(parents=True)
    (root / "data/gh_radar").mkdir(parents=True)
    (root / "site/episodes/2026-01-02").mkdir(parents=True)
    (root / "data/briefs/brief_2026-01-02.json").write_text(
        json.dumps({"thesis": "t", "stories": [], "metadata": {"total_deduped": 0}}),
        encoding="utf-8",
    )
    (root / "site/episodes/2026-01-02/playlist.json").write_text(
        json.dumps({"chunks": [{"id": 1, "host": "A", "text": "x", "start": 1.0,
                                "duration": 2.0, "audios": {"lively": "c.mp3"}}]}),
        encoding="utf-8",
    )
    (root / "site/episodes/2026-01-02.mp3").write_bytes(b"x" * 200)
    (root / "data/episodes.json").write_text("[]", encoding="utf-8")

    from ai_news_podcast.presentation.site_builder.digest_view import write_digest_outputs
    episodes = [{"id": "2026-01-02", "title": "t",
                 "enclosure_url": "https://example.com/episodes/2026-01-02.mp3"}]
    written = write_digest_outputs(
        root, episodes, {"podcast": {}}, "https://example.com",
        site_dir=root / "site", episodes_dir=root / "site/episodes",
    )
    assert (root / "site/data/digest_2026-01-02.json").exists()
    assert (root / "site/data/calendar.json").exists()
    assert written and written[0].name == "digest_2026-01-02.json"
    digest = json.loads((root / "site/data/digest_2026-01-02.json").read_text(encoding="utf-8"))
    assert digest["date"] == "2026-01-02" and digest["schema_version"] == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cli_coverage_controller.py::test_publish_writes_digest_and_calendar -q`
Expected: FAIL（`ModuleNotFoundError` / import error）

- [ ] **Step 3: publish controller 接线（`home` 路径 = `root`）**

在 `podcast_publish_controller.py` 的 `build_index_html(...)`（`:158`）之后、`_sync_tree(...)`（`:160`）之前插入：

```python
        from ai_news_podcast.presentation.site_builder.digest_view import write_digest_outputs

        try:
            write_digest_outputs(
                root,
                sorted_eps,
                cfg,
                base_url,
                site_dir=site_dir,
                episodes_dir=episodes_dir,
            )
        except (OSError, ValueError):
            log.warning("digest generation failed; continuing publish", exc_info=True)
```

文件顶部补 `import logging` 与 `log = logging.getLogger(__name__)`（若已有则复用）。

- [ ] **Step 4: daily / redeploy / rebuild 三处接线**

- `podcast_daily_controller.py`：在 `build_index_html(...)`（`:216`）之后同样调用 `write_digest_outputs(root, sorted_eps, cfg, base_url, site_dir=site_dir, episodes_dir=episodes_dir)`，包同样的 try/except（该函数内已有 `sorted_eps` 与 `episodes_dir`）。
- `scripts/redeploy_site.py`：`build_index_html(...)`（`:35`）之后调用 `write_digest_outputs(ROOT, episodes, cfg, base_url, site_dir=site_dir, episodes_dir=ROOT / str(cfg.build.episodes_dir))`（`episodes` 已在脚本内按 `published_at_iso` 降序排好）。
- `scripts/rebuild_site.py`：`build_index_html(...)`（`:49`）之后调用同款，`base_url` 用脚本里已有的 `http://localhost:8000/site`，`episodes_dir` 用 `site_dir / "episodes"`，`episodes` 用 `processed_episodes`（相对 mp3 URL 仅影响本地预览）。

- [ ] **Step 5: 验证本地预览真的生成**

Run:
```bash
uv run python scripts/rebuild_site.py
ls -la site/data/ | head
python3 -c "import json;d=json.load(open('site/data/digest_2026-10-15.json'));print(d['date'],d['issue']['number'],d['radar']['status'],len(d['sources']['items']))"
```
Expected: 生成 30 个 digest + `calendar.json`；单期输出 `2026-10-15 None empty 123`（期号按 D4 恒为 None）。（`site/` 被 gitignore，本地生成不影响仓库。）

- [ ] **Step 6: 跑全量测试确认无回归**

Run: `uv run pytest tests/ -q`
Expected: 全绿（可能新增用例数变化）；若 `test_cli_coverage_controller` 里旧的 mock 用例因新调用失败，按同样方式 patch `write_digest_outputs`。

- [ ] **Step 7: 提交**

```bash
git add src/ai_news_podcast/presentation/cli/podcast_publish_controller.py \
  src/ai_news_podcast/presentation/cli/podcast_daily_controller.py \
  scripts/redeploy_site.py scripts/rebuild_site.py tests/test_cli_coverage_controller.py
git commit -m "feat(site): generate digests on publish/daily/redeploy/rebuild"
```

---

### Task 5: CI 加固（git add -f / Validate / 部署后对账）

**Files:**
- Modify: `.github/workflows/daily.yml:327-404`

- [ ] **Step 1: Validate 步骤追加 digest 断言**

在 `Validate episode artifacts` 的 heredoc 里、`print(f"validated ...")` 之前插入：

```python
          digest_path = Path(f"site/data/digest_{ep}.json")
          assert digest_path.exists(), f"missing {digest_path}"
          digest = json.loads(digest_path.read_text(encoding="utf-8"))
          assert digest.get("schema_version") == 1, "digest schema_version mismatch"
          assert digest.get("date") == ep, "digest date mismatch"
          assert digest.get("sources", {}).get("items") is not None, "digest sources missing"
```

- [ ] **Step 2: Commit site 步骤追加跟踪**

在 `git add -f "site/episodes/${EPISODE}/playlist.json" ...`（`:357`）之后追加：

```bash
          git add -f site/data/ 2>/dev/null || echo "No site/data yet, skipped"
```

- [ ] **Step 3: 部署后对账追加 digest 检查**

在 `Verify gh-pages deployment` 的 `PLAYLIST=$(get_size ...)` 之后追加：

```bash
          DIGEST=$(get_size "data/digest_${EPISODE}.json")
          echo "gh-pages data/digest_${EPISODE}.json size: $DIGEST"
          if [ "$DIGEST" -le 0 ]; then
            echo "::error::data/digest_${EPISODE}.json missing on gh-pages"
            exit 1
          fi
```

- [ ] **Step 4: 校验 YAML 与脚本语法**

Run:
```bash
python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/daily.yml')); print('yaml ok')"
bash -n <(sed -n '/Commit site/,/Remove .gitignore/p' .github/workflows/daily.yml) 2>/dev/null || true
```
Expected: `yaml ok`；无 bash 语法错误。

- [ ] **Step 5: 提交**

```bash
git add .github/workflows/daily.yml
git commit -m "ci: track and reconcile site/data digests on gh-pages"
```

---

### Task 6: 新页面骨架（html_gen_view 重写 + 内联 `site-data`）

**Files:**
- Rewrite: `src/ai_news_podcast/presentation/site_builder/html_gen_view.py`
- Test: `tests/test_html_gen_view.py`（整体重写）

**移植策略：直接沿用原型 DOM 的 id/class 命名**（`01-prototype-inventory.md` §2 全表），只做下表 9 处改动。这样 CSS 可以从原型逐块搬，app.js/player.js 的 id 全部有据可查；旧生产 id 一律不保留（`player.js` 本来就重写）。

| 原型位置 | 改动 |
|---|---|
| `index.html:1634-1636` | 删「390px 手机版」按钮（评审工具） |
| `index.html:1655-1660` | 日期 pill 清空，改由 JS 从 `DATES` 渲染；前后日按钮绑定 `data-step` |
| `index.html:1612` | 品牌徽章 `<span class="brand-badge" id="issue-badge">` 改由 JS 填日期（D4：不显示期号） |
| `index.html:1687-1690` | 加 `id="podcast-duration"`、`id="podcast-title"`、`id="podcast-subtitle"` |
| `index.html:1748` | 导语容器加 `id="lead-text"`，内容由 JS 填 |
| `index.html:1752-1786` | 指标条 `#metrics-strip` 与 `#chapters-row` 清空，由 JS 渲染 |
| `index.html:1804-1902` | 日报整体改为 `#daily-report-body` 空容器 + `#daily-toc`；解析后的结构化内容由 JS 渲染 |
| `index.html:1951-2041` | 雷达整块改为 `#radar-root` 空容器（空态/命中态由 JS 渲染，canvas 在 JS 里创建） |
| `index.html:2124-2132,2139-2143` | 删 `#mobile-modal`、删 4 个 mock 数据常量与全部内联 `<script>`；改为内联 `app.js` + `player.js` |

- [ ] **Step 1: 写新测试（骨架/数据块/mock 残留/体积）**

```python
"""新首页生成契约：结构、数据块、DOM id、体积预算。"""

import gzip
import json
import re

from ai_news_podcast.presentation.site_builder.html_gen_view import build_index_html

_EPISODES = [
    {"id": "2026-10-15", "title": "AI 新闻快报 | 2026-10-15",
     "enclosure_url": "https://example.com/episodes/2026-10-15.mp3",
     "description": "<p>desc</p>"},
]
_MOCK_MARKERS = ("PLAYLIST_DATA", "SOURCES_DATA", "RADAR_DATA", "CANDIDATES_DATA",
                 "Lorem ipsum", "第 286 期", "toggleRadarMockState", "simulateDateSwitch",
                 "cdn.jsdelivr.net")


def _build(tmp_path, episodes=None, cfg=None) -> str:
    site = tmp_path / "site"
    build_index_html(site, "AI 每日先锋", episodes if episodes is not None else _EPISODES,
                     "https://example.com", cfg)
    return (site / "index.html").read_text(encoding="utf-8")


def _site_data(html: str) -> dict:
    match = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S)
    assert match, "缺 site-data JSON 块"
    return json.loads(match.group(1))


class TestSkeleton:
    def test_four_sections_present(self, tmp_path) -> None:
        html = _build(tmp_path)
        for anchor in ('id="section-hero"', 'id="section-daily"',
                       'id="section-sources"', 'id="section-radar"'):
            assert html.count(anchor) == 1, anchor

    def test_docked_player_and_drawer(self, tmp_path) -> None:
        html = _build(tmp_path)
        for dom_id in ("docked-player", "transcript-drawer", "main-audio", "toast-pill"):
            assert html.count(f'id="{dom_id}"') == 1, dom_id
        assert 'href="#section-hero"' in html and "./feed.xml" in html

    def test_no_legacy_tabs_or_variants(self, tmp_path) -> None:
        html = _build(tmp_path)
        for legacy in ("panel-podcast", "btn-mode-", "variant-container", "ab-variant",
                       "theme-variant", "mobile-modal"):
            assert legacy not in html, legacy

    def test_no_apple_entry_in_v1(self, tmp_path) -> None:
        """D15：Apple Podcasts 入口 v1 不做，只留 RSS。"""
        html = _build(tmp_path)
        assert "apple-link" not in html
        assert "podcasts.apple.com" not in html
        assert "./feed.xml" in html


class TestSiteData:
    def test_parsable_and_scoped(self, tmp_path) -> None:
        data = _site_data(_build(tmp_path))
        assert data["dates"] == ["2026-10-15"]
        assert data["episodes"]["2026-10-15"]["mp3"].endswith("2026-10-15.mp3")
        assert set(data["voices"]) == {"host_a", "host_b"}

    def test_empty_episodes(self, tmp_path) -> None:
        data = _site_data(_build(tmp_path, episodes=[]))
        assert data["dates"] == []

    def test_desc_not_inlined(self, tmp_path) -> None:
        html = _build(tmp_path)
        assert "<p>desc</p>" not in html, "desc 不应再内联（体积红线）"


class TestNoMockLeftovers:
    def test_no_prototype_constants(self, tmp_path) -> None:
        html = _build(tmp_path)
        for marker in _MOCK_MARKERS:
            assert marker not in html, f"原型 mock 残留: {marker}"


class TestVoiceLabelsContract:
    """音色标签收敛契约（原 test_html_gen_view.py::TestVoiceLabels 保留）。"""

    def test_defaults(self) -> None:
        from ai_news_podcast.presentation.site_builder.html_gen_view import _voice_labels
        assert set(_voice_labels({})["host_a"]) == {"professional", "lively"}

    def test_per_host_filter(self) -> None:
        from ai_news_podcast.presentation.site_builder.html_gen_view import _voice_labels
        cfg = {"tts": {"cosyvoice": {"synth_variants": {"host_a": ["lively"], "host_b": ["professional"]}}}}
        labels = _voice_labels(cfg)
        assert set(labels["host_a"]) == {"lively"} and set(labels["host_b"]) == {"professional"}


class TestSizeBudget:
    def test_inline_budget(self, tmp_path) -> None:
        html = _build(tmp_path, episodes=[{"id": f"2026-09-{d:02d}", "title": "t"} for d in range(1, 31)])
        raw = len(html.encode("utf-8"))
        assert raw < 1_500_000, raw
        assert len(gzip.compress(html.encode("utf-8"))) < 300_000
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_html_gen_view.py -q`
Expected: FAIL（旧模板无 `site-data`、无新 section）

- [ ] **Step 3: 实现 `build_index_html`（模板骨架）**

```python
"""站点首页生成：骨架 + 内联 site-data + 内联 style/app/player。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_news_podcast.data.config.models_dao import AppConfig

_STATIC_DIR = Path(__file__).parent / "static"
_DEFAULT_VOICE_NAMES = {
    "host_a": {"professional": "亲切女声", "lively": "青春女声"},
    "host_b": {"professional": "专业男声", "lively": "活力男声"},
}


def _read_static(filename: str) -> str:
    return (_STATIC_DIR / filename).read_text(encoding="utf-8")


def _norm_host_key(key: str) -> str | None:
    k = str(key).strip().lower()
    if k in ("host_a", "a"):
        return "host_a"
    if k in ("host_b", "b"):
        return "host_b"
    return None


def _voice_labels(cfg: Any) -> dict[str, dict[str, str]]:
    """注入播放器的音色标签；按 synth_variants（每 host）收敛，按钮只指向真实存在的音频。"""
    if cfg is None:
        return {key: dict(value) for key, value in _DEFAULT_VOICE_NAMES.items()}

    if isinstance(cfg, dict):
        custom = cfg.get("tts", {}).get("voice_names")
        raw_variants = cfg.get("tts", {}).get("cosyvoice", {}).get("synth_variants") or []
    else:
        custom = None
        raw_variants = cfg.tts.cosyvoice.synth_variants

    labels = {key: dict(value) for key, value in _DEFAULT_VOICE_NAMES.items()}
    if isinstance(custom, dict):
        labels.update(custom)
    if isinstance(raw_variants, dict):
        allowed: dict[str, list[str]] = {}
        for key, values in raw_variants.items():
            host = _norm_host_key(key)
            if host is not None and isinstance(values, (list, tuple)):
                allowed[host] = [str(value).strip() for value in values if str(value).strip()]
    elif raw_variants:
        allowed = {
            "host_a": [str(value).strip() for value in raw_variants],
            "host_b": [str(value).strip() for value in raw_variants],
        }
    else:
        allowed = {}

    for host, allowed_variants in allowed.items():
        if allowed_variants and host in labels:
            labels[host] = {
                variant: name for variant, name in labels[host].items()
                if variant in allowed_variants
            }
    return labels


def _safe_json(payload: Any) -> str:
    """内联 JSON：转义 </ 防止提前闭合 script；ensure_ascii=False 保留中文。"""
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


_PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <header class="site-header">
    <div class="header-inner">
      <div class="brand-group">
        <img src="./logo.png" alt="{title}" class="brand-logo-img">
        <div class="brand-text">
          <div class="brand-title">{title}<span class="brand-badge" id="issue-badge">—</span></div>
          <div class="brand-subtitle">每日 AI 情报刊</div>
        </div>
      </div>
      <nav class="header-nav">
        <a href="#section-hero" class="nav-anchor-link active">🎧 播客速览</a>
        <a href="#section-daily" class="nav-anchor-link">📰 科技日报</a>
        <a href="#section-sources" class="nav-anchor-link" id="nav-sources">🌐 来源速递</a>
        <a href="#section-radar" class="nav-anchor-link">📡 项目雷达</a>
      </nav>
      <div class="header-actions">
        <a href="./feed.xml" target="_blank" class="btn-header-pill btn-header-rss">📡 RSS 订阅</a>
      </div>
    </div>
  </header>

  <main class="app-layout">
    <section class="shared-date-axis">
      <div class="date-nav-controls">
        <button class="date-step-btn" id="date-prev">← 前一日</button>
        <button class="date-step-btn" id="date-next">后一日 →</button>
      </div>
      <div class="date-pills-carousel" id="date-pills"></div>
      <div class="date-archive-link" id="date-latest" hidden>回到最新一期</div>
    </section>

    <section id="section-hero" class="content-section">
      <div class="hero-chronicle-grid">
        <div class="podcast-hero-card" id="podcast-card">
          <div class="podcast-cover-meta">
            <div class="podcast-disc-cover"><div class="disc-wave-bars">
              <div class="disc-wave-bar"></div><div class="disc-wave-bar"></div>
              <div class="disc-wave-bar"></div><div class="disc-wave-bar"></div>
              <div class="disc-wave-bar"></div>
            </div></div>
            <div class="podcast-title-area">
              <div class="podcast-tag-row">
                <span class="podcast-tag">双主持对谈</span>
                <span class="podcast-duration" id="podcast-duration">—</span>
              </div>
              <h2 class="podcast-main-heading" id="podcast-title">AI 新闻快报</h2>
              <div class="podcast-subhead" id="podcast-subtitle"></div>
            </div>
          </div>
          <div class="host-control-box">
            <div class="host-box-header">
              <span>双主持音色设定</span>
              <span class="speaker-live-pill" id="live-speaker-badge"><span class="badge-dot">●</span> 待播放</span>
            </div>
            <div class="host-rows">
              <div class="host-voice-row">
                <div class="host-identity"><div class="host-avatar host-a">A</div><span class="host-name">苏晴</span></div>
                <div class="voice-segmented" id="host-a-voices"></div>
              </div>
              <div class="host-voice-row">
                <div class="host-identity"><div class="host-avatar host-b">B</div><span class="host-name">周航</span></div>
                <div class="voice-segmented" id="host-b-voices"></div>
              </div>
            </div>
          </div>
          <div class="podcast-cta-row">
            <button class="btn-play-primary" id="hero-play-btn">
              <span id="hero-play-icon">▶</span><span id="hero-play-text">立即收听本期</span>
            </button>
            <button class="btn-transcript-cta" id="hero-transcript-btn">📜 逐字稿</button>
          </div>
        </div>

        <div class="editorial-lead-card">
          <div class="editorial-header-stamp">
            <div class="stamp-label">✨ 今日总览导语</div>
            <div class="stamp-date" id="lead-date">—</div>
          </div>
          <div class="lead-quote-text" id="lead-text"></div>
          <div class="metrics-summary-strip" id="metrics-strip"></div>
          <div class="chapters-row" id="chapters-row"></div>
        </div>
      </div>
    </section>

    <section id="section-daily" class="content-section">
      <div class="daily-reading-layout">
        <div class="section-header-row">
          <div class="section-tag-title"><span>📰</span> 《科技日报》深度解读
            <span class="section-counter-badge" id="daily-counter"></span></div>
          <div class="section-desc-hint">黄金阅读行宽 760px · 结构化长篇研判</div>
        </div>
        <article class="daily-report-wrapper">
          <div class="daily-meta-bar">
            <div class="daily-meta-left">
              <span class="daily-pill">深度研判</span>
              <span class="daily-read-time" id="daily-meta"></span>
            </div>
            <div class="daily-toc-pills" id="daily-toc"></div>
          </div>
          <div id="daily-report-body"></div>
        </article>
      </div>
    </section>

    <section id="section-sources" class="content-section">
      <div class="section-header-row">
        <div class="section-tag-title"><span>🌐</span> 今日参考信源速览
          <span class="section-counter-badge" id="source-count-badge">—</span></div>
        <div class="section-desc-hint">按媒体分类聚合 · 支持实时检索与全文直达</div>
      </div>
      <div class="sources-controls-row">
        <div class="source-filter-pills" id="source-filters"></div>
        <div class="source-search-box">
          <span class="source-search-icon">🔍</span>
          <input type="text" class="source-search-input" id="source-search"
                 placeholder="检索标题或来源…">
        </div>
      </div>
      <div class="sources-grid-layout" id="sources-grid"></div>
      <div class="source-expand-bar" id="sources-expand-wrap" hidden>
        <button class="btn-expand-sources" id="btn-expand-sources">展开全部信源 ▾</button>
      </div>
    </section>

    <section id="section-radar" class="content-section">
      <div class="section-header-row">
        <div class="section-tag-title"><span>📡</span> GitHub 项目雷达
          <span class="section-counter-badge" id="radar-badge">—</span></div>
        <div class="section-desc-hint" id="radar-rules"></div>
      </div>
      <div class="radar-card-container" id="radar-root"></div>
    </section>
  </main>

  <footer class="docked-audio-player" id="docked-player">
    <div class="player-inner">
      <div class="player-left">
        <img src="./logo.png" alt="cover" class="player-thumb-img">
        <div class="player-track-info">
          <div class="player-track-title" id="player-title">AI 新闻快报</div>
          <div class="player-live-sentence" id="player-sentence">点击播放开始收听…</div>
        </div>
      </div>
      <div class="player-center">
        <div class="player-button-row">
          <button class="btn-player-skip" id="skip-back">⏪ 15s</button>
          <button class="btn-player-play" id="dock-play-btn"><span id="dock-play-icon">▶</span></button>
          <button class="btn-player-skip" id="skip-forward">15s ⏩</button>
        </div>
        <div class="player-scrubber-row">
          <span class="player-time" id="time-current">00:00</span>
          <div class="scrubber-track" id="scrubber-track"><div class="scrubber-progress" id="scrubber-fill"><div class="scrubber-thumb"></div></div></div>
          <span class="player-time" id="time-duration">00:00</span>
        </div>
      </div>
      <div class="player-right">
        <button class="btn-speed" id="speed-btn">1.0x</button>
        <div class="volume-box">
          <button class="btn-vol" id="mute-btn">🔊</button>
          <input type="range" class="vol-slider" id="vol-slider" min="0" max="1" step="0.05" value="0.8">
        </div>
        <button class="btn-transcript-drawer-toggle" id="btn-drawer-toggle">📜 逐字稿</button>
      </div>
    </div>
  </footer>

  <aside class="transcript-drawer" id="transcript-drawer">
    <div class="drawer-header">
      <div class="drawer-title"><span>🎙️</span> 对话逐字稿 <span class="brand-badge" id="drawer-count"></span></div>
      <button class="drawer-close-btn" id="drawer-close">✕</button>
    </div>
    <div class="drawer-filters">
      <div class="drawer-filter-group">
        <button class="btn-drawer-filter active" data-speaker="all">全部</button>
        <button class="btn-drawer-filter" data-speaker="A">只看苏晴</button>
        <button class="btn-drawer-filter" data-speaker="B">只看周航</button>
      </div>
      <span class="drawer-hint">点击任意句子快速跳转播放</span>
    </div>
    <div class="drawer-body" id="drawer-chunk-list"></div>
  </aside>

  <div class="toast-pill" id="toast-pill"></div>
  <audio id="main-audio" preload="metadata"></audio>

  <script type="application/json" id="site-data">{site_data}</script>
  <script>{app_js}</script>
  <script>{player_js}</script>
</body>
</html>
"""


def build_index_html(
    site_dir: Path,
    podcast_title: str,
    episodes: list[dict[str, Any]],
    base_url: str,
    cfg: dict[str, Any] | AppConfig | None = None,
) -> None:
    dates_list: list[str] = []
    episodes_map: dict[str, dict[str, str]] = {}
    for episode in episodes:
        ep_id = str(episode.get("id") or episode.get("guid") or "").rsplit("/", 1)[-1]
        ep_id = ep_id.replace(".mp3", "")
        if not ep_id or ep_id in episodes_map:
            continue
        dates_list.append(ep_id)
        episodes_map[ep_id] = {
            "title": str(episode.get("title") or f"AI 新闻快报 | {ep_id}"),
            "mp3": str(episode.get("enclosure_url") or f"{base_url}/episodes/{ep_id}.mp3"),
        }
    dates_list.sort(reverse=True)

    site_data = {
        "dates": dates_list,
        "episodes": episodes_map,
        "voices": _voice_labels(cfg),
    }
    html = _PAGE_TEMPLATE
    for placeholder, value in (
        ("{title}", str(podcast_title)),
        ("{css}", _read_static("style.css")),
        ("{app_js}", _read_static("app.js")),
        ("{player_js}", _read_static("player.js")),
        ("{site_data}", _safe_json(site_data)),
    ):
        html = html.replace(placeholder, value)
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "index.html").write_text(html, encoding="utf-8")
```

说明：
- `_voice_labels` / `_norm_host_key` 原样保留（测试与 digest 复用）。
- `format_friendly_date` 生产零引用（`02 §4.7`），**删除**并同步删旧测试。
- Apple Podcasts 入口 v1 不做（D15）：头部只有 RSS；等 Podcasts Connect 上架后再加，属于 backlog（Task 11 记录）。
- 若 `cfg` 为 `None`（旧测试调用），`_voice_labels(None)` 返回默认标签。

- [ ] **Step 4: 建临时静态文件占位以跑通生成（本步只为让测试可运行）**

```bash
# 注意：Task 7/8/9 会用真实内容替换；此处先写最小可用文件，避免 import/生成失败。
printf '/* placeholder: replaced in Task 7 */\n' > src/ai_news_podcast/presentation/site_builder/static/style.css
printf '/* placeholder: replaced in Task 8 */\n' > src/ai_news_podcast/presentation/site_builder/static/app.js
printf '/* placeholder: replaced in Task 9 */\n' > src/ai_news_podcast/presentation/site_builder/static/player.js
```

- [ ] **Step 5: 跑测试**

Run: `uv run pytest tests/test_html_gen_view.py -q`
Expected: PASS（约 12 passed）

- [ ] **Step 6: 提交**

```bash
git add src/ai_news_podcast/presentation/site_builder/html_gen_view.py \
  src/ai_news_podcast/presentation/site_builder/static/ tests/test_html_gen_view.py
git commit -m "feat(site): new long-scroll page skeleton with inline site-data"
```

---

### Task 7: 设计系统 CSS（从原型移植 + 响应式）

**Files:**
- Rewrite: `src/ai_news_podcast/presentation/site_builder/static/style.css`

- [ ] **Step 1: 抽取原型 CSS 作为基线**

```bash
python3 - <<'PY'
from pathlib import Path

src = Path("design/2026-10-redesign/directions/A/index.html").read_text(encoding="utf-8")
start = src.index("<style>") + len("<style>")
end = src.index("</style>")
css = src[start:end].strip("\n")
header = (
    "/* AI 每日先锋 · 站点主题（方向 A：每日刊）\n"
    "   来源：design/2026-10-redesign/directions/A/index.html 的 <style> 块（移植基线）。\n"
    "   维护约定：设计令牌在 :root；桌面布局在前；响应式断点在文件末尾（1024/720）。*/\n\n"
)
Path("src/ai_news_podcast/presentation/site_builder/static/style.css").write_text(
    header + css + "\n", encoding="utf-8"
)
print("css bytes:", len(css))
PY
```
Expected: `css bytes: 约 40KB`；`uv run ruff check` 不涉及 CSS。

- [ ] **Step 2: 删除评审/模拟专用样式块**

在 `static/style.css` 中删除以下选择器**整块**（含子选择器）：
- `.mobile-modal-overlay`、`.mobile-frame-*`（原型评审 iframe 预览）
- `#radar-hit-box` 的「模拟命中态切换」相关按钮样式 `.btn-radar-mock-toggle`
- `.radar-candidates-table-wrap`、`.candidates-table`（v1 不做候选表，见 D10；v1.5 再加回）

校验：`grep -n "mobile-modal\|mock-toggle\|candidates-table" static/style.css` → 无输出。

- [ ] **Step 3: 补齐 JS 会用到的状态类**

在文件末尾（响应式断点**之前**）追加：

```css
/* ---------- 交互状态与加载/空态 ---------- */
[hidden] { display: none !important; }
.date-step-btn:disabled { opacity: .4; cursor: not-allowed; }
.is-loading { opacity: .55; pointer-events: none; }
.state-placeholder {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 10px; padding: 40px 16px; color: var(--text-muted); text-align: center;
}
.state-placeholder .state-icon { font-size: 1.6rem; }
.source-card.hidden, .transcript-chunk-row.hidden { display: none; }
.brand-badge[hidden], .btn-header-pill[hidden] { display: none; }
.scrubber-track { cursor: pointer; }
/* ---------- 雷达命中卡（app.js 动态渲染所需的补充样式） ---------- */
.radar-hit-title {
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px; flex-wrap: wrap;
}
.radar-hit-title h3 {
  font-size: 1.25rem; font-weight: 800; color: #fff; font-family: var(--font-mono);
}
.radar-stars { font-family: var(--font-mono); font-weight: 700; color: var(--accent-emerald); }
.radar-hit-meta {
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  font-size: .8rem; color: var(--text-muted);
}
.radar-hit-meta a { color: var(--accent-light); }
.disc-wave-bars { animation-play-state: paused; }
#podcast-card.is-playing .disc-wave-bars { animation-play-state: running; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .001ms !important; transition-duration: .001ms !important; }
}
```

- [ ] **Step 4: 追加响应式断点（替代 mobile.html）**

在文件末尾追加：

```css
/* ---------- 响应式：1024（平板/小桌面） ---------- */
@media (max-width: 1024px) {
  .app-layout { padding: 0 20px; }
  .hero-chronicle-grid { grid-template-columns: 1fr; }
  .metrics-summary-strip { grid-template-columns: repeat(2, 1fr); }
  .sources-grid-layout { grid-template-columns: repeat(2, 1fr); }
  .player-inner { grid-template-columns: 1fr auto; }
  .player-right .volume-box { display: none; }
}

/* ---------- 响应式：720（手机 390 目标） ---------- */
@media (max-width: 720px) {
  body { padding-bottom: 76px; }
  .app-layout { padding: 0 16px; }
  .header-inner { padding: 10px 16px; gap: 8px; }
  .header-nav { display: none; }               /* 移动端用板块纵向流，不显示锚点导航 */
  .brand-subtitle { display: none; }
  .brand-title { font-size: 1rem; }
  .shared-date-axis { gap: 8px; padding: 10px 0; }
  .date-pills-carousel {
    overflow-x: auto; flex-wrap: nowrap; scrollbar-width: none;
    -webkit-overflow-scrolling: touch;
  }
  .date-pills-carousel::-webkit-scrollbar { display: none; }
  .date-pill { flex: 0 0 auto; padding: 6px 10px; font-size: .74rem; }
  .date-nav-controls .date-step-btn { padding: 6px 8px; font-size: .74rem; }
  .date-archive-link { display: none; }

  .section-header-row { flex-direction: column; align-items: flex-start; gap: 4px; }
  .podcast-hero-card { padding: 16px; }
  .podcast-main-heading { font-size: 1.1rem; }
  .editorial-lead-card { padding: 16px; }
  .lead-quote-text { font-size: .92rem; line-height: 1.68; }
  .daily-reading-layout { max-width: 100%; }
  .daily-report-wrapper { padding: 20px 18px; }
  .daily-article-title { font-size: 1.15rem; }
  .daily-toc-pills { display: none; }
  .deep-dive-card { padding: 14px; }
  .sources-controls-row { flex-direction: column; align-items: stretch; gap: 8px; }
  .source-search-box { width: 100%; }
  .sources-grid-layout { grid-template-columns: 1fr; }
  .radar-empty-state-view { flex-direction: column; text-align: center; }

  /* 吸底播放器 → 64px 迷你条：只留封面 + 当前句 + 播放键 */
  .docked-audio-player { height: 64px; }
  .player-inner {
    grid-template-columns: 1fr auto; padding: 0 12px; height: 64px; gap: 8px;
  }
  .player-left .player-thumb-img { width: 40px; height: 40px; }
  .player-track-title { font-size: .78rem; }
  .player-live-sentence { font-size: .7rem; max-width: 62vw; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }
  .player-center .player-scrubber-row { display: none; }
  .player-center .btn-player-skip { display: none; }
  .player-right .btn-speed, .player-right .volume-box { display: none; }
  .player-right .btn-transcript-drawer-toggle { padding: 6px 8px; font-size: .7rem; }

  /* 逐字稿抽屉 → 全屏 bottom sheet */
  .transcript-drawer {
    inset: auto 0 0 0; width: auto; height: 78vh; border-radius: 18px 18px 0 0;
    transform: translateY(100%);
  }
  .transcript-drawer.open { transform: translateY(0); }
  .drawer-body { padding-bottom: 24px; }
}
```

- [ ] **Step 5: 校验无重复/无残留**

Run:
```bash
grep -c "@media" src/ai_news_podcast/presentation/site_builder/static/style.css   # 期望 2
grep -c "mobile-modal\|mock-toggle\|candidates-table" src/ai_news_podcast/presentation/site_builder/static/style.css  # 期望 0
grep -c "var(--bg-canvas)" src/ai_news_podcast/presentation/site_builder/static/style.css  # >0，证明 token 已带上
```
Expected: `2` / `0` / 正数。

- [ ] **Step 6: 提交**

```bash
git add src/ai_news_podcast/presentation/site_builder/static/style.css
git commit -m "feat(site): port direction-A design tokens and responsive layout"
```

---

### Task 8: 页面渲染与交互（app.js）

**Files:**
- Rewrite: `src/ai_news_podcast/presentation/site_builder/static/app.js`

依赖约定：`player.js` 先加载并暴露 `window.PodcastEngine`；`app.js` 后加载，负责一切 DOM 渲染与事件。

- [ ] **Step 1: 实现 app.js（完整实现，可直接落盘）**

```javascript
(function () {
  'use strict';

  var SITE = JSON.parse(document.getElementById('site-data').textContent);
  var DATES = SITE.dates || [];
  var EPISODES = SITE.episodes || {};
  var VOICES = SITE.voices || {};

  var state = {
    date: null,
    digest: null,
    chunks: [],
    sourceGroup: 'all',
    sourceQuery: '',
    sourcesExpanded: false,
    drawerFilter: 'all',
    player: null,
  };

  var el = function (id) { return document.getElementById(id); };
  var HOST_NAMES = { A: '苏晴', B: '周航' };
  var SOURCE_PREVIEW = 12;
  var radarLoop = { running: false, raf: 0, reduceMotion: false };

  function showToast(message) {
    var toast = el('toast-pill');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(function () { toast.classList.remove('show'); }, 2200);
  }

  function fmtTime(seconds) {
    var total = Math.max(0, Math.floor(seconds || 0));
    var mm = String(Math.floor(total / 60)).padStart(2, '0');
    var ss = String(total % 60).padStart(2, '0');
    return mm + ':' + ss;
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function placeholder(icon, text) {
    return '<div class="state-placeholder"><div class="state-icon">' + icon + '</div><div>' +
      escapeHtml(text) + '</div></div>';
  }

  async function fetchJson(url) {
    var response = await fetch(url, { cache: 'no-cache' });
    if (!response.ok) throw new Error(url + ' ' + response.status);
    return response.json();
  }

  /* ---------- 日期轴 ---------- */
  function renderDateAxis() {
    var wrap = el('date-pills');
    wrap.innerHTML = '';
    var calendar = state.calendar || {};
    var byDate = {};
    (calendar.dates || []).forEach(function (item) { byDate[item.date] = item; });
    DATES.forEach(function (date) {
      var meta = byDate[date] || {};
      var issue = meta.issue_number ? ' · 第 ' + meta.issue_number + ' 期' : '';
      var pill = document.createElement('button');
      pill.className = 'date-pill' + (date === state.date ? ' active' : '');
      pill.setAttribute('data-date', date);
      pill.textContent = (date === DATES[0] ? date + ' (最新' + issue + ')' : date + issue);
      pill.addEventListener('click', function () { loadDate(date); });
      wrap.appendChild(pill);
    });
    var index = DATES.indexOf(state.date);
    el('date-prev').disabled = index < 0 || index >= DATES.length - 1;
    el('date-next').disabled = index <= 0;
    el('date-latest').hidden = index <= 0;
    document.querySelectorAll('.nav-anchor-link').forEach(function (link) {
      link.classList.toggle('active', link.getAttribute('href') === '#section-hero');
    });
  }

  /* ---------- 首屏静态区 ---------- */
  function renderHero() {
    var digest = state.digest;
    var podcast = digest.podcast || {};
    el('issue-badge').textContent = digest.issue && digest.issue.label
      ? digest.date + ' · ' + digest.issue.label
      : digest.date;
    el('podcast-title').textContent = podcast.title || ('AI 新闻快报 | ' + digest.date);
    el('podcast-subtitle').textContent = podcast.subtitle || '';
    el('podcast-duration').textContent = [
      podcast.duration_label || '--:--',
      podcast.segment_count ? podcast.segment_count + ' 段逐字稿' : '',
    ].filter(Boolean).join(' · ');
    el('player-title').textContent = podcast.title || '';
    el('lead-date').textContent = digest.date_label + ' · 编辑部按';
    el('lead-text').innerHTML = digest.lead ? escapeHtml(digest.lead.text) : placeholder('✨', '本期暂无导语');
    el('nav-sources').textContent = '🌐 来源速递 (' + ((digest.sources || {}).shown || 0) + ')';
    renderMetrics(digest.metrics || []);
    renderChapters(digest.chapters || []);
    renderVoiceButtons((podcast.hosts || []));
  }

  function renderMetrics(metrics) {
    el('metrics-strip').innerHTML = metrics.map(function (metric) {
      return '<div class="metric-item"><div class="metric-value">' + escapeHtml(metric.value) +
        '<span class="metric-unit">' + escapeHtml(metric.unit) + '</span></div>' +
        '<div class="metric-label">' + escapeHtml(metric.label) + '</div></div>';
    }).join('');
  }

  function renderChapters(chapters) {
    var wrap = el('chapters-row');
    if (!chapters.length) { wrap.innerHTML = ''; wrap.hidden = true; return; }
    wrap.hidden = false;
    wrap.innerHTML = '<span class="chapters-title">快速定位章节：</span>' + chapters.map(function (chapter) {
      return '<button class="chapter-chip" data-start="' + chapter.start + '">' +
        '<span>🎧</span> ' + escapeHtml(chapter.title) +
        ' <span class="chapter-time">' + escapeHtml(chapter.label) + '</span></button>';
    }).join('');
    wrap.querySelectorAll('.chapter-chip').forEach(function (chip) {
      chip.addEventListener('click', function () {
        state.player.seek(Number(chip.getAttribute('data-start')) + 0.01, true);
        showToast('已跳转至 ' + fmtTime(Number(chip.getAttribute('data-start'))));
      });
    });
  }

  function renderVoiceButtons(hosts) {
    var engine = state.player;
    hosts.forEach(function (host) {
      var container = el(host.key === 'A' ? 'host-a-voices' : 'host-b-voices');
      container.innerHTML = '';
      var selected = engine ? engine.voiceOf(host.key) : null;
      (host.variants || []).forEach(function (variant, index) {
        var isActive = selected ? selected === variant.id : index === 0;
        var button = document.createElement('button');
        button.className = 'voice-btn host-' + host.key.toLowerCase() + '-btn' +
          (isActive ? ' active' : '');
        button.textContent = variant.label;
        button.addEventListener('click', function () {
          engine.setVoice(host.key, variant.id);
          renderVoiceButtons(hosts);
          showToast('已切换 ' + host.name + ' 音色为：' + variant.label);
        });
        container.appendChild(button);
      });
      if (!host.variants || !host.variants.length) {
        container.innerHTML = placeholder('🎙️', '该主持人暂无可选音色');
      }
    });
  }

  /* ---------- 日报 ---------- */
  function renderReport(report) {
    var body = el('daily-report-body');
    var toc = el('daily-toc');
    if (!report) {
      body.innerHTML = placeholder('📰', '该日期暂无深度日报');
      toc.innerHTML = '';
      el('daily-meta').textContent = '';
      el('daily-counter').textContent = '';
      return;
    }
    el('daily-counter').textContent = '全文 ' + report.char_count + ' 字';
    el('daily-meta').textContent = '阅读时间：约 ' + report.reading_minutes + ' 分钟 · ' +
      (state.digest.date_label || '');
    var html = '';
    html += '<h1 class="daily-article-title">' + escapeHtml(report.title || '') + '</h1>';
    (report.deep_dives || []).forEach(function (dive, index) {
      html += '<div class="article-section-block" id="deep-dive-' + (index + 1) + '">' +
        '<div class="deep-dive-card">' +
        '<h3 class="deep-dive-title">' + escapeHtml(dive.heading) + '</h3>' +
        pointBlock('📌 事件背景', dive.background) +
        pointBlock('⚡ 核心突破点', dive.breakthrough) +
        pointBlock('🌐 对行业与用户的深远影响', dive.impact) +
        (dive.raw_sections || []).map(function (text) {
          return '<div class="deep-dive-point">' + escapeHtml(text) + '</div>';
        }).join('') +
        '</div></div>';
    });
    if ((report.frontier || []).length) {
      html += '<div class="article-section-block"><h2 class="article-section-header">⚡ 前沿情报</h2>' +
        '<div class="frontier-list">' + report.frontier.map(function (item) {
          return '<div class="frontier-item"><span class="frontier-bullet">🔹</span><div>' +
            '<strong>' + escapeHtml(item.title) + '</strong>：' + escapeHtml(item.summary) +
            '</div></div>';
        }).join('') + '</div></div>';
    }
    if (report.editor_verdict) {
      html += '<div class="article-section-block"><div class="editor-quote-card">' +
        '<div class="editor-quote-header"><span>💡</span> AI 小编深度点评</div>' +
        '<div class="editor-quote-body">' + escapeHtml(report.editor_verdict) + '</div>' +
        '</div></div>';
    }
    body.innerHTML = html;
    toc.innerHTML = (report.toc || []).map(function (item) {
      return '<a href="#' + escapeHtml(item.anchor) + '" class="daily-toc-link">' +
        escapeHtml(item.title.length > 12 ? item.title.slice(0, 12) + '…' : item.title) + '</a>';
    }).join('');
  }

  function pointBlock(label, text) {
    if (!text) return '';
    return '<div class="deep-dive-point"><strong>' + label + '：</strong>' +
      escapeHtml(text) + '</div>';
  }

  /* ---------- 来源 ---------- */
  function filteredSources() {
    var block = state.digest.sources || { items: [] };
    var query = state.sourceQuery.trim().toLowerCase();
    return (block.items || []).filter(function (item) {
      var groupOk = state.sourceGroup === 'all' || item.group === state.sourceGroup;
      var queryOk = !query ||
        (item.title || '').toLowerCase().indexOf(query) >= 0 ||
        (item.publisher || '').toLowerCase().indexOf(query) >= 0;
      return groupOk && queryOk;
    });
  }

  function renderSources() {
    var block = state.digest.sources || { items: [], groups: [] };
    if (!(block.items || []).length) {
      el('source-count-badge').textContent = '—';
      el('sources-grid').innerHTML = placeholder('🌐', '该日期暂无来源数据');
      el('sources-expand-wrap').hidden = true;
      el('source-filters').innerHTML = '';
      return;
    }
    el('source-count-badge').textContent = block.shown + ' 条条目已收录';
    el('source-filters').innerHTML =
      '<button class="source-filter-btn' + (state.sourceGroup === 'all' ? ' active' : '') +
      '" data-group="all">全部 (' + block.shown + ')</button>' +
      (block.groups || []).map(function (group) {
        return '<button class="source-filter-btn' +
          (state.sourceGroup === group.group ? ' active' : '') +
          '" data-group="' + escapeHtml(group.group) + '">' +
          escapeHtml(group.group) + ' (' + group.count + ')</button>';
      }).join('');
    el('source-filters').querySelectorAll('.source-filter-btn').forEach(function (button) {
      button.addEventListener('click', function () {
        state.sourceGroup = button.getAttribute('data-group');
        state.sourcesExpanded = false;
        renderSources();
      });
    });

    var items = filteredSources();
    var visible = state.sourcesExpanded ? items : items.slice(0, SOURCE_PREVIEW);
    el('sources-grid').innerHTML = visible.map(function (item) {
      return '<div class="source-card">' +
        '<div class="source-card-top"><span class="source-badge">' +
        escapeHtml(item.publisher) + '</span><span class="source-role-pill">' +
        escapeHtml(item.role_label) + '</span></div>' +
        '<a href="' + escapeHtml(item.link) + '" target="_blank" rel="noopener" ' +
        'class="source-title-link" title="' + escapeHtml(item.title) + '">' +
        escapeHtml(item.title) + '</a>' +
        '<div class="source-card-footer"><span>直达原文 ↗</span></div></div>';
    }).join('');
    var canExpand = items.length > SOURCE_PREVIEW;
    el('sources-expand-wrap').hidden = !canExpand;
    if (canExpand) {
      el('btn-expand-sources').textContent = state.sourcesExpanded
        ? '收起部分信源 ▴' : '展开全部 ' + items.length + ' 条信源 ▾';
    }
  }

  /* ---------- 雷达 ---------- */
  function renderRadar() {
    var radar = state.digest.radar || {};
    var root = el('radar-root');
    el('radar-badge').textContent = radar.available
      ? (radar.status === 'pick' ? '主推 ' + radar.pick.repo : '本期无主推项目')
      : '该期无雷达';
    if (radar.rules) {
      el('radar-rules').textContent = '门槛：≥ ' + radar.rules.min_stars + ' 星 · 近 ' +
        radar.rules.created_window_days + ' 天创建 · ' + radar.rules.recent_push_days + ' 天内活跃';
    }
    if (!radar.available) {
      root.innerHTML = placeholder('📡', '该日期暂无项目雷达');
      return;
    }
    if (radar.status === 'pick' && radar.pick) {
      var pick = radar.pick;
      var delta = pick.delta_stars == null ? '' : ' (+' + pick.delta_stars + '/日)';
      root.innerHTML = '<div class="radar-hit-view active"><div class="radar-hit-header">' +
        '<div class="radar-status-badge">● 今日主推开源项目 · 深度挖掘</div></div>' +
        '<div class="radar-hit-card"><div class="radar-hit-title">' +
        '<h3>' + escapeHtml(pick.repo) + '</h3>' +
        '<span class="source-badge">' + escapeHtml(pick.language || '—') + '</span>' +
        '<span class="radar-stars">★ ' + escapeHtml(pick.stars) + delta + '</span></div>' +
        '<p>' + escapeHtml(pick.readme_intro || pick.description) + '</p>' +
        '<div class="radar-hit-meta"><span>许可证：' + escapeHtml(pick.license || '—') +
        '</span><a href="' + escapeHtml(pick.url) + '" target="_blank" rel="noopener">' +
        '在 GitHub 上查阅仓库 ↗</a></div></div></div>';
      return;
    }
    var candidates = radar.candidates || {};
    var message = candidates.total
      ? '今日全网共扫描 <strong>' + candidates.total + ' 个候选仓库</strong>，' +
        (candidates.excluded_recent ? '其中 <strong>' + candidates.excluded_recent +
          ' 个</strong>为近 30 天已推荐或重复收录，' : '') +
        '经工程稳定性与代码审查，编辑部决定今日<strong>宁缺毋滥、轮空主推</strong>。'
      : '今日候选项不足，编辑部决定<strong>轮空主推</strong>。';
    root.innerHTML = '<div class="radar-empty-state-view"><div class="radar-visual-box">' +
      '<canvas id="radarCanvas" class="radar-sweep-canvas" width="180" height="180"></canvas>' +
      '</div><div class="radar-empty-content">' +
      '<span class="radar-status-badge"><span>●</span> 今日雷达状态：暂无主推项目</span>' +
      '<h3 class="radar-empty-headline">今日候选池经多轮筛查，未检出达标项目</h3>' +
      '<p class="radar-empty-paragraph">「AI 每日先锋」每日严选 1 个具备重大架构革新且近期 ' +
      'Star 增速稳定的开源项目。' + message + '</p>' +
      '</div></div>';
    startRadarSweep();
  }

  function startRadarSweep() {
    var canvas = el('radarCanvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var angle = 0;
    cancelAnimationFrame(radarLoop.raf);
    radarLoop.running = true;
    function draw() {
      if (!radarLoop.running) return;
      var w = canvas.width, h = canvas.height, cx = w / 2, cy = h / 2, r = cx - 6;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = 'rgba(56,189,248,0.18)';
      ctx.lineWidth = 1;
      for (var rad = r / 4; rad <= r; rad += r / 4) {
        ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2); ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();
      ctx.save(); ctx.translate(cx, cy); ctx.rotate(angle);
      var grad = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
      grad.addColorStop(0, 'rgba(56,189,248,0.4)');
      grad.addColorStop(1, 'rgba(56,189,248,0)');
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.arc(0, 0, r, -0.4, 0); ctx.closePath();
      ctx.fillStyle = grad; ctx.fill();
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(r, 0);
      ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2; ctx.stroke();
      ctx.restore();
      angle += 0.035;
      if (!radarLoop.reduceMotion) radarLoop.raf = requestAnimationFrame(draw);
    }
    draw();
  }

  /* ---------- 逐字稿抽屉 ---------- */
  function renderTranscript() {
    var drawer = el('drawer-chunk-list');
    if (!state.chunks.length) {
      drawer.innerHTML = placeholder('🎙️', '该日期暂无逐字稿');
      el('drawer-count').textContent = '';
      return;
    }
    el('drawer-count').textContent = state.chunks.length + ' 句对谈';
    var visible = state.chunks.filter(function (chunk) {
      return state.drawerFilter === 'all' || chunk.host === state.drawerFilter;
    });
    drawer.innerHTML = visible.map(function (chunk) {
      return '<div class="transcript-chunk-row" data-index="' +
        state.chunks.indexOf(chunk) + '">' +
        '<div class="chunk-header"><span class="chunk-speaker-badge host-' +
        (chunk.host === 'A' ? 'a' : 'b') + '">● ' + HOST_NAMES[chunk.host] +
        '</span><span class="chunk-time">' + fmtTime(chunk.start) + '</span></div>' +
        '<div class="chunk-text">' + escapeHtml(chunk.text) + '</div></div>';
    }).join('');
    drawer.querySelectorAll('.transcript-chunk-row').forEach(function (row) {
      row.addEventListener('click', function () {
        var chunk = state.chunks[Number(row.getAttribute('data-index'))];
        if (!chunk) return;
        state.player.seek(chunk.start + 0.01, true);
        highlightChunk(state.chunks.indexOf(chunk));
      });
    });
  }

  function highlightChunk(index) {
    document.querySelectorAll('.transcript-chunk-row').forEach(function (row) {
      row.classList.toggle('active', Number(row.getAttribute('data-index')) === index);
    });
    var chunk = state.chunks[index];
    if (!chunk) return;
    el('player-sentence').textContent = HOST_NAMES[chunk.host] + '：' + chunk.text;
    var badge = el('live-speaker-badge');
    badge.classList.toggle('host-b-speaking', chunk.host === 'B');
    badge.innerHTML = '<span class="badge-dot">●</span> 正在发言：' + HOST_NAMES[chunk.host];
  }

  function toggleDrawer(open) {
    var drawer = el('transcript-drawer');
    var shouldOpen = open === undefined ? !drawer.classList.contains('open') : open;
    drawer.classList.toggle('open', shouldOpen);
    el('btn-drawer-toggle').classList.toggle('active', shouldOpen);
  }

  /* ---------- 播放器 UI 绑定 ---------- */
  function bindPlayer() {
    var engine = state.player;
    engine.onState(function (playing) {
      var icon = playing ? '⏸' : '▶';
      el('hero-play-icon').textContent = icon;
      el('dock-play-icon').textContent = icon;
      el('hero-play-text').textContent = playing ? '暂停播放' : '立即收听本期';
      el('podcast-card').classList.toggle('is-playing', playing);
    });
    engine.onTime(function (current, duration) {
      el('time-current').textContent = fmtTime(current);
      el('time-duration').textContent = fmtTime(duration);
      el('scrubber-fill').style.width = (duration ? (current / duration) * 100 : 0) + '%';
      highlightChunk(engine.index);
    });
    document.querySelectorAll('#hero-play-btn, #dock-play-btn').forEach(function (button) {
      button.addEventListener('click', function () { engine.toggle(); });
    });
    el('skip-back').addEventListener('click', function () { engine.skip(-15); });
    el('skip-forward').addEventListener('click', function () { engine.skip(15); });
    el('scrubber-track').addEventListener('click', function (event) {
      var rect = event.currentTarget.getBoundingClientRect();
      engine.seek(((event.clientX - rect.left) / rect.width) * engine.duration, false);
    });
    el('speed-btn').addEventListener('click', function () {
      var label = engine.cycleRate();
      el('speed-btn').textContent = label;
      showToast('播放倍速：' + label);
    });
    el('vol-slider').addEventListener('input', function (event) {
      engine.setVolume(Number(event.target.value));
      el('mute-btn').textContent = Number(event.target.value) === 0 ? '🔇' : '🔊';
    });
    el('mute-btn').addEventListener('click', function () {
      var muted = engine.toggleMute();
      el('mute-btn').textContent = muted ? '🔇' : '🔊';
      el('vol-slider').value = muted ? 0 : engine.volume;
    });
    el('btn-drawer-toggle').addEventListener('click', function () { toggleDrawer(); });
    el('hero-transcript-btn').addEventListener('click', function () { toggleDrawer(true); });
    el('drawer-close').addEventListener('click', function () { toggleDrawer(false); });
    document.querySelectorAll('.btn-drawer-filter').forEach(function (button) {
      button.addEventListener('click', function () {
        document.querySelectorAll('.btn-drawer-filter').forEach(function (other) {
          other.classList.remove('active');
        });
        button.classList.add('active');
        state.drawerFilter = button.getAttribute('data-speaker');
        renderTranscript();
      });
    });
    el('source-search').addEventListener('input', function (event) {
      state.sourceQuery = event.target.value;
      state.sourcesExpanded = false;
      renderSources();
    });
    el('btn-expand-sources').addEventListener('click', function () {
      state.sourcesExpanded = !state.sourcesExpanded;
      renderSources();
    });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') toggleDrawer(false);
    });
    el('date-prev').addEventListener('click', function () { stepDate(1); });
    el('date-next').addEventListener('click', function () { stepDate(-1); });
    el('date-latest').addEventListener('click', function () { loadDate(DATES[0]); });
  }

  function stepDate(direction) {
    var index = DATES.indexOf(state.date);
    var next = DATES[index + direction];
    if (next) loadDate(next);
  }

  /* ---------- 日期加载主流程 ---------- */
  async function loadDate(date) {
    state.date = date;
    renderDateAxis();
    var body = el('daily-report-body');
    body.classList.add('is-loading');
    try {
      var results = await Promise.all([
        fetchJson('./data/digest_' + date + '.json'),
        fetchJson('./episodes/' + date + '/playlist.json').catch(function () { return { chunks: [] }; }),
      ]);
      state.digest = results[0];
      state.chunks = (results[1].chunks || []);
    } catch (error) {
      console.warn('digest load failed', date, error);
      state.digest = {
        schema_version: 1, date: date, date_label: date,
        issue: { number: null, label: null, source: 'unavailable' },
        availability: {}, lead: null, metrics: [], chapters: [],
        report: null, sources: { items: [], groups: [], shown: 0 },
        radar: { available: false, status: 'unavailable', rules: null,
                 pick: null, runner_ups: [], candidates: {}, message: '' },
        podcast: { title: 'AI 新闻快报 | ' + date, subtitle: null, duration_label: '--:--',
                   segment_count: 0, hosts: [] },
      };
      state.chunks = [];
      showToast('该日期数据暂不可用');
    } finally {
      body.classList.remove('is-loading');
    }
    renderHero();
    renderReport(state.digest.report);
    renderSources();
    renderRadar();
    renderTranscript();
    state.player.load(state.chunks, date);
    ((state.digest.podcast || {}).hosts || []).forEach(function (host) {
      if (!state.player.voiceOf(host.key) && (host.variants || []).length) {
        state.player.setVoice(host.key, host.variants[0].id);
      }
    });
    renderVoiceButtons(((state.digest.podcast || {}).hosts || []));
    el('player-sentence').textContent = '点击播放开始收听…';
    el('live-speaker-badge').innerHTML = '<span class="badge-dot">●</span> 待播放';
  }

  async function boot() {
    state.player = new window.PodcastEngine(el('main-audio'));
    bindPlayer();
    radarLoop.reduceMotion = !!(window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) {
        radarLoop.running = false;
        cancelAnimationFrame(radarLoop.raf);
      } else if (el('radarCanvas') && !radarLoop.reduceMotion) {
        startRadarSweep();
      }
    });
    try {
      state.calendar = await fetchJson('./data/calendar.json');
    } catch (error) {
      state.calendar = { dates: DATES.map(function (date) {
        return { date: date, issue_number: null };
      }) };
    }
    var initial = DATES.indexOf(location.hash.replace('#', '')) >= 0
      ? location.hash.replace('#', '') : DATES[0];
    if (!initial) {
      el('date-pills').innerHTML = '';
      el('daily-report-body').innerHTML = placeholder('📰', '暂无任何已发布期数');
      return;
    }
    await loadDate(initial);
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
```

- [ ] **Step 2: 校验语法（Node 可用时）**

Run:
```bash
node --check src/ai_news_podcast/presentation/site_builder/static/app.js && echo "app.js syntax ok"
```
Expected: `app.js syntax ok`（无 node 时跳到 Step 3 的浏览器验证，并在 PR 里注明）。

- [ ] **Step 3: 提交**

```bash
git add src/ai_news_podcast/presentation/site_builder/static/app.js
git commit -m "feat(site): render sections, date axis, sources, radar and drawer"
```

---

### Task 9: 播放引擎（player.js 重写）

**Files:**
- Rewrite: `src/ai_news_podcast/presentation/site_builder/static/player.js`

引擎行为契约（从旧 `player.js:764-1294` 移植，不许丢）：
1. 逐 chunk 播放，`chunk.audios[variant] → chunk.audios.professional → 首个可用键 → chunk.audio → chunk_NNN.mp3` 回退链；
2. 全局时间轴 = `chunk.start + audio.currentTime`，跨 chunk 的 ±15s / 进度条拖拽按全局时间换算；
3. 双主持音色：切换后从当前句起生效（`setVoice` 后重载当前 chunk）；
4. 倍速 1.0/1.25/1.5/1.75/2.0 循环；音量/静音；句切换回调；播放结束自动下一句；
5. 不碰 DOM（DOM 全部由 app.js 更新，通过回调）。

- [ ] **Step 1: 实现 player.js**

```javascript
(function () {
  'use strict';

  var SPEEDS = [1.0, 1.25, 1.5, 1.75, 2.0];
  var DEFAULT_VOLUME = 0.8;

  class PodcastEngine {
    constructor(audio) {
      this.audio = audio;
      this.chunks = [];
      this.index = 0;
      this.voices = { A: null, B: null };
      this.rateIndex = 0;
      this.volume = DEFAULT_VOLUME;
      this.lastVolume = DEFAULT_VOLUME;
      this._handlers = { state: [], time: [], duration: [] };
      audio.volume = this.volume;
      audio.addEventListener('play', () => this._emitState(true));
      audio.addEventListener('pause', () => this._emitState(false));
      audio.addEventListener('timeupdate', () => this._emitTime());
      audio.addEventListener('loadedmetadata', () => this._emitTime());
      audio.addEventListener('ended', () => this._onChunkEnded());
    }

    /* ---------- 事件订阅 ---------- */
    onState(handler) { this._handlers.state.push(handler); }
    onTime(handler) { this._handlers.time.push(handler); }
    onDuration(handler) { this._handlers.duration.push(handler); }
    _emitState(playing) { this._handlers.state.forEach(function (fn) { fn(playing); }); }
    _emitTime() {
      var current = this.currentTime;
      var duration = this.duration;
      this._handlers.time.forEach(function (fn) { fn(current, duration); });
    }

    /* ---------- 状态 ---------- */
    get duration() {
      if (!this.chunks.length) return 0;
      var last = this.chunks[this.chunks.length - 1];
      return Number(last.start || 0) + Number(last.duration || 0);
    }

    get currentTime() {
      var chunk = this.chunks[this.index];
      if (!chunk) return 0;
      return Number(chunk.start || 0) + (this.audio.currentTime || 0);
    }

    get playing() { return !this.audio.paused && !!this.audio.currentSrc; }

    /* ---------- 数据装载 ---------- */
    load(chunks, episodeDate) {
      this.chunks = chunks || [];
      this._episodeDate = episodeDate || '';
      this.index = 0;
      this.audio.pause();
      this.audio.removeAttribute('src');
      this.audio.load();
      if (this.chunks.length && this._episodeDate) this._setChunk(0, false);
      this._emitTime();
    }

    /* ---------- 播放控制 ---------- */
    toggle() { if (this.playing) { this.pause(); } else { this.play(); } }

    play() {
      if (!this.chunks.length) return Promise.resolve();
      return this.audio.play().catch(function (error) {
        console.warn('audio play blocked or failed', error);
      });
    }

    pause() { this.audio.pause(); }

    seek(globalSeconds, autoplay) {
      if (!this.chunks.length) return;
      var target = Math.max(0, Math.min(this.duration, Number(globalSeconds) || 0));
      var index = this._indexAt(target);
      var offset = target - Number(this.chunks[index].start || 0);
      var shouldPlay = autoplay === undefined ? this.playing : autoplay;
      this._setChunk(index, false);
      this.audio.addEventListener('loadedmetadata', () => {
        this.audio.currentTime = Math.max(0, offset);
      }, { once: true });
      this.audio.currentTime = Math.max(0, offset);
      if (shouldPlay) this.play();
      this._emitTime();
    }

    skip(delta) {
      if (!this.chunks.length) return;
      this.seek(this.currentTime + Number(delta), this.playing);
    }

    /* ---------- 音色 / 倍速 / 音量 ---------- */
    setVoice(host, variant) {
      this.voices[host] = variant;
      if (this.chunks[this.index] && this.chunks[this.index].host === host) {
        var wasPlaying = this.playing;
        this._setChunk(this.index, false);
        if (wasPlaying) this.play();
      }
    }

    voiceOf(host) {
      var current = this.chunks[this.index];
      if (current && current.host === host && this.voices[host]) return this.voices[host];
      return this.voices[host];
    }

    cycleRate() {
      this.rateIndex = (this.rateIndex + 1) % SPEEDS.length;
      this.audio.playbackRate = SPEEDS[this.rateIndex];
      return SPEEDS[this.rateIndex].toFixed(this.rateIndex % 2 === 0 ? 1 : 2) + 'x';
    }

    setVolume(volume) {
      this.volume = Math.max(0, Math.min(1, Number(volume) || 0));
      this.audio.volume = this.volume;
      if (this.volume > 0) this.lastVolume = this.volume;
    }

    toggleMute() {
      if (this.audio.volume > 0) {
        this.lastVolume = this.audio.volume;
        this.audio.volume = 0;
        return true;
      }
      this.audio.volume = this.lastVolume || DEFAULT_VOLUME;
      return false;
    }

    /* ---------- 内部 ---------- */
    _indexAt(globalSeconds) {
      for (var i = this.chunks.length - 1; i >= 0; i -= 1) {
        if (globalSeconds >= Number(this.chunks[i].start || 0)) return i;
      }
      return 0;
    }

    _audioFile(chunk, index) {
      var variant = this.voices[chunk.host];
      var audios = chunk.audios || {};
      var file = (variant && audios[variant]) || audios.professional || Object.values(audios)[0];
      if (file) return file;
      if (chunk.audio) return chunk.audio;
      return 'chunk_' + String(index + 1).padStart(3, '0') + '.mp3';
    }

    _setChunk(index, autoplay) {
      if (!this.chunks.length) return;
      this.index = Math.max(0, Math.min(this.chunks.length - 1, index));
      var chunk = this.chunks[this.index];
      this.audio.src = './episodes/' + this._episodeDate + '/' + this._audioFile(chunk, this.index);
      this.audio.playbackRate = SPEEDS[this.rateIndex];
      if (autoplay) this.play();
      this._emitTime();
    }

    _onChunkEnded() {
      if (this.index < this.chunks.length - 1) {
        this._setChunk(this.index + 1, true);
      } else {
        this._emitState(false);
      }
    }

  }

  window.PodcastEngine = PodcastEngine;
})();
```

注意：`load(chunks, date)` 的第二个参数是每期音频目录名（切片 URL 为 `./episodes/{date}/{chunkFile}`）；app.js 已按此签名调用并在 `load` 之后设置默认音色。若将来退化为整集播放，只改引擎内部的 `_audioFile`/`_setChunk`，UI 与回调不动。

- [ ] **Step 2: 语法检查**

Run:
```bash
node --check src/ai_news_podcast/presentation/site_builder/static/player.js && echo "player.js syntax ok"
```
Expected: `player.js syntax ok`

- [ ] **Step 3: 生成站点并在本地起服务做冒烟**

Run:
```bash
uv run python scripts/rebuild_site.py && python3 -m http.server 8000 --directory "$PWD/site"
```
打开 `http://localhost:8000/`，人工确认：播放/暂停、±15s、倍速、音量、音色按钮、逐字稿抽屉、日期切换、来源筛选/搜索/展开、雷达空态、390px 无横向滚动。
（本地 `site/episodes/2026-10-15/` 有 playlist 但 mp3 切片只在 gh-pages——没有音频时播放静默失败属预期，控制台不应有 JS 报错。）

- [ ] **Step 4: 提交**

```bash
git add src/ai_news_podcast/presentation/site_builder/static/player.js
git commit -m "feat(site): rewrite chunk-based player engine without DOM coupling"
```

---

### Task 10: 移动端验收与响应式修正

**Files:**
- Modify: `static/style.css`（仅断点内调整）
- Modify: `static/app.js`（必要时，如 `matchMedia` 分支）

- [ ] **Step 1: 390px 视口逐项核对（DevTools iPhone 12/390）**

清单：
1. `document.documentElement.scrollWidth <= 390`（brief 硬约束）；
2. 日期条可横向滑动、选日期生效；
3. 播客卡、导语、指标 2×2、日报、来源单列、雷达空态均不溢出；
4. 吸底条为 64px 迷你播放器（封面 + 当前句 + 播放键），不遮挡正文（`body padding-bottom` 生效）；
5. 逐字稿按钮打开全屏 bottom sheet，`Esc`/✕ 可关；
6. 屏幕宽度 720–1024 时两栏降级为单栏，无横向滚动。

- [ ] **Step 2: 修复发现的溢出（示例：长 URL/长标题）**

在 `style.css` 追加兜底（放在断点内）：

```css
@media (max-width: 720px) {
  .source-title-link, .chunk-text, .lead-quote-text, .daily-report-wrapper {
    overflow-wrap: anywhere; word-break: break-word;
  }
  .radar-hit-card .radar-hit-title { flex-wrap: wrap; gap: 6px; }
  .player-left { min-width: 0; }
}
```

- [ ] **Step 3: 提交**

```bash
git add src/ai_news_podcast/presentation/site_builder/static/style.css \
  src/ai_news_podcast/presentation/site_builder/static/app.js
git commit -m "fix(site): mobile 390px overflow and mini-player polish"
```

---

### Task 11: 收尾（删死代码、更新文档、v1.5 可选项）

**Files:**
- Modify: `src/ai_news_podcast/presentation/site_builder/show_notes_view.py`（删 `generate_show_notes` Markdown 版）
- Modify: `tests/test_show_notes_view.py`（删对应用例，保留 HTML 版）
- Modify: `scripts/redeploy_site.py`、`scripts/rebuild_site.py`（确认无旧模板假设）
- Modify: `docs/development.md`、`AGENTS.md`
- Modify: `docs/index.md`（如项目文档索引需要指向新 plan/digest 契约）

- [ ] **Step 1: 删死代码并跑测试**

删除：`show_notes_view.generate_show_notes`（Markdown 版，生产零引用，`02 §4.7`）与 `tests/test_show_notes_view.py` 中仅测它的用例；确认 `html_gen_view` 里已无 `format_friendly_date`（Task 6 已删）。

Run: `uv run pytest tests/ -q && uv run ruff check src/ tests/ scripts/ && uv run lint-imports`
Expected: 全绿；`Contracts: 4 kept, 0 broken`。

- [ ] **Step 2: 更新文档**

`docs/development.md` 增「站点数据契约」小节：digest/calendar 路径、生成入口（publish/daily/redeploy/rebuild）、`site/data` 需 `git add -f`、prune 语义。
`AGENTS.md` 的 Gotchas 增两条：
- `site/index.html` 现在是骨架 + 内联 `site-data` + `static/{style,app,player}`；不要重新内联 `episodes.json.desc`（体积红线 1.5MB）。
- `site/data/digest_*.json` 由 publish 生成并 `git add -f` 跟踪；**不要**把它挪出 `site/`，否则月度 prune 后会从 gh-pages 消失。

- [ ] **Step 3（可选 v1.5，默认不做）: 雷达候选审计**

若要还原设计稿的候选明细表：按 `03` §5.1 路 A 改 `gh_radar_service`（`_hard_filter` 返回原因、`_select_probe_list` 记录排除、`build_radar` 写 `meta.candidate_audit`），digest 透传 `candidates.preview`，前端加回表格。必须先加测试再动，且保持「雷达可失败」语义。工作量约 1 人日。

- [ ] **Step 4（可选 v2）: 章节真标题**

若要编辑级标题而非转场词截断：改 Stage 3 writer prompt 让 LLM 输出 `<chapter title="…">` 标记（仅对新期生效，历史 30 期无法回填），`tts_parser_service` 解析后写入 playlist chunk 元数据。工作量约 1-1.5 人日。

- [ ] **Step 5（backlog，v1 不做）: Apple Podcasts 入口**

等节目在 Podcasts Connect 上架后：`PodcastConfig` 增 `apple_podcasts_url: str = ""`，`config.yaml` 填节目链接，Task 6 模板头部加回 `<a id="apple-link" hidden>`，app.js `boot()` 里有 href 时 `hidden=false`（该逻辑 v1 已按 D15 删除，届时补 3 行）。**前提是 feed 已经在 Apple 侧验证通过**（`itunes:*` 字段、cover 图、guid 稳定），否则入口无意义。

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "chore(site): remove dead code and document digest contract"
```

---

## 4. 验收阶梯（每个 PR 必跑；上线前全跑）

| 级别 | 命令 | 预期 |
|---|---|---|
| L0 | `uv run pytest tests/test_digest_view.py tests/test_report_digest_view.py tests/test_sources_digest_view.py tests/test_html_gen_view.py -q` | 全绿，0 failed |
| L1 | `uv run pytest tests/ -q` | 全绿（CI 同口径 `.github/workflows/ci.yml:42`） |
| L2 | `uv run ruff check src/ tests/ scripts/ && uv run ruff format --check src/ tests/ scripts/ && uv run lint-imports` | `All checks passed!`；无 diff；`Contracts: 4 kept, 0 broken` |
| L3 | `uv run python scripts/rebuild_site.py` | 生成 `site/index.html` + `site/data/*`（30 期 + calendar）；日志无异常 |
| L3b | `python3 -c "import json,pathlib;d=json.loads(pathlib.Path('site/data/digest_2026-10-15.json').read_text());print(d['date'],d['issue']['number'],d['radar']['status'])"` | `2026-10-15 None empty` |
| L3c | `python3 -c "import gzip,pathlib;b=pathlib.Path('site/index.html').read_bytes();print(len(b),len(gzip.compress(b)))"` | raw < 1_500_000；gzip < 300_000 |
| L4 | `python3 -m http.server 8000 --directory "$PWD/site"` | 四板块可读、切日期 30 期、控制台无 JS 报错（音频 404 属预期） |
| L5 | feed 零 diff（见 §5.1） | `ZERO-DIFF` |
| L6 | 390px `document.documentElement.scrollWidth` | `<= 390` |
| L7 | 人工验收清单 10 条（§5.2） | 全部通过并截图存档 |

### 4.1 feed.xml 零 diff（红线）

```bash
PODCAST_BASE_URL=https://nicekai-jpg.github.io/ai-news-podcast uv run python - <<'PY'
import json, re
from pathlib import Path
from ai_news_podcast.data.config.loader_dao import load_config
from ai_news_podcast.presentation.cli.episode_utils_controller import get_base_url
from ai_news_podcast.presentation.site_builder.rss_gen_view import build_feed_xml
cfg = load_config(Path("config/config.yaml"))
eps = json.loads(Path("data/episodes.json").read_text(encoding="utf-8"))
eps.sort(key=lambda e: str(e.get("published_at_iso") or ""), reverse=True)
p = cfg.podcast
new = build_feed_xml(base_url=get_base_url(cfg, None), podcast_title=p.title,
                     podcast_description=p.description, podcast_language=p.language,
                     podcast_author=p.author, podcast_category=p.category,
                     podcast_explicit=p.explicit, episodes=eps)
old = Path("site/feed.xml").read_text(encoding="utf-8")
norm = lambda s: re.sub(r"<lastBuildDate>[^<]*</lastBuildDate>", "", s)
print("ZERO-DIFF" if norm(new) == norm(old) else "DIFF")
PY
```
Expected: `ZERO-DIFF`。**必须设 `PODCAST_BASE_URL`**，否则 `get_base_url` 回落 `http://localhost`，channel 三元组全变（`04 §6.1` 实测踩坑记录）。

### 4.2 人工验收清单（写进 PR 描述）

1. 桌面 1440：播放/暂停、±15s、倍速循环、音量/静音；
2. 音色按钮只显示真实合成变体（当前每 host 1 个），点击后下一句生效；
3. 逐字稿抽屉随播放高亮并居中滚动；点击任意句跳播；
4. 章节 chip 跳播时间正确（有转场词锚点时显示，无则整行隐藏）；
5. 日期轴切 30 期，前后日/回到最新正常，切换后播放器不会串期；
6. 日报 30 期全部渲染（2 篇重磅解读 + 前沿情报 + 小编点评）；
7. 来源：分组 pill 计数与条数一致、搜索、展开全部；
8. 雷达：空态（10-15）与有主推日期都正确，无 mock 切换按钮；
9. 390px 无横向滚动，迷你播放器与 bottom sheet 正常；
10. RSS 入口 `./feed.xml` 可用；期号徽章不出现（只有日期）。

---

## 5. 发布、回滚与 PR 拆分

### 5.1 合并顺序（4 个 PR）

| PR | 内容（对应 Task） | 合并门 |
|---|---|---|
| **PR1 数据层** | Task 1-5：日报解析、来源归一、digest/calendar、四链路接线、CI 跟踪 | L0-L2 + L5 `ZERO-DIFF`；**零视觉变化**，可先合；合完跑一次 `Redeploy Site` 确认 gh-pages 上出现 `data/digest_*.json` |
| **PR2 渲染与交互** | Task 6-10：新骨架、CSS、app.js、player.js、响应式、测试重写 | L0-L4 + L6 + 人工清单 10 条 + 体积预算；PR 必须**同时**更新 `tests/test_html_gen_view.py`（护栏与改动同 PR） |
| **PR3 收尾** | Task 11：删死代码、文档、可选 v1.5/v2 | L1-L2 + 一次回滚演练 |
| 上线动作 | `gh workflow run "Redeploy Site"`（合并 PR2 后手动触发；合并代码**不会**自动上线） | 等 `pages-build-deployment` 完成 → L5 → 抽查 4 板块 + 历史 3 期 + 390px |

### 5.2 回滚（R1 首选，2-4 分钟）

```bash
git revert <merge-commit> && git push origin main
gh workflow run "Redeploy Site"
```
`keep_files: true` 保留音频/切片；新 digest 文件作为残留留在 gh-pages（无害）。**注意**：`scripts/redeploy_site.py` 若不刷新 digest（PR1 已接），回滚后旧页面会取不到 digest → 这是 PR1 必须先把 redeploy 接上的原因。

### 5.3 上线后 48 小时观察

- 抽查 `https://nicekai-jpg.github.io/ai-news-podcast/data/digest_2026-10-15.json` 与当天日期；
- 下一次 daily 后确认：`git ls-files site/data | wc -l` ≥ 31（30 期 + calendar）、gh-pages 对账步骤绿；
- 月度 prune 后（每月 1 号）抽查 `data/digest_*.json` 仍 200（tracked → `copy_site_assets` 会拷回）；顺带验证 `radar/*.json` 的已知 404 窗口是否仍存在（`04 §4.2`）；
- Apple Podcasts：feed 零 diff、`enclosure` URL 未变，无需任何操作。

---

## 6. 风险与红线

| 级别 | 风险 | 止损 |
|---|---|---|
| P0 | digest 未 `git add -f` → 月度 prune 后历史数据消失 | Task 5 的 CI 改动 + 上线首月验证 prune 后 URL |
| P0 | `site/index.html` 超 pre-commit 2MB 上限 / 首屏崩塌 | Task 6 体积预算测试；禁止再内联 `episodes.json.desc` |
| P1 | digest 404 但 CI 全绿 | Task 5 的 Validate + 部署后对账 |
| P1 | DOM id 漂移导致播放器/逐字稿静默失效（无 JS 测试） | Task 6 的 DOM id 契约断言 + 人工清单 1-4 |
| P1 | `Redeploy Site` 未刷新 digest → 上线后修 CSS 打坏页面 | Task 4 四链路统一接线；PR1 验收项 |
| P2 | 章节启发式在个别日期失效 | 不足 2 锚点自动隐藏；不影响播放 |
| P2 | 旧历史日报（06-07 月）解析为空 | 30 期窗口全部新格式；超出窗口不影响 |
| P2 | 移动端 390 仍有横向滚动 | Task 10 兜底 CSS + L6 断言 |

---

## 7. 产品口径（2026-11-04 已拍板，执行时不再询问）

| # | 问题 | 结论 | 落地位置 |
|---|---|---|---|
| 1 | 期号「第 286 期」是否真实 | **不真实 → v1 不显示期号**，UI 只显示日期；digest 保留 `issue` 字段为 `null/unavailable`，不持久化 | D4、Task 3 `resolve_issue`（恒 unavailable）、Task 6 模板只留日期徽章、Task 8 `renderHero`/`renderDateAxis` 已处理 null |
| 2 | 雷达 12 个候选明细 | **v1 删表**，按真实 `meta` 计数渲染空态；v1.5 再改管线落盘审计 | D10、Task 8 `renderRadar`、Task 11 Step 3 |
| 3 | 章节 chip | **采用转场词锚点 + 首 12 字标题**（不追求编辑级标题）；不足 2 个锚点整行隐藏 | D9、Task 3 `detect_chapters`、Task 8 `renderChapters` |
| 4 | 移动端 | **单一响应式构建**，弃用 `mobile.html` 独立稿 | D12、Task 7（3 个断点）、Task 10（验收修正） |
| 5 | Apple Podcasts 入口 | **v1 不做**（从未运营），头部只留 RSS；feed.xml 不变；真正上架后再加 | D15、Task 6 模板、Task 11 Step 5 backlog |

> 唯一仍需在验收时确认的小口径（不阻塞开发）：来源速览 v1 展示**全部非 skip 来源**（真实约 123 条/期），默认折叠前 12 条 + 分组筛选；若产品希望只展示 `role=main`（约 30 条），在 digest 里把 `role_filter` 改成 `"main"` 即可（`sources_view.build_sources_block` 已支持）。

---

## 8. 工作量估算

| 阶段 | 内容 | 估时 |
|---|---|---|
| PR1 数据层 | Task 1-5（含测试、CI） | 2-2.5 人日 |
| PR2 前端 | Task 6-10（骨架/CSS/app.js/引擎/响应式） | 4-6 人日（其中播放器引擎 1-1.5 天，CSS 移植 1-1.5 天） |
| PR3 收尾 | Task 11 + 文档 + 回滚演练 | 0.5 人日 |
| 验收与上线 | L0-L7 + 人工清单 + 观察 48h | 1 人日 |
| **合计** | | **7.5-10 人日**（不含设计返工；Apple 入口已移出 v1） |

测试面：新增 3 个测试文件约 38 个用例；重写 `tests/test_html_gen_view.py` 约 12 个用例；其余测试 KEEP。
