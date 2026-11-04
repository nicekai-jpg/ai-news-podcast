# 04 · 测试 / CI / 部署 / 回滚风险与验收阶梯

范围：站点改版移植（`design/2026-10-redesign` → `src/ai_news_podcast/presentation/site_builder/`）对
**测试面、CI 面、部署面、回滚面**的影响。只读分析，未改动任何生产代码；所有结论标注 `file:line`。


## 0. 摘要（最需要记住的 7 条）

1. `tests/test_html_gen_view.py` 24 个用例全绿是当前基线（实测 `24 passed in 0.04s`），但其中 **9 条断言的是旧 DOM/class/JS 函数名**，改版后会「静默变绿或静默失效」，必须逐条重写而不是整文件删掉（见 §2）。
2. 仓库**没有 JS 测试设施**（无 `package.json`、无 playwright/puppeteer 依赖：`pyproject.toml:40-48`）。务实方案 = Python 侧生成物结构断言 + 10 条人工验收清单；**不建议**为此引入 headless browser（见 §3）。
3. `daily.yml` 的 publish 预校验只验 mp3 大小 / feed enclosure / episodes.json（`.github/workflows/daily.yml:327-349`），部署后对账只验 mp3 + playlist.json（`:376-404`）——**任何新增的 per-date 数据文件都不会被校验**，404 也能全绿。
4. 新增 `site/data/digest_{date}.json` 这类文件：`git add -u site/`（`daily.yml:355`）**不会**把新文件入库，只有 `:356-357` 显式列出的两个路径会被 `git add -f`。漏掉 → 文件只在 gh-pages 上活到下一次月度 prune。
5. `scripts/prune_gh_pages.py` 是 orphan 重建：只从 gh-pages 备份「episodes.json 里有效的 30 期的 mp3/html/txt + chunk 目录」（`:92-119`），其余只从 **main 的 `site/` 工作区**重拷（`:139-153`）。`site/` 被 gitignore（`.gitignore:32`），所以未 `git add -f` 的生成物在 prune 后必然消失。
6. pre-commit 有 **2048KB 单文件上限**（`.pre-commit-config.yaml:15-16`），而 `site/index.html` 现在已 763,731 B（raw 745.8KB / gzip 181.3KB）。把 30 期数据内联进 HTML 会直接撞墙。
7. 回滚很快：`git revert` + 手动跑 `Redeploy Site`（`.github/workflows/redeploy_site.yml:7-8`）2–4 分钟，`keep_files: true`（`daily.yml:374`）保住音频。**不建议**做 `?layout=legacy` 长期开关（理由见 §6.3）。


## 1. 基线事实（可复核）

| 事实 | 值 | 证据 |
|---|---|---|
| `test_html_gen_view.py` 基线 | 24 passed | `uv run pytest tests/test_html_gen_view.py -q` |
| 生成物大小 | index.html 763,731 B (gzip 181.3KB)；feed.xml 627KB (gzip 152.8KB) | `wc -c < site/index.html` |
| 内联资源 | style.css 1956 行 / 86.8KB；player.js 1294 行 / 50.8KB | `src/.../static/` |
| 单期数据量 | 55 行脚本（HostA 28 / HostB 27）、playlist chunks、30 条来源、radar `projects: []` | `site/episodes/2026-10-15.txt`、`data/gh_radar/radar_2026-10-15.json` |
| main 上被跟踪的 site 文件 | 178 个：`index.html`、`feed.xml`、`logo.png`、`.nojekyll`、`reports/*.md`、`episodes/*.txt`、`episodes/*/playlist.json` | `git ls-files site/` |
| **未被跟踪**的 site 子目录 | `site/radar/`（0 个 tracked）与任何新目录 | `git ls-files site/radar \| wc -l` → 0 |
| 音频从不进 main | 走 artifact 传 publish | `daily.yml:280-287`、`.pre-commit-config.yaml:26-31` |
| feed 可复现性 | 除 `<lastBuildDate>` 外零 diff（已实测） | §6.1 |

> 结论：**site/ 上「只有被 git 跟踪的文件才是持久的」**，这是后面所有部署风险的根。


## 2. 测试清单与改版判定（逐条 file:line）

判定口径：**KEEP** = 与新页面无关或契约不变；**UPDATE** = 语义保留、断言对象必须换（换成新 DOM/新数据通道）；**DELETE** = 断言对象是即将删除的旧结构；**ADD** = 改版必须新增。

### 2.1 `tests/test_html_gen_view.py`

| file:line | 断言内容 | 判定 | 说明/改法 |
|---|---|---|---|
| `:10-13` `test_iso_format` | ISO 日期转出来含 `2026`、`06-04` | KEEP | `format_friendly_date` 与 DOM 无关（`html_gen_view.py:70-85`） |
| `:15-17` `test_rfc_format` | RFC822 日期格式化成 `06-04` | KEEP | 同上 |
| `:19-20` `test_empty_string` | 空串返回空串 | KEEP | `:71-72` |
| `:22-23` `test_invalid_returns_original` | 非法日期原样返回 | KEEP | `:84-85` |
| `:25-26` `test_none_like_empty` | 与 `:19-20` 重复 | KEEP（可删冗余） | 无害 |
| `:30-40` `test_creates_index_file` | `site/index.html` 存在 | KEEP | 产物路径契约（`html_gen_view.py:521`） |
| `:42-53` `test_index_contains_title` | 品牌名出现在 HTML | KEEP | 顶部品牌块可能改名（设计 brief 提及工作名「AI 每日先锋」），断言仍成立 |
| `:55-67` `test_index_contains_episodes_map` | HTML 内含 `"2026-06-04"` 与 `2026-06-04.mp3` | **UPDATE** | 现在靠 `player.js:1-3` 的 `{dates_json}/{episodes_map_json}` 字符串注入（`html_gen_view.py:512-519`）。若改成 JSON script 块或 per-date fetch，子串断言即失效；改为「解析数据块后比对键/值」 |
| `:69-74` `test_index_no_episodes` | 空列表时 `DATES = []` 出现 | **UPDATE** | 硬绑 `player.js:1` 的 `const DATES = {dates_json};` 字面量。JS 一改名必破；改为解析数据块断言 `dates == []` |
| `:76-90` `test_index_contains_key_elements` | `layout-grid`/`podcast-workspace`、`player-bar`/`console-player`、`./feed.xml`、`parseTranscript` | **UPDATE** | 实测 `layout-grid`、`player-bar` 在 html/js/css 中**都不存在**，这两条只靠 `or` 的另一半（`style.css:385` `.podcast-workspace`、`style.css:731` `.console-player`）才绿——是典型的「看似在护栏、其实已失效」。改版后换成新骨架 class 或删；`./feed.xml` 必须 KEEP（`html_gen_view.py:185`） |
| `:92-102` `test_index_base_url` | base_url 进 HTML | KEEP | 但注意：base_url 由 `get_base_url()` 推（`episode_utils_controller.py:15-37`），本地默认 `http://localhost` |
| `:104-119` `test_index_transcript_parser` | `parseTranscript`、`dialogue-line`/`transcript-row`、`speaker-label`/`speaker-meta`、`host-a`、`host-b` | **UPDATE** | `dialogue-line` 与 `speaker-label` 今日已不存在（只靠 `or` 后半段 `transcript-row`(player.js:462)/`speaker-meta` 撑住）。改版若用服务端渲染逐字稿，改为断言真实行数与 speaker 类名 |
| `:122-196` `TestVoiceLabels`（7 例） | `_voice_labels` 的默认值/host 过滤/AppConfig 分支（`html_gen_view.py:35-67`） | **KEEP** | 音色契约不变（`config/config.yaml` `synth_variants` host_a→lively、host_b→professional），与 DOM 无关 |
| `:212-217` `test_tab_button_and_panel_present`（来源） | `btn-mode-sources`、`switchMode('sources')`、`id="panel-sources"`、`🏢 来源速览` | **UPDATE** | 四 Tab 顶部导航（`html_gen_view.py:160-177`）在改版中大概率被替换为日期轴 + 板块锚点；`panel-*`/`btn-mode-*` id 契约要重新定义 |
| `:219-222` `test_board_containers_present` | `sources-date-pills`、`sources-board-body` | **UPDATE** | 同上；id 契约若保留则 KEEP，若改为统一 `date-axis` 则 UPDATE |
| `:224-227` `test_js_helpers_present` | `parseSourceItems`、`renderSourcesBoard` | **UPDATE** | 函数在 `player.js:551/592`；改名或迁移即失效。建议改为「行为契约」测试（见 §3） |
| `:243-248` `test_tab_button_and_panel_present`（雷达） | `btn-mode-radar`、`switchMode('radar')`、`id="panel-radar"`、`📡 项目雷达` | **UPDATE** | 同上 |
| `:250-254` `test_containers_and_renderer` | `radar-date-pills`、`radar-body`、`renderRadar` | **UPDATE** | `renderRadar` 在 `player.js:693`；空态 `📡 暂无项目雷达`（`player.js:151,696`）要保留断言 |

### 2.2 其他站点/CLI 相关测试

| file:line | 断言 | 判定 | 说明 |
|---|---|---|---|
| `tests/test_site_builder_view.py:9-54` | feed XML 头、`<title>`、enclosure url/type、itunes 字段；空 episodes 不产生 `<item>` | **KEEP（红线）** | `rss_gen_view.py:7-64` 改版**不应触碰**；这是 Apple Podcasts 契约的自动化护栏 |
| `tests/test_show_notes_view.py:10-110` | show notes Markdown 结构/角色分区/items 上限 5 | KEEP | `show_notes_view.py` 生成 `.html` 备注，与页面骨架无关；改版不动 |
| `tests/test_cli_coverage_controller.py:99-130` | `PublishCommand` 缺 mp3 → 1；正常路径 patch 掉 `build_index_html`/`build_feed_xml` | **KEEP + ADD** | 该用例把渲染全 mock 掉了（`:123-127`），所以**不会**因为改版失败——是「假保护」。ADD：新增不 mock 渲染的 publish 契约用例（断言 digest 文件落盘） |
| `tests/test_cli_controller.py:84-142` | `prune_episodes` 删旧文件、orphan sweep、保留在途未来日期、非日期 id 不清扫 | **KEEP + ADD** | 若新增 per-date 数据文件写在 `site/data/`，sweep 不覆盖（正则 `episode_utils_controller.py:52` 只匹配日期形状且作用于 episodes_dir），需 ADD 针对新目录的清理/保留用例 |
| `tests/test_run_daily_full_controller.py:176-182` | 产物 `.txt/.html/.mp3` 与 `site/feed.xml`、`site/index.html` 存在 | KEEP | 只验存在性，对新骨架兼容；ADD digest 文件存在性断言 |
| `tests/test_pipeline_e2e_controller.py:218-225`、`tests/test_run_daily_branches_controller.py` | `--no-audio` 下 feed/index 不生成；配置/无 story 分支返回码 | KEEP | 无站点耦合 |
| `tests/test_prune_gh_pages.py:26-83` | clone 探测/初始化/clone 三条路径 | KEEP | 若扩展 prune 备份 `site/data`，需 ADD restore 用例 |
| `tests/test_prune_gh_pages.py:87-129` | 目录跳过、文件覆盖、有效 id 过滤 | KEEP + ADD | 新 per-date 文件的备份/恢复需同款断言 |
| `tests/test_prune_gh_pages.py:133-166` | episodes.json 解析、备份范围 | KEEP | — |
| `tests/test_daily_report_controller.py` / `test_daily_report_e2e_controller.py` | 日报 prompt / md 输出 | KEEP | 页面只消费 md，不耦合 |

### 2.3 ADD 清单（改版必须新增的测试）

- `tests/test_digest_view.py`（新文件）：per-date 数据契约（字段、期号、时长、来源分组、radar 空态、确定性）。
- `tests/test_html_gen_view.py`：数据块可解析、DOM id 唯一性、无原型 mock 残留、体积预算（§3 给草稿）。
- `tests/test_site_contract.py`（可选）：`feed.xml` 归一化零 diff（§6.1 的自动化版）；`tests/test_prune_gh_pages.py`：新数据目录被备份/恢复。


## 3. 新页面的测试策略（无 JS 测试设施）

### 3.1 现状证据

- 仓库无 `package.json` / `node_modules`，`pyproject.toml:40-48` 的 dev 组只有 ruff/pre-commit/import-linter/pytest，CI 只跑 ruff + format + lint-imports + pytest（`.github/workflows/ci.yml:32-42`）。
- 站点是「一个内联 HTML 产物」：CSS 内联（`html_gen_view.py:144-148`）、JS 内联（`:503-507`），DOM id 硬编码在模板里。所以**真正的可测边界在 Python 生成物**，不在浏览器。

### 3.2 推荐方案：生成物结构断言 + 人工清单（**采用**）

先做一件结构性改进：把三处字符串注入（`html_gen_view.py:512-519`）换成**一个 JSON 数据块**，它是唯一可被 Python 稳定解析的入口：

```html
<script type="application/json" id="site-data">{...}</script>
```

（注意把 `</` 转义成 `<\/`，`episodes.json` 的 desc 含 HTML；`player.js:1-3` 改为 `JSON.parse(document.getElementById('site-data').textContent)`。）

在此基础上给 4 个可直接落地的测试草稿（草稿基于现有 `build_index_html` 签名 `html_gen_view.py:93-99`）：

```python
# tests/test_html_gen_view.py（新增）
import gzip, json, re

from ai_news_podcast.presentation.site_builder.html_gen_view import build_index_html

_MOCK_MARKERS = ("PLAYLIST_DATA", "SOURCES_DATA", "RADAR_DATA", "CANDIDATES_DATA",
                 "Lorem ipsum", "示例主播", "第 286 期")

def _build(tmp_path, episodes=None, cfg=None) -> str:
    site = tmp_path / "site"
    build_index_html(site, "Test Podcast", episodes or [], "https://example.com", cfg)
    return (site / "index.html").read_text(encoding="utf-8")

def _site_data(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S)
    assert m, "缺 site-data JSON 块"
    return json.loads(m.group(1))

def test_no_prototype_mock_leftovers(tmp_path):
    """原型里的硬编码常量/假数字一个都不许进生产 HTML。"""
    html = _build(tmp_path, [{"id": "2026-06-04", "title": "t"}])
    for marker in _MOCK_MARKERS:
        assert marker not in html, f"原型 mock 残留: {marker}"

def test_site_data_block_is_valid_and_scoped(tmp_path):
    """数据块可解析、只含当天数据。"""
    eps = [{"id": "2026-06-04", "title": "t",
            "enclosure_url": "https://example.com/episodes/2026-06-04.mp3"}]
    data = _site_data(_build(tmp_path, eps))
    assert data["dates"] == ["2026-06-04"]
    assert data["episodes"]["2026-06-04"]["mp3"].endswith("2026-06-04.mp3")
    assert "2026-06-04" not in json.dumps(_site_data(_build(tmp_path / "d2",
        [{"id": "2026-06-05", "title": "x"}])), ensure_ascii=False)  # 不串味

def test_player_dom_id_contract(tmp_path):
    """播放器必需 id 必须存在且唯一（改版最容易丢的就是这几个）。"""
    html = _build(tmp_path, [{"id": "2026-06-04", "title": "t"}])
    required = ["main-audio", "console-progress-fill", "console-progress-track",
                "console-btn-play", "host-a-voice-pills", "host-b-voice-pills",
                "cast-panel-body", "date-pills"]
    for rid in required:
        assert html.count(f'id="{rid}"') == 1, f"{rid} 缺失或重复"

def test_index_size_budget(tmp_path):
    """体积预算：raw < 1.5MB，gzip < 300KB（今天 745.8KB/181.3KB，留 2x 余量）。"""
    html = _build(tmp_path, [{"id": "2026-06-04", "title": "t"}])
    assert len(html.encode("utf-8")) < 1_500_000
    assert len(gzip.compress(html.encode("utf-8"))) < 300_000
```

设计建议：**把「是否为空 / 空态文案 / 期号 / 时长」的判断放到 Python（新 `digest_view.py`），JS 只做展示**，这样空态（radar 今天就是空的）才能被无浏览器测试断言；`player.js:149-155` 那种 `setEmpty(...)` 文案散落十处的写法无法断言。

### 3.3 大 HTML 快照测试：不采用

风险：`build_time`（`html_gen_view.py:131`）每次构建都变；`lastBuildDate` 同理（`rss_gen_view.py:30`）；内容随每天真实新闻变；763KB 的 diff 无法 review，任何 CSS 微调都会让快照全量失效 → 维护成本远高于收益。替代：**结构断言 + 关键文本断言 + 体积预算**（上）。

### 3.4 播放器行为验证：人工清单（**推荐**）vs headless browser

| 方案 | 成本 | 能覆盖 | 建议 |
|---|---|---|---|
| A. 人工验收清单（10 条）+ Python 结构断言 | 0.5 人日 | 播放/切音色/滚动同步/空态/移动端 | **采用**（每次发版跑一遍，5 分钟） |
| B. pytest-playwright 单文件 smoke（`page.goto` + 点播放 + 断言 `audio.src`/类名切换） | 1–2 人日 + CI 装浏览器（~+60-90s/run，`uv sync` 之外要 `playwright install`），且音频 autoplay/真实解码仍不可可靠断言 | 播放按钮、tab 切换、日期轴、进度条点击 | **本期不做**；若做，只放 3 条 smoke 并允许 `continue-on-error` |
| C. 完整 e2e（真音频 + 逐字稿同步） | 3-5 人日 + 需真实 mp3（artifact/gh-pages 拉取）+ 时间轴断言 flaky | — | 不推荐 |

人工清单（写进 PR 模板）：①桌面 1440 播放/暂停/±15s/倍速/音量；②逐字稿随音频高亮且可点句跳转；③切 host A/B 音色，音色按钮只出现真实存在的变体（`_voice_labels` 收敛逻辑 `html_gen_view.py:64-67`）；④切日期轴换期；⑤日报 tab 渲染 md（marked CDN 可用）；⑥来源 tab 分组/筛选；⑦雷达 tab 空态；⑧移动端 390px 无横向滚动（`document.documentElement.scrollWidth <= 390`）；⑨RSS 入口 `./feed.xml` 可开；⑩刷新后不回到错误日期。


## 4. CI / 部署风险（具体到 job / step）

### 4.1 publish job 的四道闸门（`daily.yml:289-404`）

| 步骤 | file:line | 行为 | 改版相关风险 |
|---|---|---|---|
| env | `:293-295` | `MIN_MP3_BYTES: '100000'`（约 100KB，真实单期 2-3MB） | 不动 |
| Publish | `:321-325` | `uv run podcast-publish --date "$EPISODE"` | **新数据文件的唯一「生产出口」**：`podcast_publish_controller.py:158-162` 里 index → feed → `_sync_tree(data/reports→site/reports)` → `_sync_tree(data/gh_radar→site/radar)`。新文件必须在这里写入 `site/`，否则根本没产物 |
| Validate | `:327-349` | mp3 存在且 `> MIN_MP3_BYTES`；`feed.xml` 含 `episodes/{ep}.mp3`；`episodes.json` 有该 id | **只验这三样**。新增 digest 文件不校验 → 生成失败也绿 |
| Commit site | `:351-364` | `git add -u site/ data/episodes.json`；再 `git add -f` report md 与 `playlist.json` | 新文件不被 `-u` 捕获，必须新增 `git add -f` |
| Remove .gitignore | `:366-367` | 部署前删 `.gitignore`（否则 peaceiris 的 `git add` 会跳过 `site/`，见 `.gitignore:32`） | 必须保持「删 .gitignore → deploy」的顺序 |
| Deploy | `:369-374` | `publish_dir: ./site` + `keep_files: true` | keep_files 让「本次 site/ 里没有的文件」保留在 gh-pages（音频靠这个），但**删除的文件也永远删不掉** |
| Verify | `:376-404` | `gh api contents/...?ref=gh-pages` 取 size，比 mp3 与 `playlist.json` | **不检查任何新数据文件** → 最容易漏 |

### 4.2 prune（月度）为什么是最大风险

- 触发：每月 1 号 12:00 UTC（`prune_pages.yml:4-6`），与 daily 共用 `podcast-pipeline` 并发组（`prune_pages.yml:15-17`）；跑完后 daily 仍要等约 9 小时才补。
- 行为：clone gh-pages → 只备份 valid id 的 `{id}.mp3/.html/.txt` + `{id}/` 目录（`prune_gh_pages.py:92-119`，valid ids 来自 `data/episodes.json`，`keep_last: 30`，`config/config.yaml` `podcast.keep_last`）→ `git checkout --orphan` 清空（`:122-136`）→ 从 **main 工作区** `site/` 拷静态资源（`:139-153`，跳过 `episodes`）→ 合并（`:163-201`）→ `push --force`（`:204-221`）。
- 推论（可验证）：**任何没被 git 跟踪的 `site/` 文件，prune 后从 gh-pages 消失**。
- 现成的同构 bug（可当证据）：`site/radar/*.json` 由 publish 从 `data/gh_radar` 同步（`podcast_publish_controller.py:162`），但 `git ls-files site/radar` = **0**。因此 prune 重建后 `radar/` 目录不在 gh-pages 上，`player.js:236` 的 `fetch('./radar/radar_'+d+'.json')` 会 404，直到下一次 daily publish（`_sync_tree` 会一次性补回 34 个文件，`data/gh_radar` tracked 34 个）才自愈。**这就是新 digest 文件会遇到的同款故障，只是数据更大、用户更明显。**
- 验证命令：prune 跑完后 `curl -s -o /dev/null -w '%{http_code}\n' https://nicekai-jpg.github.io/ai-news-podcast/radar/radar_2026-10-15.json`。

### 4.3 其它 CI/本地风险

| 风险 | 证据 | 后果 | 缓解 |
|---|---|---|---|
| pre-commit 2MB 单文件 | `.pre-commit-config.yaml:15-16` | 本地提交 `site/index.html`（tracked）超 2MB 被拦；bot 提交绕过（CI 不跑 hooks，`ci.yml:32-42` 只跑 ruff/pytest） | index.html 预算 ≤ 1.5MB（§3.2）；不要内联大 JSON/图 |
| `site/` 被 gitignore | `.gitignore:32` | PR 里看不到渲染 diff，review 盲区 | PR 附本地截图 + `scripts/rebuild_site.py` 预览（`rebuild_site.py:48-71` 会同步 `data/_preview/`） |
| 音频阻断 hook | `.pre-commit-config.yaml:26-31` | 新设计若放 `.mp3/.wav` 音频素材（除 `assets/`）会被拦 | 新素材放 `assets/` 或转成非音频格式 |
| 第三方 CDN | `html_gen_view.py:143` 内联 `<script src="https://cdn.jsdelivr.net/npm/marked/...">`，且在 `<head>` 无 defer | jsdelivr 被墙/超时 → 日报 tab 空白，且阻塞首屏 | 自托管 marked（~40KB 进 2MB 预算）或改 `defer` + 失败降级为 `<pre>` |
| 无 push 触发部署 | `daily.yml:3-6` 只有 schedule+dispatch；`redeploy_site.yml:7-8` 只有 dispatch；`ci.yml:3-7` 仅质量门 | 合并代码**不会**改变线上，必须手动 dispatch 或等次日 | 合并后立刻 `gh workflow run "Redeploy Site"` |
| 生成物破坏性 | `podcast-publish` 会改写 `data/episodes.json`（`podcast_publish_controller.py:120-137`）与 `site/index.html/feed.xml`（`:157-158`） | 本地拿不在 episodes.json 的日期跑 publish 会污染索引并触发删文件（`:136` → `prune_episodes`） | 只在干净 worktree 里跑，或跑完 `git checkout -- site/index.html site/feed.xml data/episodes.json` |

### 4.4 新增 per-date 数据文件的完整链路（**必读**）

以 `site/data/digest_2026-10-15.json` 为例：

| 环节 | 发生在哪 | 现状是否覆盖 | 需要改什么 |
|---|---|---|---|
| ① 数据源 | `data/briefs`(134)、`data/reports`(133)、`data/gh_radar`(34)、`site/episodes/*.txt`(30 tracked)、`data/episodes.json` | ✅ 全在 main | 无 |
| ② 生成 | publish job：`daily.yml:321-325` → `podcast_publish_controller.py:158-162` | ❌ 现有代码不生成 | 新 `digest_view.py`，并在 `execute_async` 里调用（同时改 `scripts/redeploy_site.py:35-55`、`scripts/rebuild_site.py:48-49`、`podcast_daily_controller`） |
| ③ 部署到 gh-pages | `daily.yml:366-374`（`publish_dir: ./site`，keep_files） | ✅ 只要文件被写进 `site/` 就会上传 | 无（前提：`.gitignore` 仍被删） |
| ④ 进 main（持久化） | `daily.yml:355` `git add -u site/` **不覆盖新文件** | ❌ | 两个选择：**(a)** 在 `:356-357` 旁加 `git add -f "site/data/digest_${EPISODE}.json"`；**(b)** 不落 main，靠 publish 每天重新生成全量（radar 模式，`podcast_publish_controller.py:160-162` + `_sync_tree`） |
| ⑤ 月度 prune 后是否还在 | prune 重建只恢复 ① 的 tracked 文件 + 有效 30 期音频资源（`prune_gh_pages.py:92-119,139-153`） | ❌ 未落 main 的必然消失；落 main 的会被 `copy_site_assets` 递归拷贝 | 若走 (b)，扩展 prune：备份/恢复 `site/data`（约 15-25 行 + `test_prune_gh_pages.py` 用例） |
| ⑥ 部署后对账 | `daily.yml:376-404` | ❌ 不检查新文件 | 加一段 `get_size "data/digest_${EPISODE}.json"` 且 `> 0` 断言 |
| ⑦ 前端读取 | `player.js` 新增 `fetch('./data/digest_'+d+'.json')` | — | 必须与 `player.js:203/236/266/287` 一样有 catch + `setEmpty` 空态 |

**最容易漏的一步**：`④`（`daily.yml:355-357` 的 `git add -f`）与 `⑥`（对账）。前者导致文件只在 gh-pages 上「活一个月」，后者导致漏了也不会红。第二容易漏的是 `②` 的**旁路**：只改了 `podcast_publish_controller.py`，忘了 `scripts/redeploy_site.py`——而 `Redeploy Site` 正是你上线后第二天修 CSS 时唯一会跑的工作流，它跑完会把一个「引用了不存在 digest 的新 index.html」推上去。

**推荐**：v1 走 **(b) 不落 main + publish 时全量重建 + 扩展 prune 备份**。理由：单期 digest 估 10-30KB，30 期 300KB-1MB 塞进 main 的 git 历史不划算；而 radar 已经证明「tracked 源 + `_sync_tree` 全量重建」这套可行（唯一代价是 prune 后 ~9 小时窗口 + 需要补 prune 备份）。


## 5. 本地验收阶梯（可直接执行）

| 级 | 命令 | 预期输出 | 是否写文件 |
|---|---|---|---|
| L0 | `uv run pytest tests/test_html_gen_view.py -q` | `24 passed`（改版后数字会变，0 failed） | 否 |
| L1 | `uv run pytest tests/ -q` | 全绿（CI 同口径 `ci.yml:42`） | 仅 tmp_path |
| L2 | `uv run ruff check src/ tests/ scripts/`；`uv run ruff format --check src/ tests/ scripts/`；`uv run lint-imports` | `All checks passed!`；无 diff；`Contracts: N kept, 0 broken` | 否 |
| L3 | `uv run python scripts/rebuild_site.py` | `Rebuilding index.html in .../site...` → `Rebuild and sync complete!` | 写 `site/index.html`（tracked，验完 `git checkout -- site/index.html`）与 `data/_preview/`（gitignored） |
| L3b | `grep -c -E '\{dates_json\}|\{episodes_map_json\}|\{voices_config_json\}|\{podcast_title\}' site/index.html`；`python3 -c "import gzip,pathlib;b=pathlib.Path('site/index.html').read_bytes();print(len(b),len(gzip.compress(b)))"` | 占位符残留 `0`（改成 JSON 数据块则换 §3.2 解析测试）；raw < 1_500_000 且 gzip < 300_000（今天 763731 / 185653） | 否 |
| L4 | `python3 -m http.server 8000 --directory "$PWD/site"` 然后开 `http://localhost:8000/` | `Serving HTTP on :: port 8000`；四个板块可切、控制台无 404（除音频 404，本地无 mp3 属预期） | 否 |
| L5 | 真发布（**破坏性**，见 §4.3）：先 `cp -R site /tmp/site-preview`，再 `uv run podcast-publish --date 2026-10-15` | 本地**预期失败**：`MP3 not found: .../site/episodes/2026-10-15.mp3`，退出码 1（音频只在 gh-pages）→ 想跑通先 `curl -o site/episodes/2026-10-15.mp3 https://nicekai-jpg.github.io/ai-news-podcast/episodes/2026-10-15.mp3`（该文件被 gitignore，安全） | 是（改 episodes.json + index + feed） |
| L6 | 390px：DevTools 设备模式选 iPhone 12，控制台跑 `document.documentElement.scrollWidth` | `<= 390`（brief 硬约束：移动端不许横向滚动） | 否 |

性能预算（手工在 DevTools Performance 面板，Fast 3G + 4x CPU throttle）：

| 指标 | 预算 | 依据 |
|---|---|---|
| LCP | ≤ 2.5s | 今天 gzip 181KB 内联 HTML；改版后 ≤ 300KB |
| index.html raw | ≤ 1.5MB | 2MB pre-commit 上限 `.pre-commit-config.yaml:16` 的安全边距 |
| 首屏 DOM 节点 | ≤ 1500 | 55 段逐字稿 × ~4 节点 + 30 来源 ≈ 350，留余量 |
| 55 段逐字稿渲染 | ≤ 100ms | 一次性 innerHTML（`player.js:462` 现在就是字符串拼接） |
| 板块/日期切换 | ≤ 100ms | 按需渲染，别在首屏渲染全部 4 个 Tab |
| 外部请求数 | ≤ 1（marked CDN） | 今天只有 jsdelivr 一个外部依赖（`html_gen_view.py:143`） |


## 6. 发布与回滚

### 6.1 feed.xml 零 diff 校验（已实测通过）

`build_feed_xml` 里唯一每次变的是 `<lastBuildDate>`（`rss_gen_view.py:30`）；item 全部来自 `episodes.json` 的持久字段（`:45-61`）。因此「HTML 改版不能动 feed」可以用归一化比对机械验证：

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

预期输出 `ZERO-DIFF`；实测今天的 `site/feed.xml` 满足（30 个 `<item>` 完全一致）。**必须设 `PODCAST_BASE_URL`**，否则 `get_base_url` 回落到 `http://localhost`（`episode_utils_controller.py:31-37`），channel 里 `link/atom:link/itunes:image` 会全变，输出 `DIFF`（我第一次跑就踩了这个）。
若差分，按 `lastBuildDate` → channel(link) → item 逐层定位；`enclosure_length` 变化说明 mp3 被重编码（`podcast_publish_controller.py:114,131`）。

### 6.2 回滚三档

| 档 | 操作 | 耗时 | 影响 | 适用 |
|---|---|---|---|---|
| R1（首选） | `git revert <commit>` 推 main，然后 `gh workflow run "Redeploy Site"`（`redeploy_site.yml:7-8`）。它只重建 index/feed/reports/radar（`scripts/redeploy_site.py:27-55`）并 `keep_files` 部署（`redeploy_site.yml:42-47`） | 2–4 min | 音频/切片/备注全保留；新设计的 per-date 数据文件会作为残留留在 gh-pages（无害，可择机清） | 视觉/JS bug |
| R2 | `gh workflow run daily.yml -f date=<today> -f retry=true`（`daily.yml:3-16`） | 40 min+ | 会重跑 writer（覆盖 `site/episodes/{date}.txt`，`AGENTS.md` gotcha）与 TTS，风险大 | 仅当当天产物本身坏了 |
| R3 | 直接回滚 gh-pages：clone → `git reset --hard <旧 sha>` → force push | 10 min | 下一次 daily/prune 会覆盖回去，只是临时止血；大 push 受 HTTP/1.1 约束（`AGENTS.md`） | 灾难性部署 |

注意：合并代码**不会**自动上线（`daily.yml:3-6` 无 push 触发），所以「revert 了但线上还是新的」是正常现象——必须显式跑 R1。

### 6.3 要不要保留旧页面入口 / 灰度？

判断：**不做 `?layout=legacy` 长期开关**。证据与理由：

1. 站点是单文件内联产物（`html_gen_view.py:136-510`），双布局 = 双倍体积（现在 763KB）与双倍测试面，直接顶穿 2MB pre-commit 上限（`.pre-commit-config.yaml:16`）。
2. 现在已有一套内部 A/B 开关（`html_gen_view.py:179-186` + `player.js:20-34` 的 `localStorage['ab_variant']`），设计 brief 明确「A/B 测试开关是内部调试，不应出现在产品里」。再加一个 `?layout=legacy` 是把同样技术债再写一遍。
3. 回滚成本已经很低（R1 2–4 分钟）。

**可选低成本逃生门**：首次上线那一次部署，额外把一个静态副本 `site/index_legacy.html`（旧模板产物）放进 `site/`（一次性脚本，不进模板、不进日常构建），保留 48 小时后删除。它不是灰度开关，只是出事时能立刻让人看到旧版截图/对照。若坚持要可切换，用 `/legacy/` 静态目录而非查询参数，避免 JS 分支污染主流程。

### 6.4 Apple Podcasts 侧：无需任何变更（确认点）

- `feed.xml` 由 `rss_gen_view.py` 生成，改版不触碰 → §6.1 零 diff。
- `enclosure_url = {base_url}/episodes/{date}.mp3`（`podcast_publish_controller.py:112,130`），音频文件名/路径不变 → enclosure URL 稳定，Apple 侧缓存不会失效。
- `guid = {base_url}/episodes/{date}`（`:118,128`）不变；`link` 仍指 `episodes/{date}.html`（`:113`）。
- `base_url` 由 `get_base_url()`（`episode_utils_controller.py:15-37`）决定，CI 从 `GITHUB_REPOSITORY` 推导，无需改配置。
- **唯一红线**：不要移动 `feed.xml` 在 gh-pages 上的路径、不要改 `base_url`。动了才需要去 Podcasts Connect 重新提交。


## 7. 工作量、风险分级与 PR 拆分

### 7.1 改动量级（纯代码，不含设计稿）

| 文件 | 现在 | 预计改动 |
|---|---|---|
| `presentation/site_builder/html_gen_view.py` | 521 行 | 模板重写 700-900 行（删 A/B 双布局 ≈200 行，加新骨架/数据块） |
| `static/style.css`（1956 行）+ `static/player.js`（1294 行） | — | CSS 净变 1200-2000 行（四套主题收敛为一套）；JS 改 40-60%（600-900 行：数据块解析、板块渲染、播放器 dock、移动端抽屉） |
| `presentation/site_builder/digest_view.py` | 新增 | 150-250 行 |
| `presentation/cli/podcast_publish_controller.py` | 173 行 | +20-40 行 |
| `scripts/{redeploy_site,rebuild_site}.py` | 62 / 77 行 | 各 +5-15 行 |
| `.github/workflows/daily.yml` | 478 行 | +8-15 行（git add -f、对账） |
| `scripts/prune_gh_pages.py` | 276 行 | +15-25 行（`site/data` 备份/恢复） |
| `tests/` | — | `test_html_gen_view.py` 254→300-400 行；新增 `test_digest_view.py` 150-250 行；`test_prune_gh_pages.py` +30-50 行 |

合计约 **6-9 个文件、2500-4000 行改动、2-3 人周**（不含设计返工与真机验收）。

### 7.2 风险分级

| 级别 | 风险 | 触发点 | 止损 |
|---|---|---|---|
| **P0** | 新 per-date 数据文件未落 main，月度 prune 后历史数据消失 | `prune_gh_pages.py:122-153` + `daily.yml:355` | 走 §4.4(b) 并扩展 prune 备份；上线首月手动验证 prune 后 URL |
| **P0** | index.html 超 2MB，本地提交被 pre-commit 拦、首屏崩塌 | `.pre-commit-config.yaml:15-16` | 体积预算测试（§3.2）+ 不内联历史数据 |
| **P1** | 新数据文件 404 但 CI 全绿（对账不覆盖） | `daily.yml:376-404` | 加 `get_size ... > 0` 断言 |
| **P1** | DOM id 漂移导致播放器/逐字稿静默失效（无 JS 测试） | `player.js:11,250-261,868-871` | id 契约测试 + 人工清单①-③ |
| **P1** | `Redeploy Site` 未同步生成新数据 → 上线后第二天修 CSS 反而打坏页面 | `scripts/redeploy_site.py:27-55` | 同一生成函数，两条链路都调用（PR3 验收项） |
| **P2** | marked.js CDN 不可达 → 日报空白 | `html_gen_view.py:143` | 自托管或 defer + 降级 |
| **P2** | keep_files 残留（改版删掉的旧资源仍可访问） | `daily.yml:374` | 可接受；需要时在 gh-pages 手动 `git rm` |
| **P2** | PR 无法 review 渲染结果（site/ gitignore） | `.gitignore:32` | PR 模板要求附本地截图 + L3/L4 输出 |

### 7.3 建议合并顺序（4 个 PR）

| PR | 内容 | 为什么这个顺序 | 合并门 |
|---|---|---|---|
| **PR1 数据层** | 新 `digest_view.py` + 调用点（publish/redeploy/rebuild/daily）+ `test_digest_view.py` + 期号/时长/来源/radar 空态推导 | **零视觉变化**，可先合、可回滚几乎无感；它把 §3.2 的数据块结构定下来 | L0-L2 + `feed.xml` `ZERO-DIFF` |
| **PR2 渲染与交互** | `html_gen_view.py` 模板重写 + `style.css` + `player.js`；**同一 PR 内**更新 `tests/test_html_gen_view.py` 的 UPDATE 项并 ADD 契约/体积/空态测试 | 渲染和它的护栏必须同时进，否则中间态是「无护栏的大改」 | L0-L4 + 人工清单 10 条 + §3.2 四条测试 |
| **PR3 CI/部署加固** | `daily.yml` 加 `git add -f`（若走方案 a）或加对账断言（方案 b 也必须有）+ `prune_gh_pages.py` 备份 `site/data` + `test_prune_gh_pages.py` 用例 + `docs` 回滚 runbook | 必须在 PR2 真正上线前合入；可与 PR2 并行开发但**先合** | L1-L2 + prune 演练（手动 dispatch 一次 `Prune GitHub Pages History` 后验证 URL） |
| **PR4 收尾** | 删 A/B variant 开关（`html_gen_view.py:179-186`、`player.js:20-34`）、删死 id/死 CSS、`README`/`docs/development.md` 更新 | 确认新页面稳定 1-2 天后做，避免和回滚窗口重叠 | L1-L4 + 一次 R1 回滚演练 |

上线动作（PR2 合并后手动执行）：`gh workflow run "Redeploy Site"` → 等 `pages-build-deployment` → §6.1 feed 零 diff → 抽查 4 个板块 + 历史 3 个日期 + 390px；48 小时内保留 R1 回滚文档与旧版截图。


## 8. 附：可复核命令

`uv run pytest tests/test_html_gen_view.py -q`（24 passed）· `git ls-files site/ | wc -l`（178）· `git ls-files site/radar | wc -l`（0，prune 后必失的现成证据）· `wc -c < site/index.html`（763731）· `grep -rn "MIN_MP3_BYTES\|keep_files\|git add -f" .github/workflows/daily.yml`
