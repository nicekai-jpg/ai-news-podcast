# 成品质量：S4–S7 的检查、缺陷与修复

> 记录 2026-11-25 对「日报 / 文稿 / 点评」做的一轮质量排查。
> 结论先行：**每一处「生成了但没展示」或「展示了但没校验」的链路都已补上程序检查**，
> 剩下未解决的都是数据本身的问题（来源缺年份、单一来源），不是代码缺陷。

## 0 一页速览

| 环节 | 检查项 | 谁在跑 | 状态 |
|---|---|---|---|
| S4 日报 | 数字必须逐字出现在证据中 | `_parse_fact` | ✅ |
| S4 日报 | 事实必须带 evidence_ids | `_parse_fact` | ✅ |
| S4 日报 | 禁用词拒收／观点清洗 | `_parse_fact` / `sanitize_text` | ✅ |
| S4 日报 | **原文出处**（可点击链接） | `_merge_sources` + 站点渲染 | ✅ 本轮 |
| S4 日报 | **同标题转载不算多源** | `_effective_independent_sources` | ✅ 本轮 |
| S4 日报 | 客观性审计指标 | `commentary_quality` → `ops_snapshot` | ✅ 本轮接线 |
| S5 点评 | 结构完整度（overall/items） | `review_quality` | ✅ |
| S5 点评 | **verdict 标签展示** | `digest_view.editor_verdict` | ✅ 本轮 |
| S6 文稿 | 字数／章节／主持占比／禁用词 | `check_script` | ✅ |
| S6 文稿 | **数字核对**（线上真正执行） | `scripts/script_factcheck.py` | ✅ 本轮 |
| S7 点评 | **章节标题与文稿 `[CHAPTER]` 对齐** | `_align_chapters` | ✅ 本轮 |
| S7 点评 | **fact_check_notes 确定性核对** | `_deterministic_fact_notes` | ✅ |
| S7 点评 | 结构完整度 | `review_quality` | ✅ |

## 1 本轮修掉的四个真问题

### 1.1 日报没有任何原文链接

**现象**：2026-11-25 生产日报 5 条里 4 条 `sources` 为空，读者拿不到出处。

**根因**：出处完全依赖 LLM 输出 `sources` 字段，模型经常不填。

**修法**：
- `load_item_sources(ocean_dir)` 从 `items.json` 读 `item_id → {label, link, official, title}`；
- `_merge_sources` 合并 LLM 出处与元数据出处：LLM 给的有效 link 优先，
  只给 label 的用元数据补 URL，按 link 去重。

**验证**：缺出处 4/5 → **0/5**。

### 1.2 出处生成后到不了读者眼前

**现象**：1.1 修完后，站点上一个链接都没有。

**根因**：`digest_view._report_from_rows` 只取
`facts / background / impact / counterpoint`，`row["sources"]` 被直接丢弃；
前端 `dives` 结构里也没有 `sources` 字段。

**修法**（三层）：
- 数据：每条 dive 增加 `sources`（过滤无 link 的条目）；
- 前端：`digestToData` 存 `chapter_N_sources`，`renderDiveSources()` 渲染
  「原文出处」链接行（最多 4 条，官方源标「· 官方」，`rel="noopener"`）；
- 样式：`.deep-dive-sources` 用虚线分隔、主题令牌配色。

**验证**：5 条 dive 全部带出处，例如「量子位 / Hugging Face 模型页 / ModelScope / 雷峰网」。

### 1.3 同标题转载被算成两个独立来源

**现象**：S5 点评指出「两条证据正文近乎同稿，需核实来源独立性」——核实属实：

| 事件 | 两家媒体标题 | 正文相似度 | 被算作 |
|---|---|---|---|
| OceanBase OceanLake | 完全相同 | **0.98** | 2 个独立来源 |
| 文心 5.5 Flash 开源 | 完全相同 | **0.85** | 2 个独立来源 |

日报因此不给这些事实标「（单一来源）」，读起来像有交叉印证。

**修法**：`_effective_independent_sources` 按 item 标题去重得到有效独立来源数，
`_parse_fact` 用它决定 `single_source`；拿不到标题时退回原值，不凭空降低。

**验证**：上述 2 条事件的 facts 由 `False` → `True`；
有 4 家不同标题报道的 Nvidia 事件保持多源，未被误降。

**未开启** `evidence.dedup_syndicated_titles`：它会同时改变 `evidence_score`
与推荐排序，而缺 30 天历史时无法做公平 A/B（重建后 shortlist 有 4/8 变化，
混杂了历史缺失的影响）。该开关维持 `false`，待有完整历史再评。

### 1.4 S6 数字核对在线上从未执行

**现象**：GHA 的 `Check script quality` 只调用 `quality(text)`，没传事实底稿，
`check_script` 里 `fact_checked=False` —— 核对被静默跳过。本地有完整测试，
但线上这条路一直是静的。

**修法**：新增 `scripts/script_factcheck.py`，从当日产物
（`events/items/shortlist/brief.radar`）按 S6 完全相同口径重建底稿后核对，
有未核对数字输出 `::warning::`，`--strict` 可返回非 0。GHA 已接入。

**验证**：2026-11-25 真实文稿 29 个数字全部可核对，与 S7 的
`fact_check_notes` 一致。

## 2 顺手修掉的两处小问题

### 2.1 S5 的 verdict 标签被丢弃

`editor_verdict` 只拼 `title + comment`，丢掉了 S5 的
`verdict`（准确/需修正/补充）—— 那正是复核结论本身。
现在渲染成 `【需修正】标题：点评`，`comment` 为空的行不再占位。

### 2.2 日报路径指向不存在的目录

`digest` 与站点重建都读 `data/reports/daily_report_{date}.md`，
而 v3 实际写到 `data/editorial/{date}/daily_report_{date}.md`。

后果：`source_file` 永远指向幽灵文件（无人读取，但会误导排障）；
站点重建时 `report_md` 永远是 None，只因 `report_rows` 存在走了结构化路径才没暴露。

现在统一指向 v3 路径，重建时优先读 `data/editorial/`，找不到退回旧 `data/reports/`。

## 3 核实后**不成立**的两个问题

留着记录，避免以后重复排查。

### 3.1 「日报 fact 长度严重超标」

我一度判断「3/4 条超出 150–300 区间、像论文」。实测两份生产数据：

| 数据 | 字数分布 | 超 300 |
|---|---|---|
| 11-25 完整运行 | 144–205 | 0 条 |
| 11-25 快速运行 | 144–226 | 0 条 |

告警全部来自**略低于 150 的下限**（140/144/148/149），属告警级噪声，
不影响阅读。`fact_min/max_chars` 只是审计告警，不是硬性要求。

### 3.2 「shortlist 排序错乱」

`rank=5` 的分高于 `rank=6`，看着像 bug。实际排序键是 **`mmr_utility`（降序）**，
不是 `score`：MMR 的多样性惩罚会故意把相似的高分条目往后压。属设计意图。

## 4 核查过、确认链路完整的部分

| 产物 | 展示位置 | 结论 |
|---|---|---|
| S6 文稿 | `episodes/{date}.txt` 文本 + 音频 | ✅ |
| S7 show_notes | 播客导语 + `episodes/{date}.html` | ✅ |
| S7 chapters | show notes 附录 + 章节导航 | ✅ |
| S7 items | `review_quality` 结构评分 | ✅ 不是死数据 |
| 非 AI 新闻 | shortlist 8 条全部 `ai_relevance=core` | ✅ 门槛正常 |

## 5 还没解决、且**代码解决不了**的问题

| 问题 | 性质 | 说明 |
|---|---|---|
| 证据缺年份（「11月24日」） | 来源质量 | 原文就这样，无法补全 |
| 单一来源占比 100% | 语料事实 | AI 新闻 98.7% 只有一家报道，已如实标注 |
| 同一天两条 OceanBase 内容 | 选题判断 | 产品发布与经营数据确实是两个事件，是否合并需人判断 |
| S5 的 corrections | 需人工/额外证据 | 「补统计期」「补分母」都需要证据里没有的信息 |

**S5 的 corrections 没有做「自动应用」**：抽查 5 条全部需要理解语义改写或补充
外部信息，机械套用会引入新错误。S5 的定位是**发现问题**，修复仍需人或针对性代码。

## 6 本地复现命令

```bash
# 日报客观性审计（单日）
uv run python -c "
import sys; sys.path.insert(0,'src')
from ai_news_podcast.business.contracts import Commentary, Fact
from ai_news_podcast.business.editorial.quality import commentary_quality
..."

# 运维快照（含日报事实/单源/缺出处/章节对齐）
uv run python scripts/ops_snapshot.py --date 2026-11-25 --data-dir data

# 文稿数字核对（从产物重建底稿）
uv run python scripts/script_factcheck.py --date 2026-11-25 --strict

# 端到端构建 digest，检查出处是否到达站点
uv run python scripts/redeploy_site.py --dry-run   # 或直接看 data/digest 产物
```

## 7 测试覆盖

| 文件 | 覆盖 |
|---|---|
| `tests/test_editorial_commentary.py` | 客观性闸门、出处兜底、LLM 优先、空 link 回填、转载降单源 |
| `tests/test_editorial_review.py` | S7 章节对齐（漂移校正/数量不符重建/缺失生成骨架） |
| `tests/test_editorial_quality.py` | 快照审计、文稿数字核对、claim number 口径 |
| `tests/test_digest_view.py` | 出处透传、verdict 标签渲染 |
| `tests/test_editorial_script.py` | 文稿数字核对触发重写 |

总计 1135 项通过（全量）。
