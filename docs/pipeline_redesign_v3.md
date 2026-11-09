# 流程细化设计 v3（Stage 1–7）：可执行细节

> 状态：待审核。审核通过后动工。
> 本文在 v2 大框架基础上，补齐每个 Stage 的接口、算法、参数、数据结构、失败处理、验收与调参计划。

## 0. 命名与编号

**03 板块定名候选（含“资讯”，避免“新闻/信源/证据”的严肃感）：**

| 候选 | 感觉 |
|---|---|
| **资讯潮**（推荐） | 短、有动态感，像每天的资讯浪潮 |
| 资讯海 | 量大、可下潜 |
| 资讯汇 | 聚合感 |
| 资讯流 | 信息流 |
| 资讯星图 | 偏技术/图谱 |

推荐 **「资讯潮」**；备选 资讯海 / 资讯汇 / 资讯流。等你拍板。

**Stage 编号**：

```
Stage 1 采集入海     → artifact（统一 item，含全文）
Stage 2 资讯潮       → 每日海洋版本（去重/聚类/连续性/标注/证据）
Stage 3 选题推荐     → shortlist（粗排 + 精排 + 理由 + 证据包）
Stage 4 编辑点评     → 02 深度日报
Stage 5 播客文稿     → 01 今日主推播客文稿
Stage 6 音频合成
Stage 7 发布
```

## 1. Stage 1：采集入海（细化）

### 1.1 接口

每个通道一个 fetcher 模块，统一签名：

```python
def fetch(source: SourceConfig, ctx: FetchContext) -> list[RawItem]
```

- `SourceConfig`：name、label、channel、url/route、parser、limit、enabled、failover；
- `FetchContext`：today、window_hours、http_client、rsshub_base、logger；
- 返回 `list[RawItem]`，失败抛异常或返回空列表（由调度层捕获并记录）。

### 1.2 源配置（config/sources_v3.yaml）

```yaml
sources:
  - name: anthropic_news
    label: Anthropic News
    channel: c3_rsshub
    route: /anthropic/news
    category: official
    limit: 15
    enabled: true
    failover: {channel: c1_rss, url: "https://www.anthropic.com/news/rss.xml"}
  - name: openai_news
    label: OpenAI News
    channel: c1_rss
    url: "https://openai.com/news/rss.xml"
    category: official
    limit: 30
```

- 字段校验：name 唯一、channel 合法、url 或 route 二选一；
- 一个源只出现一次；failover 仅主通道失败时启用。

### 1.3 时间窗口与条数

- `window_hours: 72`（可配）：只抓发布时间在窗口内的条目；
- `per_source_limit: 15–30`（按通道配置）；`max_pages: 3`；
- 无发布时间的条目：保留但标记 `published_at=null`，由 Stage 2 决定去留。

### 1.4 正文获取优先级

1. RSS/Atom 的 `content:encoded` / `content`（wechat2rss、部分博客）；
2. `description`/`summary` 若长度 ≥ 800 字，视为可用正文；
3. 否则采集时抓文章页，用 readability 抽正文（单页超时 20s，失败即标记）；
4. 全部失败：`full_text_status=failed`，`full_text` 留空，摘要保留。

### 1.5 URL 规范化（采集端唯一去重手段）

- 转小写 host、去 `www.` 前缀；
- 删除跟踪参数：`utm_*`、`fbclid`、`gclid`、`spm`、`from`、`ref`；
- 去 fragment（`#...`）；
- 去末尾 `/`；保留 query 中有语义的参数；
- 不跟随重定向（避免额外请求和反爬风险）。

### 1.6 采集端只做三件事

1. 同源内 URL 去重；
2. 时效过滤（窗口）；
3. 条数上限。

**不做**：跨源去重、聚类、打分、重要性判断。

### 1.7 ID 与输出

- `item_id = sha1(normalized_link)`；无链接时 `sha1(title + source)`；
- 输出 `artifact raw-{date}.json.gz`：

```json
{
  "date": "2026-11-09",
  "generated_at": "2026-11-09T16:06:00+08:00",
  "window_hours": 72,
  "sources": [{"name": "openai_news", "channel": "c1_rss", "count": 12, "status": "ok"}],
  "items": [ { "item_id": "...", "channel": "...", "source": "...", "title": "...", "link": "...", "published_at": "...", "fetched_at": "...", "summary": "...", "full_text": "...", "full_text_status": "ok", "lang": "en", "extra": {} } ]
}
```

### 1.8 失败处理

- 单源失败：记录 `status=failed` + 原因，继续其他源；
- 重试 2 次（指数退避 1s/3s），仍失败则跳过；
- RSSHub 容器起不来：C3 全部标记 skipped，其他通道照常；
- 全通道失败：Stage 1 失败，走日报告警（不阻塞历史数据）。

### 1.9 验收

- schema 校验 100% 通过；
- 每个启用源都有 count/status 记录；
- 抽 5 个源人工核对：标题/链接/时间/正文正确。

## 2. Stage 2：资讯潮（细化）

### 2.1 输入输出

- 输入：今日 artifact + 最近 30 天 ocean 版本 + `coverage/ledger.json`；
- 输出（每日版本，`ocean/{date}/`）：
  - `items.json`：条目级（含 event_id、continuity_id、labels）；
  - `events.json`：事件级（代表标题、成员、来源数、标签、证据分）；
  - `continuity.json`：事件线（continuity_id、sequence、历史成员）；
  - `manifest.json`：版本元数据（输入 artifact、模型、参数、统计）。

### 2.2 预处理

- 文本清洗（去 HTML、空白、emoji 归一）；
- 语言识别（zh/en，其余标记 other）；
- 关键词抽取（jieba TF-IDF，top 15）；
- 向量：优先 sentence-transformers（`paraphrase-multilingual-MiniLM-L12-v2`），
  未安装时降级 TF-IDF（与现有 dedup_service 一致）。

### 2.3 去重（四层，按顺序执行）

| 层 | 规则 | 参数（初值） | 输出 |
|---|---|---|---|
| L1 URL | 规范化后完全相同 | — | 丢弃后到者 |
| L2 标题 | 字符 n-gram Jaccard 或 rapidfuzz ratio | ≥ 0.90 | 合并为同一条 |
| L3 语义 | embedding cosine（或 TF-IDF 相似度） | embedding ≥ 0.80；TF-IDF ≥ 0.30 | 合并，保留最早/最全 |
| L4 跨日 | 与历史 ocean + ledger 比对 | 同 L1–L3 | 已选中 → 过滤；未选中 → 保留并标记 `reappeared=true` |

- 合并时保留规则：正文更长 > 官方源 > 更早发布时间；
- 每条 item 记录 `dup_of`（被合并到谁）与 `dup_reason`。

### 2.4 事件聚类

- **候选生成（分块）**：按“共享实体/关键词/时间窗（±48h）”生成候选对，避免 O(N²)；
- **相似度**：embedding cosine 0.55–0.80 区间 + 实体 Jaccard ≥ 0.5 → 视为候选；
- **聚类**：对候选图做连通分量 / HDBSCAN（min_cluster_size=2）；
- **代表标题**：选成员中“标题最短但含实体+动作”的，或来源数最多的标题；
- **输出**：`event_id`、`representative_title`、`members`（item_id 列表）、`source_count`、`first_seen_at`。

### 2.5 连续性识别

对今日每个事件，与历史事件（30 天）匹配：

- 匹配条件：实体集合 Jaccard ≥ 0.6，或代表标题相似 ≥ 0.75；
- 判定：
  - **无匹配** → 新事件（`sequence=1`，新 `continuity_id`）；
  - **有匹配且无新进展**（无新数字/版本/动作词）→ 同一事件（sequence 不变，标记 `update=false`）；
  - **有匹配且有新进展** → 持续事件（`sequence+1`，`update=true`）；
- 新进展检测：新数字（金额/参数/日期）、新版本号、新动作词（发布/开源/融资/合作/监管/事故）；
- 已选中且无新进展 → 过滤；未选中 → 保留。

### 2.6 多维标注

| 维度 | 抽取方式 |
|---|---|
| 主体 | 实体词典（公司/实验室/人物/产品） + 正则 |
| 对象 | 模型/产品/论文/政策/芯片（关键词表） |
| 动作 | 动词表（发布/开源/融资/合作/诉讼/事故/评测） |
| 技术栈 | LLM/Agent/多模态/芯片/机器人（关键词表） |
| 地域 | 国家/地区词典 |
| 生命周期 | 由 sequence + 来源数增长推导（爆发/演进/长尾/消退） |
| 证据强度 | 官方源标记 + 来源数 + 事实一致性（数字是否一致） |
| 主题 | 技术/商业/政策/研究/产品（关键词 + 规则） |

- 实体/关键词表放 `config/nlp/*.yaml`，可持续扩充；
- LLM 只用于“前 20 条事件的标签复核”（成本可控），默认规则优先。

### 2.7 证据分（event 级）

```
evidence_score =
  2.0 * official_source
  + 1.0 * min(source_count, 5)
  + 1.0 * consistent_facts
  + 0.5 * full_text_ratio
  - 1.0 * single_source
  - 1.0 * rumor_flag
```

### 2.8 每日版本语义

- 每天生成一个 `ocean/{date}` 快照（不可变版本，便于回溯）；
- 输入是“今日新增 + 最近 3 天”的滚动窗口，保证连续性和跨日去重；
- 历史版本保留 30 天（artifact），摘要可入 git。

### 2.9 性能预算

- 1600 条/天，分块后候选对 < 5 万，embedding 一次算完（约 1–3 分钟）；
- TF-IDF 降级路径更快；
- 内存 < 2GB。

### 2.10 质量评估与调参

- 每周抽 2 天做人工/LLM 评估：聚类纯度、连续性准确率、跨日去重准确率；
- 指标：precision/recall（抽样）、重复率、误杀率（把新进展当重复）；
- 阈值配置化，先给默认值，跑 2 周再调。

## 3. Stage 3：选题推荐（细化）

> 按你的意见：先跑一轮再调参，所有规则/权重放配置。

### 3.1 粗排（规则，高召回）

```yaml
stage3:
  coarse:
    window_hours: 48          # 新事件
    continuity_window_days: 7 # 持续事件放宽
    min_sources: 2            # 或 official=true
    exclude_keywords: [招聘, 课程, 广告, 优惠, 榜单, 抽奖]
    require_ai_topic: true
    dedupe_against_covered_days: 30
```

- 规则全部可配；每步记录淘汰原因（便于审计和调参）。

### 3.2 精排（加权 + 多样性）

| 特征 | 初值权重 |
|---|---|
| impact（影响力） | 0.30 |
| novelty（新颖度） | 0.25 |
| momentum（势头/连续性） | 0.15 |
| audience_fit（受众相关） | 0.10 |
| narratability（可讲性） | 0.10 |
| evidence（证据强度） | 0.10 |
| risk（风险，负向） | -0.10 |

- **多样性约束（MMR）**：同一主体/同一技术栈最多 2 条；避免全是一个事件；
- **LLM 复核**：只对 top 20 逐条问“值不值得讲”，返回 0–5 分与一句理由；与规则分加权（规则 0.7 + LLM 0.3）；
- 输出 `shortlist.json`：rank、score、reasons（规则命中 + LLM 理由）、evidence_items。

### 3.3 调参与评估（边跑边调）

1. **离线回测**：用最近 14 天 artifact 重跑 Stage 3，得到每天 top 8；
2. **评估**：人工或 LLM 按 rubric 打分（重要性/新颖/可讲/证据），算 precision@5、覆盖率、重复率；
3. **调参**：每次只改 1–2 个参数（权重/阈值），记录版本与结果；
4. **上线**：先 shadow 运行（不影响正式节目），对比 2 周后切换。

### 3.4 边界情况

- 无候选：降级到“当日来源数最多”的事件；
- 全是同一事件：强制多样性，取不同主体；
- 突发大事件：允许同一事件 1 条主推 + 1 条“多角度”；
- 疑似谣言：`rumor_flag` 降权，不推荐。

## 4. Stage 4：编辑点评（02 深度日报）

### 4.1 输入

shortlist 前 N（默认 6）+ 事件证据包（成员 item、来源、连续性上下文）。

### 4.2 输出结构（每条）

```json
{
  "event_id": "...",
  "headline": "……",
  "one_liner": "一句话摘要",
  "why_matters": "为什么重要",
  "facts": [{"text": "……", "evidence_ids": ["item_id"]}],
  "background": "背景脉络（来自连续性）",
  "impact": "影响与观点",
  "risk": "风险/不确定性",
  "sources": [{"label": "...", "link": "..."}]
}
```

### 4.3 约束

- 事实必须来自证据包（facts 带 evidence_ids）；
- 禁止编造数字；不确定就写“据报道”；
- 每条 150–300 字；
- 日报按“重要性降序 + 主题分组”。

## 5. Stage 5：播客文稿（01 今日主推播客）

### 5.1 输入

shortlist + Stage 4 点评 + 连续性上下文；主持人口径：林夏 / 陈默。

### 5.2 结构

- 开场（今日主题一句话）
- 3–5 个章节（对应主推新闻，每章含事实/背景/观点/互动）
- 结尾（总结 + 明日看点）
- 长度：25–35 分钟口播（约 4000–5500 字）

### 5.3 约束

- 事实与点评一致，不引入未在证据中的信息；
- 禁止列表式的“第一、第二”，保持对话感；
- 保留 `[CHAPTER]` 标记供章节导航；
- 输出 `site/episodes/{date}.txt`。

## 6. Stage 6–7（保持现有）

- Stage 6：CosyVoice2 双主持多音色合成，24kHz，产物 `{date}.mp3`；
- Stage 7：重建站点/feed、更新 episodes.json、部署 gh-pages。

## 7. 调度与编排

- daily.yml Job 链：`collect(Stage1) → ocean(Stage2) → select(Stage3) → commentary(Stage4) → writer(Stage5) → tts(Stage6) → publish(Stage7)`；
- Stage 1–2 的产物通过 artifact 传递（30 天）；
- 单 job 失败重试 1 次；notify 告警逻辑沿用现有。

## 8. 配置汇总（config/pipeline_v3.yaml）

```yaml
stage1:
  window_hours: 72
  per_source_limit: 15
  http_timeout: 20
stage2:
  title_dedup_threshold: 0.90
  semantic_embedding_threshold: 0.80
  semantic_tfidf_threshold: 0.30
  cluster_sim_threshold: 0.60
  entity_jaccard_threshold: 0.50
  continuity_entity_threshold: 0.60
  continuity_title_threshold: 0.75
  history_days: 30
stage3:
  shortlist_size: 8
  llm_judge_top_k: 20
  weights: {impact: 0.30, novelty: 0.25, momentum: 0.15, audience_fit: 0.10, narratability: 0.10, evidence: 0.10, risk: -0.10}
stage4:
  report_top_n: 6
stage5:
  script_minutes: 30
```

## 9. 测试与验收

- 单元测试：每层去重、聚类、连续性、粗排/精排规则、schema 校验；
- 集成测试：用 2 天历史 artifact 跑通 Stage 1→3，产出 shortlist；
- E2E：mock LLM/TTS 跑全链；
- 验收指标：Stage 2 聚类纯度 ≥ 0.85（抽样）、跨日误杀率 ≤ 5%、Stage 3 top5 人工可接受率 ≥ 70%（首版）。

## 10. 迁移计划

1. 新增 Stage 1/2 模块与 artifact 契约，旧 pipeline 保持运行；
2. Stage 3 先 shadow 运行，与旧选材对比；
3. 切换节目生成到新链，旧 brief 兼容 2 周；
4. 清理旧代码。

## 11. 待确认

1. 03 板块名：**资讯潮**（推荐）/ 资讯海 / 资讯汇 / 资讯流？
2. Stage 3 首版精排权重是否按上表默认值？
3. Stage 4 点评条数 6 条、Stage 5 时长 30 分钟，是否合适？
