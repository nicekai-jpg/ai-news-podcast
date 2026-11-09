# 流程细化设计 v4：Stage 1–7（标注、推荐、点评、文稿、模型）

> 状态：待审核。本文回答 v3 评审中的 8 个问题，并给出可执行细节。

## 0. 03 板块命名（带“新闻”）

| 候选 | 说明 |
|---|---|
| **新闻潮汐**（推荐） | 有“潮起潮落”的连续性意象，比“海洋/星图”轻，适合每日节奏 |
| 新闻潮 | 短、有力 |
| 新闻海 | 量大、可下潜 |
| 新闻浪 | 轻快 |
| 新闻汇 / 新闻流 | 聚合感 / 信息流 |

推荐 **「新闻潮汐」**；备选 新闻潮 / 新闻海 / 新闻浪。

## 1. Stage 2 标注：怎么标、后面怎么用

### 1.1 标注原则

1. **规则优先**：实体词典 + 关键词表 + 正则，保证可解释、可迭代；
2. **LLM 只补充**：只对“前 20 条事件”做标签复核，控制成本；
3. **每条标签带置信度和来源**（rule/llm），便于审计；
4. **多标签**：一个事件可同时有多个主体、技术栈、主题；
5. **词表可迭代**：`config/nlp/*.yaml` 可持续扩充，不写死在代码里。

### 1.2 词典结构（config/nlp/）

```yaml
# entities.yaml
companies:   # 主体（公司/实验室）
  - id: openai
    name: OpenAI
    aliases: [Open AI, 开放人工智能]
    type: company
    weight: 1.0          # 影响力权重（决定 impact）
  - id: deepseek
    name: DeepSeek
    aliases: [深度求索]
    type: company
    weight: 0.9
labs:
  - id: deepmind
    name: DeepMind
    aliases: [谷歌深度思维]
people:
  - id: sama
    name: Sam Altman
    aliases: [萨姆·奥特曼]
# objects.yaml：模型/产品/论文/政策/芯片
# actions.yaml：发布/开源/融资/合作/诉讼/事故/评测/招聘/监管
# tech.yaml：LLM/Agent/多模态/芯片/机器人/推理/训练/数据
# regions.yaml：中国/美国/欧洲/日本/韩国/全球 + 城市
```

### 1.3 各维度抽取方法

| 维度 | 方法 | 输出 | 置信度 |
|---|---|---|---|
| **主体** | 实体词典 + 别名归一（Aho-Corasick/多模匹配） | `subjects[]`（id、canonical、weight） | 高（词典命中） |
| **对象** | 对象词典 + 句式模式（“发布/开源/上线/推出 + 对象”） | `object_type`（model/product/paper/policy/chip） | 中 |
| **动作** | 动词表 + 优先级规则（事故>诉讼>融资>发布>评测） | `action`（主动作）+ `actions[]` | 中 |
| **技术栈** | 关键词表（LLM/Agent/多模态/芯片/机器人…） | `tech_stack[]` | 高 |
| **地域** | 国家/城市词典 + 公司总部映射 | `regions[]` | 中 |
| **生命周期** | 由 sequence、来源数趋势、距首见天数推导 | `lifecycle`（burst/evolving/long_tail/fading） | 高（规则） |
| **证据强度** | 公式（见 1.6） | `evidence_tier`（强/中/弱/传闻）+ 分数 | 高（可计算） |
| **主题** | 主题词典（技术/商业/政策/研究/产品/资本/监管） | `topics[]` | 中 |

### 1.4 生命周期判定规则

| 状态 | 条件 |
|---|---|
| `burst`（爆发） | sequence=1 且 24h 内来源数 ≥3 |
| `evolving`（演进） | sequence≥2 且存在新进展 |
| `long_tail`（长尾） | 距首见 ≥7 天且仍有零散新报道 |
| `fading`（消退） | 连续 3 天无新增报道 |

### 1.5 标注流水线

```
Stage 2 规则标注（全部事件）
   → 置信度低的字段打标 needs_review
   → 前 20 条事件送 LLM 复核（rubric：主体/动作/主题/证据）
   → 合并：规则优先，LLM 只补缺/纠错
   → 每周人工抽检 20 条，更新词典
```

### 1.6 证据强度公式与分级

```
evidence_score =
  2.0 * official_source
  + 1.0 * min(independent_sources, 5)
  + 1.0 * fact_consistency      # 跨源数字/版本一致
  + 0.5 * full_text_ratio
  - 1.0 * single_source
  - 1.5 * rumor_flag
```

| 分级 | 阈值 |
|---|---|
| 强 | ≥ 4.0 |
| 中 | 2.5–4.0 |
| 弱 | 1.0–2.5 |
| 传闻 | < 1.0 或 rumor_flag |

### 1.7 标注的下游用途（不白标）

| 下游 | 用到哪些标签 | 怎么用 |
|---|---|---|
| **Stage 3 粗排** | topics / tech_stack / regions / action / lifecycle / evidence_tier / subjects | 相关性过滤、证据门槛、时效分层、排除招聘/广告 |
| **Stage 3 精排** | subject.weight（impact）、object/version（novelty）、source_count/sequence（momentum）、tech_stack（audience_fit）、action+数字（narratability）、rumor/single_source（risk） | 打分特征 |
| **Stage 4 点评** | continuity + subjects（background）、evidence（facts 引用）、topics/regions（impact） | 背景脉络与事实核对 |
| **Stage 5 文稿** | topics / subjects（章节分组）、lifecycle（“后续进展”）、tech_stack（受众解释深度） | 章节编排与多样性 |
| **02 日报** | topics / regions / lifecycle | 分组与排序 |
| **运营分析** | 全部标签 | 来源质量、主题分布、连续性统计 |

---

## 2. 跨日 30 天窗口是什么

**三个不同的窗口，别混：**

| 窗口 | 含义 | 用途 |
|---|---|---|
| 采集窗口 72h | 每个源只抓最近 72h 的条目 | 控制 Stage 1 体积 |
| 滚动输入 3 天 | Stage 2 每天输入“今日新增 + 最近 3 天” | 保证连续性判断有上下文 |
| **去重比较窗口 30 天** | 与最近 30 天的 ocean 版本 + 覆盖账本比对 | 识别“第 5 天/第 3 周再次出现”的旧闻 |

- **为什么是 30 天**：与 artifact 保留期一致；覆盖一个自然月；48k 条量级可承受；
- **超过 30 天怎么办**：`coverage/ledger.json` 永久保留“已选中故事”的指纹（标题/URL/实体/关键词），仍能挡住老闻重现；
- **性能**：URL 精确匹配用哈希索引；语义比对只对“实体/关键词分块后的候选对”计算，不做全量 O(N²)；
- **配置**：`history_days: 30` 可调。

---

## 3. Stage 3 粗排：扩展维度（不只看时效和来源数）

| # | 维度 | 规则（初值） | 作用 |
|---|---|---|---|
| 1 | 时效分层 | 新事件 ≤48h；持续事件 ≤7 天；长尾 ≤30 天且须有新进展 | 保新 |
| 2 | 独立来源数 | ≥2 个不同来源（同一集团算 1） | 保真 |
| 3 | 来源质量 | 官方源直接通过；只有自媒体且 <3 源 → 降权 | 保真 |
| 4 | 事实一致性 | 跨源数字/版本一致率 <0.5 → 标记待核 | 防错 |
| 5 | 主体重要性 | 主体权重 <0.3 且无知名实体 → 降权 | 保重要性 |
| 6 | 主题相关性 | topics 命中 AI 主题；纯财经/招聘/课程 → 淘汰 | 保相关 |
| 7 | 技术栈匹配 | 命中 LLM/Agent/芯片/机器人等 | 保相关 |
| 8 | 地域偏好 | 中国/全球优先；纯地方小事 → 降权 | 受众 |
| 9 | 动作类型 | 发布/开源/融资/政策/事故优先；软文/评测 → 降权 | 价值 |
| 10 | 生命周期 | burst/evolving 优先；fading → 淘汰 | 势头 |
| 11 | 新颖度 | 新版本/新数字/新主体 → 加分；纯复述 → 淘汰 | 防重复 |
| 12 | 可讲性 | 全文 ≥800 字且有明确数字/对比/冲突 | 可讲 |
| 13 | 风险合规 | 传闻/单一来源/敏感/标题党 → 降权或淘汰 | 安全 |
| 14 | 去重/覆盖 | 命中 ledger 已选 → 淘汰；未选重现 → 保留 | 防重复 |
| 15 | 多样性预筛 | 同主体/同技术栈先各留 2 条 | 防扎堆 |
| 16 | 当日饱和度 | 某主题当日已选 ≥2 → 降权 | 平衡 |
| 17 | 突发加权 | 当日来源数激增（≥5）且新主体 → 加分 | 抓热点 |

每条规则记录**淘汰/降权原因**，便于审计和调参。

---

## 4. Stage 3 精排：不止一张权重表

### 4.1 四种方案对比

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| A 线性加权 | 特征 × 权重求和 | 可解释、易实现 | 权重主观、难评价 |
| B LTR（学习排序） | 用人工/LLM 标注训练 pairwise/pointwise | 数据够时效果好 | 冷启动无标注 |
| C LLM 成对比较 | 对 top K 两两比较，Bradley-Terry/ELO 出排名 | 相对判断稳定，接近人评 | 成本高、需控 K |
| **D 混合（推荐）** | 规则分初筛 + LLM 按 rubric 打分/成对比较 + 加权融合 | 兼顾可解释与质量，可评估 | 实现稍复杂 |

### 4.2 推荐方案 D 的流程

```
粗排 → 候选 top 20
  → LLM 按 rubric 打分（0–5）：重要性 / 新颖度 / 影响范围 / 证据强度 / 可讲性 / 风险
  → 规则分归一化 + LLM 分归一化
  → final = 0.6 * rule_score + 0.4 * llm_score
  → MMR 多样性约束（同主体/同技术栈 ≤2）
  → top 8
```

- LLM 只处理 top 20，成本可控；
- rubric 固定、可复现（同一 rubric 可跨天比较）；
- 规则分保证“兜底不跑偏”。

### 4.3 怎么评价好坏（这是关键）

1. **离线回测**：最近 14–30 天 artifact 重跑 Stage 3，得到每天 top 8；
2. **评测集**：人工或强 LLM 对 top 8 逐条按 rubric 打分（1–5），得到偏好标签；
3. **指标**：NDCG@5、precision@3（人工可接受率）、主题覆盖率、同主体重复率；
4. **成对一致性**：LLM pairwise 与人工 pairwise 的一致率（目标 ≥70%）；
5. **shadow 运行**：新排序与现网并行 2 周，对比最终选用结果；
6. **调参纪律**：每次只改 1–2 个参数，记录配置版本与指标变化。

### 4.4 冷启动策略

- 前 2 周：规则主导（0.8 规则 + 0.2 LLM），先积累偏好数据；
- 之后：逐步提高 LLM 权重（0.6/0.4 → 0.5/0.5）；
- 标注数据足够后，可升级到方案 B（LTR）。

---

## 5. Stage 4 编辑点评：格式与客观性

### 5.1 输出 schema（事实与观点分离）

```json
{
  "event_id": "...",
  "headline": "……",
  "one_liner": "一句话事实摘要",
  "facts": [
    {"text": "……", "evidence_ids": ["item_1", "item_2"], "single_source": false}
  ],
  "why_matters": "为什么重要（分析）",
  "background": "背景脉络（来自连续性）",
  "impact": "影响（分析，标注为分析）",
  "counterpoint": "反方/不同观点（有则写，无则空）",
  "uncertainty": "不确定/待核实项",
  "sources": [{"label": "...", "link": "...", "official": true}]
}
```

### 5.2 客观性规则（写进 prompt + 自动检查）

1. 事实必须对应 `evidence_ids`；无出处不写；
2. 数字、版本、金额**不得改写/四舍五入到失真**；
3. 单一来源的事实标注“据 X 报道”；
4. **事实与观点分栏**：impact/why_matters 属分析，不冒充事实；
5. 不偏袒：有反方观点必须写 counterpoint；无则写“暂无公开反方信息”；
6. 禁用营销词/夸张词（“颠覆”“史上最强”“吊打”等）；
7. 不确定就写不确定，不补脑；
8. 涉及事故/监管等敏感话题，措辞克制。

### 5.3 自动质量检查

- facts 的 evidence_ids 必须存在；
- 数字在原文可找到；
- 禁用词表扫描；
- 每条字数 150–300；
- 单一来源比例统计。

---

## 6. Stage 5 播客文稿：格式、模型、时长

### 6.1 15 分钟的长度预算

| 项 | 目标 |
|---|---|
| 总时长 | **≤15 分钟** |
| 语速估算 | 220–260 字/分钟（CosyVoice 中文） |
| 目标字数 | **3000–3600 字**（按 15 分钟 × 230 字/分 ≈ 3450） |
| 章节数 | 3–4 个 |
| 每章 | 约 3–4 分钟（700–900 字） |
| 开场/结尾 | 各 30 秒（约 120 字） |

### 6.2 文稿格式（模板）

```
[CHAPTER] 开场
林夏：……（今日一句话）
陈默：……（为什么值得听）

[CHAPTER] 主推 1：标题
林夏：事实播报（含数字/来源）
陈默：解读/背景（来自连续性）
林夏：影响/争议
陈默：一句过渡

[CHAPTER] 主推 2：标题
……

[CHAPTER] 收尾
林夏：总结
陈默：明日看点
```

- 主持人口径：**林夏（Host A）/ 陈默（Host B）**；
- 保留 `[CHAPTER]` 供章节导航；
- 禁止“第一、第二、第三”列表式；保持对话感；
- 事实必须与 Stage 4 点评一致；不引入证据外信息。

### 6.3 模型：现在用什么、还有什么

| 模型 | 现状/能力 | 建议用途 |
|---|---|---|
| **MiniMax-M3** | 当前配置（`config.yaml: model: MiniMax-M3`） | Stage 4 点评 + Stage 5 文稿（质量优先） |
| **MiniMax-M3.1 Flash** | 2026-10-19 发布；官方称推理快约 3 倍、成本降约 70%；强项 agent/coding/computer use | Stage 3 的 LLM 复核、Stage 2 标签复核（便宜快）；文稿可 A/B 试 |
| 其他可插拔 | `llm_backends` 已支持 OpenAI 兼容后端（DeepSeek/GLM 等） | 可做备选/降本，但需评估中文文风 |

**建议**：
1. 加 `stage_models` 配置，支持每个 Stage 指定模型；
2. Stage 3/2 用 M3.1 Flash 试跑（成本敏感）；
3. Stage 4/5 先保持 M3，做 A/B（M3 vs M3.1 Flash）看文稿质量；可接受再切换；
4. CI 用真实 key 验证 M3.1 Flash 的模型名（以平台文档为准，可能是 `MiniMax-M3.1-Flash`）。

### 6.4 文稿质量检查（自动）

- 字数 3000–3600；章节 3–4；主持人字符占比均衡；
- 禁用词扫描；`[CHAPTER]` 完整；
- 与点评事实一致性抽查；
- 超长/过短触发重写。

---

## 7. 配置更新示例

```yaml
stage2:
  history_days: 30
  llm_label_top_k: 20
stage3:
  shortlist_size: 8
  llm_judge_top_k: 20
  weights: {rule: 0.6, llm: 0.4}
  rubric: [importance, novelty, impact, evidence, narratability, risk]
stage4:
  report_top_n: 5
stage5:
  target_minutes: 15
  min_chars: 3000
  max_chars: 3600
  chapters: 4
llm:
  model: "MiniMax-M3"
  stage_models:
    stage2: "MiniMax-M3.1-Flash"
    stage3: "MiniMax-M3.1-Flash"
    stage4: "MiniMax-M3"
    stage5: "MiniMax-M3"
```

## 8. 待确认

1. 03 板块名用「**新闻潮汐**」还是 新闻潮/新闻海/新闻浪？
2. Stage 3 采用方案 D（规则 0.6 + LLM 0.4，start）？
3. M3.1 Flash 先用于 Stage 2/3（低成本任务），Stage 4/5 A/B，可以吗？
4. 15 分钟对应 3000–3600 字、3–4 章，点评 5 条，是否合适？
