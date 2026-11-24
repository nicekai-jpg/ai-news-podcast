# 推荐流程评测（Stage 3）

目标：把“推荐准不准”从感觉变成可复现的数字。骨架：

```
events.json（Stage 2 产物）
        │
        ▼
标注 JSONL（人工/助审，200 条）
        │
        ▼
scripts/eval_recommend.py
        │
        ├── 召回：正样本召回率 / 负样本漏入率
        ├── 粗排：正样本保留率 / 负样本漏入率
        └── top8：精确率 / 正样本覆盖 / 负样本占比
```

## 标注格式

每行一个 JSON：

```json
{"title":"OpenAI 发布新模型","summary":"...","ai_relevance":"core","worth_publishing":true,"label_source":"human","reviewed":true,"note":""}
```

| 字段 | 取值 | 说明 |
|---|---|---|
| `title` | 事件标题 | 必须和 `events.json` 的 `representative_title` 归一化后一致 |
| `ai_relevance` | `core` | AI 模型、Agent、算法、AI 产品、AI 研究 |
| `ai_relevance` | `adjacent` | AI 芯片、算力、机器人、自动驾驶、具身智能 |
| `ai_relevance` | `none` | 与 AI 无关（汽车、消费、股市、社会） |
| `ai_relevance` | `unknown` | 信息不足，不参与正/负样本统计 |
| `worth_publishing` | `true` | 值得上节目 |
| `worth_publishing` | `false` | 不值得上节目 |
| `label_source` | `human` | 人工标注（金标准） |
| `label_source` | `assistant_seed` | 助审种子，仅供冒烟测试 |
| `reviewed` | `true` | 已人工复核 |

## 生成标注模板（多天分层）

```bash
uv run python scripts/build_eval_template.py \
  --events data/ocean/2026-11-09/events.json \
           data/ocean/2026-11-10/events.json \
           data/ocean/2026-11-11/events.json \
  --out data/eval/recommend/review_queue_2026-11.jsonl \
  --quota core=250,adjacent=100,none=100,unknown=150
```

抽样规则：

- 跨天合并、按标题去重；
- 先按配额抽 `core / adjacent / none / unknown`，某组不足时用其他组补齐；
- 稀有负样本（`none`）全部纳入，不随机丢弃；
- 每题带 `suggested_ai_relevance`（机器建议，仅加速人工，不是金标准）。

填好后把每条 `ai_relevance` 和 `worth_publishing` 写上，`label_source` 保持 `human`、
`reviewed` 改为 `true`。

## 跑评测

```bash
uv run python scripts/eval_recommend.py \
  --events data/ocean/2026-11-11/events.json \
  --labels data/eval/recommend/seed_2026-11-11.jsonl \
  --reference-time 2026-11-11T08:00:00+00:00 \
  --out-md data/eval/recommend/report_2026-11-11.md
```

`--reference-time` 建议固定，否则“时效衰减”会随运行时间变化，指标不可复现。

## 指标口径

| 指标 | 计算 | 看什么 |
|---|---|---|
| 召回正样本召回率 | 被召回的正样本 / 标注正样本 | 漏没漏 AI 新闻 |
| 召回负样本漏入率 | 被召回的负样本 / 标注负样本 | 汽车/消费新闻有没有混进来 |
| 粗排正样本保留率 | 粗排留下的正样本 / 召回正样本 | 粗排误杀多少 |
| 粗排负样本漏入率 | 粗排留下的负样本 / 召回负样本 | 粗排挡住多少 |
| top8 精确率 | top8 正样本 / top8 已标注条目 | 节目质量 |
| top8 正样本覆盖 | top8 正样本 / 全部标注正样本 | 有没有漏掉重要新闻 |
| top8 负样本占比 | top8 负样本 / top8 条目 | 有没有“不 AI”混进节目 |

未标注的 top8 条目单独计入 `unknown`，不污染精确率分母。

## 当前状态（2026-11-11）

- 指标脚本、标注模板、多天汇总已落地；
- 种子标注 `seed_2026-11-11.jsonl` 16 条，`assistant_seed`、`reviewed=false`，只能冒烟；
- 已生成多天分层队列 `review_queue_2026-11.jsonl` 600 条：core 320、adjacent 82、none 48、unknown 150；
- 可用的 v3 artifact 只有 2026-11-09 / 11-10 / 11-11 三天，所以队列覆盖 3 天；
  负样本池本身只有 48 条（AI 源里汽车/消费新闻少），已全部纳入；
- 人工标注 600 条预计 1.5–2 小时；至少完成 200 条后指标才有参考价值，600 条做完再用于调权重；
- 种子冒烟结果仅作链路验证：召回正样本 8/8、负样本漏入 2/8、top8 精确率 80%。

## 样本规模怎么算够

| 用途 | 最少标注 | 说明 |
|---|---|---|
| 链路冒烟 | 20 | 只验证代码和指标能算 |
| 阶段阈值调整 | 200 | 每类至少 30 条 |
| 权重/LLM 校准 | 600 | 每类至少 100 条 |
| 上线前回归 | 1000 | 覆盖 7 天以上、多来源 |

样本要**跨天**、**分层**、**包含负样本**；单天 16 条不能用来下结论。

## 局限

- 标题精确匹配：事件合并后标题可能被改写，未匹配会列在报告里；
- 只评推荐链路，不评日报/文稿/音频质量；
- `worth_publishing` 目前只记录，不参与指标（后续可做“值得讲”的排序评测）。

## 分次标注（推荐）

先按天拆分，再逐天标，避免一次 600 条压力太大：

```bash
# 1. 按天拆分
uv run python scripts/split_review_queue.py \
  --queue data/eval/recommend/review_queue_2026-11.jsonl \
  --out-dir data/eval/recommend/days

# 2. 交互式标注（逐条写盘，Ctrl-C 可中断，重跑自动跳过已标）
uv run python scripts/label_queue.py \
  --queue data/eval/recommend/days/2026-11-11.jsonl \
  --out data/eval/recommend/labels_2026-11-11.jsonl \
  --limit 50

# 3. 标完当天就跑一次评测
uv run python scripts/eval_recommend.py \
  --events data/ocean/2026-11-11/events.json \
  --labels data/eval/recommend/labels_2026-11-11.jsonl \
  --reference-time 2026-11-11T08:00:00+00:00
```

回车 = 接受机器建议；`s` = 跳过；`c/a/n/u` 手动指定 AI 相关性；`y/n` 选是否值得上节目。

## Stage 2 的 ai_relevance 分级（规则版已上线）

Stage 2 标注新增两个字段：

| 字段 | 取值 | 说明 |
|---|---|---|
| `ai_relevance` | core | AI 模型、Agent、算法、AI 产品、AI 研究 |
| `ai_relevance` | adjacent | 芯片、算力、机器人、自动驾驶、数据中心 |
| `ai_relevance` | none | 无任何 AI 信号 |
| `ai_relevance` | unknown | 信息不足 |
| `ai_evidence` | 关键词列表 | 命中的证据词，便于人工审计 |

判定顺序：

```
标题+摘要命中 core 词             → core   (confidence 0.8)
标题+摘要只命中 infra 词          → adjacent (0.7)
正文命中 core/infra 词（仅 official/research）→ core / adjacent (0.55 / 0.5)
无任何命中                        → none   (0.6)
标题过短                          → unknown (0.2)
```

推荐门槛比标签更严：**标题必须带 AI 信号**才放行（一手/研究来源例外）。
汽车/消费新闻即使正文提到“自动驾驶/AI”，只要标题没有 AI 信号，就不会进推荐。

推荐链路（召回/粗排）优先信标签等级：

```
core / adjacent  → 直接放行
none             → 直接淘汰
unknown          → 回退到标题+摘要关键词门槛
```

规则版先保证零成本、可解释；等评测集标注完成后再决定是否用 LLM 复核
（只对召回候选，成本可控）。

## 主题域分级（防止“今天汽车、明天别的”）

只看标题里有没有 AI 词不够：汽车、金融、消费、体育、娱乐、社会新闻都可能偶尔提到 AI。
Stage 2 再加一层**主领域**判定：

| 主领域 | 含义 |
|---|---|
| `ai_core` | 模型、Agent、算法、AI 产品、AI 研究 |
| `ai_infra` | 芯片、算力、机器人、自动驾驶、数据中心 |
| `other` | 汽车、金融、消费、体育、娱乐、社会、政治 |
| `unknown` | 信息不足 |

规则（只用标题判定，摘要/正文只给 official/research 兜底）：

```
标题命中强 AI 词            → ai_core
标题命中自动驾驶/智能驾驶   → ai_infra
标题命中汽车/金融/消费等词  → other
标题命中基础设施词          → ai_infra
否则                        → unknown（回退标题 AI 关键词）
```

推荐门槛：

```
other             → 直接淘汰
ai_core/ai_infra  → 放行
unknown           → 标题带 AI 关键词才放行
```

这样不用为“汽车/股票/手机/体育”逐个打补丁：非 AI 主领域统一被 `other` 挡住。

## 词表覆盖流程（发现一个补一个）

规则不可能理解一切，但常见类目有限，做法是**覆盖 + 回归**：

```
每天 events artifact
        ↓
scripts/domain_coverage.py
        ↓
unknown / other 标题清单（人工扫一眼）
        ↓
把漏掉的词加到 config/domain_keywords.yaml
        ↓
跑回归（domain/relevance/recall/coarse 测试 + 历史 artifact）
        ↓
合入，下一期生效
```

命令：

```bash
# 看覆盖缺口（unknown 最需要补词）
uv run python scripts/domain_coverage.py \
  --events data/ocean/2026-11-12/events.json \
  --out-md data/domain_coverage/report.md \
  --out-queue data/domain_coverage/review.jsonl
```

补词原则：

- `other_category`：明显非 AI 的类目（汽车、金融、消费、体育、娱乐、社会）；
- `auto_brands`：汽车品牌，避免“品牌出现=AI”；
- `ai_core_extra`：新模型/研究常用英文词（diffusion、tokenizer、world model 等）；
- `ai_driving`：自动驾驶/智能驾驶类，优先于汽车类目；
- 加完必须跑测试，确认没有把已通过的 AI 新闻误杀。

---

## P@k 评测与降权消融（2026-11-24）

> ⚠️ 本节前半部分的 0.85 / 0.92 只基于 3 天金标。扩到 8 天后 P@8 实测为
> **0.49**，3 天样本过于乐观，结论以文末「8 天复核」为准。

### 为什么要单独一个 P@k 脚本

`scripts/eval_recommend.py` 用 `ai_relevance` 判正样本，等于评「是不是 AI 新闻」；
真实目标是「值不值得播」。新脚本 `scripts/recommend_p8.py` 按金标
`worth_publishing` 判，分天输出 P@k，已判样本不足的单列 `?`。

```bash
# 规则分（默认）
uv run python scripts/recommend_p8.py --show-picks
# 生产同款 LLM 精排（需要 MINIMAX_API_KEY，约 5 分钟）
uv run python scripts/recommend_p8.py --llm --show-picks
```

### 基线

| 日期 | shortlist | 已判 | 正例 | P@8 |
|---|---|---|---|---|
| 2026-11-09 | 8 | 6 | 5 | 0.83 |
| 2026-11-10 | 8 | 3 | 2 | 0.67 |
| 2026-11-11 | 8 | 4 | 4 | 1.00 |
| 合计 | — | 13 | 11 | **0.85** |

### 活动/综述降权消融

| 配置 | 正例/已判 | P@8 |
|---|---|---|
| 基线 | 11/13 | 0.85 |
| 仅活动词（读书会/沙龙/报名/直播） | 11/12 | 0.92 |
| 活动词 + 评审 | 11/12 | 0.92 |
| 再加论坛/峰会/大赛/综述/盘点/回顾/白皮书 | 12/13 | 0.92 |
| 再加解析/详解/入门/科普/教程/指南/如何实现 | 12/13 | 0.92 |

结论：**收益几乎全部来自去掉「读书会」这类活动宣传**。宽词表与解析类词没有
增量，反而可能误伤真新闻（论坛/峰会常是真事件）。最终只保留最小一组：
`读书会 / 沙龙 / 报名 / 直播 / 评审`，全部走 `demote`（-0.6 降权）而非淘汰。

### LLM 精排对比（同一金标）

| 配置 | 正例/已判 | P@8 | 说明 |
|---|---|---|---|
| 规则分（最小降权集） | 11/12 | 0.92 | 11-09 5/5、11-10 2/3、11-11 4/4 |
| 规则 + LLM rubric（生产同款） | 11/12 | 0.92 | 11-09 5/6、11-10 3/3、11-11 3/3 |

结论：**LLM 精排没有提升 P@8**。它确实修正了 11-10 的「RSI 框架解析」，
但在 11-09 引入了另一个误选（Nvidia 科普向英文稿），且会让部分已判样本
掉出 shortlist（分母变化）。在 n=12 的样本上这个差异属于噪声。

因此下一步不该继续调精排词表/权重，而应扩大金标到 8 天以上再下结论
（n≥30 才有统计意义）。

### 已知盲区

- 评测只覆盖推荐链路，不评日报/文稿/音频；
- `recommend_p8.py` 用标题精确匹配金标，事件标题被改写会漏匹配（记为 `?`）；
- LLM 精排有随机性（temperature 0.2），单次结果不代表稳定水平。

### 8 天复核（2026-11-24 追加）：3 天样本过于乐观

把评测扩到 8 天（2026-11-09 ~ 11-16）。新增 5 天只标注「可能进 shortlist 的候选池」
（每天 top20 重排 + 新旧 shortlist 并集，共 55 条），保证改配置后仍有标注可用。

| 日期 | 已判 | 正例 | P@8 |
|---|---|---|---|
| 11-09 | 5 | 4 | 0.80 |
| 11-10 | 3 | 2 | 0.67 |
| 11-11 | 5 | 5 | 1.00 |
| 11-12 | 8 | 3 | 0.38 |
| 11-13 | 8 | 3 | 0.38 |
| 11-14 | 8 | 6 | 0.75 |
| 11-15 | 8 | 2 | 0.25 |
| 11-16 | 8 | 1 | 0.12 |
| **合计** | **53** | **26** | **0.49** |

**结论：真实 P@8 约 0.49，不是 0.85~0.92。** 前 3 天恰好是表现好的样本。

#### 误选构成

对 8 天全部误选按金标理由归类：

| 类型 | 条数 | 例 |
|---|---|---|
| 观点/经验/方法论 | 10 | 「AI 时代，公司的新厂房长什么样？」 |
| 其他（难归类） | 7 | 海外芯片/设备类报道 |
| 评测/推广/软文 | 4 | 「How Amplitude ships product improvements」 |
| 论文/学术综述 | 2 | TPAMI 综述、AV reasoning 论文 |
| 旧闻/重复 | 2 | 同一篇观点稿连续 3 天进 shortlist |
| 汇总/清单 | 2 | 早报汇编、AI graveyard |

#### 已做且有效的修正

`junk_keywords` 里补回 9 个汇总词（早知道/早报/周报/盘点/汇总/一览/合集/速览/简报）。
代码默认有这批词，但 YAML 提供该键后会覆盖默认值，线上一直没生效。
P@8 从 0.46 升到 0.49（小幅，但方向正确）。

#### 已验证但无效、已撤回的实验

| 假设 | 做法 | 结果 |
|---|---|---|
| 观点稿的 `primary_action` 来自正文（如「故障」→accident），粗排据此加分 | 正文动作不参与 `primary_action`，粗排只看 `primary_action` | **P@8 0.49 → 0.43，变差** |

原因：不少真新闻的标题里没有动作词，它们的 `primary_action` 本来就靠正文补；
砍掉正文动作同时削掉了这些真新闻的加分。该改动已撤回（代码回到原状）。

#### 重复条目的机制问题（待修）

同一篇「AI 时代，公司的新厂房」连续 3 天进 shortlist：

- 它从没进过日报 → `covered_before=False`、账本不记录；
- 但 `reappeared=True`，精排的 momentum 给了 **+0.4** 加分；
- 结果「越没被选中，越容易被再次推荐」。

8 天里 shortlist 跨天重复 5 条（4 条无新进展），约占 64 个槽位的 8%。
修法需要引入「已进 shortlist 但未选中」的历史记录，并在无新进展时不再加分——
这属于选择逻辑改动，需单独评估，不在本次范围。

#### 给后续的结论

1. **不要再用 3 天金标调任何权重/词表**，样本太小，差异都是噪声；
2. 主要误选是「观点/经验/汇总」类，靠关键词列表只能治标（汇总词有效、观点词难）；
3. 真正的下一步是给 S2 加一个「是不是具体新闻事件」的判别信号（规则或小模型），
   而不是继续在 S3 上堆关键词；
4. 现有 `actions` 标签有明显噪声（观点稿被判 accident），直接删除正文动作会伤真新闻，
   需要「动作来源」标记后再决定权重。

### 事件性信号：`narrative_angle=howto` 降权（2026-11-24，有效）

先对比 shortlist 里正/负例的特征，结论是**现有特征几乎不可分**：

| 特征 | 正例均值 | 负例均值 |
|---|---|---|
| 粗排分 | 2.68 | 2.32 |
| impact | 4.83 | 4.67 |
| novelty | 4.22 | 4.34 |
| momentum | 0.95 | 0.96 |
| audience_fit | 3.94 | 3.88 |
| narratability | 3.70 | 4.01 |
| evidence | 1.52 | 1.42 |
| 独立来源 | 1.12 | 1.04 |

召回通道也不是区分器：`continuity` 命中 26/27、`subject` 命中 27/27——
好坏两组的准入方式完全一样。

唯一出现的分离信号是 `narrative_angle`：shortlist 里带 `howto` 的条目
**9 条负例 / 1 条正例**。于是把 howto 作为粗排降权项：

```yaml
coarse:
  demote_angles: [howto]   # 教程/方法论类，不是具体新闻事件
  angle_penalty: 0.8
```

公平对比（把规则改动后新进 shortlist 的 9 条也补标）后：

| 配置 | 正例/已判 | P@8 |
|---|---|---|
| 无角度降权 | 26/53 | 0.49 |
| 降权 0.4 | 29/51 | 0.57 |
| **降权 0.8（采用）** | **31/53** | **0.58** |
| 降权 1.2 / 2.0 | 31/53 | 0.58（无额外收益） |

分天变化：11-12 0.38→0.50、11-13 0.38→0.75、11-14 0.75→0.88，
11-09/10/11 持平，11-15/16 不变。没有一天变差。

**注意**：
- 这是在同一份 8 天金标上的 in-sample 调参，只能说明方向正确，不能当泛化保证；
- `howto` 标签本身噪声较大（7/9 的负例是正文命中，不是标题），后续应改成
  「标题/摘要含教程句式」的更严格判据；
- 单靠角度词无法解决 11-15/16 的低分（0.25 / 0.12），这两天的负例是重复条目
  与泛观点稿，需要事件性判别模型。
