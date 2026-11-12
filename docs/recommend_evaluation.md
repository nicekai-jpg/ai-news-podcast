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
