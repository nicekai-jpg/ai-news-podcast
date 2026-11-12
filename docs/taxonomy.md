# 标签体系（先定骨架，再填词）

目标：把“标签”拆成**受控枚举**和**抽取值**两类，先定维度与枚举，再往有限清单里补词，
避免今天加“汽车”、明天加“股票”式打补丁。

```
config/taxonomy/
├── labels.yaml       维度 + 受控枚举（决策标签）
├── entities.yaml     公司/实验室/人物/产品（有限，可穷举）
└── industries.yaml   行业树（有限，可穷举）
        │
        ▼
business/ocean/taxonomy.py  （加载 + 校验 + 别名映射）
        │
        ├── 规则标注：词典命中 → 映射到 id
        └── LLM 标注：只允许输出枚举/id + 证据句
```

## 两类标签

| 类型 | 例子 | 怎么来 | 能不能自由发挥 |
|---|---|---|---|
| 决策标签 | `ai_centrality`、`primary_domain`、`object_type`、`primary_action` | 规则或 LLM 从**受控枚举**里选 | 不能 |
| 抽取标签 | 公司名、产品名、版本号、金额 | 从全文抽，再映射 `entities.yaml` 的 id | 值开放，id 受控 |

## 当前维度

| 维度 | 取值 |
|---|---|
| `ai_centrality` | central、infra、incidental、none、unknown |
| `primary_domain` | ai_core、ai_infra、ai_application、other、unknown |
| `object_type` | model、product_feature、research_paper、funding、policy、hardware、incident…… |
| `primary_action` | release、update、research、funding、regulation、pricing、lawsuit…… |
| `lifecycle` | burst、evolving、long_tail、fading、unknown |
| `evidence_tier` | strong、medium、weak、rumor、unknown |

## 主体与行业

- `entities.yaml`：全球知名 AI 公司/实验室就那么多，按 id + 别名维护；抽取时先映射别名；
- `industries.yaml`：行业是有限集合（AI、半导体、机器人、汽车、金融、消费、娱乐……），
  每个行业带 `ai_related`（core/infra/application/none）和关键词；
- 推荐门槛最终看 `ai_centrality` / `industry.ai_related`，不再靠“有没有 AI 词”。

## 标注技术路线

```
标题 + 摘要/导语
      ↓
规则层（便宜、可解释）
  - 实体别名匹配 → entity id
  - 行业关键词 → industry id
  - 动作/对象词典 → 枚举
      ↓
LLM 层（只对候选，受控输出）
  - 输出：ai_centrality / primary_domain / object_type / action
  - 必带 evidence（原文短句）和 confidence
  - 校验：枚举合法、evidence 必须出现在输入里
      ↓
低置信 / 校验失败 → 回退规则值，标 unknown
```

## 补词流程

1. 跑 `scripts/domain_coverage.py` 看 `unknown/other` 标题；
2. 判断缺的是**实体**、**行业词**还是**动作/对象词**；
3. 加到对应 YAML（不是加到代码里）；
4. 跑 `tests/test_taxonomy.py`、`test_ocean_domain.py`、推荐回归；
5. 合入，下一期生效。

## 现状与下一步

- 骨架已落地：3 个 YAML + 加载/校验 + 别名/行业查询 + 测试；
- 实体和行业清单是首批种子（AI 公司/常见行业），后续按“有限清单”补齐；
- 下一步：规则标注接入 `taxonomy`（用 industry/entity id），再上 LLM 受控标注；
- 旧的 `config/domain_keywords.yaml` 暂时保留作补充词表，后续逐步收敛到 taxonomy。


## 字段全景与穷尽方法（v1）

字段分四类，穷尽方式不同：

| 类别 | 字段例子 | 穷尽方式 |
|---|---|---|
| 封闭枚举 | ai_centrality、object_type、primary_action、sentiment、risk_labels | 自上而下定类别，值只能从枚举选 |
| 有限清单 | 公司/实验室/人物、产品/模型、行业、地区、语言 | 建 registry，用标准库 + 持续补别名 |
| 组合生成 | event_type、novelty_type、narrative_angle | 不穷举字符串，穷举维度组合（action × object） |
| 开放抽取 | 数字、benchmark、引语、链接 | 正则/NER 抽取，不做硬门槛 |
| 派生计算 | lifecycle、timeliness、credibility | 公式/规则，不人工填 |

### 形容词类：刻度是输出表示，词表仍要穷尽

刻度**不是**“不用穷举同义词”。输入侧仍靠词表/线索，只是把同义词归到
“极性 × 强度”上，再叠加否定词/程度词组合：

```
“极其重磅”   → impact +3
“不太成功”   → sentiment 负（否定翻转）
“略有下滑”   → sentiment 负 1（程度词降强）
“引发争议”   → controversy 2
```

配置在 `config/taxonomy/eval_cues.yaml`：

- `cues`：评价词 → 维度 / 方向 / 强度；
- `negators`：不、未、没有、否认、辟谣……
- `intensifiers`：非常/极其/大幅/略/稍……

词表怎么尽量穷尽：

1. 导入中文情感/评价词典（大连理工情感词汇本体、知网 HowNet、同义词词林）；
2. 从语料里挖评价句，补 AI 领域特有词（翻车、超预期、开源、落地……）；
3. 用同义词/近义词扩展（词林、WordNet、embedding 近邻）；
4. 否定词、程度词单独维护，和评价词组合，而不是给每个组合建标签；
5. 覆盖度看“评价句命中率 / unknown 率 / 人工一致性”，不是看词表长度。

判断分数时不能只看单个词：要看**局部上下文**（否定词、程度词、修饰对象），
规则的组合能力比单纯关键词强；语义/反讽边界再交给 LLM 复核。

### 穷尽的六种手段

1. **标准导入**：行业用 GICS/申万，地区用 ISO 3166，语言用 ISO 639，动作借 VerbNet/事件类型学；
2. **语料挖掘**：从历史标题/正文抽名词/动词短语，按频次补；
3. **同义词/别名扩展**：公司英文名、简称、产品名统一映射到规范 id；
4. **组合生成**：event_type = action × object，narrative_angle = 角度集合，不手写所有组合；
5. **饱和分析**：每 N 篇新增唯一词数量下降即接近覆盖；同时看 unknown 率；
6. **错误驱动**：每周抽查误判样本，补枚举/别名并跑回归。

### 覆盖度量

- 每个字段的 `unknown` 比例；
- 实体/行业命中率（命中条数 / 总条数）；
- 别名覆盖率（同一实体不同写法是否归并）；
- 饱和曲线（新增词/新增文档）；
- 误杀/漏放（评测集）。

### 字段清单

完整字段注册表见 `config/taxonomy/dimensions.yaml`（约 30 个字段，标注 type/enumerability/values_source/consumers）。

## 覆盖审计：同样的毛病还有哪些

用 2026-11-11 的 763 条事件跑了一遍（`scripts/label_coverage.py`）：

| 字段 | 空白率 | 词典规模 | 影响 |
|---|---|---|---|
| subjects | 48% | 实体词典 97 条 | 一半事件没有主体，“影响力”特征直接接近 0 |
| narrative_angle | 51% | 7 个角度 | 一半事件没有可讲性角度 |
| tech_stack | 29% | 17 个技术栈 | 技术匹配/受众特征缺失 |
| actions | 25% | 16 个动作 | 事件类型/粗排加分不稳 |
| topics | 16% | 8 个主题 | 主题相似度/多样性变粗 |
| audience_persona | 12% | 5 类受众 | 相对可接受 |

这不是单个字段的问题，是整条链的**同一个模式**：规则/词典覆盖不足 + 无覆盖度量。

### 其他位置的同类风险

| 位置 | 隐藏假设 | 可能的偏差 |
|---|---|---|
| 精排 impact | 主体词典能覆盖所有重要公司 | 长尾公司/实验室被低估 |
| 精排 novelty | 数字/版本是新颖度的代表 | 无数值但重要的发布被低估 |
| 精排 narratability | 7 个角度足够 | 新叙事（如“开源反转”“监管突袭”）漏掉 |
| 精排 risk | 关键词能代表风险 | 隐晦风险（隐私、伦理）漏判 |
| 来源分级 | category 能代表可信度 | 官方号发软文/媒体发独家，类别不等于质量 |
| 证据分 | 来源数、官方、全文比例足够 | 同源转载被当成多源 |
| 聚类/去重 | 标题相似度阈值足够 | 同事件换说法漏合并；不同事件被误合并 |
| 全文抓取 | 解析器对站点结构稳定 | 改版后静默失败，只有摘要 |
| 覆盖账本 | 标题指纹能识别“已播报” | 更新事件被误判为重复 |

### 统一治理方式

```
每个字段/规则
   ├── 值从哪来（标准/词表/registry/抽取/公式）
   ├── 覆盖度量（空白率、命中率、饱和曲线）
   ├── 消费者（谁用它做决策）
   └── 缺口报告（unknown/空白清单 → 补词/补规则 → 回归）
```

工具：`scripts/label_coverage.py` 出空白率；`scripts/domain_coverage.py` 出主题域缺口；
`scripts/source_health.py` 出采集源健康度；三份报告合起来就是“这套规则到底覆盖了多少”。
