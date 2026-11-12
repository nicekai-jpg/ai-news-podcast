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
