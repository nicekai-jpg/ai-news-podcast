# 三元组抽取方案对比（统一评测）

> 本文件是评测方法与结论的说明文档；数字表由
> `python scripts/triple_bench/cli.py report` 生成到 `docs/kg_triple_benchmark.md`。
> 两份文档分工：这里写**测什么、怎么测、历史方案清单**，那里放**实测数字**。

## 1. 为什么要重测

历史上围绕「从 AI 新闻里抽三元组」做过多轮调查，累计试过 16 个方向。
但由于三个原因，这些数字**彼此不可比**：

| 问题 | 具体表现 |
|---|---|
| 样本规模不一致 | 结论表里混着 3 篇 / 10 篇 / 18 篇 / 40 篇四种样本 |
| 方案被静默跳过 | `llm.json` / `ltp.json` / `triples.json` 用 `id` 键，金标用标题键，评测时取不到预测却仍出现在表里（实测覆盖 0/18） |
| 金标被反复调参 | dev/test 在同一批数据上迭代，72 组网格在 test 上全部回落（过拟合），被误读成「方法上限」 |

因此本目录重建了一套统一评测：**一份冻结金标、一套分句、一个匹配器、一次验收**。

## 2. 历史调查过的全部方案

### 2.1 自研规则 / 依存路线

| 方案 | 标签 | 技术路线 | 状态 |
|---|---|---|---|
| V1 模板三元组 | `v1_template` | 词典匹配 → jieba 未登录实体 → 动作锚点 + 最近距离 → spaCy 依存确认 | 已被 V2 取代 |
| V1 strict | `v1_strict` | 同上，主语只认词表、宾语限短名词短语 | 废弃分支 |
| V2 事实卡（现行） | `repo_rules` | 选句 → 分句 → 词表 span 合并 + spaCy 依存 → 谓词 → 21 类槽位 → A/B/C 判档 | **生产在跑** |
| V2 + 自定义分词 | `repo_rules_custom_tok` | 在分词上叠加「只合不拆」的词表合并 | 可选开关 |
| V2 + LTP 解析器 | `repo_rules_ltp` | LTP 标签映射成 spaCy 风格后复用 fill/grade | 可选开关 |
| 名词性事实通道 | `nominal` | 售价/估值/IPO 等触发词直接出卡（仅 B 档） | 生产在跑 |
| 主语继承 + 依存兜底 | `hybrid2` | LTP SRL + GLiNER span + 上一句主语继承 | 实验 |

### 2.2 开源模型 / 工具路线

| 方案 | 标签 | 技术路线 | 端到端出三元组 | 状态 |
|---|---|---|---|---|
| HanLP 2.1 SRL | `hanlp_srl` | SRL 帧 PRED + ARG0/ARG1 | 需拼装 | 已接入 |
| HanLP 2.1 SDP | `hanlp_sdp` | 语义依存 Agt / Pat / Cont 子树 | 需拼装 | 已接入 |
| GLiNER2 JointIE | `gliner2_jointie` | schema 约束联合抽取，char 切词 | 是 | 已接入 |
| GLiNER2 AutoExtractor | `gliner2_rel` | 逐句关系抽取 + 阈值过滤 | 是 | 已接入 |
| LTP 4.2 SRL | `ltp_srl` | SRL 帧 + 词典归一（含主语继承版本） | 需拼装 | 已接入 |
| PaddleNLP UIE | `uie` | schema 提示式信息抽取 | 是 | **N/A**：需 Docker + paddlenlp 2.8.1，实测 106–308 秒/篇 |
| OpenNRE | `opennre` | 句级关系分类（需先给实体对） | 否 | **N/A**：只有 wiki80/TACRED 英文权重 |
| DeepKE | `deepke` | ASP / PRGC / PURE 三元组模型 | 是 | **N/A**：2.2.7 安装失败，源码版锁 python3.8/torch1.5 |
| spaCy + EntityRuler | 见 `repo_rules` | 实体 + 依存 + 动作词典 | 需拼装 | 历史上最好的纯规则基线 |

### 2.3 解析器 / 分词横评（只做句法，不单独产三元组）

这些方案在 UD 中文树库上比过句法质量，结论已并入 `docs/kg_extraction.md` §6.3b：

| 方案 | GSDSimp F1 | CFL F1 | 备注 |
|---|---|---|---|
| HanLP 2.1 | 0.400 | 0.651 | 当前目标解析器 |
| SuPar 1.1 | 0.372 | 0.639 | 依赖外部分词/词性 |
| LTP 4.2 | 0.362 | 0.433 | 召回高、精确率低 |
| spaCy 3.8 | 0.303 | 0.600 | 默认基线 |
| jieba / THULAC | – | – | 仅分词，无依存 |
| Stanza / LAC | – | – | 未跑通（下载失败 / 缺 Paddle） |

### 2.4 LLM 路线（不参与排名）

本评测的目标是「不用大模型能到什么程度」，所以 LLM 只作离线参照，不计入主表。

| 方案 | 标签 | 说明 |
|---|---|---|
| MiniMax zero-shot | `llm_zero_shot` | 直接提示词 → JSON 三元组，evidence 必须回指原文 |
| 早期原型（V1 时期） | – | 656 条 / 约 10 万 token，关系发散到 408 种说法，未采纳 |

## 3. 评测标准

| 项 | 规定 |
|---|---|
| 语料 | `data/ocean/2026-11-16/items.json`，仅保留 Stage 2 判定为 AI 相关（`ai_relevance ∈ {core, adjacent}` 或 `primary_domain ∈ {ai_core, ai_infra}`）的条目 |
| 抽样 | 按 `item_id` 哈希排序，语言 × 长度分层，同一事件只取一篇；结果写入 `docs.json` 后冻结 |
| 划分 | dev 用于开发与调参；test 冻结，只在最终验收跑一次 |
| 分句 | 所有方案共用 `standard.sentence_split()` |
| 关系 | 统一 21 类，其余归 `other`；中文别名由 `RELATION_ALIASES` 收口 |
| 端点 | 默认归一化后**完全相等**（全半角、大小写、空白、首尾标点）；另报包含匹配作为宽松口径 |
| 匹配 | 每篇内最大二分匹配，不按输出顺序贪心 |
| 指标 | triple-level micro P/R/F1；另报候选召回、误差归因、分语言、bootstrap 95% CI |
| 公平性 | **缺失文档记 0 分**；不可用方案单列 N/A 并写明原因，禁止静默跳过 |

### 3.1 误差归因口径

每条金标按以下四类归档，用来区分「召回问题」与「精度问题」：

| 类别 | 含义 |
|---|---|
| `full` | 该金标被严格命中 |
| `relation_only` | 关系对上、但至少一端不对（论元选择 / 边界问题） |
| `endpoint_only` | 两端都对、但关系判错（关系分类问题） |
| `no_candidate` | 候选池里完全没有这个事实（召回问题，过滤规则救不回来） |

### 3.2 与历史口径的差异

| 项 | 历史 | 本评测 |
|---|---|---|
| 端点匹配 | 包含匹配 `a in b or b in a` | 默认精确，包含作为单独口径 |
| 关系映射 | 每个脚本各写一份 ZH→EN 表 | 单一 `normalize_relation()` |
| 匹配算法 | 顺序贪心 | 最大二分匹配 |
| 缺失文档 | 静默跳过 | 记 0 分 |
| 金标状态 | 混在一起 | `draft` / `reviewed` 分离，只有 reviewed 可冻结 |
| 样本外验收 | dev/test 同批调参 | test 冻结 |

## 4. 复现

```bash
# 1. 抽样（已冻结，一般不需重跑）
uv run python scripts/triple_bench/cli.py sample \
  --seed-v2 ~/nas/tmp/ner-eval/gold_v2.json \
  --sample ~/nas/tmp/ner-eval/sample.json \
  --total 60 --dev 20

# 2. 金标：起草 → 机器审核 → 校验 → 冻结
uv run python scripts/triple_bench/cli.py draft --model MiniMax-M3 --workers 8
uv run python scripts/triple_bench/review_gold.py --verbose
uv run python scripts/triple_bench/cli.py validate
uv run python scripts/triple_bench/cli.py freeze

# 3. 跑方案（可中断续跑；--force 才重跑）
uv run python scripts/triple_bench/cli.py run --adapter repo_rules,hanlp_srl,hanlp_sdp,ltp_srl
GLINER_MODEL=... ~/nas/tmp/ner-eval/.venv/bin/python scripts/triple_bench/cli.py run --adapter gliner2_jointie,gliner2_rel

# 4. 打分 + 报告
uv run python scripts/triple_bench/cli.py eval --bootstrap 1000
uv run python scripts/triple_bench/cli.py report
```

## 5. 目录结构

```
scripts/triple_bench/
├── standard.py   关系词表、文本归一、统一分句、数据结构
├── metrics.py    打分、误差归因、bootstrap CI
├── gold.py       抽样、校验、LLM 起草、审核清单
├── review_gold.py 机器审核：剔除不成立条目 + 证据重定位
├── adapters.py   方案的统一适配器（含 N/A 登记、并发配置）
└── cli.py        sample/draft/validate/freeze/run/eval/report

data/eval/triple_bench/
├── docs.json        冻结的 60 篇文档
├── splits.json      dev/test 划分
├── gold_v3.json     冻结金标（frozen: true，含证据类型标记）
├── preds/*.jsonl    各方案预测（可中断续跑）
└── results.json     统一打分结果
```

## 6. 已知限制

| 限制 | 说明 |
|---|---|
| 单日单批数据 | 全部来自 2026-11-16 一天，事件类型分布可能不代表长期 |
| 中文占多数 | 43 中 / 17 英，英文结论的置信区间更宽 |
| 金标含 LLM 起草 | 已用  机器审核（剔除端点不在原文/泛指/职务串/重复，证据重定位），但未逐条人工确认 |
| 关系粒度 | 21 类是项目定义，不是通用 RE 数据集（ACE/Wiki80）的标准 |
| 解析器类方案未进主表 | jieba / THULAC / Stanza / LAC 不产三元组，只在 UD 横评里出现 |

## 7. 事实层的接入现状（2026-11-23）

评测完才发现的事实层现状，记录在此以便后人：

| 项 | 现状 |
|---|---|
| 消费者 | 接入前 `data/facts/` 无任何生产消费者（只有 `scripts/kg_facts.py` 的报表 CLI 读） |
| 接入点 | S4 日报正文：`build_commentaries(verified_by_item=...)` → `build_commentary_prompt(verified_facts=...)` |
| 注入形态 | prompt 里新增「已核验事实」块，明确标注**不是新的事实来源**、与证据冲突时以证据为准、facts 仍须带 evidence_ids |
| 只取 A 档 | 每天 5~11 条；B 档 3000+ 条精度不足，注入只会引入噪声 |
| 开关 | `editorial.inject_verified_facts`，**默认 false**（实测收益为负，见下表） |
| 降级 | 开关关闭 / 事实文件缺失时返回空 dict，注入块显示「本事件无机器抽取的可核验事实」，不影响流水线 |

### 7.1 实测结论：默认关闭

2026-11-23 用 2026-11-16 的真实数据跑了注入前/后对照：

| 指标 | 数值 |
|---|---|
| A 档总量 | 7 条 |
| 能对上 A 档的事件 | **7 / 470（1.5%）** |
| 98.5% 的事件注入内容 | 「本事件无机器抽取的可核验事实」 |

两条事件各跑注入前/后，均通过客观性校验（注入不会破坏校验），但：

| 事件 | 注入前 facts | 注入后 facts |
|---|---|---|
| Anthropic Claude 制造插件 | 4 条 / 2314 字 | 3 条 / 1945 字 |
| OpenAI + Meta Poppy | 4 条 / 2394 字 | 3 条 / 2176 字 |

**原因**：注入的 A 档事实（如 `Anthropic —发布→ Claude`）LLM 从证据正文里本来就能读到，
属于冗余信息，反而挤占了正文长度预算。

**结论**：注入通道保留（可回滚、有测试保护），但默认关闭。
要让注入产生正收益，前提是 A 档能抽到「正文里读不到、但确实为真」的事实——
这需要先解决召回（当前严格 F1 0.02）。打开前先量覆盖率：

```bash
uv run python scripts/kg_facts.py report --days 1
# 或者直接数天气是否超过事件数的 10%
```

同时修掉的两个 A 档 bug：

| bug | 原因 | 修法 |
|---|---|---|
| 人事卡丢失公司 | `_fill_personnel` 只从介词短语取 company，「英伟达创始人黄仁勋」这类定语取不到 | 新增 `_owner_before()`：人名左侧 8 个 token 内找最近机构 |
| 同一事实拆成 3 张卡 | `_object_key` 用 `role` 做键，「总裁/CEO/董事」各算一条 | 人事改按 `person` 去重，`role` 只进槽位 |

## 8. A 档证据质量守卫（2026-11-23）

用 `kg_facts audit` 逐条核对 A 档后发现的真问题：**主语和关系是对的，但证据句本身不成立**。

抽查 2026-11-16 的 5 条 A 档：

| # | 事实 | 判定 | 原因 |
|---|---|---|---|
| 1 | `Anthropic —发布→ Claude` | ✅ | 宾语边界偏窄（应为 `Claude for Manufacturing`） |
| 2 | `百度 —发布→ 大模型交易平台` | ⚠️ | 宾语应为「词元所」 |
| 3 | `黄仁勋 · CEO · 英伟达` | ❌ | 证据取自文章里**虚构的高考作文题** |
| 4 | `英伟达 —发布→ 自动驾驶模型` | ⚠️ | 宾语应为 `Alpamayo 2 Edge` |
| 5 | `OpenAI —合作→ Meta` | ❌ | 正文是「**Sierra** 联合 Meta」；OpenAI 是同日另一件事，来自标题 |

严格准确率 1/5 = 20%，放宽（主语关系对）4/5 = 80%。

### 8.1 新增的两条守卫

`grade.py` 的 `grade()` 接收 `source_text` / `title` 后新增：

| 理由码 | 规则 | 挡掉的案例 |
|---|---|---|
| `evidence_from_title` | 证据句不在正文里，或与标题完全相同 | 第 5 条（标题党） |
| `evidence_quoted_material` | 证据句落在引号内，且引号前 40 字内有「作文/假设/示例/例如」等虚构标记 | 第 3 条（虚构作文题） |

刻意保守：只杀**明确假**的证据，边界错（第 1/2/4 条）留给实体吸附处理，不做二次判断。
引号未闭合的长引文也按「在引文内」处理。

### 8.2 证据改指：救回真事实

只降档会误杀「真事实、但证据取自标题」的条目（如「中金公司与蚂蚁集团达成战略合作」）。
所以加了 `_retarget_clause()`：证据句不在正文时，在正文里找**同时含两端、且两端距离 ≤40 字、且中间出现关系词**的句子，改指过去。

两个约束都是踩坑加上的：

| 约束 | 为什么 |
|---|---|
| 两端距离 ≤ 40 字 | 同一句里「OpenAI 与 Visa 均在其中……Meta 的 Muse 与 OpenAI 是竞争对手」两端都在，但谈的不是一回事 |
| 中间要有关系词 | 只有共现不能推断关系成立 |

### 8.3 宾语边界：三类同位语

审计发现剩下的错误以「宾语太泛」为主：`百度 —发布→ 大模型交易平台`（应为「词元所」）、
`Meta —发布→ 高性能AI模型`（应为 `Muse Ember`）。加了两条吸附规则：

| 规则 | 触发条件 | 例子 |
|---|---|---|
| `_quoted_appositive()` | 宾语后紧跟 `「」` / `“”` / `（）`，内容 2–20 字且非纯数字 | `推出大模型交易平台「词元所」` → `词元所` |
| `_latin_appositive()` | 宾语是泛称（模型/芯片/平台…），后接 `——` 或空格，且是含大写的拉丁专名 | `高性能AI模型——Muse Ember` → `Muse Ember` |

刻意限制：引号内超过 20 字（整句引文）不吸附；宾语本身已是专名时不吸附后面的拉丁串。

### 8.4 顺带修掉的两个守卫误杀

| 守卫 | 误杀 | 修法 |
|---|---|---|
| `a_guard_mixed_fragment` | 任何含空格且带英文的宾语都被判「中英混排」，`Memory summary`、`Instinct Coder` 被误杀 | 要求同时含中日韩字符才算混排 |
| 正则同名覆盖 | 新增的 `_LATIN_NAME` 覆盖了文件里原有的同名正则，导致英文宾语类型全变成 `person`、候选被丢弃 | 重命名为 `_LATIN_APPOSITIVE_RE` |

第二条是这次最危险的一处：测试里 `test_english_release` 立刻失败才暴露，
否则线上英文抽取会静默失效。

### 8.5 8 天实测（2026-11-09 ~ 11-16）

| 日期 | 原始 A 档 | 最终 A 档 |
|---|---|---|
| 11-09 | 0 | 0 |
| 11-10 | 1 | 2 |
| 11-11 | 2 | 4 |
| 11-12 | 3 | 1 |
| 11-13 | 5 | 1 |
| 11-14 | 7 | 5 |
| 11-15 | 11 | 6 |
| 11-16 | 7 | 4 |
| **合计** | **36** | **23** |

总量基本不变（B 档 28367 → 28293）。有升有降：假证据被挡掉，同时同位语吸附和混排误杀修复
又让少量真事实回到 A 档（如 11-11 的 `谷歌 —发布→ Memory summary`）。

顺带修掉的两个 bug：

| bug | 原因 | 修法 |
|---|---|---|
| `company=公司` | `_owner_before` 把人名左侧的泛称「公司」当成归属实体 | 加 `_GENERIC_OWNERS` 黑名单 |
| 人事卡重复 | 已在 §7 修（按 person 去重） | — |

结论：**先清证据比提召回更划算**。提召回是在放大错误；清证据是把假的先挡掉，
再把被误杀的真的捞回来。最终 A 档每天 0–6 条（合计 23 条），
每条都要求证据句能在正文里逐字定位，且不在虚构引文里。

## 9. 提召回的实测结论（2026-11-23）

### 9.1 修了什么

拉丁专名原先一律按「人名」处理（两个大写词 → `person`），导致
`release` 等关系的宾语类型校验失败、整条候选被丢弃（`Memory summary`、
`ChatGPT Wallet` 之类）。现在按后缀、版本号、大小写规律判断：

| 输入 | 旧 | 新 |
|---|---|---|
| `ChatGPT Wallet` | person | product（后缀 wallet） |
| `Gemini Robotics 2.5` | person | product（含版本号） |
| `Memory summary` | person | product（第二个词小写，不像人名） |
| `Alice Huang` | person | person（保持） |

### 9.2 但 8 天 A 档没有变化

| 版本 | A 档合计 |
|---|---|
| 上一轮（守卫 + 同位语） | 23 |
| 本轮（再加类型推断） | 23 |

因为真正卡住召回的**不是类型判断**，是下面两条。

### 9.3 真正的两个瓶颈

**① 中英混排句解析失败 24%**

2026-11-16 一天里，「中文 + 英文 + 动作词」的句子 776 句，其中 **186 句（24%）
解析不出宾语**：

```
机器之心发布 11 月 9 日，Anthropic 发布 Claude Fable 5.2。
→ OpenAI 推出 ChatGPT Wallet。
→ 解析结果：OpenAI 是 ROOT(PUNCT)，推出是 dep，宾语为空
```

试过用正则兜底，但两天里只有 19 句符合「拉丁主语 + 动作词 + 宾语」，
且两端都命中词典的只有 2 句——**不值得为它写规则**。

**② A 档事实上要求「两端都在词典里」**

| 理由码 | 命中次数（8 天） | 含义 |
|---|---|---|
| `subject_inherited` | 26,568 | 主语是继承来的（跨句指代） |
| `object_weak` | 25,944 | 宾语不在词典、类型未知 |
| `subject_open` | 21,736 | 主语不在词典 |
| `noun_trigger` | 15,902 | 名词性事实（只出 B） |

抽查 `object_weak` 的宾语：`美元`、`公告`、`价格`、`风险`、`测试`、`故障`……
**绝大多数确实是噪声**，说明这道门没有判错。想提 A 档，只能让两端的专名进词典。

### 9.4 所以：把缺口列出来，而不是继续改代码

新增 `scripts/kg_lexicon_gaps.py`：扫描最近 N 天的 ocean 数据，把
「出现在发布/推出/开源…之后、但不在 `config/nlp/*.yaml` 里」的拉丁专名按频次排出。

```bash
python scripts/kg_lexicon_gaps.py --days 8 --out-md data/facts/lexicon_gaps.md
```

2026-11-09 ~ 11-16 的 top 缺口：

| 次数 | 片段 |
|---|---|
| 21 | `LongCat-2` |
| 14 | `Seedance 2` |
| 12 | `Xperience 2` |
| 11 | `Seedream 5` |
| 9 | `Apollo 2` |
| 8 | `LingBot-Touch` |
| 6 | `NeoHorse-1`、`DisplayPort 2` |
| 4 | `GR00T 2`、`RISC-V` |

这些正是新闻里真实出现、但词典没收录的模型/产品名。把它们补进 `objects.yaml`
是唯一能把对应事实升到 A 档的办法。

**结论**：召回不再是「算法问题」，而是「词表维护问题」。

### 9.5 按缺口补词典的效果（2026-11-23）

补了 23 条（`LongCat-2.5`、`Seedance 2.5 flash`、`Seedream 5.5`、`Xperience 2`、
`LingBot-Touch`、`GR00T 2`、`Edge0-35B/145B`、`Gemini 4 Hydrogen`、`TraceRouter++` 等），
词典从 53 条增至 76 条。

| 指标 | 补词典前 | 补词典后 |
|---|---|---|
| A 档（8 天） | 23 | **24** |

同时修了三个真 bug：

| bug | 影响 | 修法 |
|---|---|---|
| 缺口工具在小数点处截断 | 把 `Seedance 2.5 flash` 记成 `Seedance 2`，词典补的是半截名字 | 小数点不再作分隔符 |
| 带版本号的产品名被标 `nmod:range` | `字节推出 Seedance 2.5 flash` 抽不出宾语 | `_OBJECT_DEPS` 加入 `nmod:range` |
| 词典别名用了错误版本号 | `LongCat-2` / `Seedream 5` 与新闻写法不符 | 改为 `LongCat-2.5` / `Seedream 5.5` |

**为什么只涨了 1 条**：这些产品名大多出现在「主语是继承来的」句子里，
真正卡住它们的是下一节的 `subject_inherited`，不是词典。

### 9.6 最后一个门槛：主语继承（201 条）

8 天里，**只差 `subject_inherited` 这一个理由**就能进 A 的卡片有 201 条（约 25 条/天）。
其中 78% 的宾语是词典内的产品名，质量看着不错：

```
谷歌 —发布→ Gemini 4 Hydrogen      ✅
Cursor —发布→ Origin               ✅
百度 —发布→ 两款文心应用模型 API    ✅
```

但抽样 10 条里约一半是错的，错误形态是**跨句指代取错主语**：

```
Anthropic —funding→ 100 亿美元     ❌ 原句在讲药品销售峰值
蚂蚁集团 —regulation→ 微软公司      ❌ 原句是另一条新闻（摘要流）
谷歌 —发布→ 原生模型开发的灵活性     ❌ 宾语是抽象短语
```

试过用「摘要/推荐」标记过滤，但只能挡住 1/200——错误是语义层面的，规则分不开。
**放开继承大约会让 A 档精度从 ~80% 掉到 ~50%**，所以没有把它混进 A。

### 9.7 处理方式：新增 A2 中间档

`grade()` 现在返回三档：

| 档位 | 条件 | 数量（8 天） | 抽样精度 |
|---|---|---|---|
| `A` | 无任何问题 | 24 | ~80% |
| `A2` | **只差**主语继承（两端都在词典、证据可定位） | 139 | ~50% |
| `B` | 其余 | ~28,000 | 低 |

**A 的定义一个字没改**，A2 是纯增量。实现要点：

| 位置 | 改动 |
|---|---|
| `grade.py` | `reasons == ["subject_inherited"]` 且守卫通过 → `A2`；守卫里的 `a_guard_inherited` / 继承场景的 `a_guard_subject_absent` 移除（由 A2 接管） |
| `store.py` | 合并优先级 `A > A2 > B`，高檔位胜出 |
| `types.py` | 统计新增 `cards_a2` |
| `verified_facts.py` | `load_a_cards(grades=("A",))` 可按档位取 |
| `editorial_v3.yaml` | `fact_grades: [A]`，想连 A2 一起注入就写 `[A, A2]` |
| `kg_facts.py` | `audit` 分开统计 A/A2；`validate` 要求 A2 必须带 `subject_inherited`；`prune` 长期保留 A/A2 |

8 天实测（2026-11-09 ~ 11-16）：

| 日期 | A | A2 | 总卡 |
|---|---|---|---|
| 11-09 | 0 | 11 | 2213 |
| 11-10 | 2 | 16 | 3260 |
| 11-11 | 4 | 17 | 4112 |
| 11-12 | 1 | 17 | 3823 |
| 11-13 | 2 | 25 | 4291 |
| 11-14 | 5 | 19 | 3988 |
| 11-15 | 6 | 19 | 3539 |
| 11-16 | 4 | 15 | 3100 |
| **合计** | **24** | **139** | |

每天可用事实从 ~3 条升到 **~20 条**（A + 2×A2 折算前）。

抽查 2026-11-16 的 15 条 A2，确认「约一半对」：

| 判定 | 例子 |
|---|---|
| ✅ | `谷歌 —发布→ Gemini 4 Hydrogen` |
| ✅ | `Jensen Huang · CEO（人事）` |
| ✅ | `微软 —发布→ Copilot` |
| ❌ | `Anthropic —融资上市→ 100 亿美元`（原句讲药品销售峰值） |
| ❌ | `苹果 —融资上市→ 1100 万美元`（原文是 a16z 领投，主语取错） |
| ❌ | `快手 —研究→ 业务、会决策、能调工具的大模型`（宾语是抽象短语） |

所以：**A 用于日报，A2 只用于「素材不够」和人工抽查**，两者不能混用。

## 10. 实施中踩到的坑（供以后复用）

| 现象 | 根因 | 处理 |
|---|---|---|
| 某方案整轮预测为 0，日志无异常 | GLiNER2 的推理对象不是线程安全的，并发调用静默返回空 | `CONCURRENCY` 里把 `gliner2_*` 固定为 1 |
| 单句异常被吞掉，结果全空但看不出原因 | `except: continue` | `_record_error()` 计数并写进 meta，报告里可查 |
| 旧金标 `[13]` 这类 key 解析不到 | 旧 `sample.json` 用 `id` 字段而不是 `item_id` | 解析时同时兼容 `item_id` / `id` |
| 旧金标文档被抽样丢掉 | 按 event 去重 + `dup_of` 指向自身时被误判为重复 | 去重只在 `dup_of != item_id` 时生效 |
| 抽出纸飞机、GTA 这类非 AI 新闻 | 抽样没有按 AI 相关性过滤 | 只保留 `ai_relevance ∈ {core, adjacent}` 或 `primary_domain ∈ {ai_core, ai_infra}` |
| 英文长文让 HanLP 慢到 90 秒/篇 | 中文 MTL 模型跑英文 | 中文专用方案对 `lang != zh` 直接返回空 |
| 中英混排句抽不出事实 | spaCy 中文模型把 `OpenAI 推出 ChatGPT Wallet` 解析成 PUNCT/ROOT，宾语为空 | 已知限制；正则兜底收益太小（两天仅 2 句），未实现 |
| 提升召回迟迟无效果 | A 档要求两端都在词典（`object_weak` 25,944 次），而开放集合宾语多为噪声 | 转向词表维护：`scripts/kg_lexicon_gaps.py` 输出缺口清单 |
| 全量跑通过、单跑 `test_kg_fill` 也通过，但组合跑就失败 | 别的测试 `monkeypatch.chdir()` 后，`KGlexicon.load()` 的相对路径 `config/nlp` 失效 | 测试 fixture 改用绝对路径 `Path(__file__).parents[1]/"config"/"nlp"` |
