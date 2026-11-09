# 流程重设计 v2：Stage 1–7 与三大产品板块

> 状态：设计稿，待审核。审核通过后再动工。
> 本文重新定义流水线阶段与产品板块，取代旧版“Stage 1 fetch → Stage 3 writer”的编号。

## 0. 总体结构

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1  采集入海（C1–C5 通道，各一条 fetcher 线）           │
│          统一 item → artifact（含全文）                       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2  新闻海洋（去源化 → 去重 → 事件聚类 → 连续性 → 多维标注）│
│          每日一个“海洋版本”：ocean-{date}                     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3  选题推荐（粗排过滤 → 精排排序 → 候选集 + 推荐理由）    │
└──────────────┬───────────────────────────────┬──────────────┘
               ▼                               ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐
│ Stage 4  编辑点评             │   │ Stage 5  播客文稿             │
│  → 02 深度日报                │   │  (海洋+点评 → 双主持稿)       │
└──────────────────────────────┘   └──────────────┬───────────────┘
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │ Stage 6  音频合成 → 01 今日主推│
                                   │ Stage 7  发布（站点/feed）    │
                                   └──────────────────────────────┘
```

### 阶段 vs 板块

| 产品板块 | 名称 | 由哪些 Stage 产出 |
|---|---|---|
| **01** | **今日主推播客** | Stage 3 选题 + Stage 2 海洋 + Stage 4 点评 → Stage 5 文稿 → Stage 6 音频 → Stage 7 发布 |
| **02** | **深度日报** | Stage 3 选题 + Stage 4 编辑点评 + 证据链 |
| **03** | **新闻海洋**（原“信源与证据”，改名） | Stage 1 采集 + Stage 2 去重/聚类/连续性/标注 |

> **03 板块新名字建议：新闻海洋（News Ocean）**
> 理由：它不只是“信源与证据”，而是每天一个版本的全域新闻池 + 事件聚类 + 连续性追踪 + 证据链，用“海洋”能表达“量大、可下潜、有洋流（连续性）”的意象。
> 备选名：新闻星图（强调实体/关系）、新闻图谱（强调事件图谱）、信息海洋（更泛）。推荐用**新闻海洋**。

## 1. Stage 1：采集入海

**目标**：一次性抓完整（元数据+正文），统一格式，落盘 artifact；不做任何重要性判断。

### 1.1 通道 = 独立 fetcher 线

| 通道 | fetcher 模块 | 输入 | 输出 |
|---|---|---|---|
| C1 固定 RSS | `fetch_rss` | 31 个 RSS URL | item[] |
| C2 Atom | `fetch_atom` | GitHub Releases + YouTube | item[] |
| C3 RSSHub | `fetch_rsshub` | 66 条路由（临时容器） | item[] |
| C4 网页抓取 | `fetch_web` | 7 个 SSR 站点抓取规则 | item[] |
| C5 wechat2rss | `fetch_wechat2rss` | 17 个公众号 feed | item[] |

- **一个通道一条流水线代码**（独立模块、独立配置、独立失败隔离）；
- 所有通道输出**同一个 item schema**；
- 通道之间并行或串行都可，失败互不影响。

### 1.2 统一 item schema

```json
{
  "item_id": "sha1(channel+link)",
  "channel": "c1_rss|c2_atom|c3_rsshub|c4_web|c5_wechat2rss",
  "source": "anthropic_news",
  "source_label": "Anthropic News",
  "category": "official|news|research|community",
  "title": "……",
  "link": "https://……",
  "published_at": "2026-11-09T00:00:00+00:00",
  "fetched_at": "2026-11-09T16:05:00+00:00",
  "summary": "……（≤500 字）",
  "full_text": "……（采集时一次性抓到的正文）",
  "full_text_status": "ok|failed|truncated",
  "lang": "zh|en",
  "extra": {}
}
```

### 1.3 采集规则

- 每个源抓**最近 72 小时**（可配置）内条目，每源上限 15–30 条；
- feed 有 `content:encoded` 用全文；没有则采集时抓文章页抽正文；失败标记 `full_text_status=failed`；
- 只做 URL 规范化、同源内去重、时效过滤、条数上限；
- **不做**：跨源去重、重要性判断、聚类（全部交给 Stage 2）。

### 1.4 落盘

```
artifact: raw-{date}.json.gz        # 全量 item（含全文），30 天
git(可选): data/raw_summary/{date}.json  # 仅摘要，审计用
```

## 2. Stage 2：新闻海洋（03 板块）

**目标**：把多源数据变成“一个源头”的新闻池，输出每日海洋版本。

### 2.1 去源化

- `source` 降级为**溯源字段（provenance）**，不再作为分组依据；
- 所有 item 进入同一个池子，视为“同一条新闻的不同报道”。

### 2.2 全局去重（自上而下）

1. **URL 精确去重**（规范化后）；
2. **标题近似去重**（编辑距离/SimHash）；
3. **语义去重**（embedding/TF-IDF，同一事件不同标题）；
4. **跨日去重**：与历史海洋（30 天）+ 覆盖账本比对，已选中的旧闻过滤，未选中的保留并标记。

### 2.3 事件聚类

- 同一天内：同一事件的多源报道合并成 **event cluster**；
- 聚类信号：语义相似度 + 实体重叠（公司/模型/人物/数字）+ 时间接近；
- 每个事件输出：`event_id`、代表标题、成员 item、来源数、首见时间。

### 2.4 连续性识别（关键新增）

- 与历史事件比对，判断是：**新事件 / 持续事件 / 长尾事件**；
- 输出 `continuity_id`（同一条事件线的编号）与 `sequence`（第几天）；
- 例：Day1「DeepSeek 发布 V4.1」→ Day3「V4.1 实测」→ Day5「基于 V4.1 的应用」= 同一条 continuity，不同 sequence。

### 2.5 多维标注（归纳维度）

| 维度 | 取值示例 |
|---|---|
| 主体 | OpenAI、DeepSeek、NVIDIA、某高校 |
| 对象 | 模型 / 产品 / 论文 / 政策 / 芯片 |
| 动作 | 发布 / 开源 / 融资 / 合作 / 诉讼 / 事故 / 评测 |
| 技术栈 | LLM / Agent / 多模态 / 芯片 / 机器人 |
| 地域 | 中国 / 美国 / 欧洲 / 全球 |
| 生命周期 | 爆发 / 演进 / 长尾 / 消退 |
| 证据强度 | 官方 / 多源印证 / 单一来源 / 传闻 |
| 主题 | 技术 / 商业 / 政策 / 研究 / 产品 |

### 2.6 证据链

每个事件保留：
- 全部成员 item（标题/链接/来源/时间/全文摘要）；
- 来源列表（provenance）与来源数；
- `first_seen_at` / `last_update_at`；
- 关联事件（同 continuity、因果/对标关系）。

### 2.7 输出（每日一个版本）

```
ocean/{date}/
  events.json      # 事件级：event_id、代表标题、成员、标签、证据
  items.json       # 条目级：item + event_id + continuity_id + labels
  continuity.json  # 事件线：continuity_id、sequence、历史成员
```

## 3. Stage 3：选题推荐

**目标**：从海洋里推荐“最有价值的几条新闻”，输出候选集 + 推荐理由 + 证据包。

### 3.1 粗排（规则过滤，高召回）

- 时效：新事件 ≤48h，持续事件可放宽；
- 去重：已播报/已选过的过滤；
- 相关性：AI 主题/实体命中；
- 证据：来源数 ≥2 或官方来源；
- 排除：广告、招聘、课程、纯榜单。

### 3.2 精排（打分排序）

| 特征 | 说明 |
|---|---|
| 影响力 | 主体重要性、影响范围（全球/中国/行业） |
| 新颖度 | 首见时间、新实体/新版本/新数字 |
| 势头 | 来源数增长、连续性（sequence 越高越有价值） |
| 受众相关 | 开发者/从业者关注度 |
| 可讲性 | 全文质量、信息密度、是否有可解释的背景 |
| 风险 | 单一来源/传闻降权 |

- 规则加权分 + 可选 LLM 复核（对 top K 逐条判断“值不值得讲”）；
- 输出：`shortlist.json`，含每条候选的分数、推荐理由、证据包（成员 item + 来源）。

## 4. Stage 4：编辑点评（02 深度日报）

- 对 shortlist 前 N 条（建议 5–8），编辑 LLM 逐条写：
  - 一句话摘要
  - 为什么重要
  - 关键事实/数字（必须来自证据）
  - 背景脉络（来自海洋连续性）
  - 影响与观点
  - 风险提示
- 产出：`editorial/{date}/commentary.json` + `daily_report_{date}.md` → **02 深度日报**。

## 5. Stage 5：播客文稿（01 今日主推播客）

- 输入：shortlist + 点评 + 海洋上下文（连续性/证据）；
- 双主持（林夏/陈默）对话稿，带章节；
- 产出：`site/episodes/{date}.txt`（01 文稿）。

## 6. Stage 6：音频合成

- CosyVoice2 零样本克隆，双主持音色，多版本；
- 产出：`site/episodes/{date}.mp3`。

## 7. Stage 7：发布

- 重建站点/feed、更新 episodes.json、部署 gh-pages；
- 产出：01 播客上线、02 日报页面。

## 8. 数据契约汇总

| 阶段 | 产物 | schema 关键字段 |
|---|---|---|
| Stage 1 | item[] | channel、source、title、link、published_at、full_text |
| Stage 2 | events[]、items[]、continuity[] | event_id、continuity_id、labels、source_count、evidence |
| Stage 3 | shortlist[] | rank、score、reasons、evidence_items |
| Stage 4 | commentary[] | summary、why_matters、facts、background、impact、risk |
| Stage 5 | script | 章节 + 对话 |
| Stage 6 | mp3 | 音频 |
| Stage 7 | site/feed | 页面/RSS |

## 9. 目录与存储

```
data/
  raw/{date}.json.gz            # Stage 1 artifact（含全文，30 天）
  ocean/{date}/                 # Stage 2（artifact；摘要可入 git）
  selection/{date}/shortlist.json
  editorial/{date}/commentary.json + daily_report_{date}.md   # 02
  coverage/ledger.json          # 跨日覆盖账本（git）
  episodes/...                  # 现有节目产物
site/
  episodes/{date}.txt/.mp3      # 01
```

## 10. 与现有代码的迁移

| 现有 | 新位置 |
|---|---|
| `runner_service.run_pipeline`（Stage 1 fetch+dedup+cluster+score+brief） | 拆成 Stage 1（采集）与 Stage 2（海洋） |
| `fetcher_service` | Stage 1 的 C1/C2/C4 基础 |
| `dedup_service` | Stage 2 的去重/跨日 |
| `clustering` | Stage 2 的事件聚类 |
| `strategies` + `material_service` | Stage 3 的粗排/精排 |
| `podcast_writer_service` | Stage 4 点评 + Stage 5 文稿 |
| `gh_radar` | 保留为 Stage 2 的“项目轨”子模块或独立板块 |
| `data/briefs` | 由 `ocean/{date}` + `selection/{date}` 取代（迁移期兼容） |

## 11. 待确认

1. 03 板块命名用「新闻海洋（News Ocean）」还是备选？
2. Stage 3 的专家规则权重/阈值是否先按默认值，跑一轮再调？
3. 编辑点评是先做 02 日报，再复用到播客，还是并行两套提示词？
4. 迁移期是否保留旧 brief 作为兼容层（推荐保留 1–2 周）？
5. 连续性判定阈值（语义相似 + 实体重叠 + 时间窗）先给默认值，需实测调参。
