# 多通道信息采集系统 · 整体设计总览（定稿）

> 版本：v1.0（2026-11-09）
> 状态：通道与源已完成验证并定稿；本文是最终设计汇总，细化清单见 `docs/source_channels.md`。
> 目标：每天一次性完整抓取 AI 相关新闻（含正文），之后全程零二次抓取，交给 Stage 1 做去重/聚类/打分/选材，产出播客与日报。

## 1. 设计原则

1. **一次性全量抓取**：采集阶段把元数据 + 正文全部抓下来并落盘；之后不再访问网络（文章可能 404/反爬/需登录）。
2. **通道 = 抓取机制**：RSS、Atom、RSSHub、网页抓取、公众号 feed，各自独立成通道。
3. **源唯一归属但允许跨通道重复**：同一源可同时出现在不同通道（如 OpenAI 同时在 C1 和 C3），不做跨通道去重，去重统一在 Stage 1。
4. **采集端只做轻量过滤**：URL 规范化、同源内去重、时效过滤、条数上限；重要性判断全部交给 Stage 1。
5. **失败隔离**：单源失败保留旧快照，不阻塞其他源；RSSHub 容器失败只影响 C3。
6. **全文不进 git**：全文走 GHA artifact（同一次 run 内传递）；git 只存摘要快照用于审计。

## 2. 通道总表（定稿）

| 通道 | 机制 | 源数量 | 验证状态 |
|---|---|---|---|
| C1 固定 RSS | 直连 RSS 2.0 URL | 31 | ✅ 格式逐个确认 |
| C2 Atom | 直连 Atom 1.0 URL（GitHub Releases、YouTube） | 8 + 3 | ✅ 逐条实测 |
| C3 RSSHub | 临时 RSSHub 容器 + 路由 | 66（38 一级 + 6 二级 + 22 三级） | ✅ 全部本地实测 |
| C4 网页抓取 | HTML 解析（SSR 页面） | 7 | ✅ 逐家实测 |
| C5 wechat2rss | 公共公众号 feed | 17 | ✅ 全部实测（含全文） |
| C6 私有桥 | 登录态（wewe-rss / 私有部署） | — | 排除，另立项 |

合计约 **132 个源条目**（跨通道重复按条目计）。

## 3. 各通道定稿清单（摘要）

### C1 固定 RSS（31）

- 中文媒体：量子位、少数派、机器之心（wechat2rss）、新智元（wechat2rss）；
- 官方博客：OpenAI、DeepMind、Google AI、NVIDIA、Microsoft Research、Apple ML、HuggingFace、Qwen；
- 英文媒体：TechCrunch AI、Synced Review、VentureBeat AI；
- arXiv：cs.CV / cs.CL / cs.RO / cs.AI / cs.LG + 10 个主题查询；
- 其它：36Kr 原生（已坏，主通道改 C3）。

### C2 Atom（11）

- **GitHub Releases（8）**：deepseek-harness、DeepGEMM、DeepEP、vllm、llama.cpp、transformers、langchain、ollama；
- **YouTube（3）**：OpenAI、Google DeepMind、Anthropic。

### C3 RSSHub（66）

- **一级 AI 垂直/官方（38）**：OpenAI、Anthropic、Claude Blog、Claude Code Changelog、DeepMind、DeepSeek、HuggingFace、Ollama、Qwen Research、Cohere、LangChain、Kiro、Inception Labs、Thinking Machines、Transformer Circuits、The Decoder、Cool Papers、MIT TR AI、NVIDIA Newsroom、NVIDIA Developer、智源 BAAI、DeepLearning.AI、BFL、Cursor、Cline、Coze、Civitai、Chub、Podwise、Zread、Forward Future、Augment Code、DeepL、Mathpix、AI工具集、AIbase、极术社区、量子位、Google Research、PixelsTech；
- **二级 AI 栏目（6）**：IT之家 `/tag/AI`、少数派 `/tag/AI`、NYT Technology、Nature News、WIRED AI、Bloomberg Technology；
- **三级 泛科技/产业（22，可选）**：36Kr、极客公园、虎嗅、钛媒体、雷峰网、爱范儿、InfoQ、澎湃、cnBeta、OFweek、Engadget、Scientific American、TrendForce、半导体协会、开源中国、掘金、快科技、蓝点网、数字尾巴、MIT 科技评论中文、Benedict Evans、Gwern。

### C4 网页抓取（7）

DeepSeek、智谱、MiniMax、字节 Seed、商汤、工信部、网信办（含待验证：零一万物）。
二期/暂不接：Kimi、文心、混元、阶跃、讯飞（SPA，需 Playwright）。

### C5 wechat2rss（17）

机器之心、新智元、量子位、极客公园、PaperWeekly、我爱计算机视觉、夕小瑶科技说、字节跳动技术团队、阿里技术、阿里云开发者、阿里巴巴中间件、腾讯技术工程、美团技术团队、小米技术、哔哩哔哩技术、爱奇艺技术产品团队、得物技术。

## 4. 统一数据契约

```json
{
  "channel": "c1_rss|c2_atom|c3_rsshub|c4_web|c5_wechat2rss",
  "source": "anthropic_news",
  "source_label": "Anthropic News",
  "category": "official|news|research|community|podcast|video",
  "title": "……",
  "link": "https://……",
  "published_at": "2026-11-09T00:00:00+00:00",
  "summary": "……（≤500 字）",
  "full_text": "……（采集时一次性抓到的正文）",
  "full_text_status": "ok|failed|truncated",
  "extra": {"score": 120, "comments": 14}
}
```

落盘：

```
data/cache/
  c1_rss/{source}.json
  c2_atom/{source}.json
  c3_rsshub/{source}.json
  c4_web/{source}.json
  c5_wechat2rss/{account}.json
```

## 5. 存储与传递

| 层 | 内容 | 位置 | 保留 |
|---|---|---|---|
| 采集产物 | 元数据 + 摘要 + **全文** | GHA artifact（同一次 run 传给 Stage 1） | 30 天 |
| 摘要快照 | 元数据 + 摘要（≤500 字） | `data/cache/`（git） | 审计/回看 |
| 节目产物 | brief / 脚本 / 音频 | 现有流程 | 不变 |

- 全文不进 git（避免仓库膨胀）；artifact 默认保留 30 天（与去重窗口一致）；
- 采集 Job 上传 `fulltext-{date}.json.gz`，Stage 1 下载解压后本地处理，**零二次抓取**。

## 6. 调度

- 每天 **16:00 UTC（北京 00:00）** 执行一次；GitHub 调度延迟 0–3h，仍早于日报流水线；
- 采集作为 daily.yml 的 Job-0：起临时 RSSHub 容器 → 抓 C1–C5 → 上传 artifact → Stage 1 消费；
- C3 的 RSSHub 容器起不来时，C3 源跳过，其余通道照常；
- 手动触发支持 `dry_run`（只抓不写）与单通道调试。

## 7. 跨期去重

- **窗口 30 天**；比较用的历史数据来自历史采集 artifact（下载后本地比对），不重新抓取；
- 叠加历史 brief 的“已选中”标记：只有“出现过且已选中”的新闻才过滤；
- 历史 artifact 缺失时，用 git 里的覆盖账本（`data/coverage/ledger.json`）兜底；
- 新进展保护：出现新数字/日期/版本/动作时保留，避免误杀事件更新。

## 8. 失败处理

- 每源 try/except + 重试；单源失败保留旧快照；
- 单通道失败不影响其他通道；
- 采集产物缺失/损坏时，Stage 1 用最近一次摘要快照降级运行。

## 9. 容量与耗时预算

| 通道 | 条数上限/天（估） |
|---|---|
| C1 | ~300 |
| C2 | ~250 |
| C3 | ~700 |
| C4 | ~150 |
| C5 | ~200 |
| **合计** | **~1600 条/天（去重前）** |

- 全文 artifact 约 7–15MB/天（gzip 后 2–5MB）；摘要快照约 1MB/天（git）；
- 单轮采集耗时约 3–6 分钟（含 RSSHub 镜像启动）。

## 10. 边界与已知缺口

- **不纳入**：微信公众号私有号、微博、小红书、知乎热榜、X/Twitter、Reddit、Bluesky（登录态/付费/不稳定）；
- **RSSHub 没有的 AI 源**：机器之心、新智元、VentureBeat、Synced、The Information、Import AI、Simon Willison、Mistral、xAI、智谱、Kimi、MiniMax、文心、混元、阶跃、零一、商汤、讯飞等（靠官网直抓/公众号/媒体覆盖）；
- **SPA 二期**：Kimi、文心、混元、阶跃、讯飞（Playwright 渲染或放弃）；
- **暂缓**：CAAI、国科大 AI（学术机构公告类）；
- **Telegram**：待选定可信频道。

## 11. 实现顺序（建议）

1. 采集 Job-0 骨架 + 统一 item schema + artifact 传递；
2. C1/C2（简单直连）→ C5（wechat2rss）→ C3（RSSHub 容器 + 66 路由）→ C4（7 个 SSR 抓取器）；
3. Stage 1 改为消费 artifact（含全文）；
4. 跨期去重（30 天 artifact + 账本）；
5. 摘要快照提交 git（审计用）；
6. 逐步接入三级源与二期源。
