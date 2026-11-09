# RSSHub 快照缓存中转（GHA 临时实例）

> 目的：把"没有稳定原生 RSS"的少数源（Anthropic News、IT之家、36Kr、DeepSeek Harness
> Releases 等）通过 **GitHub Actions 里临时启动的 RSSHub 实例**抓取成小 JSON 快照，提交回
> 仓库，供日报 Stage 1 消费。不依赖任何第三方公共 RSSHub 实例，不 fork RSSHub。

## 背景与决策

| 方案 | 结论 |
|---|---|
| 第三方公共 RSSHub 实例（rsshub.app / rsshub.ktachibana.party 等） | ❌ 不用：公共实例随时 403/503/停服，可用性绑在别人服务器上 |
| 常驻自建 RSSHub（NAS / 小主机 / 云服务器） | ⏸ 备选：稳定但要一台常开设备，先不投产 |
| **GHA 里临时起官方 RSSHub 镜像** | ✅ 采用：0 元、不依赖第三方、每次跑完即销毁 |
| fork RSSHub 二次开发 | ❌ 不需要：官方镜像 + 官方路由足够；自定义路由可用 `USER_ROUTES_PATH` 外挂 |

公共实例的本质是"中间商"：请求先到第三方 RSSHub，再由它去抓目标网站。任何一跳断了都不行；
而 GHA 临时实例把链路变成"GHA → 本机容器 → 目标网站"，少一跳、自己可控。

## 架构

```
GitHub Actions（rsshub_cache.yml，每天 16:00 UTC / 北京 00:00）
  └─ job 内 service container：ghcr.io/diygod/rsshub:latest（端口 1200）
       └─ scripts/build_rsshub_cache.py
            ├─ 路由源：curl http://localhost:1200/<route>（RSSHub 路由）
            └─ 直连源：配置里给绝对 url（如 GitHub Releases atom）
                 └─ 归一化 → data/cache/<source>.json
                      └─ git commit "cache: rsshub snapshots YYYY-MM-DD [skip ci]" → push main
日报流水线（daily.yml，约 20:30 UTC 起）
  └─ Stage 1 读取 data/cache/*.json（后续实现），失败再实时抓
```

为什么不是"在线 RSSHub"：GHA 没有公网入站、runner 是临时的，不能当常驻服务；它只能做
**定时批量抓取 + 落盘**。这也正是本项目需要的东西。

## 数据契约

`data/cache/<source>.json`：

```json
{
  "source": "ithome_it",
  "label": "IT之家 IT 资讯",
  "category": "news",
  "url": "http://localhost:1200/ithome/it",
  "fetched_at": "2026-11-07T14:14:40+00:00",
  "elapsed_seconds": 1.87,
  "item_count": 7,
  "items": [
    {
      "title": "……",
      "link": "https://www.ithome.com/0/…",
      "published_at": "2026-11-07T08:30:00+00:00",
      "summary": "……（去标签后的纯文本，最多 2000 字）"
    }
  ]
}
```

- 只存 `title / link / published_at / summary`，不存全文 HTML，单文件几十 KB；
- 按 `link` 去重、只保留最近 `--days`（默认 14）天、每源最多 `--max-items`（默认 30）条；
- 单个源失败**不覆盖旧快照**，只是本轮跳过，下轮再试。

## 路由清单

配置在 `config/rsshub_routes.json`（JSON 而非 YAML：脚本只用标准库，GHA 里无需装依赖）。
每条二选一：`route`（相对 RSSHub 实例）或 `url`（绝对地址，如 GitHub Releases atom）。

当前 11 条（本地实测 11/11 成功，见下）：

| name | 来源 | 方式 |
|---|---|---|
| `anthropic_news` | Anthropic News（官网 RSS 已 404） | RSSHub `/anthropic/news` |
| `anthropic_research` | Anthropic Research | RSSHub `/anthropic/research` |
| `google_research` | Google Research Blog | RSSHub `/google/research` |
| `dsh_releases` | DeepSeek Harness 版本发布 | GitHub Releases atom（直连） |
| `ithome_it` | IT之家 IT 资讯（官网 feed 404） | RSSHub `/ithome/it` |
| `36kr_newsflashes` | 36Kr 快讯（官网 feed 返回 HTML） | RSSHub `/36kr/newsflashes` |
| `geekpark` | 极客公园 | RSSHub `/geekpark` |
| `huxiu_moment` | 虎嗅 24 小时 | RSSHub `/huxiu/moment` |
| `aijishu_ai` | 极术社区 AI | RSSHub `/aijishu/channel/ai` |
| `aibase_news` | AIBase 新闻 | RSSHub `/aibase/news` |
| `baai_hub` | 智源社区 | RSSHub `/baai/hub` |

### 试过但淘汰的（2026-11-08 本地实测）

| 路由 | 原因 |
|---|---|
| `/deepmind/blog` | 本地 30s 超时（官网原生 RSS 已覆盖，不必绕） |
| `/producthunt/today` | 503 |
| `/github/trending/daily` | 404（该路由已变化；GitHub Trending 走官方 API） |
| `/huggingface/blog-zh` | 最新条目停在 2026-06（陈旧） |
| `/qbitai/tag/大语言模型` | 最新 2026-05（陈旧；量子位走原生 feed） |
| `/hellogithub/article` | 最新 2026-01（陈旧） |
| `/zhihu/pin/hotlist` | 能返回但日期解析成 2020 年（不可用） |
| `/leiphone/newsflash` | 时而超时，暂不纳入（可作二线候选） |
| `/openai/*`、`/techcrunch/*`、`/hackernews`、`/qbitai/category/资讯` | 已有原生 RSS / 官方 API，不重复接 |
| `/tmtpost/new`、`/ifanr/digest`、`/infoq/recommend`、`/theverge`、`/wired/tag/...` | 可用但同质化/偏泛科技，留作二线候选 |

加新源：往 `config/rsshub_routes.json` 的 `sources` 里加一条即可，不用改脚本/workflow。

## 调度与提交

- **频率**：每天 1 次（16:00 UTC）。GitHub 调度会延迟 0–3h，即使最晚 19:00 UTC 完成，
  也早于日报流水线（约 20:30 UTC 起）。需要更实时可改 cron（如 `0 */3 * * *`），但会变吵。
- **提交**：`data/cache/` 有变化才提交，commit message 带 `[skip ci]`（和仓库现有约定一致），
  push 前 `git pull --rebase`，失败重试 3 次。
- **并发**：独立并发组 `rsshub-cache`，避免排在 4–5 小时的日报主链后面。
- **手动测速**：`workflow_dispatch` 勾选 `dry_run`，只抓取并打印耗时，不写文件、不提交。

## 实测数据（本机，2026-11-07）

| 项目 | 数值 |
|---|---|
| 镜像 `diygod/rsshub:latest` | 压缩 126.7 MB（不要用 chromium-bundled，376 MB） |
| 容器启动到可响应 | 0.9–2 s |
| 11 条源首次抓取（含上游） | **约 18 s**（单条 0.01–4s，最慢极客公园/Anthropic 约 4s） |
| 快照体积 | 11 个文件合计 **约 219 KB**（115 条，`--max-items 20`、单条摘要 ≤1200 字） |
| GHA 预估单次总耗时 | 1–2 分钟（主要花在拉镜像 20–60s） |

## 已知限制

- GHA cron 不准点（延迟 0–3h），快照是"小时级"不是实时；
- 从 GHA 海外 IP 抓国内站点可能更慢，个别站点可能被墙/限流（失败只跳过，不影响其他源）；
- 快照只有摘要，不含全文；需要全文时由 Stage 1 再按链接抓（可选）；
- 微信公众号/小红书/知乎等**登录态内容**不适用本方案，仍需桥接（wewe-rss 等）；
- RSSHub 路由可能随上游改版失效，失效时该源保留旧快照并告警。

## 后续步骤（Phase 2）

1. **Stage 1 消费**：新增 `fetcher_type: local_cache`，读 `data/cache/*.json` 生成 RawItem
   （走普通去重/打分/聚类）；缓存缺失或过期时回退实时抓取；
2. **扩源**：按需加 RSSHub 路由（如 `/openai/news`、`/huggingface/blog-zh`、`/zhihu/pin/hotlist`）
   或更多 GitHub Releases atom；
3. **提频**：确认时效不足时，把 cron 改为每 3 小时，并让 workflow 只在内容变化时提交；
4. **监控**：在 notify 的 Step Summary 里输出每个源的成功/失败与条目数。

## 回滚

- 删除 `.github/workflows/rsshub_cache.yml` 即停用采集；
- 删除 `data/cache/` 或忽略该目录即停用消费（Phase 2 实现前，缓存文件不影响现有流水线）。
