# 采集源健康度

目标：从多天的 `raw-*.json.gz` 里看出“哪些源长期失败、长期空、全文抓不到”，
让修源/调并发有依据，而不是等节目内容出问题才发现。

```
data/raw/raw-2026-11-09.json.gz ─┐
data/raw/raw-2026-11-10.json.gz ─┼─→ scripts/source_health.py ─→ 按源聚合
data/raw/raw-2026-11-11.json.gz ─┘                              ├─ 天数 / 条目
                                                                ├─ 失败 / 空源天数
                                                                ├─ 全文成功率
                                                                └─ 平均耗时
```

## 用法

```bash
# 本地已有 raw artifact
uv run python scripts/source_health.py --raw-dir data/raw --days 7 \
  --out-md data/source_health/report.md --out-json data/source_health/report.json
```

GHA 里的 raw artifact 可以先用 `gh run download` 下载，或从 `v3-output-*` 之外的
raw artifact 取；脚本只读 `raw-*.json.gz`，不依赖网络。

## 健康标记

| 标记 | 条件 |
|---|---|
| bad | 失败天数 ≥ 半数 |
| bad | 连续 ≥2 天 0 条（且不是禁用） |
| warn | 有失败但未过半 |
| warn | 空源天数 ≥ 半数 |
| warn | 全文成功率 < 50% |
| ok | 以上都不命中 |

全文成功率 = `full_text_status == ok` 的条目 / 该源条目数；`truncated` 单独计数。

## 报告怎么用

1. `bad` 源逐个看：路由失效？反爬？改走 failover/原生 RSS；
2. `warn` 且全文率低的源：检查是否被反爬，必要时保留摘要降级；
3. 平均耗时特别高的源：评估是否值得为它拖慢整条采集；
4. 连续多天空源：考虑禁用，避免白占并发。

## 当前状态

- 首版聚合脚本 + 测试已落地；
- 真正的“趋势看板”还没接 GHA（先把报告跑起来，再决定要不要上页面/定时任务）。

---

## C3（RSSHub）并发陷阱与诊断工具（2026-11-25）

### 现象

2026-11-25 定时任务里有 8 个 C3 源失败（503 / 超时），而此前 11-16 只有 1 个、11-24 有 4 个。
表面像「路由集体坏掉」，实际不是。

### 根因：所有 C3 路由共用一个 RSSHub 容器

49 个 C3 路由全部指向 `http://127.0.0.1:1200`（同一域），而限流器 `HttpClient._wait`
是**按域**做的；`max_workers` 却是**全局**的（`a3910ec0` 于 2026-11-11 把 8 → 24）。
于是 24 个 worker 抢一个容器的 0.5s 闸门，RSSHub 内部还要并发抓上游，排队超时即 503。

**决定性实验**：在 GHA 容器里按生产配置（0.5s 间隔、20s 超时）**串行**探测全部 50 个路由，
`kiro_blog` / `mathpix_blog` / `wired_ai` / `pixelstech_ai` 全部 200；
同一天 24 并发采集时它们全部失败。→ 是并发压力，不是路由损坏。

恶化轨迹（C3 failed 数）：11-16 = 1 → 11-24 = 4 → 11-25 = 8。

### 修复

```yaml
stage1:
  max_workers: 24        # 外部站点
  rsshub_workers: 4      # C3 独立池，不与外部站点抢
  rsshub_timeout: 45     # RSSHub 现抓上游再渲染，比直连慢
```

`runner.collect_all` 里 C3 走独立 `ThreadPoolExecutor` 与独立 `FetchContext`
（更长超时）。修复后首个真实采集：C3 failed 5 → 2、ok 33 → 37、全站条目 627 → 640。

### 真正损坏的路由（已删除）

| 路由 | 证据 |
|---|---|
| `/inceptionlabs/blog` | 上游博客下线（`/blog` 308 → 首页），稳定 503 |
| `/deeplearning/the-batch` | 上游改版，详情页取不到正文，三天全挂 |
| `/papers/category/arxiv/cs.AI` | papers.cool 不可达，超时 20s |

顺带修正配置笔误：`/tmtpost/news`→`/new`、`/benedictevans`→`/ben-evans`、
`/qbitai/category/资讯`(400)→`/qbitai/tag/大语言模型`；
删除已禁用且路由不存在的 `scientificamerican`、`csia_news`。

### 诊断工具（保留）

| workflow | 用途 |
|---|---|
| `rsshub_probe.yml` | 探测任意路由（可传 `routes` 参数），3 次重试 + 并发模拟 |
| `rsshub_sweep.yml` | 按生产配置串行全量探测所有 C3 路由，输出状态/耗时/字节 |

两者都是 `workflow_dispatch`，不占用定时资源。**排查 C3 失败时先跑 sweep**：
串行成功而采集失败 = 并发问题，两者都失败 = 路由问题。

### 注意

`rsshub_cache.yml`（快照缓存）已于 2026-11-10 去掉定时，仅保留手动 dispatch 作回滚，
不是「缺失的兜底机制」——v3 的设计就是实时容器。
