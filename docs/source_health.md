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
