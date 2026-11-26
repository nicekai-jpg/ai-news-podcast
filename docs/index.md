# AI 每日先锋 文档

## 目录

- [**架构说明**](architecture.md) — 系统工作流与核心模块设计
- [**知识图谱事实层**](kg_extraction.md) — 零 LLM 抽事实卡（已实施 2026-11-16）
- [**三元组抽取统一评测**](kg_triple_benchmark.md) — 非 LLM 方案同口径对比（方法见 [说明](kg_triple_benchmark_method.md)）
- [**从采集到推荐（第三套方案）**](source_to_recommendation.md) — L1–L4 分层设计；**槽位已放宽 8→15（每天捞回正例 28→42），见 todo N1**
- [**成品质量：S4–S7**](editorial_quality.md) — 日报/文稿/点评的检查项、已修缺陷与残留问题
- [**推荐评测**](recommend_evaluation.md) — 推荐指标口径（P@k 的局限）、降权消融与 S3 上限证据
- [**开发指南**](development.md) — 本地环境搭建、项目结构与代码规范
- [**贡献指南**](contributing.md) — 如何参与项目贡献
- [**生产计划**](plan.md) — MVP 阶段的设计文档与技术选型（⚠️ 与实际代码存在差异）
- [**更新日志**](../CHANGELOG.md) — 版本更新记录
