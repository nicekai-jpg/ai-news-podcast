# 参与贡献 AI 每日先锋

首先，非常感谢您有意向为 `ai-news-podcast` 贡献代码或提出建议！开源社区正是因为有了您的参与才变得更加美好。

## 我应该从哪里开始？

如果您发现了一个 Bug 或者有任何疑问，请先在我们的 [Issue 追踪器](https://github.com/nicekai-jpg/ai-news-podcast/issues) 中搜索一下，看看是否已经有人提出了相同的问题。如果没有，请随时创建一个新的 Issue！

## 如何参与贡献

我们欢迎所有形式的贡献，无论是修复问题还是提交新功能。

### 1. Fork 并创建一个分支
1. Fork 本仓库。
2. 创建一个新的分支：`git checkout -b my-new-feature`
3. 提交您的修改。

### 2. 遵循开发者指南
请参阅我们的 [开发指南](development.md) 获取关于设置本地环境（使用 `uv`）、理解项目结构以及运行测试的详细说明。

### 3. 提交您的修改
拉取请求 (Pull Request) 需要附带一个描述清晰的提交信息：
```bash
git commit -m "feat: 增加了一个全新的 AI 调研类 RSS 订阅源"
```

### 4. 推送并创建 Pull Request
1. 将分支推送到你的远程仓库：`git push origin my-new-feature`
2. 提交 Pull Request。请在 PR 描述中详细说明您的修改内容，并关联任何相关的 Issue 编号。

## 添加新的 RSS 新闻源
如果您准备向 `config/sources.yaml` 中添加全新的新闻源（Feeds）：
- 请确保该源是稳定且仍在积极维护的。
- 尽量选择那些能提供全文输出，或至少能提供详细摘要的 RSS 源。
- 添加后，请运行 `uv run pytest` 以确保该数据源的格式没有破坏现有的 RSS 解析器的逻辑。

## 报告 Bug / 申请新功能
请使用项目内置的 GitHub Issue 模板提交报告，这样有助于确保您提供了充分的上下文，以便我们能更快地复现并解决您的问题。

再次感谢您对改进 `ai-news-podcast` 项目的关注！
