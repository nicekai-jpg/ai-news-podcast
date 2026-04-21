.PHONY: help install lint format check test clean build

help: ## 显示可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## 安装依赖（含开发依赖）
	uv sync

lint: ## 运行代码检查
	uv run ruff check src/ tests/ scripts/

format: ## 格式化代码
	uv run ruff format src/ tests/ scripts/

fix: ## 自动修复代码问题
	uv run ruff check --fix src/ tests/ scripts/

lock: ## 更新依赖锁定文件
	uv lock


daily: ## 生成一期播客（本地测试）
	uv run podcast-daily --base-url http://localhost

report: ## 生成 Markdown 日报
	uv run podcast-report

clean: ## 清理缓存和生成文件
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
