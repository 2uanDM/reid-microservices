lint: ## Run linters
	@echo "🚀 Running Ruff lint"
	@ruff check . --fix

dev:
	@echo "🚀 Running the development server"
	@cd app && python main.py

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help