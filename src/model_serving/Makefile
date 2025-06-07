lint: ## Run linters
	@echo "🚀 Running Ruff lint"
	@ruff check . --fix

dev:
	@echo "🚀 Running the development server"
	@cd app && python main.py

download: ## Download the latest data from the source
	@echo "🚀 Downloading the latest data from the source"
	@huggingface-cli download --resume-download --repo-type dataset 2uanDM/reid-pipeline --local-dir .
	@mv ./assets/ ./src
	@rm -rf .cache

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'



.DEFAULT_GOAL := help