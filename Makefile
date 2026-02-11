# Define the virtual environment directory
VENV_DIR := .venv

##@ Utility
.PHONY: help
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: uv
# uv:  ## Install uv if it's not present.
# 	@command -v uv >/dev/null 2>&1 || brew install uv

.PHONY: venv
venv: uv ## Create a virtual environment
	uv venv $(VENV_DIR)

.PHONY: install
install: venv ## Install dependencies
	uv sync

.PHONY: lock
lock: ## Lock dependencies
	uv lock

.PHONY: sync
sync: ## Synchronize environment with lock file
	uv sync

.PHONY: test
# test: ## Run tests
# 	uv run pytest -vv --cov=main --cov=mylib test_*.py

.PHONY: format
format: ## Format code
	uv run black src/cognidoc/

.PHONY: lint
lint: ## Run linters
	uv run pylint src/cognidoc/

.PHONY: container-lint
container-lint: ## Lint Dockerfile
	docker run --rm -i hadolint/hadolint < Dockerfile

.PHONY: refactor
refactor: format lint ## Format code and run linters

.PHONY: deploy
deploy: ## Deploy application
	# Deploy commands go here

.PHONY: all
all: install lint test format deploy ## Run all tasks