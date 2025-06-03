# Makefile for Analystt1 - AI-Powered Financial Crime Analysis Platform

.PHONY: help dev test lint format type-check clean migrate migrate-create smoke-test smoke-test-all ws-test ws-demo ws-curl

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development
dev: ## Start development environment with docker-compose
	@echo "Starting development environment..."
	docker-compose up -d

# Testing
test: ## Run all tests
	@echo "Running tests..."
	pytest

lint: ## Run linting
	@echo "Running linter..."
	ruff check .

format: ## Format code
	@echo "Formatting code..."
	ruff format .

type-check: ## Run type checking
	@echo "Running type checking..."
	mypy .

# Database migrations
migrate: ## Run Alembic migrations
	@echo "Running database migrations..."
	alembic upgrade head
	
migrate-create: ## Create a new migration
	@echo "Creating new migration..."
	alembic revision --autogenerate -m "$(MSG)"

# Smoke test
smoke-test: ## Run end-to-end smoke test
	@echo "Running smoke test..."
	python scripts/smoke_test.py || echo "Smoke test failed"
	
smoke-test-all: dev smoke-test ## Start dev environment and run smoke test

# WebSocket testing
ws-test: ## Test WebSocket connection
	@echo "Testing WebSocket connection..."
	python -m pytest tests/test_websocket_progress.py -v

ws-demo: ## Run WebSocket demo client
	@echo "Running WebSocket demo..."
	python scripts/websocket_demo.py

ws-curl: ## Test WebSocket endpoint with curl
	@echo "Testing WebSocket with curl..."
	@curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" -H "Sec-WebSocket-Version: 13" http://localhost:8000/api/v1/ws/tasks/test?token=$(TOKEN)

# Cleanup
clean: ## Remove temporary files and artifacts
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache
