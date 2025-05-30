# Analyst's Augmentation Agent - Makefile
# Provides simplified commands for development and operations

# Colors for output
BLUE=\033[0;34m
GREEN=\033[0;32m
YELLOW=\033[1;33m
RED=\033[0;31m
NC=\033[0m # No Color

# Configuration
DOCKER_COMPOSE=docker-compose
DOCKER_COMPOSE_FILE=docker-compose.yml
DOCKER_COMPOSE_DEV=$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_FILE) --profile dev
DOCKER_COMPOSE_PROD=$(DOCKER_COMPOSE) -f $(DOCKER_COMPOSE_FILE) --profile prod
BACKEND_DIR=backend
FRONTEND_DIR=frontend
LOG_DIR=logs
VENV_DIR=venv
PYTHON=python3
PIP=$(VENV_DIR)/bin/pip
PYTEST=$(VENV_DIR)/bin/pytest

# Default target when just running 'make'
.PHONY: help
help: ## Show this help message
	@echo "Analyst's Augmentation Agent - Development Commands"
	@echo "=================================================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-20s$(NC) %s\n", $$1, $$2}'

# ===========================================
# Setup Commands
# ===========================================
.PHONY: setup
setup: env deps dirs ## Initialize development environment (env file, dependencies, directories)
	@echo "$(GREEN)Setup complete!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "1. Edit .env with your API keys"
	@echo "2. Run 'make start-dev' to start services"

.PHONY: env
env: ## Create .env file from example if it doesn't exist
	@if [ ! -f .env ]; then \
		echo "$(BLUE)Creating .env from .env.example...$(NC)"; \
		cp .env.example .env; \
		echo "$(YELLOW)Please edit .env with your API keys!$(NC)"; \
	else \
		echo "$(BLUE).env file already exists.$(NC)"; \
	fi

.PHONY: deps
deps: deps-backend deps-frontend ## Install all dependencies

.PHONY: deps-backend
deps-backend: ## Install backend dependencies
	@echo "$(BLUE)Installing backend dependencies...$(NC)"
	@if [ ! -d $(VENV_DIR) ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)Backend dependencies installed.$(NC)"

.PHONY: deps-frontend
deps-frontend: ## Install frontend dependencies
	@echo "$(BLUE)Installing frontend dependencies...$(NC)"
	@cd $(FRONTEND_DIR) && npm install
	@echo "$(GREEN)Frontend dependencies installed.$(NC)"

.PHONY: dirs
dirs: ## Create necessary directories
	@echo "$(BLUE)Creating necessary directories...$(NC)"
	@mkdir -p $(LOG_DIR)
	@mkdir -p $(BACKEND_DIR)/agents/configs/crews
	@echo "$(GREEN)Directories created.$(NC)"

# ===========================================
# Service Management
# ===========================================
.PHONY: start-dev
start-dev: env ## Start all services in development mode
	@echo "$(BLUE)Starting services in development mode...$(NC)"
	@$(DOCKER_COMPOSE_DEV) up -d
	@echo "$(GREEN)Services started in development mode.$(NC)"
	@echo "$(BLUE)Frontend:$(NC) http://localhost:3000"
	@echo "$(BLUE)Backend:$(NC) http://localhost:8000"
	@echo "$(BLUE)API Docs:$(NC) http://localhost:8000/docs"
	@echo "$(BLUE)Neo4j:$(NC) http://localhost:7474 (neo4j/analyst123)"

.PHONY: start-prod
start-prod: env ## Start all services in production mode
	@echo "$(BLUE)Starting services in production mode...$(NC)"
	@$(DOCKER_COMPOSE_PROD) up -d
	@echo "$(GREEN)Services started in production mode.$(NC)"
	@echo "$(BLUE)Frontend:$(NC) http://localhost:3000"
	@echo "$(BLUE)Backend:$(NC) http://localhost:8000"
	@echo "$(BLUE)API Docs:$(NC) http://localhost:8000/docs"

.PHONY: start-dbs
start-dbs: ## Start only database services
	@echo "$(BLUE)Starting database services...$(NC)"
	@$(DOCKER_COMPOSE) up -d neo4j postgres redis
	@echo "$(GREEN)Database services started.$(NC)"

.PHONY: stop
stop: ## Stop all services
	@echo "$(BLUE)Stopping all services...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)All services stopped.$(NC)"

.PHONY: restart
restart: stop start-dev ## Restart all services in development mode

.PHONY: restart-prod
restart-prod: stop start-prod ## Restart all services in production mode

# ===========================================
# Development Tools
# ===========================================
.PHONY: logs
logs: ## View logs from all services
	@echo "$(BLUE)Viewing logs from all services...$(NC)"
	@$(DOCKER_COMPOSE) logs -f

.PHONY: logs-backend
logs-backend: ## View logs from backend service
	@echo "$(BLUE)Viewing logs from backend service...$(NC)"
	@$(DOCKER_COMPOSE) logs -f backend

.PHONY: logs-frontend
logs-frontend: ## View logs from frontend service
	@echo "$(BLUE)Viewing logs from frontend service...$(NC)"
	@$(DOCKER_COMPOSE) logs -f frontend

.PHONY: logs-neo4j
logs-neo4j: ## View logs from Neo4j service
	@echo "$(BLUE)Viewing logs from Neo4j service...$(NC)"
	@$(DOCKER_COMPOSE) logs -f neo4j

.PHONY: logs-postgres
logs-postgres: ## View logs from Postgres service
	@echo "$(BLUE)Viewing logs from Postgres service...$(NC)"
	@$(DOCKER_COMPOSE) logs -f postgres

.PHONY: test
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	@$(PYTEST)
	@echo "$(GREEN)Tests completed.$(NC)"

.PHONY: test-backend
test-backend: ## Run backend tests
	@echo "$(BLUE)Running backend tests...$(NC)"
	@$(PYTEST) $(BACKEND_DIR)
	@echo "$(GREEN)Backend tests completed.$(NC)"

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@$(PYTEST) --cov=backend --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	@$(VENV_DIR)/bin/ptw -- --testmon

.PHONY: test-unit
test-unit: ## Run only unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	@$(PYTEST) -m unit

.PHONY: test-integration
test-integration: ## Run only integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	@$(PYTEST) -m integration

.PHONY: lint
lint: ## Run linting with ruff
	@echo "$(BLUE)Running ruff linter...$(NC)"
	@$(VENV_DIR)/bin/ruff check backend tests
	@echo "$(GREEN)Linting completed.$(NC)"

.PHONY: lint-fix
lint-fix: ## Run linting with automatic fixes
	@echo "$(BLUE)Running ruff linter with fixes...$(NC)"
	@$(VENV_DIR)/bin/ruff check --fix backend tests
	@echo "$(GREEN)Linting with fixes completed.$(NC)"

.PHONY: format
format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code with black...$(NC)"
	@$(VENV_DIR)/bin/black backend tests
	@echo "$(BLUE)Sorting imports with isort...$(NC)"
	@$(VENV_DIR)/bin/isort backend tests
	@echo "$(GREEN)Formatting completed.$(NC)"

.PHONY: format-check
format-check: ## Check code formatting without changes
	@echo "$(BLUE)Checking code formatting...$(NC)"
	@$(VENV_DIR)/bin/black --check backend tests
	@$(VENV_DIR)/bin/isort --check-only backend tests
	@echo "$(GREEN)Format check completed.$(NC)"

.PHONY: type-check
type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running mypy type checker...$(NC)"
	@$(VENV_DIR)/bin/mypy backend
	@echo "$(GREEN)Type checking completed.$(NC)"

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	@$(VENV_DIR)/bin/pre-commit run --all-files
	@echo "$(GREEN)Pre-commit hooks completed.$(NC)"

.PHONY: pre-commit-install
pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	@$(VENV_DIR)/bin/pre-commit install
	@echo "$(GREEN)Pre-commit hooks installed.$(NC)"

.PHONY: ci
ci: lint type-check test-coverage ## Run all CI checks locally
	@echo "$(GREEN)All CI checks completed successfully!$(NC)"

# ===========================================
# Health Checks
# ===========================================
.PHONY: health
health: ## Check health of all services
	@echo "$(BLUE)Checking health of all services...$(NC)"
	@curl -s http://localhost:8000/health || echo "$(RED)Backend not responding!$(NC)"
	@curl -s http://localhost:3000 > /dev/null && echo "$(GREEN)Frontend OK$(NC)" || echo "$(RED)Frontend not responding!$(NC)"
	@curl -s http://localhost:8000/health/neo4j || echo "$(RED)Neo4j not responding!$(NC)"
	@curl -s http://localhost:8000/health/gemini || echo "$(RED)Gemini not responding!$(NC)"
	@curl -s http://localhost:8000/health/crew || echo "$(RED)Crew not responding!$(NC)"

.PHONY: health-backend
health-backend: ## Check health of backend service
	@echo "$(BLUE)Checking health of backend service...$(NC)"
	@curl -s http://localhost:8000/health || echo "$(RED)Backend not responding!$(NC)"

.PHONY: health-neo4j
health-neo4j: ## Check health of Neo4j service
	@echo "$(BLUE)Checking health of Neo4j service...$(NC)"
	@curl -s http://localhost:8000/health/neo4j || echo "$(RED)Neo4j not responding!$(NC)"

# ===========================================
# Database Operations
# ===========================================
.PHONY: db-init
db-init: ## Initialize database schema
	@echo "$(BLUE)Initializing database schema...$(NC)"
	@$(DOCKER_COMPOSE) exec neo4j cypher-shell -u neo4j -p analyst123 -f /docker-entrypoint-initdb.d/001-schema.cypher || true
	@echo "$(GREEN)Database schema initialized.$(NC)"

.PHONY: db-reset
db-reset: ## Reset database (WARNING: Deletes all data)
	@echo "$(RED)WARNING: This will delete all data in the database!$(NC)"
	@echo "$(RED)Are you sure you want to continue? [y/N]$(NC)"
	@read -p "" confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "$(BLUE)Resetting database...$(NC)"; \
		$(DOCKER_COMPOSE) down -v; \
		$(DOCKER_COMPOSE) up -d neo4j postgres redis; \
		sleep 10; \
		$(MAKE) db-init; \
		echo "$(GREEN)Database reset complete.$(NC)"; \
	else \
		echo "$(BLUE)Database reset cancelled.$(NC)"; \
	fi

.PHONY: db-shell-neo4j
db-shell-neo4j: ## Open Neo4j shell
	@echo "$(BLUE)Opening Neo4j shell...$(NC)"
	@$(DOCKER_COMPOSE) exec neo4j cypher-shell -u neo4j -p analyst123

.PHONY: db-shell-postgres
db-shell-postgres: ## Open PostgreSQL shell
	@echo "$(BLUE)Opening PostgreSQL shell...$(NC)"
	@$(DOCKER_COMPOSE) exec postgres psql -U analyst -d analyst_agent

# ===========================================
# Cleanup Commands
# ===========================================
.PHONY: clean
clean: ## Clean up generated files
	@echo "$(BLUE)Cleaning up generated files...$(NC)"
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type d -name "*.egg" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".coverage" -exec rm -rf {} +
	@find . -type d -name "htmlcov" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "$(GREEN)Cleanup complete.$(NC)"

.PHONY: clean-logs
clean-logs: ## Clean up log files
	@echo "$(BLUE)Cleaning up log files...$(NC)"
	@rm -rf $(LOG_DIR)/*
	@mkdir -p $(LOG_DIR)
	@echo "$(GREEN)Log cleanup complete.$(NC)"

.PHONY: clean-all
clean-all: clean clean-logs ## Clean up everything including logs
	@echo "$(BLUE)Complete cleanup finished.$(NC)"

.PHONY: purge
purge: clean-all ## Purge all data and containers (WARNING: Destructive)
	@echo "$(RED)WARNING: This will delete all containers, volumes, and data!$(NC)"
	@echo "$(RED)Are you sure you want to continue? [y/N]$(NC)"
	@read -p "" confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "$(BLUE)Purging all data and containers...$(NC)"; \
		$(DOCKER_COMPOSE) down -v; \
		echo "$(GREEN)Purge complete.$(NC)"; \
	else \
		echo "$(BLUE)Purge cancelled.$(NC)"; \
	fi
