# dr01d0ne Makefile
# Comprehensive development workflow automation

# Configuration
SHELL := /bin/bash
.PHONY: all setup dev dev-backend dev-frontend test test-backend test-frontend lint format clean help docker docker-build docker-up docker-down docker-logs redis-cli redis-cache-cli db-shell neo4j-shell init-vector-index generate-api-types

# Colors for pretty output
BLUE := \033[1;34m
GREEN := \033[1;32m
YELLOW := \033[1;33m
RED := \033[1;31m
RESET := \033[0m

# Default target
all: help

# ==============================================================================
# Development Targets
# ==============================================================================

# Setup development environment
setup:
	@echo -e "${BLUE}Setting up development environment...${RESET}"
	@pip install -r requirements.txt
	@cd frontend && npm install
	@cp -n .env.example .env || true
	@echo -e "${GREEN}Setup complete!${RESET}"

# Run backend with hot-reload
dev-backend:
	@echo -e "${BLUE}Starting backend with hot-reload...${RESET}"
	@uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Run frontend with hot-reload
dev-frontend:
	@echo -e "${BLUE}Starting frontend with hot-reload...${RESET}"
	@cd frontend && npm run dev

# Run both backend and frontend concurrently with hot-reload
dev:
	@echo -e "${BLUE}Starting full-stack development environment...${RESET}"
	@echo -e "${YELLOW}Backend: http://localhost:8000${RESET}"
	@echo -e "${YELLOW}Frontend: http://localhost:3000${RESET}"
	@echo -e "${YELLOW}API Docs: http://localhost:8000/api/docs${RESET}"
	@./scripts/start.sh

# Initialize Redis vector index for development
init-vector-index:
	@echo -e "${BLUE}Initializing Redis vector index...${RESET}"
	@python scripts/init_redis_vector_index.py --redis-url redis://localhost:6380/1 --retry

# Generate TypeScript API client from OpenAPI spec
generate-api-types:
	@echo -e "${BLUE}Generating TypeScript API client...${RESET}"
	@./scripts/generate_openapi_types.sh

# ==============================================================================
# Testing Targets
# ==============================================================================

# Run all tests
test: test-backend test-frontend

# Run backend tests
test-backend:
	@echo -e "${BLUE}Running backend tests...${RESET}"
	@pytest

# Run backend tests with coverage
test-backend-cov:
	@echo -e "${BLUE}Running backend tests with coverage...${RESET}"
	@pytest --cov=backend --cov-report=term --cov-report=html

# Run frontend tests
test-frontend:
	@echo -e "${BLUE}Running frontend tests...${RESET}"
	@cd frontend && npm test

# Run e2e tests
test-e2e:
	@echo -e "${BLUE}Running end-to-end tests...${RESET}"
	@cd frontend && npm run test:e2e

# Run smoke tests
test-smoke:
	@echo -e "${BLUE}Running smoke tests...${RESET}"
	@python scripts/smoke_test.py

# ==============================================================================
# Linting and Formatting
# ==============================================================================

# Run all linting
lint: lint-backend lint-frontend

# Run backend linting
lint-backend:
	@echo -e "${BLUE}Linting backend code...${RESET}"
	@ruff check backend tests
	@mypy backend

# Run frontend linting
lint-frontend:
	@echo -e "${BLUE}Linting frontend code...${RESET}"
	@cd frontend && npm run lint

# Format all code
format: format-backend format-frontend

# Format backend code
format-backend:
	@echo -e "${BLUE}Formatting backend code...${RESET}"
	@black backend tests
	@isort backend tests

# Format frontend code
format-frontend:
	@echo -e "${BLUE}Formatting frontend code...${RESET}"
	@cd frontend && npm run format

# ==============================================================================
# Docker Targets
# ==============================================================================

# Build all Docker images
docker-build:
	@echo -e "${BLUE}Building Docker images...${RESET}"
	@docker compose build

# Start all services with Docker Compose
docker-up:
	@echo -e "${BLUE}Starting all services...${RESET}"
	@docker compose up -d
	@echo -e "${GREEN}Services started!${RESET}"
	@echo -e "${YELLOW}Backend: http://localhost:8000${RESET}"
	@echo -e "${YELLOW}Frontend: http://localhost:3000${RESET}"

# Stop all services
docker-down:
	@echo -e "${BLUE}Stopping all services...${RESET}"
	@docker compose down
	@echo -e "${GREEN}Services stopped!${RESET}"

# View logs from all services
docker-logs:
	@echo -e "${BLUE}Viewing logs...${RESET}"
	@docker compose logs -f

# Start production-like environment
docker-prod:
	@echo -e "${BLUE}Starting production-like environment...${RESET}"
	@docker compose -f docker-compose.prod.yml up -d
	@echo -e "${GREEN}Production environment started!${RESET}"
	@echo -e "${YELLOW}Backend: http://localhost:8000${RESET}"
	@echo -e "${YELLOW}Frontend: http://localhost:3000${RESET}"

# ==============================================================================
# Database Shells
# ==============================================================================

# Open Redis CLI
redis-cli:
	@echo -e "${BLUE}Opening Redis CLI...${RESET}"
	@docker exec -it analyst_agent_redis redis-cli

# Open Redis Cache CLI
redis-cache-cli:
	@echo -e "${BLUE}Opening Redis Cache CLI...${RESET}"
	@docker exec -it analyst_agent_redis_cache redis-cli

# Open PostgreSQL shell
db-shell:
	@echo -e "${BLUE}Opening PostgreSQL shell...${RESET}"
	@docker exec -it analyst_agent_postgres psql -U analyst -d analyst_agent

# Open Neo4j shell
neo4j-shell:
	@echo -e "${BLUE}Opening Neo4j shell...${RESET}"
	@docker exec -it analyst_agent_neo4j cypher-shell -u neo4j -p analyst123

# ==============================================================================
# Utility Targets
# ==============================================================================

# Clean up generated files
clean:
	@echo -e "${BLUE}Cleaning up...${RESET}"
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type d -name "*.egg" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name "htmlcov" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type d -name ".ruff_cache" -exec rm -rf {} +
	@find . -type d -name "node_modules" -exec rm -rf {} +
	@find . -type d -name ".next" -exec rm -rf {} +
	@echo -e "${GREEN}Cleanup complete!${RESET}"

# Install pre-commit hooks
install-hooks:
	@echo -e "${BLUE}Installing pre-commit hooks...${RESET}"
	@pre-commit install
	@echo -e "${GREEN}Pre-commit hooks installed!${RESET}"

# ==============================================================================
# Help Target
# ==============================================================================

# Show help
help:
	@echo -e "${BLUE}Analyst Droid Makefile Help${RESET}"
	@echo -e "${YELLOW}Development:${RESET}"
	@echo -e "  ${GREEN}make setup${RESET}              - Set up development environment"
	@echo -e "  ${GREEN}make dev${RESET}                - Run backend and frontend with hot-reload"
	@echo -e "  ${GREEN}make dev-backend${RESET}        - Run backend with hot-reload"
	@echo -e "  ${GREEN}make dev-frontend${RESET}       - Run frontend with hot-reload"
	@echo -e "  ${GREEN}make init-vector-index${RESET}  - Initialize Redis vector index"
	@echo -e "  ${GREEN}make generate-api-types${RESET} - Generate TypeScript API client"
	@echo -e "${YELLOW}Testing:${RESET}"
	@echo -e "  ${GREEN}make test${RESET}               - Run all tests"
	@echo -e "  ${GREEN}make test-backend${RESET}       - Run backend tests"
	@echo -e "  ${GREEN}make test-frontend${RESET}      - Run frontend tests"
	@echo -e "  ${GREEN}make test-e2e${RESET}           - Run end-to-end tests"
	@echo -e "  ${GREEN}make test-smoke${RESET}         - Run smoke tests"
	@echo -e "${YELLOW}Linting & Formatting:${RESET}"
	@echo -e "  ${GREEN}make lint${RESET}               - Run all linting"
	@echo -e "  ${GREEN}make format${RESET}             - Format all code"
	@echo -e "${YELLOW}Docker:${RESET}"
	@echo -e "  ${GREEN}make docker-build${RESET}       - Build all Docker images"
	@echo -e "  ${GREEN}make docker-up${RESET}          - Start all services"
	@echo -e "  ${GREEN}make docker-down${RESET}        - Stop all services"
	@echo -e "  ${GREEN}make docker-logs${RESET}        - View logs from all services"
	@echo -e "  ${GREEN}make docker-prod${RESET}        - Start production-like environment"
	@echo -e "${YELLOW}Database Shells:${RESET}"
	@echo -e "  ${GREEN}make redis-cli${RESET}          - Open Redis CLI"
	@echo -e "  ${GREEN}make redis-cache-cli${RESET}    - Open Redis Cache CLI"
	@echo -e "  ${GREEN}make db-shell${RESET}           - Open PostgreSQL shell"
	@echo -e "  ${GREEN}make neo4j-shell${RESET}        - Open Neo4j shell"
	@echo -e "${YELLOW}Utilities:${RESET}"
	@echo -e "  ${GREEN}make clean${RESET}              - Clean up generated files"
	@echo -e "  ${GREEN}make install-hooks${RESET}      - Install pre-commit hooks"
