.PHONY: help install install-dev format lint type-check test test-unit test-integration test-e2e test-cov run up down logs clean

# Default target
help:
	@echo "Enterprise Agentic RAG Platform - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install all dependencies including dev tools"
	@echo ""
	@echo "Development:"
	@echo "  run           Start development server with hot reload"
	@echo "  format        Format code with Ruff"
	@echo "  lint          Run linter checks"
	@echo "  type-check    Run type checking with mypy"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-e2e      Run end-to-end tests only"
	@echo "  test-cov      Run tests with coverage report"
	@echo ""
	@echo "Docker:"
	@echo "  up            Start all services (Qdrant, Redis, Postgres)"
	@echo "  down          Stop all services"
	@echo "  logs          View service logs"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean         Remove cache and temporary files"

# =============================================================================
# Setup
# =============================================================================

install:
	uv sync --no-dev

install-dev:
	uv sync --all-extras
	uv run pre-commit install

# =============================================================================
# Development
# =============================================================================

run:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

format:
	uv run ruff format .
	uv run ruff check . --fix

lint:
	uv run ruff check .

type-check:
	uv run mypy src/

# =============================================================================
# Testing
# =============================================================================

test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v -m unit

test-integration:
	uv run pytest tests/integration/ -v -m integration

test-e2e:
	uv run pytest tests/e2e/ -v -m e2e

test-cov:
	uv run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# =============================================================================
# Docker
# =============================================================================

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

# =============================================================================
# Database Migrations (Alembic)
# =============================================================================

migrate-init:
	uv run alembic init migrations

migrate-create:
	@read -p "Enter migration message: " msg; \
	uv run alembic revision --autogenerate -m "$$msg"

migrate-up:
	uv run alembic upgrade head

migrate-down:
	uv run alembic downgrade -1

# =============================================================================
# Maintenance
# =============================================================================

clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# =============================================================================
# Evaluation (Phase 3)
# =============================================================================

eval-setup:
	uv sync --extra eval

eval-run:
	uv run python -m scripts.evaluate
