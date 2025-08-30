.PHONY: help build up down logs clean dev prod test

# Default target
help:
	@echo "PDF Watermark Service - Docker Commands"
	@echo "======================================"
	@echo ""
	@echo "Development Commands:"
	@echo "  make dev          - Start development environment"
	@echo "  make dev-build    - Build development containers"
	@echo "  make dev-logs     - View development logs"
	@echo ""
	@echo "Production Commands:"
	@echo "  make prod         - Start production environment"
	@echo "  make prod-build   - Build production containers"
	@echo "  make prod-logs    - View production logs"
	@echo ""
	@echo "General Commands:"
	@echo "  make build        - Build all containers"
	@echo "  make up           - Start all services"
	@echo "  make down         - Stop all services"
	@echo "  make logs         - View logs for all services"
	@echo "  make clean        - Clean up containers, images, and volumes"
	@echo "  make test         - Run tests"
	@echo ""

# Development environment
dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

dev-build:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml build

dev-logs:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

dev-down:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down

# Production environment
prod:
	docker compose -f docker-compose.prod.yml up -d

prod-build:
	docker compose -f docker-compose.prod.yml build

prod-logs:
	docker compose -f docker-compose.prod.yml logs -f

prod-down:
	docker compose -f docker-compose.prod.yml down

# General commands
build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

# Cleanup
clean:
	docker compose down -v --rmi all
	docker system prune -f
	docker volume prune -f

# Testing
test:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm backend python -m pytest

# Health checks
health:
	@echo "Checking service health..."
	@curl -f http://localhost:5001/api/health || echo "Backend health check failed"
	@curl -f http://localhost:80/health || echo "Frontend health check failed"

# Database operations (if using Redis)
redis-cli:
	docker compose exec redis redis-cli

# Monitoring
monitor:
	@echo "Service Status:"
	@docker compose ps
	@echo ""
	@echo "Resource Usage:"
	@docker stats --no-stream

# Backup
backup:
	@echo "Creating backup..."
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@docker compose exec backend tar czf - /app/uploads /app/outputs > backups/$(shell date +%Y%m%d_%H%M%S)/data.tar.gz

# Restore
restore:
	@echo "Restoring from backup..."
	@docker compose exec -T backend tar xzf - < backups/$(shell date +%Y%m%d_%H%M%S)/data.tar.gz
