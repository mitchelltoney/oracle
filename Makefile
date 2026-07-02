.PHONY: dev ingest predict sim backtest test lint

dev:
	@echo "TODO(P0/P2): run FastAPI (services/api) + Vite dev server (apps/hud)" && exit 1

ingest:
	uv run python -m services.ingest.cli

predict:
	uv run python -m services.predict

sim:
	uv run python -m services.sim.cli

backtest:
	uv run python -m services.backtest

test:
	uv run pytest

lint:
	uv run ruff check . && uv run mypy services tests
