.PHONY: dev ingest predict sim backtest test lint

dev:
	@echo "TODO(P0/P2): run FastAPI (services/api) + Vite dev server (apps/hud)" && exit 1

ingest:
	uv run python -m services.ingest.cli

predict:
	uv run python -m services.predict

sim:
	@echo "TODO(P1): Monte Carlo bracket -> data/sim/latest.json" && exit 1

backtest:
	@echo "TODO(P1): historical calibration report (Brier / log loss)" && exit 1

test:
	uv run pytest

lint:
	uv run ruff check . && uv run mypy services tests
