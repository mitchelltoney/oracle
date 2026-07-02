.PHONY: dev ingest predict sim backtest test lint

# Targets are stubs on purpose — Claude Code implements them in P0.
# Each stub exits 1 so /goal conditions can't pass vacuously.

dev:
	@echo "TODO(P0/P2): run FastAPI (services/api) + Vite dev server (apps/hud)" && exit 1

ingest:
	@echo "TODO(P0): provider-abstracted data pull -> data/raw/ snapshots" && exit 1

predict:
	@echo "TODO(P0): generate + append predictions -> data/predictions/*.jsonl" && exit 1

sim:
	@echo "TODO(P1): Monte Carlo bracket -> data/sim/latest.json" && exit 1

backtest:
	@echo "TODO(P1): historical calibration report (Brier / log loss)" && exit 1

test:
	@echo "TODO(P0): pytest services/ + vitest apps/hud" && exit 1

lint:
	@echo "TODO(P0): ruff + mypy (services), eslint + tsc (hud)" && exit 1
