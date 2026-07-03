.PHONY: dev ingest predict sim backtest test lint

dev:
	@test -d apps/hud/node_modules || npm --prefix apps/hud install; \
	trap 'kill 0' EXIT; \
	uv run uvicorn --factory services.api.app:create_app --port 8000 --reload & \
	npm --prefix apps/hud run dev & \
	wait

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
	npm --prefix apps/hud run test

lint:
	uv run ruff check . && uv run mypy services tests
	npm --prefix apps/hud run lint && npm --prefix apps/hud run typecheck
