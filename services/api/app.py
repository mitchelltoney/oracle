"""Minimal read-only API over snapshots and the prediction log.

Run with: uv run uvicorn --factory services.api.app:create_app
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from services.ingest import Snapshot, load_latest_snapshot
from services.metrics import score_predictions
from services.prediction_log import PredictionLog


def create_app(data_dir: Path = Path("data")) -> FastAPI:
    app = FastAPI(title="WC Oracle API")
    snapshots_dir = data_dir / "snapshots"
    log = PredictionLog(data_dir / "predictions" / "predictions.jsonl")

    def latest_snapshot() -> Snapshot:
        try:
            return load_latest_snapshot(snapshots_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/fixtures")
    def fixtures() -> list[dict[str, Any]]:
        snapshot = latest_snapshot()
        return [
            {
                "id": m.id,
                "home": m.home,
                "away": m.away,
                "kickoff_utc": m.utc_kickoff.isoformat(),
                "stage": m.stage,
                "status": m.status.value,
            }
            for m in snapshot.upcoming(datetime.now(UTC))
        ]

    @app.get("/predictions")
    def predictions(model_version: str | None = None) -> list[dict[str, Any]]:
        latest = log.latest_per_fixture(model_version)
        return [
            asdict(record)
            for record in sorted(
                latest.values(), key=lambda r: (r.kickoff_utc, r.model_version)
            )
        ]

    @app.get("/calibration")
    def calibration() -> list[dict[str, Any]]:
        snapshot = latest_snapshot()
        reports = score_predictions(
            log.latest_per_fixture().values(), snapshot.matches
        )
        return [asdict(report) for _, report in sorted(reports.items())]

    @app.get("/sim")
    def sim() -> dict[str, Any]:
        path = data_dir / "sim" / "latest.json"
        if not path.exists():
            raise HTTPException(
                status_code=404, detail="no bracket simulation yet — run `make sim`"
            )
        body: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return body

    return app
