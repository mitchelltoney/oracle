"""Generate and log predictions for all upcoming fixtures. Entry point for ``make predict``.

Fits Dixon-Coles once at cutoff=now on the latest snapshot — leakage-safe for every
future fixture — then appends one row per upcoming fixture to the append-only log.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from services.ingest import load_latest_snapshot
from services.models.dixon_coles import DixonColesModel
from services.prediction_log import PredictionLog, PredictionRecord


def run(data_dir: Path = Path("data")) -> int:
    try:
        snapshot = load_latest_snapshot(data_dir / "snapshots")
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    now = datetime.now(UTC)
    model = DixonColesModel()
    try:
        model.fit(snapshot.finished_before(now), cutoff=now)
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    upcoming = snapshot.upcoming(now)
    if not upcoming:
        print("no upcoming fixtures in the latest snapshot")
        return 0

    log = PredictionLog(data_dir / "predictions" / "predictions.jsonl")
    for fixture in upcoming:
        prediction = model.predict(fixture)
        record = PredictionRecord(
            fixture_id=fixture.id,
            home=fixture.home,
            away=fixture.away,
            kickoff_utc=fixture.utc_kickoff.isoformat(),
            model=prediction.model,
            model_version=prediction.model_version,
            probs={
                "home": prediction.p_home,
                "draw": prediction.p_draw,
                "away": prediction.p_away,
            },
            top_scorelines=[
                [float(h), float(a), p] for h, a, p in prediction.top_scorelines
            ],
            snapshot_as_of=snapshot.as_of_utc.isoformat(),
        )
        log.append(record)
        print(
            f"{fixture.utc_kickoff:%Y-%m-%d %H:%M}Z  {fixture.home} v {fixture.away}: "
            f"H {prediction.p_home:.3f}  D {prediction.p_draw:.3f}  "
            f"A {prediction.p_away:.3f}"
        )
    print(f"appended {len(upcoming)} predictions to {log.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
