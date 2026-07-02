"""Generate and log predictions for all upcoming fixtures. Entry point for ``make predict``.

Fits every model once at cutoff=now — leakage-safe for every future fixture —
then appends one row per (fixture, model) to the append-only log, each with its
own ``model_version`` (hard rule 1). A model that cannot fit (e.g. the GBM's
minimum-rows guard on a thin corpus) is skipped with a warning; it never blocks
the others. Dixon-Coles keeps its P0 diet (the WC snapshot corpus); Elo, GBM,
and the ensemble train on the combined WC + historical corpus.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from services.ingest import Match, load_latest_snapshot
from services.models.base import MatchModel
from services.models.dixon_coles import DixonColesModel
from services.models.elo import EloModel
from services.models.ensemble import EnsembleModel
from services.models.features import combine_corpora
from services.models.gbm import GbmModel
from services.prediction_log import PredictionLog, PredictionRecord


def run(data_dir: Path = Path("data")) -> int:
    snapshots_dir = data_dir / "snapshots"
    try:
        snapshot = load_latest_snapshot(snapshots_dir)
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    history_matches: list[Match] = []
    try:
        history_matches = load_latest_snapshot(snapshots_dir, kind="history").matches
    except FileNotFoundError as exc:
        print(f"WARNING: {exc}; training on the WC corpus only", file=sys.stderr)

    now = datetime.now(UTC)
    combined = combine_corpora(history_matches, snapshot.matches)
    to_fit: list[tuple[MatchModel, list[Match]]] = [
        (DixonColesModel(), snapshot.finished_before(now)),  # P0 corpus, unchanged
        (EloModel(), combined),
        (GbmModel(), combined),
        (EnsembleModel(weights_dir=data_dir / "sim"), combined),
    ]
    models: list[MatchModel] = []
    for model, corpus in to_fit:
        try:
            model.fit(corpus, cutoff=now)
        except ValueError as exc:
            print(f"WARNING: skipping {model.name}: {exc}", file=sys.stderr)
            continue
        models.append(model)
    if not models:
        print("FATAL: no model could be fitted", file=sys.stderr)
        return 1

    upcoming = snapshot.upcoming(now)
    if not upcoming:
        print("no upcoming fixtures in the latest snapshot")
        return 0

    log = PredictionLog(data_dir / "predictions" / "predictions.jsonl")
    appended = 0
    for fixture in upcoming:
        for model in models:
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
            appended += 1
            print(
                f"{fixture.utc_kickoff:%Y-%m-%d %H:%M}Z  {fixture.home} v {fixture.away}  "
                f"[{prediction.model_version}]: H {prediction.p_home:.3f}  "
                f"D {prediction.p_draw:.3f}  A {prediction.p_away:.3f}"
            )
    print(
        f"appended {appended} predictions ({len(models)} models x "
        f"{len(upcoming)} fixtures) to {log.path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
