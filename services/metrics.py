"""Calibration metrics computed strictly from the prediction log vs completed results.

The 1X2 outcome is derived from the full-time score (extra time included, shootout
excluded) — consistent with what the models are fit on; a knockout match decided on
penalties therefore counts as a draw for calibration purposes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from services.ingest import Match, MatchStatus
from services.prediction_log import PredictionRecord

_CLIP = 1e-15


@dataclass(frozen=True)
class CalibrationReport:
    model_version: str
    n: int
    brier: float
    log_loss: float


def _outcome(match: Match) -> str | None:
    if match.status is not MatchStatus.FINISHED:
        return None
    if match.home_goals is None or match.away_goals is None:
        return None
    if match.home_goals > match.away_goals:
        return "home"
    if match.home_goals < match.away_goals:
        return "away"
    return "draw"


def score_predictions(
    predictions: Iterable[PredictionRecord], results: Iterable[Match]
) -> dict[str, CalibrationReport]:
    """Brier score and log loss per model_version, joined on fixture id."""
    outcome_by_fixture: dict[int, str] = {}
    for match in results:
        outcome = _outcome(match)
        if outcome is not None:
            outcome_by_fixture[match.id] = outcome

    scored: dict[str, list[tuple[float, float]]] = {}  # version -> [(brier, ll)]
    for record in predictions:
        outcome = outcome_by_fixture.get(record.fixture_id)
        if outcome is None:
            continue
        brier = sum(
            (record.probs[k] - (1.0 if k == outcome else 0.0)) ** 2
            for k in ("home", "draw", "away")
        )
        log_loss = -math.log(max(record.probs[outcome], _CLIP))
        scored.setdefault(record.model_version, []).append((brier, log_loss))

    return {
        version: CalibrationReport(
            model_version=version,
            n=len(pairs),
            brier=sum(b for b, _ in pairs) / len(pairs),
            log_loss=sum(ll for _, ll in pairs) / len(pairs),
        )
        for version, pairs in scored.items()
    }
