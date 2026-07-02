"""Common model interface: every match model implements ``MatchModel``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from services.ingest import Match


@dataclass(frozen=True)
class Prediction:
    fixture_id: int
    p_home: float
    p_draw: float
    p_away: float
    top_scorelines: list[tuple[int, int, float]]  # (home_goals, away_goals, prob)
    model: str
    model_version: str


class LeakageError(RuntimeError):
    pass


class MatchModel(Protocol):
    name: str
    version: str

    def fit(self, matches: Sequence[Match], cutoff: datetime) -> None:
        """Fit on finished matches that kicked off strictly before ``cutoff``."""
        ...

    def predict(self, fixture: Match) -> Prediction:
        """Predict a fixture kicking off after the fitted cutoff (else LeakageError)."""
        ...
