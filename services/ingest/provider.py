"""Domain types and the provider-neutral data interface.

Hard rule 3: models, API, and orchestration code import only this interface
(via ``services.ingest``) — never a concrete provider module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol


class MatchStatus(StrEnum):
    SCHEDULED = "SCHEDULED"
    TIMED = "TIMED"
    IN_PLAY = "IN_PLAY"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    SUSPENDED = "SUSPENDED"
    POSTPONED = "POSTPONED"
    CANCELLED = "CANCELLED"
    AWARDED = "AWARDED"


@dataclass(frozen=True)
class Match:
    id: int
    season: int
    utc_kickoff: datetime  # tz-aware UTC
    stage: str
    status: MatchStatus
    home_id: int
    home: str
    away_id: int
    away: str
    home_goals: int | None  # full-time score incl. extra time, excl. shootout
    away_goals: int | None
    duration: str  # REGULAR | EXTRA_TIME | PENALTY_SHOOTOUT
    winner: str | None  # HOME_TEAM | AWAY_TEAM | DRAW


class MissingApiKeyError(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


class DataProvider(Protocol):
    def fetch_matches(self, season: int) -> list[Match]: ...

    def fetch_standings(self, season: int) -> dict[str, Any]: ...
