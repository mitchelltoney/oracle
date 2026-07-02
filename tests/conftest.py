from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from services.ingest import Match, MatchStatus

KICKOFF_BASE = datetime(2026, 6, 15, 18, 0, tzinfo=UTC)

MatchFactory = Callable[..., Match]


@pytest.fixture
def make_match() -> MatchFactory:
    counter = iter(range(1, 10_000))

    def _make(
        *,
        id: int | None = None,
        season: int = 2026,
        kickoff: datetime | None = None,
        days_ago: float | None = None,
        stage: str = "GROUP_STAGE",
        status: MatchStatus | None = None,
        home_id: int = 1,
        home: str = "Team A",
        away_id: int = 2,
        away: str = "Team B",
        home_goals: int | None = None,
        away_goals: int | None = None,
        duration: str = "REGULAR",
    ) -> Match:
        if kickoff is None:
            if days_ago is not None:
                kickoff = KICKOFF_BASE - timedelta(days=days_ago)
            else:
                kickoff = KICKOFF_BASE
        finished = home_goals is not None and away_goals is not None
        if status is None:
            status = MatchStatus.FINISHED if finished else MatchStatus.TIMED
        winner: str | None = None
        if finished:
            assert home_goals is not None and away_goals is not None
            if home_goals > away_goals:
                winner = "HOME_TEAM"
            elif home_goals < away_goals:
                winner = "AWAY_TEAM"
            else:
                winner = "DRAW"
        return Match(
            id=id if id is not None else next(counter),
            season=season,
            utc_kickoff=kickoff,
            stage=stage,
            status=status,
            home_id=home_id,
            home=home,
            away_id=away_id,
            away=away,
            home_goals=home_goals,
            away_goals=away_goals,
            duration=duration,
            winner=winner,
        )

    return _make
