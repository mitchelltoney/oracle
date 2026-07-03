"""GBM feature matrix — one chronological pass, leakage-free by construction.

Every training row is emitted BEFORE the row's match updates the rolling state,
so each row sees exactly the information available strictly before its kickoff
(hard rule 2). The identical ``featurize`` path serves prediction-time fixtures
from a ``TeamState`` built the same way.

Features and their as-of-time sources:
- ``elo_diff``      pre-match Elo rating differential (home − away), replayed from
                    matches strictly before this kickoff (``elo.core.replay``).
                    Venue effects are deliberately NOT baked in here — they are
                    separate features below.
- ``ppg_diff_5``    points/game over each team's last ≤5 finished matches before
                    this kickoff, home − away.
- ``ppg_diff_10``   same, window ≤10.
- ``gd_pg_diff_5``  goal difference/game over last ≤5 finished matches, home − away.
- ``rest_diff``     days since each team's previous match before this kickoff
                    (clipped to ``REST_CLIP_DAYS``; fully rested if none), home − away.
- ``is_true_home``  1.0 iff the data says the home side truly plays at home
                    (``Match.neutral is False``) — static match metadata.
- ``host_diff``     host-nation flag (home − away) per ``HOST_NATIONS_BY_SEASON``,
                    applied when venue is unknown (``neutral is None``, WC corpus)
                    — static match metadata.

Friendlies feed the rolling state but are never training targets: their outcome
process (rotation, low stakes) differs systematically from competitive matches,
and mixing them in would trade bias for very little variance (hard rule 6).
Rows are also skipped while either team has fewer than ``MIN_PRIOR_MATCHES``
prior matches, so windows never emit garbage defaults as training signal.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from services.ingest import Match, MatchStatus
from services.models.elo.core import RATING_INIT, replay
from services.models.features.corpus import Tier, combine_corpora, importance_tier
from services.models.features.team_names import HOST_NATIONS_BY_SEASON, normalize_team

FloatArray = NDArray[np.float64]

FEATURE_NAMES = (
    "elo_diff",
    "ppg_diff_5",
    "ppg_diff_10",
    "gd_pg_diff_5",
    "rest_diff",
    "is_true_home",
    "host_diff",
)

MIN_PRIOR_MATCHES = 5
REST_CLIP_DAYS = 30.0
_DEFAULT_PPG = 1.0  # neutral prior for a team with no window yet (draw-rate points)


class TeamState:
    """Rolling per-team state, keyed by normalized team name."""

    def __init__(self) -> None:
        self.ratings: dict[str, float] = {}
        self._recent: dict[str, deque[tuple[float, int]]] = {}  # (points, goal_diff)
        self._last_kickoff: dict[str, datetime] = {}
        self._n_prior: dict[str, int] = {}

    def n_prior(self, norm_name: str) -> int:
        return self._n_prior.get(norm_name, 0)

    def update(self, match: Match) -> None:
        """Record a finished match (Elo ratings are updated by ``replay``)."""
        assert match.home_goals is not None and match.away_goals is not None
        diff = match.home_goals - match.away_goals
        home_points = 3.0 if diff > 0 else 1.0 if diff == 0 else 0.0
        away_points = 3.0 if diff < 0 else 1.0 if diff == 0 else 0.0
        for name, points, gd in (
            (normalize_team(match.home), home_points, diff),
            (normalize_team(match.away), away_points, -diff),
        ):
            self._recent.setdefault(name, deque(maxlen=10)).append((points, gd))
            self._last_kickoff[name] = match.utc_kickoff
            self._n_prior[name] = self._n_prior.get(name, 0) + 1

    def _windows(self, name: str, kickoff: datetime) -> tuple[float, float, float, float]:
        recent = self._recent.get(name)
        if not recent:
            return _DEFAULT_PPG, _DEFAULT_PPG, 0.0, REST_CLIP_DAYS
        last5 = list(recent)[-5:]
        ppg5 = sum(p for p, _ in last5) / len(last5)
        ppg10 = sum(p for p, _ in recent) / len(recent)
        gd5 = sum(g for _, g in last5) / len(last5)
        rest = (kickoff - self._last_kickoff[name]).total_seconds() / 86400.0
        return ppg5, ppg10, gd5, min(max(rest, 0.0), REST_CLIP_DAYS)


def featurize(fixture: Match, state: TeamState) -> FloatArray:
    home = normalize_team(fixture.home)
    away = normalize_team(fixture.away)
    elo_diff = state.ratings.get(home, RATING_INIT) - state.ratings.get(away, RATING_INIT)
    h_ppg5, h_ppg10, h_gd5, h_rest = state._windows(home, fixture.utc_kickoff)
    a_ppg5, a_ppg10, a_gd5, a_rest = state._windows(away, fixture.utc_kickoff)
    if fixture.neutral is None:
        hosts = HOST_NATIONS_BY_SEASON.get(fixture.season, frozenset())
        host_diff = float(home in hosts) - float(away in hosts)
    else:
        host_diff = 0.0
    return np.array(
        [
            elo_diff,
            h_ppg5 - a_ppg5,
            h_ppg10 - a_ppg10,
            h_gd5 - a_gd5,
            h_rest - a_rest,
            1.0 if fixture.neutral is False else 0.0,
            host_diff,
        ],
        dtype=np.float64,
    )


def _training_stream(matches: Sequence[Match], cutoff: datetime) -> list[Match]:
    finished = [
        m
        for m in matches
        if m.status is MatchStatus.FINISHED
        and m.utc_kickoff < cutoff  # strict: hard rule 2
        and m.home_goals is not None
        and m.away_goals is not None
    ]
    # dedupe + chronological sort + name-keyed ids; idempotent on combined input
    return combine_corpora(finished, [])


def build_training(
    matches: Sequence[Match], cutoff: datetime
) -> tuple[FloatArray, NDArray[np.intp], list[Match]]:
    """Feature matrix X, labels y (0 home / 1 draw / 2 away), and the kept matches."""
    train = _training_stream(matches, cutoff)
    state = TeamState()
    rows: list[FloatArray] = []
    labels: list[int] = []
    kept: list[Match] = []
    for step in replay(train, state.ratings):  # replay owns the Elo updates
        match = step.match
        assert match.home_goals is not None and match.away_goals is not None
        emit = (
            importance_tier(match.stage) is not Tier.FRIENDLY
            and state.n_prior(normalize_team(match.home)) >= MIN_PRIOR_MATCHES
            and state.n_prior(normalize_team(match.away)) >= MIN_PRIOR_MATCHES
        )
        if emit:  # row first, then update: the row never sees its own match
            rows.append(featurize(match, state))
            diff = match.home_goals - match.away_goals
            labels.append(0 if diff > 0 else 2 if diff < 0 else 1)
            kept.append(match)
        state.update(match)
    x = np.vstack(rows) if rows else np.empty((0, len(FEATURE_NAMES)))
    return x, np.array(labels, dtype=np.intp), kept


def state_at(matches: Sequence[Match], cutoff: datetime) -> TeamState:
    """Rolling state as of ``cutoff`` — for featurizing prediction-time fixtures."""
    state = TeamState()
    for step in replay(_training_stream(matches, cutoff), state.ratings):
        state.update(step.match)
    return state
