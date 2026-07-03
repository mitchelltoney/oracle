"""Elo-style team rating model behind ``MatchModel``.

Ratings replay historical internationals chronologically (K by importance tier,
margin-of-victory multiplier, home/host advantage — see ``core``). The rating
differential maps to home/draw/away via a Davidson model whose single draw
parameter ``nu`` is fit by MLE on the training matches (modern era only:
draw rates before ``NU_FIT_START_YEAR`` reflect a different game).

Ratings used to predict match M reflect only matches completed before M's
kickoff: ``fit`` filters strictly below the cutoff and ``predict`` refuses
fixtures at/before it (hard rule 2). Changing any constant in ``core`` or here
requires a new model version string (hard rule 1).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from services.ingest import Match, MatchStatus
from services.models.base import LeakageError, Prediction
from services.models.elo.core import (
    RATING_INIT,
    davidson_probs,
    rating_adjustments,
    replay,
)
from services.models.features import combine_corpora, normalize_team

logger = logging.getLogger(__name__)

NU_FIT_START_YEAR = 1990
_NU_BOUNDS = (0.01, 5.0)
# Weak log-space prior shrinking nu toward a ~25% draw rate between equal sides
# (nu/(2+nu) = 0.25): worth ~a handful of matches, so it only matters when the
# training sample is tiny (hard rule 6).
NU_PRIOR = 2.0 / 3.0
NU_PRIOR_WEIGHT = 5.0
_EPS = 1e-15

FloatArray = NDArray[np.float64]


class EloModel:
    name = "elo"
    version = "elo-1.0.1"  # 1.0.1: cross-corpus date-boundary dedup fix changed the training corpus

    def __init__(self) -> None:
        self._cutoff: datetime | None = None
        self._ratings: dict[str, float] = {}
        self._nu: float = 1.0

    @property
    def ratings(self) -> dict[str, float]:
        """Fitted ratings keyed by normalized team name (copy; read-only)."""
        return dict(self._ratings)

    def fit(self, matches: Sequence[Match], cutoff: datetime) -> None:
        finished = [
            m
            for m in matches
            if m.status is MatchStatus.FINISHED
            and m.utc_kickoff < cutoff  # strict: hard rule 2
            and m.home_goals is not None
            and m.away_goals is not None
        ]
        if not finished:
            raise ValueError("no finished matches before cutoff to fit on")
        # combine_corpora dedupes, sorts chronologically, and rekeys by name;
        # idempotent when the caller already passed a combined corpus.
        train = combine_corpora(finished, [])

        ratings: dict[str, float] = {}
        home_effs: list[float] = []
        away_effs: list[float] = []
        outcomes: list[int] = []  # 0 home, 1 draw, 2 away
        seasons: list[int] = []
        for step in replay(train, ratings):
            assert step.match.home_goals is not None
            assert step.match.away_goals is not None
            diff = step.match.home_goals - step.match.away_goals
            home_effs.append(step.home_eff)
            away_effs.append(step.away_eff)
            outcomes.append(0 if diff > 0 else 2 if diff < 0 else 1)
            seasons.append(step.match.season)

        self._ratings = ratings
        self._nu = self._fit_nu(
            np.array(home_effs),
            np.array(away_effs),
            np.array(outcomes, dtype=np.intp),
            np.array(seasons, dtype=np.intp),
        )
        self._cutoff = cutoff
        logger.info(
            "fitted Elo on %d matches (%d teams), nu=%.3f, cutoff %s",
            len(train),
            len(ratings),
            self._nu,
            cutoff.isoformat(),
        )

    def predict(self, fixture: Match) -> Prediction:
        if self._cutoff is None:
            raise RuntimeError("model is not fitted")
        if fixture.utc_kickoff <= self._cutoff:
            raise LeakageError(
                f"fixture {fixture.id} kicks off at {fixture.utc_kickoff.isoformat()}, "
                f"not after the fitted cutoff {self._cutoff.isoformat()}"
            )
        home = normalize_team(fixture.home)
        away = normalize_team(fixture.away)
        adj_home, adj_away = rating_adjustments(
            home, away, fixture.season, fixture.neutral
        )
        home_eff = self._team_rating(home, fixture.home) + adj_home
        away_eff = self._team_rating(away, fixture.away) + adj_away
        p_home, p_draw, p_away = davidson_probs(home_eff, away_eff, self._nu)
        return Prediction(
            fixture_id=fixture.id,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            top_scorelines=[],  # Elo has no scoreline distribution
            model=self.name,
            model_version=self.version,
        )

    def _team_rating(self, norm_name: str, display_name: str) -> float:
        rating = self._ratings.get(norm_name)
        if rating is None:
            logger.warning(
                "team %s unseen in training data; using initial rating", display_name
            )
            return RATING_INIT
        return rating

    @staticmethod
    def _fit_nu(
        home_effs: FloatArray,
        away_effs: FloatArray,
        outcomes: NDArray[np.intp],
        seasons: NDArray[np.intp],
    ) -> float:
        modern = seasons >= NU_FIT_START_YEAR
        if modern.any():  # pre-modern-era draw rates would bias nu
            home_effs, away_effs, outcomes = (
                home_effs[modern],
                away_effs[modern],
                outcomes[modern],
            )
        pi_home = 10.0 ** (home_effs / 400.0)
        pi_away = 10.0 ** (away_effs / 400.0)
        tie_base = np.sqrt(pi_home * pi_away)
        rows = np.arange(len(outcomes))

        def nll(nu: float) -> float:
            tie = nu * tie_base
            denom = pi_home + pi_away + tie
            probs = np.stack([pi_home, tie, pi_away], axis=1) / denom[:, None]
            penalty = NU_PRIOR_WEIGHT * (np.log(nu) - np.log(NU_PRIOR)) ** 2
            return float(
                -np.log(np.clip(probs[rows, outcomes], _EPS, None)).sum() + penalty
            )

        result = minimize_scalar(nll, bounds=_NU_BOUNDS, method="bounded")
        return float(result.x)
