"""Dixon-Coles (1997) match model, implemented from scratch.

Goal rates on neutral ground (no home-advantage parameter in P0: World Cup 2026
venue/host attribution is unreliable in the provider payload, and a host-advantage
term fit on a handful of USA/MEX/CAN home matches would be noise — hard rule 6):

    lambda = exp(c + attack_home - defense_away)
    mu     = exp(c + attack_away - defense_home)

with the Dixon-Coles low-score correction tau(x, y; lambda, mu, rho), exponential
time-decay weights w_i = exp(-xi * days_before_cutoff), and an L2 penalty
kappa * (|attack|^2 + |defense|^2) that shrinks every team toward the average team
(and pins down the additive identifiability invariance).

Hyperparameters xi (default: one-year half-life) and kappa (default 1.0) are fixed,
never fitted, and provisional until the P1 backtest exists; changing them requires a
new model version string (hard rule 1).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln

from services.ingest import Match, MatchStatus
from services.models.base import LeakageError, Prediction

logger = logging.getLogger(__name__)

MAX_GOALS = 10  # scoreline grid is (0..MAX_GOALS) x (0..MAX_GOALS)
_EPS = 1e-10

FloatArray = NDArray[np.float64]


class DixonColesModel:
    name = "dixon_coles"
    version = "dc-1.0.0"

    def __init__(
        self,
        *,
        xi: float = math.log(2) / 365.0,
        kappa: float = 1.0,
    ) -> None:
        self._xi = xi
        self._kappa = kappa
        self._cutoff: datetime | None = None
        self._team_index: dict[int, int] = {}
        self._attack: FloatArray = np.zeros(0)
        self._defense: FloatArray = np.zeros(0)
        self._intercept: float = 0.0
        self._rho: float = 0.0

    def fit(self, matches: Sequence[Match], cutoff: datetime) -> None:
        train = [
            m
            for m in matches
            if m.status is MatchStatus.FINISHED
            and m.utc_kickoff < cutoff  # strict: hard rule 2
            and m.home_goals is not None
            and m.away_goals is not None
        ]
        if not train:
            raise ValueError("no finished matches before cutoff to fit on")

        team_ids = sorted({t for m in train for t in (m.home_id, m.away_id)})
        self._team_index = {tid: i for i, tid in enumerate(team_ids)}
        n_teams = len(team_ids)

        hg = np.array([m.home_goals for m in train], dtype=np.float64)
        ag = np.array([m.away_goals for m in train], dtype=np.float64)
        hi = np.array([self._team_index[m.home_id] for m in train], dtype=np.intp)
        ai = np.array([self._team_index[m.away_id] for m in train], dtype=np.intp)
        days = np.array(
            [(cutoff - m.utc_kickoff).total_seconds() / 86400.0 for m in train],
            dtype=np.float64,
        )
        weights = np.exp(-self._xi * days)

        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)

        def penalized_nll(theta: FloatArray) -> float:
            intercept = theta[0]
            rho = theta[1]
            attack = theta[2 : 2 + n_teams]
            defense = theta[2 + n_teams :]
            lam = np.exp(intercept + attack[hi] - defense[ai])
            mu = np.exp(intercept + attack[ai] - defense[hi])
            tau = np.ones_like(lam)
            tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
            tau[m01] = 1.0 + lam[m01] * rho
            tau[m10] = 1.0 + mu[m10] * rho
            tau[m11] = 1.0 - rho
            tau = np.clip(tau, _EPS, None)
            log_lik = weights * (
                np.log(tau) + hg * np.log(lam) - lam + ag * np.log(mu) - mu
            )
            penalty = self._kappa * (attack @ attack + defense @ defense)
            return float(-log_lik.sum() + penalty)

        x0 = np.zeros(2 + 2 * n_teams)
        x0[0] = math.log(max(float((hg.sum() + ag.sum()) / (2 * len(train))), 0.1))
        bounds: list[tuple[float | None, float | None]] = [(None, None), (-0.2, 0.2)]
        bounds += [(None, None)] * (2 * n_teams)
        result = minimize(penalized_nll, x0, method="L-BFGS-B", bounds=bounds)
        if not result.success:
            logger.warning("Dixon-Coles fit did not fully converge: %s", result.message)

        theta: FloatArray = result.x
        self._intercept = float(theta[0])
        self._rho = float(theta[1])
        self._attack = theta[2 : 2 + n_teams].copy()
        self._defense = theta[2 + n_teams :].copy()
        self._cutoff = cutoff
        logger.info(
            "fitted Dixon-Coles on %d matches (%d teams), cutoff %s",
            len(train),
            n_teams,
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
        lam = math.exp(
            self._intercept
            + self._team_param(self._attack, fixture.home_id, fixture.home)
            - self._team_param(self._defense, fixture.away_id, fixture.away)
        )
        mu = math.exp(
            self._intercept
            + self._team_param(self._attack, fixture.away_id, fixture.away)
            - self._team_param(self._defense, fixture.home_id, fixture.home)
        )
        grid = self._score_grid(lam, mu)
        p_home = float(np.tril(grid, -1).sum())
        p_draw = float(np.trace(grid))
        p_away = float(np.triu(grid, 1).sum())
        flat_order = np.argsort(grid, axis=None)[::-1][:5]
        top_scorelines = [
            (int(idx // grid.shape[1]), int(idx % grid.shape[1]), float(grid.flat[idx]))
            for idx in flat_order
        ]
        return Prediction(
            fixture_id=fixture.id,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            top_scorelines=top_scorelines,
            model=self.name,
            model_version=self.version,
        )

    def _team_param(self, params: FloatArray, team_id: int, team_name: str) -> float:
        index = self._team_index.get(team_id)
        if index is None:
            logger.warning(
                "team %s (%d) unseen in training data; using average-team parameters",
                team_name,
                team_id,
            )
            return 0.0
        return float(params[index])

    def _score_grid(self, lam: float, mu: float) -> FloatArray:
        goals = np.arange(MAX_GOALS + 1, dtype=np.float64)
        log_ph = -lam + goals * math.log(lam) - gammaln(goals + 1)
        log_pa = -mu + goals * math.log(mu) - gammaln(goals + 1)
        grid: FloatArray = np.outer(np.exp(log_ph), np.exp(log_pa))
        rho = self._rho
        grid[0, 0] *= max(1.0 - lam * mu * rho, _EPS)
        grid[0, 1] *= max(1.0 + lam * rho, _EPS)
        grid[1, 0] *= max(1.0 + mu * rho, _EPS)
        grid[1, 1] *= max(1.0 - rho, _EPS)
        grid /= grid.sum()
        return grid
