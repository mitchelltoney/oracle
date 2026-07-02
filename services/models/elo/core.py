"""Pure Elo arithmetic and the chronological replay pass.

Kept free of model-class state so the GBM feature builder can consume the same
replay iterator (one O(n) pass yields every match's pre-match ratings — exactly
the as-of-kickoff value, hard rule 2).

Judgment-call constants (hard rule 6 — deliberately shrunk, never fitted):
- ``HOME_ADV_ELO`` applies only when the data says the home side truly played at
  home (``neutral is False``, historical corpus).
- ``HOST_BONUS_ELO`` applies to a World Cup host playing in its own tournament
  (``neutral is None``, provider corpus) and is smaller than real home advantage:
  a host term fit on a handful of host matches would be noise.
- K-factors come from ``features.corpus.Tier`` (World Cup > qualifiers > friendlies).
- The margin-of-victory multiplier is the eloratings.net standard.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from services.ingest import Match
from services.models.features import (
    HOST_NATIONS_BY_SEASON,
    importance_tier,
    normalize_team,
)

RATING_INIT = 1500.0
HOME_ADV_ELO = 80.0
HOST_BONUS_ELO = 60.0


def expected(home_eff: float, away_eff: float) -> float:
    """Expected home score (win=1, draw=0.5) from effective ratings."""
    return float(1.0 / (1.0 + 10.0 ** (-(home_eff - away_eff) / 400.0)))


def mov_multiplier(goal_diff: int) -> float:
    d = abs(goal_diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0


def k_factor(stage: str) -> float:
    return float(importance_tier(stage))


def rating_adjustments(
    home_norm: str, away_norm: str, season: int, neutral: bool | None
) -> tuple[float, float]:
    """Pre-match rating adjustments (home, away) from venue metadata."""
    home_adj = 0.0
    away_adj = 0.0
    if neutral is False:
        home_adj += HOME_ADV_ELO
    elif neutral is None:  # provider corpus: venue unknown, host nations flagged
        hosts = HOST_NATIONS_BY_SEASON.get(season, frozenset())
        if home_norm in hosts:
            home_adj += HOST_BONUS_ELO
        if away_norm in hosts:
            away_adj += HOST_BONUS_ELO
    return home_adj, away_adj


def davidson_probs(
    home_eff: float, away_eff: float, nu: float
) -> tuple[float, float, float]:
    """Davidson (1970) home/draw/away probabilities from effective ratings."""
    pi_home = 10.0 ** (home_eff / 400.0)
    pi_away = 10.0 ** (away_eff / 400.0)
    tie = nu * math.sqrt(pi_home * pi_away)
    denom = pi_home + pi_away + tie
    return pi_home / denom, tie / denom, pi_away / denom


@dataclass(frozen=True)
class ReplayStep:
    """Pre-match state for one training match (as-of strictly before kickoff)."""

    match: Match
    home_eff: float  # rating incl. venue adjustments, before this match's update
    away_eff: float


def replay(
    matches: Iterable[Match], ratings: dict[str, float]
) -> Iterator[ReplayStep]:
    """Chronological rating replay; mutates ``ratings`` (keyed by normalized name).

    ``matches`` must already be finished-only, deduped, and sorted by kickoff.
    Each step yields the PRE-match effective ratings, then applies the update —
    so consumers see exactly the information available before that kickoff.
    """
    for match in matches:
        assert match.home_goals is not None and match.away_goals is not None
        home = normalize_team(match.home)
        away = normalize_team(match.away)
        r_home = ratings.get(home, RATING_INIT)
        r_away = ratings.get(away, RATING_INIT)
        adj_home, adj_away = rating_adjustments(home, away, match.season, match.neutral)
        home_eff = r_home + adj_home
        away_eff = r_away + adj_away
        yield ReplayStep(match=match, home_eff=home_eff, away_eff=away_eff)

        goal_diff = match.home_goals - match.away_goals
        result = 1.0 if goal_diff > 0 else 0.0 if goal_diff < 0 else 0.5
        delta = (
            k_factor(match.stage)
            * mov_multiplier(goal_diff)
            * (result - expected(home_eff, away_eff))
        )
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
