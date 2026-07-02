"""Vectorized Monte Carlo over the remaining bracket.

The sim dimension is pure numpy: per round, per bracket slot, the (home, away)
participant arrays are grouped into their distinct pairings (``np.unique``), so
the model is asked for each pairing at most once per round — a few hundred
``predict`` calls for 100k simulated tournaments, everything else array math.

Knockout resolution collapses draw → extra time → penalties analytically into a
single advance probability (identical distribution to sampling each stage; only
advancement matters for round-reach output):
- extra time decides a share ``P_ET_DECIDED`` of level matches, winner in
  proportion to the model's win probabilities;
- the rest go to penalties, modeled as a NEAR-COIN-FLIP: a small edge
  proportional to the win-probability gap, hard-capped at ``PENS_SKILL_CAP``
  (CLAUDE.md: never strongly skill-determined).

Model probabilities are scoreline-faithful where the model provides scorelines:
Dixon-Coles's 1X2 probabilities are exact marginals of its scoreline grid, so
sampling outcomes from them equals sampling scorelines and mapping to outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from services.ingest import Match, MatchStatus
from services.models.base import MatchModel
from services.sim.bracket import BracketState, Team

P_ET_DECIDED = 0.45  # share of level knockout matches decided in extra time
PENS_SKILL_COEF = 0.10
PENS_SKILL_CAP = 0.05  # penalties stay within [0.45, 0.55] no matter the gap

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]


def penalty_home_prob(p_home: float, p_away: float) -> float:
    """Near-coin-flip shootout: small, bounded edge from the win-probability gap."""
    edge = PENS_SKILL_COEF * (p_home - p_away)
    return 0.5 + min(max(edge, -PENS_SKILL_CAP), PENS_SKILL_CAP)


def advance_home_prob(p_home: float, p_draw: float, p_away: float) -> float:
    """P(home advances) after 90', extra time, and penalties."""
    decisive = p_home + p_away
    et_home = p_home / decisive if decisive > 0.0 else 0.5
    pens_home = penalty_home_prob(p_home, p_away)
    return p_home + p_draw * (P_ET_DECIDED * et_home + (1.0 - P_ET_DECIDED) * pens_home)


@dataclass(frozen=True)
class SimResult:
    n_sims: int
    rounds: tuple[str, ...]
    teams: tuple[Team, ...]
    reach: FloatArray  # (len(rounds), len(teams)): P(team participates in round)
    win: FloatArray  # (len(teams),): P(team wins the tournament)
    pairing_source: str


def simulate(
    bracket: BracketState,
    model: MatchModel,
    *,
    n_sims: int = 100_000,
    rng: np.random.Generator | None = None,
) -> SimResult:
    rng = rng if rng is not None else np.random.default_rng(26)
    n_teams = len(bracket.teams)
    reach = np.zeros((len(bracket.rounds), n_teams))
    reach[0, :] = 1.0  # every entrant plays the entry round

    prob_cache: dict[tuple[int, int, str], float] = {}

    def resolve(home: IntArray, away: IntArray, round_name: str) -> IntArray:
        codes = home * n_teams + away
        uniq, inverse = np.unique(codes, return_inverse=True)
        out = np.empty_like(home)
        draws = rng.random(len(codes))
        for u_idx, code in enumerate(uniq):
            h, a = divmod(int(code), n_teams)
            mask = inverse == u_idx
            forced = bracket.known_winners.get(round_name, {}).get(frozenset((h, a)))
            if forced is not None:
                out[mask] = forced
                continue
            p = _advance_prob(h, a, round_name)
            out[mask] = np.where(draws[mask] < p, h, a)
        return out

    def _advance_prob(h: int, a: int, round_name: str) -> float:
        key = (h, a, round_name)
        cached = prob_cache.get(key)
        if cached is not None:
            return cached
        fixture = bracket.fixtures.get(round_name, {}).get(frozenset((h, a)))
        if fixture is None:
            fixture = _synthetic_fixture(
                bracket.teams[h],
                bracket.teams[a],
                round_name,
                bracket.round_kickoffs[round_name],
            )
        pred = model.predict(fixture)
        p_home_advances = advance_home_prob(pred.p_home, pred.p_draw, pred.p_away)
        p = (
            p_home_advances
            if fixture.home_id == bracket.teams[h].id
            else 1.0 - p_home_advances
        )
        prob_cache[key] = p
        return p

    slot_winners = np.empty((len(bracket.entry_pairs), n_sims), dtype=np.int32)
    for slot, (h, a) in enumerate(bracket.entry_pairs):
        slot_winners[slot] = resolve(
            np.full(n_sims, h, dtype=np.int32),
            np.full(n_sims, a, dtype=np.int32),
            bracket.rounds[0],
        )

    for round_idx, round_name in enumerate(bracket.rounds[1:], start=1):
        # a team appears at most once per sim across slots (disjoint sub-brackets),
        # so its count over all slots is the number of sims it reached this round
        participants = slot_winners.reshape(-1)
        reach[round_idx] = np.bincount(participants, minlength=n_teams) / float(n_sims)
        pairs = bracket.pairings[round_name]
        next_winners = np.empty((len(pairs), n_sims), dtype=np.int32)
        for slot, (i, j) in enumerate(pairs):
            next_winners[slot] = resolve(slot_winners[i], slot_winners[j], round_name)
        slot_winners = next_winners

    win = np.bincount(slot_winners[0], minlength=n_teams) / float(n_sims)
    return SimResult(
        n_sims=n_sims,
        rounds=bracket.rounds,
        teams=bracket.teams,
        reach=reach,
        win=win,
        pairing_source=bracket.pairing_source,
    )


def _synthetic_fixture(
    home: Team, away: Team, round_name: str, kickoff: datetime
) -> Match:
    return Match(
        id=-(abs(hash((home.id, away.id, round_name))) % 10**9 + 1),
        season=2026,
        utc_kickoff=kickoff,
        stage=round_name,
        status=MatchStatus.TIMED,
        home_id=home.id,
        home=home.name,
        away_id=away.id,
        away=away.name,
        home_goals=None,
        away_goals=None,
        duration="REGULAR",
        winner=None,
        neutral=None,  # venue unknown: host bonus applies via team identity
    )
