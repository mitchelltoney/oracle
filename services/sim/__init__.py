"""Monte Carlo bracket simulator — takes any ``MatchModel``."""

from services.sim.bracket import KNOCKOUT_ROUNDS, BracketState, Team, build_bracket
from services.sim.engine import (
    SimResult,
    advance_home_prob,
    penalty_home_prob,
    simulate,
)

__all__ = [
    "KNOCKOUT_ROUNDS",
    "BracketState",
    "SimResult",
    "Team",
    "advance_home_prob",
    "build_bracket",
    "penalty_home_prob",
    "simulate",
]
