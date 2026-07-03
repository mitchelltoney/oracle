"""Match-importance tiers and cross-corpus combination.

``importance_tier`` reads only static match metadata (the ``stage`` string set at
ingest time) — no time dimension, so it is leakage-free by construction.

``combine_corpora`` merges the historical corpus with the WC provider corpus into
one chronological stream with name-keyed team ids. Where both corpora carry the
same match (the CSV also contains World Cup finals matches), the provider row
wins: it has a real kickoff instant and the real fixture id.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from datetime import date
from enum import IntEnum

from services.ingest import Match

from .team_names import normalize_team, with_name_ids

# football-data.org stage tokens — all World Cup finals matches
_FD_STAGES = frozenset(
    {
        "group_stage",
        "last_32",
        "last_16",
        "round_of_32",
        "round_of_16",
        "quarter_finals",
        "semi_finals",
        "third_place",
        "final",
    }
)

_CONTINENTAL_KEYS = (
    "uefa euro",
    "copa america",
    "african cup of nations",
    "africa cup of nations",
    "afc asian cup",
    "gold cup",
    "confederations cup",
    "oceania nations cup",
)


class Tier(IntEnum):
    """Match importance; the value doubles as the Elo K-factor."""

    FRIENDLY = 20
    MINOR = 30
    QUALIFIER = 40
    CONTINENTAL = 50
    WC_FINALS = 60


def importance_tier(stage: str) -> Tier:
    folded = (
        unicodedata.normalize("NFKD", stage).encode("ascii", "ignore").decode("ascii")
    )
    s = " ".join(folded.lower().split())
    if not s:
        return Tier.MINOR
    if s in _FD_STAGES:
        return Tier.WC_FINALS
    if "friendly" in s:
        return Tier.FRIENDLY
    if "qualification" in s or "qualifier" in s or "nations league" in s:
        return Tier.QUALIFIER
    if s == "fifa world cup":
        return Tier.WC_FINALS
    if any(key in s for key in _CONTINENTAL_KEYS):
        return Tier.CONTINENTAL
    return Tier.MINOR


def _pair_key(match: Match) -> tuple[date, frozenset[str]]:
    return (
        match.utc_kickoff.date(),
        frozenset({normalize_team(match.home), normalize_team(match.away)}),
    )


def combine_corpora(history: Sequence[Match], wc: Sequence[Match]) -> list[Match]:
    """One deduped chronological corpus with name-keyed team ids.

    The provider is authoritative for every World Cup finals season it covers,
    so history rows for those (season, finals-tier) matches are dropped outright.
    The date-keyed dedup below cannot catch them: history rows sit at noon UTC
    of the LOCAL match date, while an Americas evening kickoff lands on the
    next UTC day in the provider feed — the same real match would survive twice
    under two timestamps, leaking results across walk-forward cutoffs.
    """
    covered = {m.season for m in wc if importance_tier(m.stage) is Tier.WC_FINALS}
    combined: dict[tuple[date, frozenset[str]], Match] = {}
    for match in history:
        if match.season in covered and importance_tier(match.stage) is Tier.WC_FINALS:
            continue
        combined.setdefault(_pair_key(match), with_name_ids(match))
    for match in wc:  # provider rows override same-day history duplicates
        combined[_pair_key(match)] = with_name_ids(match)
    return sorted(combined.values(), key=lambda m: (m.utc_kickoff, m.id))
