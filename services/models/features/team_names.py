"""Cross-corpus team identity: the WC provider keys teams by numeric id, the
historical corpus by name. Models that must see one team as one entity across
both corpora rekey matches with ``with_name_ids`` (fixture ``id`` is never
touched — it is the calibration join key in the prediction log).

``normalize_team`` is deliberately conservative: ascii-fold, lowercase, collapse
whitespace, then a hand-maintained alias table for known cross-corpus variants.
Unmatched names simply stay distinct — models fall back to average/initial
parameters with a warning, and the backtest prints an unmatched-name diagnostic
so ``ALIASES`` can be corrected from real data rather than guessed.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import replace

from services.ingest import Match

# variant -> canonical, both in ascii-folded lowercase form
ALIASES: dict[str, str] = {
    "usa": "united states",
    "korea republic": "south korea",
    "korea dpr": "north korea",
    "ir iran": "iran",
    "cote d'ivoire": "ivory coast",
    "cabo verde": "cape verde",
    "cape verde islands": "cape verde",  # football-data.org 2026 feed variant
    "czechia": "czech republic",
    "turkiye": "turkey",
    "congo dr": "dr congo",
    "china pr": "china",
    "bosnia-herzegovina": "bosnia and herzegovina",
    "trinidad & tobago": "trinidad and tobago",
    "republic of ireland": "ireland",
}

# normalized names of tournament hosts (host advantage applies to WC-corpus
# fixtures, where ``neutral`` is None; historical rows carry an explicit flag)
HOST_NATIONS_BY_SEASON: dict[int, frozenset[str]] = {
    2018: frozenset({"russia"}),
    2022: frozenset({"qatar"}),
    2026: frozenset({"united states", "mexico", "canada"}),
}


def normalize_team(name: str) -> str:
    folded = (
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    )
    key = " ".join(folded.lower().split())
    return ALIASES.get(key, key)


def team_name_id(name: str) -> int:
    """Deterministic 63-bit id of the normalized team name."""
    digest = hashlib.sha256(normalize_team(name).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def with_name_ids(match: Match) -> Match:
    """Rekey team ids by normalized name; the fixture ``id`` is preserved."""
    return replace(
        match,
        home_id=team_name_id(match.home),
        away_id=team_name_id(match.away),
    )
