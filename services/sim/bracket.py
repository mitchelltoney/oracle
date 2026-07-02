"""Bracket state for the remaining 2026 knockout tournament, built from the
latest snapshot.

The entry round is the earliest ``KNOCKOUT_ROUNDS`` stage present in the
snapshot; its matches (teams always determined by then) define the entrants.
Advancement to later rounds uses a sequential-pairing template — winner of
slot 0 meets winner of slot 1, and so on in schedule order — RECONCILED against
any later-round fixture the snapshot already knows: when a real fixture's two
teams can be located in distinct feeder slots, that edge is pinned and
overrides the template. ``pairing_source`` reports how much of the tree came
from real fixtures ("reconciled") vs the template.

CAVEAT (verify against the first real 2026 snapshot): the exact football-data
stage strings and the sequential-pairing assumption are encoded ONLY here, in
``KNOCKOUT_ROUNDS`` and ``_template_pairs``. THIRD_PLACE is deliberately not
simulated — it does not affect round-reach or title probabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from services.ingest import Match, MatchStatus, Snapshot

logger = logging.getLogger(__name__)

KNOCKOUT_ROUNDS = ("LAST_32", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL")

# a fixture must kick off after the model's cutoff (= now) to be predictable
_KICKOFF_MARGIN = timedelta(minutes=1)
_ROUND_GAP = timedelta(days=4)  # synthetic spacing for rounds with no known fixture

PairKey = frozenset[int]  # the two participants' team indices


@dataclass(frozen=True)
class Team:
    id: int
    name: str


@dataclass(frozen=True)
class BracketState:
    rounds: tuple[str, ...]  # entry round first, FINAL last
    teams: tuple[Team, ...]  # entrants; positions are the engine's team indices
    entry_pairs: tuple[tuple[int, int], ...]  # (home, away) team indices per slot
    # later round -> feeder-slot index pairs, in bracket order
    pairings: dict[str, tuple[tuple[int, int], ...]]
    # round -> participants -> winning team index (already-played matches)
    known_winners: dict[str, dict[PairKey, int]]
    # round -> participants -> the real fixture (kickoff clamped after ``now``)
    fixtures: dict[str, dict[PairKey, Match]]
    round_kickoffs: dict[str, datetime]  # for synthetic fixtures of unknown pairings
    pairing_source: str  # template | reconciled | mixed


def build_bracket(snapshot: Snapshot, now: datetime) -> BracketState:
    by_round: dict[str, list[Match]] = {}
    for match in snapshot.matches:
        if match.stage in KNOCKOUT_ROUNDS:
            by_round.setdefault(match.stage, []).append(match)
    entry_index = next(
        (i for i, name in enumerate(KNOCKOUT_ROUNDS) if by_round.get(name)), None
    )
    if entry_index is None:
        raise ValueError(
            "no knockout matches in the snapshot — the bracket is not determined yet"
        )
    rounds = KNOCKOUT_ROUNDS[entry_index:]

    entry = sorted(by_round[rounds[0]], key=lambda m: (m.utc_kickoff, m.id))
    n_slots = len(entry)
    if n_slots < 1 or n_slots & (n_slots - 1):
        raise ValueError(
            f"{rounds[0]} has {n_slots} matches in the snapshot; "
            "need a full power-of-two round to simulate the bracket"
        )

    teams: list[Team] = []
    team_index: dict[int, int] = {}
    entry_pairs: list[tuple[int, int]] = []
    for match in entry:
        for team_id, name in ((match.home_id, match.home), (match.away_id, match.away)):
            if team_id in team_index:
                raise ValueError(f"team {name} appears twice in {rounds[0]}")
            team_index[team_id] = len(teams)
            teams.append(Team(id=team_id, name=name))
        entry_pairs.append((team_index[match.home_id], team_index[match.away_id]))

    known_winners: dict[str, dict[PairKey, int]] = {}
    fixtures: dict[str, dict[PairKey, Match]] = {}
    round_kickoffs: dict[str, datetime] = {}

    def register(round_name: str, match: Match) -> None:
        home_idx = team_index.get(match.home_id)
        away_idx = team_index.get(match.away_id)
        if home_idx is None or away_idx is None:
            logger.warning(
                "%s fixture %d has a non-entrant team; ignoring", round_name, match.id
            )
            return
        key: PairKey = frozenset((home_idx, away_idx))
        winner_idx = _winner_index(match, home_idx, away_idx)
        if winner_idx is not None:
            known_winners.setdefault(round_name, {})[key] = winner_idx
            return
        clamped = replace(
            match, utc_kickoff=max(match.utc_kickoff, now + _KICKOFF_MARGIN)
        )
        fixtures.setdefault(round_name, {})[key] = clamped

    for match in entry:
        register(rounds[0], match)

    # possible participants per current slot, used to reconcile later-round fixtures
    possible: list[set[int]] = []
    for home_idx, away_idx in entry_pairs:
        forced = known_winners.get(rounds[0], {}).get(frozenset((home_idx, away_idx)))
        possible.append({forced} if forced is not None else {home_idx, away_idx})

    pairings: dict[str, tuple[tuple[int, int], ...]] = {}
    pinned_pairs = 0
    total_pairs = 0
    previous_kickoff = max(m.utc_kickoff for m in entry)
    round_kickoffs[rounds[0]] = max(previous_kickoff, now + _KICKOFF_MARGIN)

    for round_name in rounds[1:]:
        known = sorted(by_round.get(round_name, []), key=lambda m: (m.utc_kickoff, m.id))
        pins: list[tuple[int, int]] = []
        for match in known:
            register(round_name, match)
            edge = _locate_edge(match, team_index, possible)
            if edge is not None:
                pins.append(edge)
        pairs, n_pinned = _merge_pairs(len(possible), pins, round_name)
        pairings[round_name] = pairs
        total_pairs += len(pairs)
        pinned_pairs += n_pinned
        possible = [possible[i] | possible[j] for i, j in pairs]

        kickoffs = [m.utc_kickoff for m in known]
        previous_kickoff = max(kickoffs) if kickoffs else previous_kickoff + _ROUND_GAP
        round_kickoffs[round_name] = max(previous_kickoff, now + _KICKOFF_MARGIN)

    if total_pairs == 0 or pinned_pairs == 0:
        source = "template"
    elif pinned_pairs >= total_pairs:
        source = "reconciled"
    else:
        source = "mixed"

    return BracketState(
        rounds=rounds,
        teams=tuple(teams),
        entry_pairs=tuple(entry_pairs),
        pairings=pairings,
        known_winners=known_winners,
        fixtures=fixtures,
        round_kickoffs=round_kickoffs,
        pairing_source=source,
    )


def _winner_index(match: Match, home_idx: int, away_idx: int) -> int | None:
    if match.status is not MatchStatus.FINISHED:
        return None
    if match.winner == "HOME_TEAM":
        return home_idx
    if match.winner == "AWAY_TEAM":
        return away_idx
    logger.warning(
        "finished knockout match %d has no winner (%r); treating as unplayed",
        match.id,
        match.winner,
    )
    return None


def _locate_edge(
    match: Match, team_index: dict[int, int], possible: list[set[int]]
) -> tuple[int, int] | None:
    """Which two feeder slots does this real fixture connect?"""
    home_idx = team_index.get(match.home_id)
    away_idx = team_index.get(match.away_id)
    if home_idx is None or away_idx is None:
        return None
    home_slot = next((s for s, teams in enumerate(possible) if home_idx in teams), None)
    away_slot = next((s for s, teams in enumerate(possible) if away_idx in teams), None)
    if home_slot is None or away_slot is None or home_slot == away_slot:
        logger.warning(
            "could not reconcile fixture %d into the bracket tree; keeping template",
            match.id,
        )
        return None
    return home_slot, away_slot


def _template_pairs(n_slots: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, i + 1) for i in range(0, n_slots, 2))


def _merge_pairs(
    n_slots: int, pins: list[tuple[int, int]], round_name: str
) -> tuple[tuple[tuple[int, int], ...], int]:
    """Pinned edges plus template pairing for the rest; returns (pairs, n_pinned)."""
    used: set[int] = set()
    for i, j in pins:
        if i in used or j in used:
            logger.warning(
                "conflicting reconciled pairings in %s; falling back to the template",
                round_name,
            )
            return _template_pairs(n_slots), 0
        used.update((i, j))
    remaining = [s for s in range(n_slots) if s not in used]
    pairs = pins + [
        (remaining[k], remaining[k + 1]) for k in range(0, len(remaining), 2)
    ]
    return tuple(sorted(pairs, key=min)), len(pins)
