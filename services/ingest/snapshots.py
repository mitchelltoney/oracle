"""Versioned, as-of-timestamped snapshots — the only sanctioned source for model features.

``write_snapshot`` is the ONLY writer of ``data/snapshots/`` (hard rule 5). Snapshot files
are never overwritten; each carries its as-of instant in both filename and payload.
``Snapshot.finished_before`` is the single leakage gate (hard rule 2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from services.ingest.provider import Match, MatchStatus

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Snapshot:
    as_of_utc: datetime
    source: str
    matches: list[Match]
    standings: dict[str, Any]
    schema_version: int = SCHEMA_VERSION

    def finished_before(self, cutoff: datetime) -> list[Match]:
        """Finished matches that kicked off strictly before ``cutoff`` (hard rule 2)."""
        return [
            m
            for m in self.matches
            if m.status is MatchStatus.FINISHED and m.utc_kickoff < cutoff
        ]

    def upcoming(self, now: datetime) -> list[Match]:
        return sorted(
            (
                m
                for m in self.matches
                if m.status in (MatchStatus.SCHEDULED, MatchStatus.TIMED)
                and m.utc_kickoff > now
            ),
            key=lambda m: m.utc_kickoff,
        )


def _match_to_dict(match: Match) -> dict[str, Any]:
    return {
        "id": match.id,
        "season": match.season,
        "utc_kickoff": match.utc_kickoff.isoformat(),
        "stage": match.stage,
        "status": match.status.value,
        "home_id": match.home_id,
        "home": match.home,
        "away_id": match.away_id,
        "away": match.away,
        "home_goals": match.home_goals,
        "away_goals": match.away_goals,
        "duration": match.duration,
        "winner": match.winner,
    }


def _match_from_dict(data: dict[str, Any]) -> Match:
    return Match(
        id=data["id"],
        season=data["season"],
        utc_kickoff=datetime.fromisoformat(data["utc_kickoff"]),
        stage=data["stage"],
        status=MatchStatus(data["status"]),
        home_id=data["home_id"],
        home=data["home"],
        away_id=data["away_id"],
        away=data["away"],
        home_goals=data["home_goals"],
        away_goals=data["away_goals"],
        duration=data["duration"],
        winner=data["winner"],
    )


def write_snapshot(snapshot: Snapshot, snapshots_dir: Path) -> Path:
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    stem = f"snapshot_{snapshot.as_of_utc:%Y%m%dT%H%M%S}Z"
    path = snapshots_dir / f"{stem}.json"
    suffix = 0
    while path.exists():  # never overwrite an existing snapshot
        suffix += 1
        path = snapshots_dir / f"{stem}_{suffix}.json"
    payload = {
        "schema_version": snapshot.schema_version,
        "as_of_utc": snapshot.as_of_utc.isoformat(),
        "source": snapshot.source,
        "matches": [_match_to_dict(m) for m in snapshot.matches],
        "standings": snapshot.standings,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_snapshot(path: Path) -> Snapshot:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Snapshot(
        schema_version=payload["schema_version"],
        as_of_utc=datetime.fromisoformat(payload["as_of_utc"]),
        source=payload["source"],
        matches=[_match_from_dict(m) for m in payload["matches"]],
        standings=payload["standings"],
    )


def load_latest_snapshot(snapshots_dir: Path) -> Snapshot:
    candidates = sorted(snapshots_dir.glob("snapshot_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no snapshots in {snapshots_dir} — run `make ingest` first"
        )
    return load_snapshot(candidates[-1])
