"""One-shot data refresh: fetch WC matches + standings, write a versioned snapshot.

Entry point for ``make ingest``.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from services.ingest.football_data import FootballDataProvider
from services.ingest.provider import Match, MissingApiKeyError, ProviderError
from services.ingest.snapshots import Snapshot, write_snapshot

SEASONS = (2018, 2022, 2026)
CURRENT_SEASON = 2026


def run(data_dir: Path = Path("data")) -> int:
    load_dotenv()
    try:
        provider = FootballDataProvider.from_env(raw_dir=data_dir / "raw")
    except MissingApiKeyError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    matches: list[Match] = []
    for season in SEASONS:
        try:
            fetched = provider.fetch_matches(season)
        except ProviderError as exc:
            if season == CURRENT_SEASON:
                print(f"FATAL: could not fetch season {season}: {exc}", file=sys.stderr)
                return 1
            print(f"WARNING: skipping season {season}: {exc}", file=sys.stderr)
            continue
        print(f"season {season}: {len(fetched)} matches")
        matches.extend(fetched)

    standings: dict[str, Any] = {}
    try:
        standings = provider.fetch_standings(CURRENT_SEASON)
    except ProviderError as exc:
        print(f"WARNING: standings unavailable: {exc}", file=sys.stderr)

    snapshot = Snapshot(
        as_of_utc=datetime.now(UTC),
        source="football-data.org",
        matches=sorted(matches, key=lambda m: m.utc_kickoff),
        standings=standings,
    )
    path = write_snapshot(snapshot, data_dir / "snapshots")
    print(f"wrote snapshot {path} ({len(matches)} matches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
