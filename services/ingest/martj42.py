"""martj42/international_results adapter — historical international results.

The only module that talks to this source (hard rule 3). One CSV holds every
recorded men's international since 1872 with columns
``date,home_team,away_team,home_score,away_score,tournament,city,country,neutral``.

Caching: ETag conditional request persisted under ``.cache/ingest/``; every fresh
200 body is dumped verbatim to ``data/raw/`` with a UTC timestamp.

Mapping conventions (all documented here because the CSV lacks them):
- The dataset has no kickoff times; ``utc_kickoff`` is the match date at 12:00 UTC.
  Same-day overlap with football-data.org rows is resolved downstream
  (``services.models.features.corpus.combine_corpora`` prefers the provider row,
  which carries a real kickoff instant).
- Scores include extra time and exclude penalty shootouts — the same convention
  as ``Match.home_goals``/``away_goals``.
- ``stage`` carries the raw ``tournament`` string (e.g. "FIFA World Cup",
  "FIFA World Cup qualification", "Friendly") — the match-importance signal.
- Match and team ids are deterministic sha256-based synthetics: stable across
  re-ingests, no registry needed. Model-level cross-corpus team identity is
  established later by ``services.models.features.team_names``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from services.ingest.provider import Match, MatchStatus, ProviderError

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def synthetic_id(*parts: str) -> int:
    """Deterministic 63-bit id from string parts (stable across re-ingests)."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _parse_row(row: dict[str, str]) -> Match | None:
    try:
        date = datetime.strptime(row["date"], "%Y-%m-%d")
        home_goals = int(row["home_score"])
        away_goals = int(row["away_score"])
    except (KeyError, ValueError):
        return None  # future fixture rows have blank scores; malformed dates are rare
    home = row["home_team"].strip()
    away = row["away_team"].strip()
    if not home or not away:
        return None
    if home_goals > away_goals:
        winner = "HOME_TEAM"
    elif home_goals < away_goals:
        winner = "AWAY_TEAM"
    else:
        winner = "DRAW"
    return Match(
        id=synthetic_id(row["date"], home, away),
        season=date.year,
        utc_kickoff=date.replace(hour=12, tzinfo=UTC),  # noon-UTC convention, see module doc
        stage=row.get("tournament", "").strip(),
        status=MatchStatus.FINISHED,
        home_id=synthetic_id(home.lower()),
        home=home,
        away_id=synthetic_id(away.lower()),
        away=away,
        home_goals=home_goals,
        away_goals=away_goals,
        duration="REGULAR",
        winner=winner,
        neutral=row.get("neutral", "").strip().upper() == "TRUE",
    )


class Martj42Provider:
    SOURCE = "martj42/international_results"

    def __init__(
        self,
        *,
        cache_dir: Path = Path(".cache/ingest"),
        raw_dir: Path = Path("data/raw"),
        client: httpx.Client | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._raw_dir = raw_dir
        self._client = client or httpx.Client(timeout=60.0, follow_redirects=True)

    def fetch_all_matches(self) -> list[Match]:
        body = self._get_csv()
        matches: list[Match] = []
        skipped = 0
        for row in csv.DictReader(io.StringIO(body)):
            match = _parse_row(row)
            if match is None:
                skipped += 1
            else:
                matches.append(match)
        if skipped:
            logger.info("history: skipped %d rows without a completed result", skipped)
        if not matches:
            raise ProviderError(f"no parseable rows in {DATASET_URL}")
        return matches

    def _get_csv(self) -> str:
        cache_path = self._cache_dir / f"{hashlib.sha256(DATASET_URL.encode()).hexdigest()}.json"
        cached = self._load_cache(cache_path)
        headers: dict[str, str] = {}
        if cached is not None and cached.get("etag"):
            headers["If-None-Match"] = cached["etag"]
        try:
            response = self._client.get(DATASET_URL, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"GET {DATASET_URL} failed: {exc}") from exc

        if response.status_code == 304 and cached is not None:
            body: str = cached["body"]
            return body
        if response.status_code == 200:
            body = response.text
            self._save_cache(cache_path, response.headers.get("ETag"), body)
            self._write_raw(body)
            return body
        raise ProviderError(
            f"GET {DATASET_URL} failed with HTTP {response.status_code}: {response.text[:200]}"
        )

    def _load_cache(self, cache_path: Path) -> dict[str, Any] | None:
        if not cache_path.exists():
            return None
        try:
            data: dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8"))
            return data
        except (OSError, json.JSONDecodeError):
            return None

    def _save_cache(self, cache_path: Path, etag: str | None, body: str) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "url": DATASET_URL,
                    "etag": etag,
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "body": body,
                }
            ),
            encoding="utf-8",
        )

    def _write_raw(self, body: str) -> None:
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%S%f}Z"
        (self._raw_dir / f"international_results_{stamp}.csv").write_text(
            body, encoding="utf-8"
        )
