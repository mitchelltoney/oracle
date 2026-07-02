"""football-data.org v4 adapter — the only module that talks to this provider.

Caching: ETag conditional requests persisted under ``.cache/ingest/``.
Rate limits: honours 429 Retry-After with bounded retries and pauses when the
per-minute quota (X-Requests-Available-Minute) is exhausted — never hammers.
Every fresh 200 body is snapshotted raw to ``data/raw/`` with a UTC timestamp.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from services.ingest.provider import (
    Match,
    MatchStatus,
    MissingApiKeyError,
    ProviderError,
)

logger = logging.getLogger(__name__)

API_KEY_ENV = "FOOTBALL_DATA_API_KEY"


def _parse_match(raw: dict[str, Any], season: int) -> Match | None:
    home = raw.get("homeTeam") or {}
    away = raw.get("awayTeam") or {}
    if home.get("id") is None or away.get("id") is None:
        return None  # knockout slot not yet determined
    score = raw.get("score") or {}
    full_time = score.get("fullTime") or {}
    return Match(
        id=int(raw["id"]),
        season=season,
        utc_kickoff=datetime.fromisoformat(raw["utcDate"]),
        stage=str(raw.get("stage") or ""),
        status=MatchStatus(raw["status"]),
        home_id=int(home["id"]),
        home=str(home.get("name") or home.get("shortName") or home["id"]),
        away_id=int(away["id"]),
        away=str(away.get("name") or away.get("shortName") or away["id"]),
        home_goals=full_time.get("home"),
        away_goals=full_time.get("away"),
        duration=str(score.get("duration") or "REGULAR"),
        winner=score.get("winner"),
    )


class FootballDataProvider:
    BASE_URL = "https://api.football-data.org/v4"
    COMPETITION_ID = 2000  # FIFA World Cup

    def __init__(
        self,
        api_key: str,
        *,
        cache_dir: Path = Path(".cache/ingest"),
        raw_dir: Path = Path("data/raw"),
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise MissingApiKeyError(f"{API_KEY_ENV} must be non-empty")
        self._api_key = api_key
        self._cache_dir = cache_dir
        self._raw_dir = raw_dir
        self._client = client or httpx.Client(timeout=30.0)
        self._sleep = sleep
        self._max_retries = max_retries
        self._requests_available: int | None = None
        self._quota_reset_seconds: float = 60.0

    @classmethod
    def from_env(cls, **kwargs: Any) -> FootballDataProvider:
        api_key = os.environ.get(API_KEY_ENV, "")
        if not api_key:
            raise MissingApiKeyError(
                f"{API_KEY_ENV} is not set. Copy .env.example to .env and add your "
                "football-data.org API key "
                "(register at https://www.football-data.org/client/register)."
            )
        return cls(api_key, **kwargs)

    def fetch_matches(self, season: int) -> list[Match]:
        body = self._get(
            f"/competitions/{self.COMPETITION_ID}/matches", {"season": str(season)}
        )
        matches: list[Match] = []
        skipped = 0
        for raw in body.get("matches", []):
            match = _parse_match(raw, season)
            if match is None:
                skipped += 1
            else:
                matches.append(match)
        if skipped:
            logger.info(
                "season %d: skipped %d matches with undetermined teams", season, skipped
            )
        return matches

    def fetch_standings(self, season: int) -> dict[str, Any]:
        return self._get(
            f"/competitions/{self.COMPETITION_ID}/standings", {"season": str(season)}
        )

    def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        url = f"{self.BASE_URL}{path}"
        cache_path = self._cache_path(url, params)
        cached = self._load_cache(cache_path)
        headers = {"X-Auth-Token": self._api_key}
        if cached is not None and cached.get("etag"):
            headers["If-None-Match"] = cached["etag"]

        for attempt in range(self._max_retries + 1):
            self._wait_for_quota()
            try:
                response = self._client.get(url, params=params, headers=headers)
            except httpx.HTTPError as exc:
                raise ProviderError(f"GET {url} failed: {exc}") from exc
            self._note_quota(response)

            if response.status_code == 304 and cached is not None:
                body: dict[str, Any] = cached["body"]
                return body
            if response.status_code == 200:
                body = response.json()
                self._save_cache(cache_path, url, response.headers.get("ETag"), body)
                self._write_raw(path, params, body)
                return body
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "60"))
                if attempt < self._max_retries:
                    logger.warning(
                        "rate limited on %s; backing off %.0fs", url, retry_after
                    )
                    self._sleep(retry_after)
                    continue
                raise ProviderError(
                    f"GET {url} still rate limited after {self._max_retries} retries"
                )
            raise ProviderError(
                f"GET {url} failed with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        raise ProviderError(f"GET {url} exhausted retries")

    def _wait_for_quota(self) -> None:
        if self._requests_available == 0:
            logger.warning(
                "per-minute quota exhausted; sleeping %.0fs", self._quota_reset_seconds
            )
            self._sleep(self._quota_reset_seconds)
            self._requests_available = None

    def _note_quota(self, response: httpx.Response) -> None:
        available = response.headers.get("X-Requests-Available-Minute")
        if available is not None:
            self._requests_available = int(available)
        reset = response.headers.get("X-RequestCounter-Reset")
        if reset is not None:
            self._quota_reset_seconds = float(reset)

    def _cache_path(self, url: str, params: dict[str, str]) -> Path:
        key = hashlib.sha256(
            f"{url}?{sorted(params.items())}".encode()
        ).hexdigest()
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, cache_path: Path) -> dict[str, Any] | None:
        if not cache_path.exists():
            return None
        try:
            data: dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8"))
            return data
        except (OSError, json.JSONDecodeError):
            return None

    def _save_cache(
        self, cache_path: Path, url: str, etag: str | None, body: dict[str, Any]
    ) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "url": url,
                    "etag": etag,
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "body": body,
                }
            ),
            encoding="utf-8",
        )

    def _write_raw(
        self, path: str, params: dict[str, str], body: dict[str, Any]
    ) -> None:
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        slug = path.strip("/").replace("/", "_")
        if params:
            slug += "_" + "-".join(f"{k}{v}" for k, v in sorted(params.items()))
        stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%S%f}Z"
        (self._raw_dir / f"{slug}_{stamp}.json").write_text(
            json.dumps(body, indent=2), encoding="utf-8"
        )
