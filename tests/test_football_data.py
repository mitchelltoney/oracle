from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from services.ingest.football_data import (
    API_KEY_ENV,
    TRANSIENT_RETRY_SECONDS,
    FootballDataProvider,
)
from services.ingest.provider import MatchStatus, MissingApiKeyError, ProviderError

MATCHES_URL = "https://api.football-data.org/v4/competitions/2000/matches"

V4_PAYLOAD: dict[str, Any] = {
    "matches": [
        {
            "id": 101,
            "utcDate": "2026-06-11T20:00:00Z",
            "status": "FINISHED",
            "stage": "GROUP_STAGE",
            "homeTeam": {"id": 771, "name": "Mexico"},
            "awayTeam": {"id": 8030, "name": "South Korea"},
            "score": {
                "winner": "HOME_TEAM",
                "duration": "REGULAR",
                "fullTime": {"home": 2, "away": 1},
            },
        },
        {
            "id": 102,
            "utcDate": "2026-07-04T19:00:00Z",
            "status": "TIMED",
            "stage": "QUARTER_FINALS",
            "homeTeam": {"id": 759, "name": "Germany"},
            "awayTeam": {"id": 760, "name": "Spain"},
            "score": {
                "winner": None,
                "duration": "REGULAR",
                "fullTime": {"home": None, "away": None},
            },
        },
        {
            "id": 103,
            "utcDate": "2026-07-11T19:00:00Z",
            "status": "SCHEDULED",
            "stage": "SEMI_FINALS",
            "homeTeam": {"id": None, "name": None},
            "awayTeam": {"id": None, "name": None},
            "score": {
                "winner": None,
                "duration": "REGULAR",
                "fullTime": {"home": None, "away": None},
            },
        },
    ]
}


def make_provider(tmp_path: Path, **kwargs: Any) -> FootballDataProvider:
    return FootballDataProvider(
        "test-key",
        cache_dir=tmp_path / "cache",
        raw_dir=tmp_path / "raw",
        **kwargs,
    )


def test_from_env_fails_loudly_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(MissingApiKeyError):
        FootballDataProvider.from_env()
    monkeypatch.setenv(API_KEY_ENV, "")
    with pytest.raises(MissingApiKeyError):
        FootballDataProvider.from_env()


@respx.mock
def test_fetch_matches_sends_auth_and_normalizes(tmp_path: Path) -> None:
    route = respx.get(MATCHES_URL, params={"season": "2026"}).mock(
        return_value=httpx.Response(200, json=V4_PAYLOAD)
    )
    matches = make_provider(tmp_path).fetch_matches(2026)

    request = route.calls.last.request
    assert request.headers["X-Auth-Token"] == "test-key"

    assert [m.id for m in matches] == [101, 102]  # TBD-team match 103 skipped
    finished, upcoming = matches
    assert finished.status is MatchStatus.FINISHED
    assert (finished.home, finished.away) == ("Mexico", "South Korea")
    assert (finished.home_goals, finished.away_goals) == (2, 1)
    assert finished.winner == "HOME_TEAM"
    assert finished.utc_kickoff.isoformat() == "2026-06-11T20:00:00+00:00"
    assert upcoming.status is MatchStatus.TIMED
    assert upcoming.home_goals is None


@respx.mock
def test_etag_cache_and_conditional_requests(tmp_path: Path) -> None:
    route = respx.get(MATCHES_URL).mock(
        side_effect=[
            httpx.Response(200, json=V4_PAYLOAD, headers={"ETag": 'W/"v1"'}),
            httpx.Response(304),
        ]
    )
    provider = make_provider(tmp_path)

    first = provider.fetch_matches(2026)
    raw_files = list((tmp_path / "raw").glob("*.json"))
    assert len(raw_files) == 1  # fresh 200 body snapshotted to raw
    assert list((tmp_path / "cache").glob("*.json")), "ETag cache entry written"

    second = provider.fetch_matches(2026)
    assert route.calls.last.request.headers["If-None-Match"] == 'W/"v1"'
    assert second == first  # 304 served from cache
    assert len(list((tmp_path / "raw").glob("*.json"))) == 1  # no new raw file


@respx.mock
def test_429_backs_off_then_retries(tmp_path: Path) -> None:
    respx.get(MATCHES_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json=V4_PAYLOAD),
        ]
    )
    sleeps: list[float] = []
    provider = make_provider(tmp_path, sleep=sleeps.append)
    matches = provider.fetch_matches(2026)
    assert sleeps == [7.0]
    assert len(matches) == 2


@respx.mock
def test_429_never_hammers(tmp_path: Path) -> None:
    route = respx.get(MATCHES_URL).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "3"})
    )
    sleeps: list[float] = []
    provider = make_provider(tmp_path, sleep=sleeps.append, max_retries=2)
    with pytest.raises(ProviderError, match="rate limited"):
        provider.fetch_matches(2026)
    assert route.call_count == 3  # initial + 2 retries, then give up
    assert sleeps == [3.0, 3.0]


@respx.mock
def test_exhausted_minute_quota_pauses_before_next_request(tmp_path: Path) -> None:
    respx.get(MATCHES_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json=V4_PAYLOAD,
                headers={
                    "X-Requests-Available-Minute": "0",
                    "X-RequestCounter-Reset": "42",
                },
            ),
            httpx.Response(200, json=V4_PAYLOAD),
        ]
    )
    sleeps: list[float] = []
    provider = make_provider(tmp_path, sleep=sleeps.append)
    provider.fetch_matches(2022)
    assert sleeps == []
    provider.fetch_matches(2026)
    assert sleeps == [42.0]


@respx.mock
def test_http_error_raises_provider_error(tmp_path: Path) -> None:
    respx.get(MATCHES_URL).mock(return_value=httpx.Response(403, text="restricted"))
    with pytest.raises(ProviderError, match="403"):
        make_provider(tmp_path).fetch_matches(2018)


@respx.mock
def test_dropped_connection_retries_on_fresh_socket(tmp_path: Path) -> None:
    respx.get(MATCHES_URL).mock(
        side_effect=[
            httpx.RemoteProtocolError("Server disconnected without sending a response"),
            httpx.Response(200, json=V4_PAYLOAD),
        ]
    )
    sleeps: list[float] = []
    provider = make_provider(tmp_path, sleep=sleeps.append)
    matches = provider.fetch_matches(2026)
    assert sleeps == [TRANSIENT_RETRY_SECONDS]
    assert len(matches) == 2


@respx.mock
def test_persistent_transport_error_gives_up(tmp_path: Path) -> None:
    route = respx.get(MATCHES_URL).mock(
        side_effect=httpx.ConnectError("SSL: UNEXPECTED_EOF_WHILE_READING")
    )
    sleeps: list[float] = []
    provider = make_provider(tmp_path, sleep=sleeps.append, max_retries=2)
    with pytest.raises(ProviderError, match="UNEXPECTED_EOF"):
        provider.fetch_matches(2026)
    assert route.call_count == 3  # initial + 2 retries, then give up
    assert sleeps == [TRANSIENT_RETRY_SECONDS, TRANSIENT_RETRY_SECONDS]
