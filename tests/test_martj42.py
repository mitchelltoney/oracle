from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from services.ingest import MatchStatus, ProviderError
from services.ingest.martj42 import DATASET_URL, Martj42Provider

CSV_BODY = """\
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
1998-06-10,Brazil,Scotland,2,1,FIFA World Cup,Saint-Denis,France,TRUE
2000-09-02,Germany,Greece,2,0,FIFA World Cup qualification,Hamburg,Germany,FALSE
2001-03-28,France,Spain,0,0,Friendly,Paris,France,FALSE
2026-07-20,Winner A,Winner B,,,FIFA World Cup,New York,United States,TRUE
"""


def make_provider(tmp_path: Path) -> Martj42Provider:
    return Martj42Provider(
        cache_dir=tmp_path / "cache",
        raw_dir=tmp_path / "raw",
        client=httpx.Client(),
    )


@respx.mock
def test_fetch_parses_rows_and_dumps_raw(tmp_path: Path) -> None:
    respx.get(DATASET_URL).respond(200, text=CSV_BODY, headers={"ETag": '"v1"'})
    matches = make_provider(tmp_path).fetch_all_matches()

    assert len(matches) == 3  # the scoreless future row is skipped
    first = matches[0]
    assert first.utc_kickoff == datetime(1998, 6, 10, 12, 0, tzinfo=UTC)
    assert first.season == 1998
    assert first.stage == "FIFA World Cup"
    assert first.status is MatchStatus.FINISHED
    assert (first.home, first.away) == ("Brazil", "Scotland")
    assert (first.home_goals, first.away_goals) == (2, 1)
    assert first.winner == "HOME_TEAM"
    assert first.neutral is True
    assert matches[1].neutral is False
    assert matches[2].winner == "DRAW"

    raw_files = list((tmp_path / "raw").glob("international_results_*.csv"))
    assert len(raw_files) == 1
    assert raw_files[0].read_text() == CSV_BODY


@respx.mock
def test_ids_are_deterministic_and_distinct(tmp_path: Path) -> None:
    respx.get(DATASET_URL).respond(200, text=CSV_BODY)
    a = make_provider(tmp_path).fetch_all_matches()
    b = make_provider(tmp_path).fetch_all_matches()

    assert [m.id for m in a] == [m.id for m in b]
    assert len({m.id for m in a}) == len(a)
    brazil_1998 = a[0]
    assert brazil_1998.home_id == b[0].home_id
    assert brazil_1998.home_id != brazil_1998.away_id


@respx.mock
def test_etag_304_serves_cached_body(tmp_path: Path) -> None:
    provider = make_provider(tmp_path)
    route = respx.get(DATASET_URL)
    route.respond(200, text=CSV_BODY, headers={"ETag": '"v1"'})
    provider.fetch_all_matches()

    route.respond(304)
    matches = provider.fetch_all_matches()
    assert len(matches) == 3
    assert respx.calls[-1].request.headers["If-None-Match"] == '"v1"'
    # 304 must not produce a second raw dump
    assert len(list((tmp_path / "raw").glob("*.csv"))) == 1


@respx.mock
def test_http_error_raises_provider_error(tmp_path: Path) -> None:
    respx.get(DATASET_URL).respond(500, text="boom")
    with pytest.raises(ProviderError, match="HTTP 500"):
        make_provider(tmp_path).fetch_all_matches()
