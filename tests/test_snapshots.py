from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from services.ingest import Match, Snapshot, load_latest_snapshot, write_snapshot
from tests.conftest import MatchFactory

AS_OF = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def make_snapshot(matches: list[Match], as_of: datetime = AS_OF) -> Snapshot:
    return Snapshot(
        as_of_utc=as_of, source="football-data.org", matches=matches, standings={}
    )


def test_write_load_roundtrip(tmp_path: Path, make_match: MatchFactory) -> None:
    matches = [
        make_match(home_goals=2, away_goals=1, days_ago=10),
        make_match(),
    ]
    path = write_snapshot(make_snapshot(matches), tmp_path)
    assert path.name == "snapshot_20260701T120000Z.json"

    loaded = load_latest_snapshot(tmp_path)
    assert loaded.as_of_utc == AS_OF
    assert loaded.matches == matches
    assert loaded.schema_version == 1


def test_load_latest_picks_newest_and_never_overwrites(
    tmp_path: Path, make_match: MatchFactory
) -> None:
    for hours, name in [(2, "old"), (0, "newest"), (1, "mid")]:
        snap = make_snapshot([make_match(home=name)], as_of=AS_OF - timedelta(hours=hours))
        write_snapshot(snap, tmp_path)
    assert load_latest_snapshot(tmp_path).matches[0].home == "newest"

    # same as-of second: a new file is created, never an overwrite
    write_snapshot(make_snapshot([make_match(home="dup")]), tmp_path)
    write_snapshot(make_snapshot([make_match(home="dup2")]), tmp_path)
    assert len(list(tmp_path.glob("snapshot_*.json"))) == 5
    assert load_latest_snapshot(tmp_path).matches[0].home == "dup2"


def test_load_latest_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make ingest"):
        load_latest_snapshot(tmp_path)


def test_neutral_flag_roundtrips_and_defaults_none(
    tmp_path: Path, make_match: MatchFactory
) -> None:
    from dataclasses import replace

    neutral = replace(make_match(home_goals=1, away_goals=0, days_ago=2), neutral=True)
    at_home = replace(make_match(home_goals=0, away_goals=0, days_ago=1), neutral=False)
    unknown = make_match()
    write_snapshot(make_snapshot([neutral, at_home, unknown]), tmp_path)

    loaded = load_latest_snapshot(tmp_path).matches
    assert [m.neutral for m in loaded] == [True, False, None]


def test_history_kind_is_a_separate_stream(tmp_path: Path, make_match: MatchFactory) -> None:
    wc = make_snapshot([make_match(home="wc-team")])
    history = make_snapshot([make_match(home="hist-team")], as_of=AS_OF + timedelta(hours=1))
    write_snapshot(wc, tmp_path)
    path = write_snapshot(history, tmp_path, kind="history")
    assert path.name == "history_20260701T130000Z.json"

    assert load_latest_snapshot(tmp_path).matches[0].home == "wc-team"
    assert load_latest_snapshot(tmp_path, kind="history").matches[0].home == "hist-team"


def test_finished_before_is_strict_leakage_gate(make_match: MatchFactory) -> None:
    cutoff = datetime(2026, 6, 20, 18, 0, tzinfo=UTC)
    before = make_match(kickoff=cutoff - timedelta(days=1), home_goals=1, away_goals=0)
    at_cutoff = make_match(kickoff=cutoff, home_goals=3, away_goals=0)
    after = make_match(kickoff=cutoff + timedelta(hours=1), home_goals=2, away_goals=2)
    unfinished_before = make_match(kickoff=cutoff - timedelta(days=2))
    snap = make_snapshot([before, at_cutoff, after, unfinished_before])

    assert snap.finished_before(cutoff) == [before]


def test_upcoming_filters_and_sorts(make_match: MatchFactory) -> None:
    now = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    later = make_match(kickoff=now + timedelta(days=2))
    sooner = make_match(kickoff=now + timedelta(days=1))
    past = make_match(kickoff=now - timedelta(days=1))
    finished_future_weirdness = make_match(
        kickoff=now + timedelta(days=3), home_goals=1, away_goals=1
    )
    snap = make_snapshot([later, sooner, past, finished_future_weirdness])

    assert snap.upcoming(now) == [sooner, later]
