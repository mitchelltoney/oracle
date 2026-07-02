from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from services.prediction_log import (
    LateForKickoffError,
    PredictionLog,
    PredictionRecord,
)

KICKOFF = datetime(2026, 7, 4, 19, 0, tzinfo=UTC)
BEFORE = KICKOFF - timedelta(hours=6)


def make_record(
    fixture_id: int = 1, model_version: str = "dc-1.0.0", p_home: float = 0.5
) -> PredictionRecord:
    return PredictionRecord(
        fixture_id=fixture_id,
        home="Germany",
        away="Spain",
        kickoff_utc=KICKOFF.isoformat(),
        model="dixon_coles",
        model_version=model_version,
        probs={"home": p_home, "draw": 0.3, "away": 0.7 - p_home},
        top_scorelines=[[1.0, 0.0, 0.1]],
        snapshot_as_of="2026-07-01T00:00:00+00:00",
    )


def test_append_stamps_written_at_pre_kickoff(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "predictions.jsonl")
    stamped = log.append(make_record(), now=BEFORE)
    assert stamped.written_at == BEFORE.isoformat()
    assert datetime.fromisoformat(stamped.written_at) < KICKOFF
    assert log.read_all() == [stamped]


def test_append_preserves_existing_rows_byte_for_byte(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    log = PredictionLog(path)
    log.append(make_record(1), now=BEFORE)
    log.append(make_record(2), now=BEFORE + timedelta(minutes=1))
    original_lines = path.read_bytes().splitlines()
    assert len(original_lines) == 2

    # a brand-new instance on the same path still only appends
    PredictionLog(path).append(make_record(3), now=BEFORE + timedelta(minutes=2))
    lines = path.read_bytes().splitlines()
    assert len(lines) == 3
    assert lines[:2] == original_lines


def test_append_refuses_at_or_after_kickoff(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    log = PredictionLog(path)
    log.append(make_record(1), now=BEFORE)
    before = path.read_bytes()

    with pytest.raises(LateForKickoffError):
        log.append(make_record(2), now=KICKOFF)
    with pytest.raises(LateForKickoffError):
        log.append(make_record(3), now=KICKOFF + timedelta(seconds=1))
    assert path.read_bytes() == before  # nothing was written


def test_latest_per_fixture_last_row_wins(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "predictions.jsonl")
    log.append(make_record(1, p_home=0.40), now=BEFORE)
    log.append(make_record(1, p_home=0.45), now=BEFORE + timedelta(hours=1))
    log.append(make_record(2, p_home=0.60), now=BEFORE)
    log.append(make_record(1, model_version="dc-2.0.0", p_home=0.50), now=BEFORE)

    latest = log.latest_per_fixture()
    assert len(latest) == 3
    assert latest[(1, "dc-1.0.0")].probs["home"] == 0.45
    assert latest[(2, "dc-1.0.0")].probs["home"] == 0.60

    only_v1 = log.latest_per_fixture("dc-1.0.0")
    assert set(only_v1) == {(1, "dc-1.0.0"), (2, "dc-1.0.0")}


def test_read_all_on_missing_file(tmp_path: Path) -> None:
    assert PredictionLog(tmp_path / "nope.jsonl").read_all() == []
