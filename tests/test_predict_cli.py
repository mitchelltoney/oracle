from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from services.ingest import Match, Snapshot, write_snapshot
from services.predict import run
from tests.conftest import MatchFactory

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def planted_data_dir(tmp_path: Path, make_match: MatchFactory) -> Path:
    now = datetime.now(UTC)
    matches: list[Match] = []
    results = {
        (1, 2): (2, 0),
        (1, 3): (1, 1),
        (2, 3): (0, 1),
        (2, 1): (1, 3),
        (3, 1): (0, 2),
        (3, 2): (2, 2),
    }
    days_ago = 4.0
    for (a, b), (ga, gb) in results.items():
        matches.append(
            make_match(
                kickoff=now - timedelta(days=days_ago),
                home_id=a,
                home=f"T{a}",
                away_id=b,
                away=f"T{b}",
                home_goals=ga,
                away_goals=gb,
            )
        )
        days_ago += 3.0
    matches.append(
        make_match(
            id=900,
            kickoff=now + timedelta(days=1),
            home_id=1,
            home="T1",
            away_id=2,
            away="T2",
            stage="QUARTER_FINALS",
        )
    )
    matches.append(
        make_match(
            id=901,
            kickoff=now + timedelta(days=2),
            home_id=2,
            home="T2",
            away_id=3,
            away="T3",
            stage="QUARTER_FINALS",
        )
    )
    snapshot = Snapshot(
        as_of_utc=now, source="football-data.org", matches=matches, standings={}
    )
    write_snapshot(snapshot, tmp_path / "snapshots")
    return tmp_path


def test_predict_appends_valid_pre_kickoff_rows(planted_data_dir: Path) -> None:
    assert run(planted_data_dir) == 0

    log_path = planted_data_dir / "predictions" / "predictions.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["fixture_id"] for row in rows] == [900, 901]
    for row in rows:
        assert sum(row["probs"].values()) == pytest.approx(1.0, abs=1e-9)
        assert datetime.fromisoformat(row["written_at"]) < datetime.fromisoformat(
            row["kickoff_utc"]
        )
        assert row["model_version"] == "dc-1.0.0"
        assert row["snapshot_as_of"]

    # a re-run appends fresh rows and never touches existing ones
    first_bytes = log_path.read_bytes()
    assert run(planted_data_dir) == 0
    assert log_path.read_bytes()[: len(first_bytes)] == first_bytes
    assert len(log_path.read_text().splitlines()) == 4


def test_predict_fails_cleanly_without_snapshot(tmp_path: Path) -> None:
    assert run(tmp_path) == 1


def test_models_and_api_never_import_the_provider_directly() -> None:
    """Hard rule 3: only the ingest layer knows about football-data.org."""
    sources = [
        *(REPO_ROOT / "services" / "models").rglob("*.py"),
        *(REPO_ROOT / "services" / "api").rglob("*.py"),
        REPO_ROOT / "services" / "predict.py",
        REPO_ROOT / "services" / "metrics.py",
        REPO_ROOT / "services" / "prediction_log.py",
    ]
    assert len(sources) >= 5
    for source in sources:
        assert "football_data" not in source.read_text(encoding="utf-8"), source
