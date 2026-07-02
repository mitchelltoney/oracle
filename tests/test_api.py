from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.api.app import create_app
from services.ingest import Snapshot, write_snapshot
from services.prediction_log import PredictionLog, PredictionRecord
from tests.conftest import MatchFactory

NOW = datetime.now(UTC)


@pytest.fixture
def data_dir(tmp_path: Path, make_match: MatchFactory) -> Path:
    finished = make_match(
        id=1,
        kickoff=NOW - timedelta(days=3),
        home="Mexico",
        away="South Korea",
        home_goals=2,
        away_goals=0,
    )
    upcoming = make_match(
        id=2,
        kickoff=NOW + timedelta(days=2),
        home="Germany",
        away="Spain",
        stage="QUARTER_FINALS",
    )
    snapshot = Snapshot(
        as_of_utc=NOW - timedelta(hours=1),
        source="football-data.org",
        matches=[finished, upcoming],
        standings={},
    )
    write_snapshot(snapshot, tmp_path / "snapshots")

    log = PredictionLog(tmp_path / "predictions" / "predictions.jsonl")
    for fixture_id, kickoff, probs in [
        (1, finished.utc_kickoff, {"home": 0.5, "draw": 0.3, "away": 0.2}),
        (2, upcoming.utc_kickoff, {"home": 0.35, "draw": 0.3, "away": 0.35}),
    ]:
        log.append(
            PredictionRecord(
                fixture_id=fixture_id,
                home="H",
                away="A",
                kickoff_utc=kickoff.isoformat(),
                model="dixon_coles",
                model_version="dc-1.0.0",
                probs=probs,
                top_scorelines=[],
                snapshot_as_of=snapshot.as_of_utc.isoformat(),
            ),
            now=kickoff - timedelta(hours=6),
        )
    return tmp_path


def test_fixtures_returns_upcoming_only(data_dir: Path) -> None:
    client = TestClient(create_app(data_dir))
    response = client.get("/fixtures")
    assert response.status_code == 200
    body = response.json()
    assert [f["id"] for f in body] == [2]
    assert body[0]["home"] == "Germany"
    assert body[0]["stage"] == "QUARTER_FINALS"


def test_predictions_latest_per_fixture(data_dir: Path) -> None:
    # a later re-run supersedes the first prediction for fixture 2
    log = PredictionLog(data_dir / "predictions" / "predictions.jsonl")
    first = log.latest_per_fixture()[(2, "dc-1.0.0")]
    log.append(
        PredictionRecord(
            fixture_id=2,
            home="H",
            away="A",
            kickoff_utc=first.kickoff_utc,
            model="dixon_coles",
            model_version="dc-1.0.0",
            probs={"home": 0.4, "draw": 0.3, "away": 0.3},
            top_scorelines=[],
            snapshot_as_of=first.snapshot_as_of,
        ),
        now=datetime.fromisoformat(first.kickoff_utc) - timedelta(hours=1),
    )

    client = TestClient(create_app(data_dir))
    body = client.get("/predictions").json()
    by_fixture = {row["fixture_id"]: row for row in body}
    assert len(body) == 2
    assert by_fixture[2]["probs"]["home"] == 0.4

    filtered = client.get("/predictions", params={"model_version": "nope"}).json()
    assert filtered == []


def test_calibration_from_log_vs_results(data_dir: Path) -> None:
    client = TestClient(create_app(data_dir))
    body = client.get("/calibration").json()
    assert len(body) == 1
    report = body[0]
    # only fixture 1 is finished (home win, p_home=0.5): brier = 0.38
    assert report["model_version"] == "dc-1.0.0"
    assert report["n"] == 1
    assert report["brier"] == pytest.approx(0.38)


def test_empty_data_dir_is_graceful(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    assert client.get("/fixtures").status_code == 404
    assert client.get("/calibration").status_code == 404
    assert client.get("/predictions").json() == []
    assert client.get("/sim").status_code == 404


def test_sim_endpoint_serves_the_latest_simulation(data_dir: Path) -> None:
    import json

    client = TestClient(create_app(data_dir))
    assert client.get("/sim").status_code == 404

    payload = {
        "schema_version": 1,
        "model": "ensemble",
        "teams": {"Brazil": {"reach_final": 0.4, "win": 0.25}},
    }
    (data_dir / "sim").mkdir(parents=True, exist_ok=True)
    (data_dir / "sim" / "latest.json").write_text(json.dumps(payload))
    response = client.get("/sim")
    assert response.status_code == 200
    assert response.json() == payload
