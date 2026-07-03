from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from services.backtest import run
from services.ingest import Match, MatchStatus, Snapshot, write_snapshot
from services.models.base import Prediction
from tests.conftest import MatchFactory

WC_2018_START = datetime(2018, 6, 14, 15, 0, tzinfo=UTC)
TEAMS = ["Russia", "Brazil", "Germany", "France", "Spain", "Uruguay"]


@pytest.fixture
def planted_dir(tmp_path: Path, make_match: MatchFactory) -> Path:
    history: list[Match] = []
    kickoff = WC_2018_START - timedelta(days=700)
    i = 0
    for _round in range(8):  # round-robin-ish 2016/17 internationals
        for a in range(len(TEAMS)):
            b = (a + 1 + _round % (len(TEAMS) - 1)) % len(TEAMS)
            if a == b:
                continue
            history.append(
                make_match(
                    kickoff=kickoff,
                    season=kickoff.year,
                    stage="FIFA World Cup" if i % 3 == 0 else
                          "FIFA World Cup qualification",
                    home_id=400 + a,
                    home=TEAMS[a],
                    away_id=400 + b,
                    away=TEAMS[b],
                    home_goals=2 if a < b else 0,
                    away_goals=1,
                )
            )
            kickoff += timedelta(days=5)
            i += 1

    # past tournaments are backtested from the HISTORY corpus (the API free
    # tier does not serve them); the API snapshot carries only the live season
    targets = [
        make_match(
            id=8000 + j,
            kickoff=WC_2018_START + timedelta(days=j),
            season=2018,
            stage="FIFA World Cup",
            home_id=400 + j % 3,
            home=TEAMS[j % 3],
            away_id=403 + j % 3,
            away=TEAMS[3 + j % 3],
            home_goals=(j % 3),
            away_goals=(j + 1) % 2,
        )
        for j in range(8)
    ]
    write_snapshot(
        Snapshot(
            as_of_utc=datetime(2026, 7, 1, tzinfo=UTC),
            source="football-data.org",
            matches=[],
            standings={},
        ),
        tmp_path / "snapshots",
    )
    write_snapshot(
        Snapshot(
            as_of_utc=datetime(2026, 7, 1, tzinfo=UTC),
            source="history",
            matches=[*history, *targets],
            standings={},
        ),
        tmp_path / "snapshots",
        kind="history",
    )
    return tmp_path


class StubModel:
    name = "stub"
    version = "stub-1"

    def __init__(self) -> None:
        self.fit_calls: list[tuple[datetime, datetime]] = []
        self._cutoff: datetime | None = None

    def fit(self, matches: Sequence[Match], cutoff: datetime) -> None:
        train = [
            m
            for m in matches
            if m.status is MatchStatus.FINISHED and m.utc_kickoff < cutoff
        ]
        if not train:
            raise ValueError("no finished matches before cutoff to fit on")
        # record the RAW input's max kickoff: this asserts the CALLER filtered
        # the corpus, not just that this stub did
        self.fit_calls.append((max(m.utc_kickoff for m in matches), cutoff))
        self._cutoff = cutoff

    def predict(self, fixture: Match) -> Prediction:
        assert self._cutoff is not None
        assert fixture.utc_kickoff > self._cutoff  # target strictly after cutoff
        return Prediction(
            fixture_id=fixture.id,
            p_home=0.5,
            p_draw=0.3,
            p_away=0.2,
            top_scorelines=[],
            model=self.name,
            model_version=self.version,
        )


def test_no_training_sample_postdates_its_tournament_targets(
    planted_dir: Path,
) -> None:
    stubs: list[StubModel] = []

    def factory() -> Sequence[StubModel]:
        stub = StubModel()
        stubs.append(stub)
        return [stub]

    assert run(planted_dir, models_factory=factory) == 0

    assert stubs and all(s.fit_calls for s in stubs)
    for stub in stubs:
        for max_train_kickoff, cutoff in stub.fit_calls:
            assert max_train_kickoff < cutoff
            assert cutoff < WC_2018_START  # cutoff precedes every 2018 target
    # StubModel.predict asserts each target kicks off strictly after the cutoff


def test_report_schema_and_metrics(planted_dir: Path) -> None:
    assert run(planted_dir) == 0

    report = json.loads((planted_dir / "sim" / "backtest_report.json").read_text())
    assert report["schema_version"] == 1
    assert report["corpus"]["history_as_of"]
    assert [t["season"] for t in report["tournaments"]] == [2018]

    tournament = report["tournaments"][0]
    assert tournament["n_targets"] == 8
    versions = {m["model_version"] for m in tournament["models"]}
    # the GBM is rightly skipped on this tiny corpus; the rest must report
    assert {"dc-1.0.0", "elo-1.0.1", "ens-1.0.1"} <= versions
    for entry in tournament["models"]:
        assert entry["n"] == 8
        assert 0.0 <= entry["brier"] <= 2.0
        assert entry["log_loss"] > 0.0
    assert report["overall"] and {m["model_version"] for m in report["overall"]} == versions

    # hard rule 1: a backtest never touches the prediction log
    assert not (planted_dir / "predictions").exists()


def test_backtest_fails_cleanly_without_snapshot(tmp_path: Path) -> None:
    assert run(tmp_path) == 1
