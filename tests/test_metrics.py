from __future__ import annotations

import math

import pytest

from services.metrics import score_predictions
from services.prediction_log import PredictionRecord
from tests.conftest import MatchFactory


def make_record(
    fixture_id: int, probs: dict[str, float], model_version: str = "dc-1.0.0"
) -> PredictionRecord:
    return PredictionRecord(
        fixture_id=fixture_id,
        home="A",
        away="B",
        kickoff_utc="2026-07-04T19:00:00+00:00",
        model="dixon_coles",
        model_version=model_version,
        probs=probs,
        top_scorelines=[],
        snapshot_as_of="2026-07-01T00:00:00+00:00",
        written_at="2026-07-04T10:00:00+00:00",
    )


def test_brier_and_log_loss_hand_computed(make_match: MatchFactory) -> None:
    predictions = [
        make_record(1, {"home": 0.5, "draw": 0.3, "away": 0.2}),
        make_record(2, {"home": 0.1, "draw": 0.2, "away": 0.7}),
        make_record(3, {"home": 0.4, "draw": 0.4, "away": 0.2}),  # no result yet
    ]
    results = [
        make_match(id=1, home_goals=2, away_goals=0),  # home win
        make_match(id=2, home_goals=1, away_goals=1),  # draw
        make_match(id=3),  # not finished
    ]
    reports = score_predictions(predictions, results)
    report = reports["dc-1.0.0"]

    # match 1 (home): (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.38
    # match 2 (draw): 0.1^2 + (0.2-1)^2 + 0.7^2 = 1.14
    assert report.n == 2
    assert report.brier == pytest.approx((0.38 + 1.14) / 2)
    assert report.log_loss == pytest.approx(-(math.log(0.5) + math.log(0.2)) / 2)


def test_grouped_by_model_version(make_match: MatchFactory) -> None:
    predictions = [
        make_record(1, {"home": 1.0, "draw": 0.0, "away": 0.0}, "dc-1.0.0"),
        make_record(1, {"home": 0.0, "draw": 1.0, "away": 0.0}, "dc-2.0.0"),
    ]
    results = [make_match(id=1, home_goals=1, away_goals=0)]
    reports = score_predictions(predictions, results)

    assert reports["dc-1.0.0"].brier == pytest.approx(0.0)
    assert reports["dc-2.0.0"].brier == pytest.approx(2.0)
    # log loss of a certain-and-wrong prediction is clipped, not infinite
    assert reports["dc-2.0.0"].log_loss == pytest.approx(-math.log(1e-15))


def test_no_completed_results_yields_empty(make_match: MatchFactory) -> None:
    predictions = [make_record(1, {"home": 0.5, "draw": 0.3, "away": 0.2})]
    assert score_predictions(predictions, [make_match(id=1)]) == {}
