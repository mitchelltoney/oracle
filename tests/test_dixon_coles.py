from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from services.ingest import Match
from services.models.base import LeakageError
from services.models.dixon_coles import DixonColesModel
from tests.conftest import MatchFactory

CUTOFF = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)


def round_robin_history(make_match: MatchFactory) -> list[Match]:
    """Four teams; team 1 strong (wins big), team 4 weak (loses big)."""
    scores = {
        (1, 2): (3, 1),
        (1, 3): (2, 0),
        (1, 4): (4, 0),
        (2, 3): (1, 1),
        (2, 4): (2, 0),
        (3, 4): (2, 1),
    }
    matches: list[Match] = []
    days_ago = 5.0
    for _ in range(3):  # repeat so the fit has some signal
        for (a, b), (ga, gb) in scores.items():
            matches.append(
                make_match(
                    kickoff=CUTOFF - timedelta(days=days_ago),
                    home_id=a,
                    home=f"T{a}",
                    away_id=b,
                    away=f"T{b}",
                    home_goals=ga,
                    away_goals=gb,
                )
            )
            days_ago += 3.0
    return matches


def future_fixture(
    make_match: MatchFactory, home_id: int, away_id: int, days_ahead: float = 3.0
) -> Match:
    return make_match(
        kickoff=CUTOFF + timedelta(days=days_ahead),
        home_id=home_id,
        home=f"T{home_id}",
        away_id=away_id,
        away=f"T{away_id}",
    )


def test_probabilities_are_a_distribution(make_match: MatchFactory) -> None:
    model = DixonColesModel()
    model.fit(round_robin_history(make_match), cutoff=CUTOFF)
    pred = model.predict(future_fixture(make_match, 2, 3))

    assert pred.p_home + pred.p_draw + pred.p_away == pytest.approx(1.0, abs=1e-9)
    for p in (pred.p_home, pred.p_draw, pred.p_away):
        assert 0.0 < p < 1.0
    assert len(pred.top_scorelines) == 5
    probs = [p for _, _, p in pred.top_scorelines]
    assert probs == sorted(probs, reverse=True)
    grid = model._score_grid(1.4, 1.1)
    assert float(grid.sum()) == pytest.approx(1.0, abs=1e-12)


def test_strong_team_is_favored(make_match: MatchFactory) -> None:
    model = DixonColesModel()
    model.fit(round_robin_history(make_match), cutoff=CUTOFF)

    strong_vs_weak = model.predict(future_fixture(make_match, 1, 4))
    assert strong_vs_weak.p_home > strong_vs_weak.p_away
    weak_vs_strong = model.predict(future_fixture(make_match, 4, 1))
    assert weak_vs_strong.p_away > weak_vs_strong.p_home


def test_time_decay_downweights_old_matches(make_match: MatchFactory) -> None:
    """A beat B heavily long ago; B beat A heavily recently."""
    history: list[Match] = []
    for days_ago, (home_id, away_id) in [
        (700.0, (1, 2)),
        (703.0, (1, 2)),
        (7.0, (2, 1)),
        (10.0, (2, 1)),
    ]:
        history.append(
            make_match(
                kickoff=CUTOFF - timedelta(days=days_ago),
                home_id=home_id,
                home=f"T{home_id}",
                away_id=away_id,
                away=f"T{away_id}",
                home_goals=3,
                away_goals=0,
            )
        )
    fixture = future_fixture(make_match, 1, 2)

    decayed = DixonColesModel(kappa=0.1)
    decayed.fit(history, cutoff=CUTOFF)
    p_b_decayed = decayed.predict(fixture).p_away

    flat = DixonColesModel(xi=0.0, kappa=0.1)
    flat.fit(history, cutoff=CUTOFF)
    p_b_flat = flat.predict(fixture).p_away

    # with decay, B's recent wins dominate -> B clearly favored
    assert p_b_decayed > p_b_flat
    assert p_b_decayed > 0.5
    # without decay the record is symmetric -> near-even
    flat_pred = flat.predict(fixture)
    assert flat_pred.p_home == pytest.approx(flat_pred.p_away, abs=0.05)


def test_regularization_shrinks_team_parameters(make_match: MatchFactory) -> None:
    history = round_robin_history(make_match)
    loose = DixonColesModel(kappa=0.01)
    loose.fit(history, cutoff=CUTOFF)
    tight = DixonColesModel(kappa=100.0)
    tight.fit(history, cutoff=CUTOFF)

    assert np.linalg.norm(tight._attack) < np.linalg.norm(loose._attack)
    assert np.linalg.norm(tight._defense) < np.linalg.norm(loose._defense)


def test_tau_correction_shifts_low_scores(make_match: MatchFactory) -> None:
    model = DixonColesModel()
    model.fit(round_robin_history(make_match), cutoff=CUTOFF)

    model._rho = 0.0
    base = model._score_grid(1.3, 1.1)
    model._rho = -0.1
    corrected = model._score_grid(1.3, 1.1)

    # negative rho: (0,0) and (1,1) gain mass, (1,0) and (0,1) lose it
    assert corrected[0, 0] > base[0, 0]
    assert corrected[1, 1] > base[1, 1]
    assert corrected[1, 0] < base[1, 0]
    assert corrected[0, 1] < base[0, 1]


def test_fit_excludes_matches_at_or_after_cutoff(make_match: MatchFactory) -> None:
    history = round_robin_history(make_match)
    leaky_extra = [
        make_match(
            kickoff=CUTOFF,  # exactly at cutoff: must be excluded
            home_id=4,
            home="T4",
            away_id=1,
            away="T1",
            home_goals=9,
            away_goals=0,
        ),
        make_match(
            kickoff=CUTOFF + timedelta(days=1),
            home_id=4,
            home="T4",
            away_id=1,
            away="T1",
            home_goals=9,
            away_goals=0,
        ),
    ]
    clean = DixonColesModel()
    clean.fit(history, cutoff=CUTOFF)
    guarded = DixonColesModel()
    guarded.fit(history + leaky_extra, cutoff=CUTOFF)

    fixture = future_fixture(make_match, 1, 4)
    assert guarded.predict(fixture) == clean.predict(fixture)


def test_predict_refuses_fixture_at_or_before_cutoff(make_match: MatchFactory) -> None:
    model = DixonColesModel()
    model.fit(round_robin_history(make_match), cutoff=CUTOFF)

    with pytest.raises(LeakageError):
        model.predict(future_fixture(make_match, 1, 2, days_ahead=0.0))
    with pytest.raises(LeakageError):
        model.predict(future_fixture(make_match, 1, 2, days_ahead=-1.0))


def test_unknown_team_gets_average_parameters(make_match: MatchFactory) -> None:
    model = DixonColesModel()
    model.fit(round_robin_history(make_match), cutoff=CUTOFF)

    fixture = make_match(
        kickoff=CUTOFF + timedelta(days=2),
        home_id=99,
        home="Debutants",
        away_id=1,
        away="T1",
    )
    pred = model.predict(fixture)
    assert pred.p_home + pred.p_draw + pred.p_away == pytest.approx(1.0, abs=1e-9)
    assert pred.p_away > pred.p_home  # T1 is strong; unknown team is average


def test_unfitted_model_refuses_to_predict(make_match: MatchFactory) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        DixonColesModel().predict(future_fixture(make_match, 1, 2))


def test_fit_requires_training_data(make_match: MatchFactory) -> None:
    with pytest.raises(ValueError, match="no finished matches"):
        DixonColesModel().fit([future_fixture(make_match, 1, 2)], cutoff=CUTOFF)
