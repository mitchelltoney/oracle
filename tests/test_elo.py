from __future__ import annotations

import random
from dataclasses import replace
from datetime import timedelta

import pytest

from services.ingest import Match
from services.models.base import LeakageError
from services.models.elo import EloModel
from services.models.elo.core import k_factor, mov_multiplier
from tests.conftest import KICKOFF_BASE, MatchFactory


def friendly(
    make_match: MatchFactory,
    *,
    days_ago: float,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    neutral: bool | None = True,
) -> Match:
    return replace(
        make_match(
            days_ago=days_ago,
            home=home,
            away=away,
            home_goals=home_goals,
            away_goals=away_goals,
            stage="Friendly",
        ),
        neutral=neutral,
    )


def test_mov_multiplier_is_eloratings_standard() -> None:
    assert [mov_multiplier(d) for d in (0, 1, -1)] == [1.0, 1.0, 1.0]
    assert mov_multiplier(2) == mov_multiplier(-2) == 1.5
    assert mov_multiplier(3) == 1.75
    assert mov_multiplier(5) == 2.0


def test_k_factor_orders_importance_tiers() -> None:
    assert k_factor("Friendly") == 20.0
    assert k_factor("FIFA World Cup qualification") == 40.0
    assert k_factor("UEFA Euro") == 50.0
    assert k_factor("FIFA World Cup") == 60.0
    assert k_factor("QUARTER_FINALS") == 60.0
    assert k_factor("Friendly") < k_factor("FIFA World Cup qualification") < k_factor(
        "FIFA World Cup"
    )


def test_ratings_replay_chronologically_regardless_of_input_order(
    make_match: MatchFactory,
) -> None:
    matches = [
        friendly(make_match, days_ago=30, home="Alpha", away="Beta", home_goals=1, away_goals=0),
        friendly(make_match, days_ago=20, home="Alpha", away="Gamma", home_goals=3, away_goals=0),
        friendly(make_match, days_ago=10, home="Beta", away="Gamma", home_goals=0, away_goals=1),
    ]

    # hand-computed sequential replay (K=20 friendlies on neutral ground)
    def exp_home(rh: float, ra: float) -> float:
        return float(1.0 / (1.0 + 10.0 ** (-(rh - ra) / 400.0)))

    r = {"alpha": 1500.0, "beta": 1500.0, "gamma": 1500.0}
    d1 = 20.0 * 1.0 * (1.0 - exp_home(r["alpha"], r["beta"]))
    r["alpha"] += d1
    r["beta"] -= d1
    d2 = 20.0 * 1.75 * (1.0 - exp_home(r["alpha"], r["gamma"]))  # 3-0: mov 1.75
    r["alpha"] += d2
    r["gamma"] -= d2
    d3 = 20.0 * 1.0 * (0.0 - exp_home(r["beta"], r["gamma"]))  # away win
    r["beta"] += d3
    r["gamma"] -= d3

    shuffled = matches.copy()
    random.Random(7).shuffle(shuffled)
    model = EloModel()
    model.fit(shuffled, cutoff=KICKOFF_BASE)

    assert model._ratings == pytest.approx(r)


def test_fit_ignores_matches_at_or_after_cutoff(make_match: MatchFactory) -> None:
    early = friendly(make_match, days_ago=30, home="Alpha", away="Beta", home_goals=1, away_goals=0)
    at_cutoff = friendly(
        make_match, days_ago=0, home="Alpha", away="Beta", home_goals=9, away_goals=0
    )
    model = EloModel()
    model.fit([early, at_cutoff], cutoff=KICKOFF_BASE)

    trimmed = EloModel()
    trimmed.fit([early], cutoff=KICKOFF_BASE)
    assert model._ratings == trimmed._ratings


def test_home_advantage_applies_only_when_truly_at_home(
    make_match: MatchFactory,
) -> None:
    model = EloModel()
    model.fit(
        [friendly(make_match, days_ago=10, home="Alpha", away="Beta", home_goals=1, away_goals=1)],
        cutoff=KICKOFF_BASE,
    )
    # equal-rated sides after one draw
    fixture = make_match(kickoff=KICKOFF_BASE + timedelta(days=1), home="Alpha", away="Beta")

    at_home = model.predict(replace(fixture, neutral=False))
    neutral = model.predict(replace(fixture, neutral=True))
    assert neutral.p_home == pytest.approx(neutral.p_away)
    assert at_home.p_home > at_home.p_away


def test_host_bonus_applies_to_2026_hosts_when_venue_unknown(
    make_match: MatchFactory,
) -> None:
    model = EloModel()
    model.fit(
        [friendly(make_match, days_ago=10, home="Alpha", away="Beta", home_goals=2, away_goals=0)],
        cutoff=KICKOFF_BASE,
    )
    kickoff = KICKOFF_BASE + timedelta(days=1)
    # Canada and Gamma are both unseen (initial rating); only the host gets the bonus
    host_home = model.predict(
        make_match(kickoff=kickoff, season=2026, home="Canada", away="Alpha")
    )
    plain_home = model.predict(
        make_match(kickoff=kickoff, season=2026, home="Gamma", away="Alpha")
    )
    assert host_home.p_home > plain_home.p_home

    host_away = model.predict(
        make_match(kickoff=kickoff, season=2026, home="Alpha", away="Mexico")
    )
    plain_away = model.predict(
        make_match(kickoff=kickoff, season=2026, home="Alpha", away="Gamma")
    )
    assert host_away.p_away > plain_away.p_away


def test_probs_are_a_distribution_with_draw_component(
    make_match: MatchFactory,
) -> None:
    model = EloModel()
    model.fit(
        [friendly(make_match, days_ago=10, home="Alpha", away="Beta", home_goals=2, away_goals=0)],
        cutoff=KICKOFF_BASE,
    )
    p = model.predict(make_match(kickoff=KICKOFF_BASE + timedelta(days=1)))
    assert p.p_home + p.p_draw + p.p_away == pytest.approx(1.0)
    assert p.p_draw > 0.05
    assert p.top_scorelines == []


def test_leakage_guards(make_match: MatchFactory) -> None:
    model = EloModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(make_match())
    model.fit(
        [friendly(make_match, days_ago=10, home="Alpha", away="Beta", home_goals=1, away_goals=0)],
        cutoff=KICKOFF_BASE,
    )
    with pytest.raises(LeakageError):
        model.predict(make_match(kickoff=KICKOFF_BASE))
    with pytest.raises(ValueError, match="no finished matches"):
        EloModel().fit([], cutoff=KICKOFF_BASE)
