from __future__ import annotations

import random
from datetime import timedelta

import numpy as np
import pytest

from services.ingest import Match
from services.models.features.build import (
    FEATURE_NAMES,
    MIN_PRIOR_MATCHES,
    REST_CLIP_DAYS,
    build_training,
    featurize,
    state_at,
)
from tests.conftest import KICKOFF_BASE, MatchFactory

TEAMS = ["Alba", "Brava", "Corda", "Delta", "Echo", "Fjord", "Gale", "Harbor"]


def qualifier_corpus(make_match: MatchFactory, n: int) -> list[Match]:
    """Deterministic corpus: lower-index teams are stronger, one match per day."""
    rng = random.Random(3)
    matches: list[Match] = []
    for i in range(n):
        a, b = rng.sample(range(len(TEAMS)), 2)
        edge = b - a  # positive when the home side is the stronger (lower) index
        if edge > 2:
            goals = (2, 0)
        elif edge > 0:
            goals = (1, 0)
        elif edge == 0 or abs(edge) <= 1:
            goals = (1, 1)
        else:
            goals = (0, 1) if edge > -3 else (0, 2)
        matches.append(
            make_match(
                days_ago=float(n + 10 - i),
                home=TEAMS[a],
                away=TEAMS[b],
                home_id=100 + a,
                away_id=100 + b,
                home_goals=goals[0],
                away_goals=goals[1],
                stage="FIFA World Cup qualification",
            )
        )
    return matches


def test_rows_are_emitted_only_after_min_prior_and_never_for_friendlies(
    make_match: MatchFactory,
) -> None:
    matches: list[Match] = []
    for i in range(MIN_PRIOR_MATCHES):
        matches.append(
            make_match(
                days_ago=100.0 - i,
                home="Alba",
                away="Brava",
                home_goals=1,
                away_goals=0,
                stage="FIFA World Cup qualification",
            )
        )
    # both teams now have exactly MIN_PRIOR_MATCHES priors
    competitive = make_match(
        days_ago=50.0, home="Alba", away="Brava", home_goals=2, away_goals=0,
        stage="FIFA World Cup qualification",
    )
    friendly = make_match(
        days_ago=40.0, home="Alba", away="Brava", home_goals=0, away_goals=5,
        stage="Friendly",
    )
    x, y, kept = build_training([*matches, competitive, friendly], cutoff=KICKOFF_BASE)

    assert [m.id for m in kept] == [competitive.id]  # warmup + friendly rows skipped
    assert y.tolist() == [0]
    # ... but the friendly still updated the rolling state
    with_friendly = state_at([*matches, competitive, friendly], cutoff=KICKOFF_BASE)
    without_friendly = state_at([*matches, competitive], cutoff=KICKOFF_BASE)
    fixture = make_match(kickoff=KICKOFF_BASE + timedelta(days=1), home="Alba", away="Corda")
    i_ppg5 = FEATURE_NAMES.index("ppg_diff_5")
    ppg5_with = featurize(fixture, with_friendly)[i_ppg5]
    ppg5_without = featurize(fixture, without_friendly)[i_ppg5]
    assert ppg5_with < ppg5_without  # the 0-5 friendly loss dents Alba's form window


def test_features_are_as_of_prior_matches_only(make_match: MatchFactory) -> None:
    corpus = qualifier_corpus(make_match, 60)
    x1, y1, kept1 = build_training(corpus, cutoff=KICKOFF_BASE)

    later = make_match(
        days_ago=1.0, home="Alba", away="Brava", home_goals=4, away_goals=0,
        stage="FIFA World Cup qualification",
    )
    x2, y2, kept2 = build_training([*corpus, later], cutoff=KICKOFF_BASE)

    # appending a later match adds a row but leaves every earlier row untouched
    assert len(kept2) == len(kept1) + 1
    assert np.array_equal(x2[: len(kept1)], x1)
    assert kept2[-1].id == later.id


def test_kept_rows_are_chronological(make_match: MatchFactory) -> None:
    corpus = qualifier_corpus(make_match, 60)
    random.Random(9).shuffle(corpus)
    _, _, kept = build_training(corpus, cutoff=KICKOFF_BASE)
    kickoffs = [m.utc_kickoff for m in kept]
    assert kickoffs == sorted(kickoffs)
    assert len(kept) > 0


def test_venue_features(make_match: MatchFactory) -> None:
    from dataclasses import replace

    state = state_at([], cutoff=KICKOFF_BASE)  # empty state: only venue terms differ
    kickoff = KICKOFF_BASE + timedelta(days=1)
    i_home = FEATURE_NAMES.index("is_true_home")
    i_host = FEATURE_NAMES.index("host_diff")
    i_rest = FEATURE_NAMES.index("rest_diff")

    unknown = featurize(
        make_match(kickoff=kickoff, season=2026, home="Canada", away="Brava"), state
    )
    assert unknown[i_home] == 0.0
    assert unknown[i_host] == 1.0  # Canada hosts in 2026, venue unknown

    hosts_both = featurize(
        make_match(kickoff=kickoff, season=2026, home="Canada", away="Mexico"), state
    )
    assert hosts_both[i_host] == 0.0

    true_home = featurize(
        replace(make_match(kickoff=kickoff, season=2026, home="Canada", away="Brava"),
                neutral=False),
        state,
    )
    assert true_home[i_home] == 1.0
    assert true_home[i_host] == 0.0  # explicit venue: host flag not applied

    assert unknown[i_rest] == pytest.approx(0.0)  # both unseen: fully rested defaults


def test_rest_diff_is_clipped(make_match: MatchFactory) -> None:
    old = make_match(
        days_ago=200.0, home="Alba", away="Brava", home_goals=1, away_goals=0,
        stage="FIFA World Cup qualification",
    )
    recent = make_match(
        days_ago=2.0, home="Corda", away="Delta", home_goals=1, away_goals=0,
        stage="FIFA World Cup qualification",
    )
    state = state_at([old, recent], cutoff=KICKOFF_BASE)
    fixture = make_match(kickoff=KICKOFF_BASE + timedelta(days=1), home="Alba", away="Corda")
    rest_diff = featurize(fixture, state)[FEATURE_NAMES.index("rest_diff")]
    assert rest_diff == pytest.approx(REST_CLIP_DAYS - 3.0)  # 30-clip vs 3 days
