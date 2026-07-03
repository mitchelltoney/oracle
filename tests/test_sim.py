from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from services.ingest import Match, Snapshot
from services.models.base import Prediction
from services.sim import build_bracket, simulate
from services.sim.engine import (
    PENS_SKILL_CAP,
    advance_home_prob,
    penalty_home_prob,
)
from tests.conftest import MatchFactory

NOW = datetime(2026, 7, 2, 12, 0, tzinfo=UTC)


class StubModel:
    name = "stub"
    version = "stub-1"

    def __init__(self, probs: tuple[float, float, float] = (0.5, 0.2, 0.3)) -> None:
        self._probs = probs
        self.predicted: list[Match] = []

    def fit(self, matches: Any, cutoff: Any) -> None:
        pass

    def predict(self, fixture: Match) -> Prediction:
        self.predicted.append(fixture)
        return Prediction(
            fixture_id=fixture.id,
            p_home=self._probs[0],
            p_draw=self._probs[1],
            p_away=self._probs[2],
            top_scorelines=[],
            model=self.name,
            model_version=self.version,
        )


def r32_snapshot(make_match: MatchFactory, *, finished_slots: int = 0) -> Snapshot:
    """16 LAST_32 matches (32 teams); the first ``finished_slots`` are decided 1-0."""
    matches = []
    for slot in range(16):
        goals: tuple[int | None, int | None] = (
            (1, 0) if slot < finished_slots else (None, None)
        )
        matches.append(
            make_match(
                id=500 + slot,
                kickoff=NOW + timedelta(days=slot < finished_slots and -1 or 1, hours=slot),
                stage="LAST_32",
                home_id=1000 + 2 * slot,
                home=f"Team {2 * slot:02d}",
                away_id=1000 + 2 * slot + 1,
                away=f"Team {2 * slot + 1:02d}",
                home_goals=goals[0],
                away_goals=goals[1],
            )
        )
    return Snapshot(as_of_utc=NOW, source="test", matches=matches, standings={})


def team_probs(result: Any, name: str) -> list[float]:
    idx = next(i for i, t in enumerate(result.teams) if t.name == name)
    return [float(result.reach[r, idx]) for r in range(len(result.rounds))] + [
        float(result.win[idx])
    ]


def test_bracket_structure_from_snapshot(make_match: MatchFactory) -> None:
    bracket = build_bracket(r32_snapshot(make_match), NOW)
    assert bracket.rounds == (
        "LAST_32", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL",
    )
    assert len(bracket.teams) == 32
    assert len(bracket.entry_pairs) == 16
    assert [len(bracket.pairings[r]) for r in bracket.rounds[1:]] == [8, 4, 2, 1]
    assert bracket.pairing_source == "template"


def test_round_reach_probabilities_are_monotone_and_conserved(
    make_match: MatchFactory,
) -> None:
    result = simulate(
        build_bracket(r32_snapshot(make_match), NOW),
        StubModel(),
        n_sims=20_000,
        rng=np.random.default_rng(1),
    )
    for team in result.teams:
        r32, r16, qf, sf, final, win = team_probs(result, team.name)
        assert r32 == 1.0
        assert win <= final <= sf <= qf <= r16 <= r32

    assert float(result.win.sum()) == pytest.approx(1.0)
    # participant conservation: 32 in the R32, 16 in the R16, ... 2 in the final
    for round_idx, expected in enumerate([32, 16, 8, 4, 2]):
        assert float(result.reach[round_idx].sum()) == pytest.approx(expected)


def test_decided_matches_force_the_winner_through(make_match: MatchFactory) -> None:
    result = simulate(
        build_bracket(r32_snapshot(make_match, finished_slots=3), NOW),
        StubModel(),
        n_sims=5_000,
        rng=np.random.default_rng(2),
    )
    for slot in range(3):  # slots 0-2 finished 1-0: home advanced, away is out
        assert team_probs(result, f"Team {2 * slot:02d}")[1] == 1.0
        assert team_probs(result, f"Team {2 * slot + 1:02d}")[1] == 0.0


def test_shootout_winner_is_honoured(make_match: MatchFactory) -> None:
    snapshot = r32_snapshot(make_match)
    decided_on_pens = replace(
        snapshot.matches[0],
        home_goals=1,
        away_goals=1,
        status=snapshot.matches[0].status.FINISHED,
        duration="PENALTY_SHOOTOUT",
        winner="AWAY_TEAM",
        utc_kickoff=NOW - timedelta(days=1),
    )
    snapshot = replace(snapshot, matches=[decided_on_pens, *snapshot.matches[1:]])
    result = simulate(
        build_bracket(snapshot, NOW), StubModel(), n_sims=2_000,
        rng=np.random.default_rng(3),
    )
    assert team_probs(result, "Team 01")[1] == 1.0  # away side won the shootout
    assert team_probs(result, "Team 00")[1] == 0.0


def test_known_fixtures_reconcile_the_pairing_tree(make_match: MatchFactory) -> None:
    # entry round = QUARTER_FINALS (8 teams); QF slots 0 and 2 already decided,
    # and the snapshot knows the real SF fixture: winner(QF0) v winner(QF2) —
    # a cross-pairing that must override the sequential template ((0,1),(2,3)).
    matches = []
    for slot in range(4):
        decided = slot in (0, 2)
        matches.append(
            make_match(
                id=700 + slot,
                # same-day kickoffs, hour = slot: slot order is construction order
                kickoff=NOW - timedelta(days=1) + timedelta(hours=slot),
                stage="QUARTER_FINALS",
                home_id=2000 + 2 * slot,
                home=f"Q{2 * slot}",
                away_id=2000 + 2 * slot + 1,
                away=f"Q{2 * slot + 1}",
                home_goals=2 if decided else None,
                away_goals=0 if decided else None,
            )
        )
    semi = make_match(
        id=800,
        kickoff=NOW + timedelta(days=5),
        stage="SEMI_FINALS",
        home_id=2000,  # winner of QF slot 0
        home="Q0",
        away_id=2004,  # winner of QF slot 2
        away="Q4",
    )
    snapshot = Snapshot(
        as_of_utc=NOW, source="test", matches=[*matches, semi], standings={}
    )
    bracket = build_bracket(snapshot, NOW)
    assert bracket.rounds == ("QUARTER_FINALS", "SEMI_FINALS", "FINAL")
    assert bracket.pairings["SEMI_FINALS"] == ((0, 2), (1, 3))
    assert bracket.pairing_source == "mixed"

    # Q0 and Q4 must meet in the semi in EVERY simulation
    result = simulate(
        bracket, StubModel(), n_sims=2_000, rng=np.random.default_rng(4)
    )
    assert team_probs(result, "Q0")[1] == 1.0
    assert team_probs(result, "Q4")[1] == 1.0
    # they can never both make the final
    assert team_probs(result, "Q0")[2] + team_probs(result, "Q4")[2] == pytest.approx(1.0)


def test_penalty_model_is_a_bounded_near_coin_flip() -> None:
    assert penalty_home_prob(1.0, 0.0) == 0.5 + PENS_SKILL_CAP == 0.55
    assert penalty_home_prob(0.0, 1.0) == 0.5 - PENS_SKILL_CAP == 0.45
    assert penalty_home_prob(0.4, 0.4) == 0.5
    for p_home in np.linspace(0.0, 1.0, 21):
        p = penalty_home_prob(float(p_home), float(1.0 - p_home))
        assert 0.45 <= p <= 0.55


def test_advance_prob_is_symmetric_and_normalized() -> None:
    for p_home, p_draw, p_away in [(0.5, 0.3, 0.2), (0.1, 0.2, 0.7), (0.0, 1.0, 0.0)]:
        forward = advance_home_prob(p_home, p_draw, p_away)
        backward = advance_home_prob(p_away, p_draw, p_home)
        assert forward + backward == pytest.approx(1.0)
        assert 0.0 <= forward <= 1.0


def test_cli_writes_latest_json(tmp_path: Any, make_match: MatchFactory) -> None:
    import json

    from services.ingest import write_snapshot
    from services.sim.cli import run

    now = datetime.now(UTC)
    matches = []
    for i in range(16):  # finished group matches so Elo has ratings for all 8 teams
        home, away = i % 8, (i + 3) % 8
        matches.append(
            make_match(
                kickoff=now - timedelta(days=20 - i),
                stage="GROUP_STAGE",
                home_id=3000 + home,
                home=f"G{home}",
                away_id=3000 + away,
                away=f"G{away}",
                home_goals=2 if home < away else 0,
                away_goals=1,
            )
        )
    for slot in range(4):
        matches.append(
            make_match(
                kickoff=now + timedelta(days=1, hours=slot),
                stage="QUARTER_FINALS",
                home_id=3000 + 2 * slot,
                home=f"G{2 * slot}",
                away_id=3000 + 2 * slot + 1,
                away=f"G{2 * slot + 1}",
            )
        )
    write_snapshot(
        Snapshot(as_of_utc=now, source="test", matches=matches, standings={}),
        tmp_path / "snapshots",
    )

    assert run(["--model", "elo", "--n-sims", "2000", "--seed", "1"], data_dir=tmp_path) == 0

    body = json.loads((tmp_path / "sim" / "latest.json").read_text())
    assert body["schema_version"] == 1
    assert body["model"] == "elo"
    assert body["model_version"] == "elo-1.0.0"
    assert body["n_sims"] == 2000
    assert body["rounds"] == ["QUARTER_FINALS", "SEMI_FINALS", "FINAL"]
    assert len(body["teams"]) == 8
    assert sum(t["win"] for t in body["teams"].values()) == pytest.approx(1.0)
    for probs in body["teams"].values():
        assert probs["win"] <= probs["reach_final"] <= probs["reach_semi_finals"]


def test_engine_only_100k_sims_under_ten_seconds(make_match: MatchFactory) -> None:
    """Times ONLY the vectorized engine: ``simulate()`` on a prebuilt bracket with a
    stub model. No snapshot IO, no model fitting, no real probability computation —
    this is the numpy hot path (CLAUDE.md), ~0.15s on Apple Silicon. The full
    ``make sim`` budget is enforced separately by the pipeline test below."""
    bracket = build_bracket(r32_snapshot(make_match), NOW)
    model = StubModel()
    start = time.perf_counter()
    result = simulate(bracket, model, n_sims=100_000, rng=np.random.default_rng(5))
    elapsed = time.perf_counter() - start
    assert result.n_sims == 100_000
    assert elapsed < 10.0, f"sim took {elapsed:.2f}s"
    # grouped sampling: the model is consulted per distinct pairing, not per sim
    assert len(model.predicted) < 1000


def test_full_sim_pipeline_100k_under_ten_seconds(
    tmp_path: Any, make_match: MatchFactory
) -> None:
    """Times everything ``make sim`` runs in-process at production scale: snapshot
    load, corpus combine, ensemble fit (3 walk-forward folds + final refit of all
    bases — the dominant cost, ~8s of ~9s measured on real 2026-07-03 data), bracket
    build, 100k simulations, and the latest.json write. Only interpreter startup is
    excluded. Corpus sized to the real one: ~49.5k historical matches, 300 teams."""
    from services.ingest import write_snapshot
    from services.sim.cli import run

    now = datetime.now(UTC)
    n_teams, n_history = 300, 49_500
    names = [f"N{i:03d}" for i in range(n_teams)]
    history = []
    for i in range(n_history):
        home, away = names[i % n_teams], names[(i * 7 + 1) % n_teams]
        if home == away:
            away = names[(i * 7 + 2) % n_teams]
        stage = (
            "Friendly"
            if i % 3 == 0
            else "FIFA World Cup" if i % 25 == 0 else "FIFA World Cup qualification"
        )
        history.append(
            make_match(
                id=10_000 + i,
                kickoff=now - timedelta(hours=13 * (n_history - i)),
                stage=stage,
                home_id=20_000 + (i % n_teams),
                home=home,
                away_id=20_000 + ((i * 7 + 1) % n_teams),
                away=away,
                home_goals=i % 4,
                away_goals=(i // 3) % 3,
            )
        )
    write_snapshot(
        Snapshot(as_of_utc=now, source="test-history", matches=history, standings={}),
        tmp_path / "snapshots",
        kind="history",
    )
    wc = [  # 16-slot LAST_32 entry round mirroring the live 2026 bracket state
        make_match(
            id=900 + slot,
            kickoff=now + timedelta(days=1, hours=slot),
            stage="LAST_32",
            home_id=30_000 + 2 * slot,
            home=names[2 * slot],
            away_id=30_000 + 2 * slot + 1,
            away=names[2 * slot + 1],
        )
        for slot in range(16)
    ]
    write_snapshot(
        Snapshot(as_of_utc=now, source="test", matches=wc, standings={}),
        tmp_path / "snapshots",
    )

    start = time.perf_counter()
    exit_code = run(["--n-sims", "100000", "--seed", "1"], data_dir=tmp_path)
    elapsed = time.perf_counter() - start
    assert exit_code == 0
    assert (tmp_path / "sim" / "latest.json").exists()
    assert elapsed < 10.0, f"full make sim path took {elapsed:.2f}s (budget 10s)"
