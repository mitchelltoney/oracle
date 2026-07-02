from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from services.ingest import Match, MatchStatus
from services.models.base import LeakageError, Prediction
from services.models.ensemble import (
    FOLDS,
    MIN_VALIDATION,
    EnsembleModel,
)
from tests.conftest import KICKOFF_BASE, MatchFactory
from tests.test_features import qualifier_corpus


class StubModel:
    """Instrumented MatchModel: fixed probabilities, records every fit's inputs."""

    def __init__(self, name: str, version: str, probs: tuple[float, float, float]):
        self.name = name
        self.version = version
        self._probs = probs
        self._cutoff: datetime | None = None
        self.fit_calls: list[tuple[datetime, datetime]] = []  # (max_train_kickoff, cutoff)

    def fit(self, matches: Sequence[Match], cutoff: datetime) -> None:
        train = [
            m
            for m in matches
            if m.status is MatchStatus.FINISHED and m.utc_kickoff < cutoff
        ]
        if not train:
            raise ValueError("no finished matches before cutoff to fit on")
        self.fit_calls.append((max(m.utc_kickoff for m in train), cutoff))
        self._cutoff = cutoff

    def predict(self, fixture: Match) -> Prediction:
        assert self._cutoff is not None and fixture.utc_kickoff > self._cutoff
        return Prediction(
            fixture_id=fixture.id,
            p_home=self._probs[0],
            p_draw=self._probs[1],
            p_away=self._probs[2],
            top_scorelines=[],
            model=self.name,
            model_version=self.version,
        )


def home_win_corpus(make_match: MatchFactory, n: int) -> list[Match]:
    return [
        make_match(
            days_ago=float(n + 5 - i),
            home=f"H{i % 9}",
            away=f"A{i % 7}",
            home_id=200 + i % 9,
            away_id=300 + i % 7,
            home_goals=2,
            away_goals=0,
            stage="FIFA World Cup qualification",
        )
        for i in range(n)
    ]


def test_walk_forward_never_trains_on_or_after_its_targets(
    make_match: MatchFactory,
) -> None:
    sharp = StubModel("sharp", "sharp-1", (0.7, 0.2, 0.1))
    flat = StubModel("flat", "flat-1", (1 / 3, 1 / 3, 1 / 3))
    model = EnsembleModel([sharp, flat])
    model.fit(home_win_corpus(make_match, 90), cutoff=KICKOFF_BASE)

    for stub in (sharp, flat):
        assert len(stub.fit_calls) == FOLDS + 1  # one per fold + the final refit
        for max_train_kickoff, cutoff in stub.fit_calls:
            assert max_train_kickoff < cutoff
    # StubModel.predict itself asserts every target kicks off after the fold cutoff,
    # so train < cutoff < target holds for every validation prediction.


def test_inverse_log_loss_weights_favor_the_sharper_model(
    make_match: MatchFactory,
) -> None:
    sharp = StubModel("sharp", "sharp-1", (0.7, 0.2, 0.1))  # corpus is all home wins
    flat = StubModel("flat", "flat-1", (1 / 3, 1 / 3, 1 / 3))
    model = EnsembleModel([sharp, flat])
    model.fit(home_win_corpus(make_match, 90), cutoff=KICKOFF_BASE)

    weights = model.weights
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["sharp-1"] > weights["flat-1"]

    p = model.predict(make_match(kickoff=KICKOFF_BASE + timedelta(days=1)))
    assert p.p_home + p.p_draw + p.p_away == pytest.approx(1.0)
    assert p.model_version == "ens-1.0.0"
    # blend must sit strictly between the two bases
    assert flat._probs[0] < p.p_home < sharp._probs[0]


def test_uniform_fallback_when_validation_history_is_thin(
    make_match: MatchFactory,
) -> None:
    bases = [
        StubModel("sharp", "sharp-1", (0.7, 0.2, 0.1)),
        StubModel("flat", "flat-1", (1 / 3, 1 / 3, 1 / 3)),
    ]
    model = EnsembleModel(bases)
    corpus = home_win_corpus(make_match, MIN_VALIDATION - 10)
    model.fit(corpus, cutoff=KICKOFF_BASE)

    assert model.weights == pytest.approx({"sharp-1": 0.5, "flat-1": 0.5})
    assert model._fallback_reason is not None


def test_weights_file_schema_and_provenance(
    make_match: MatchFactory, tmp_path: Path
) -> None:
    bases = [
        StubModel("sharp", "sharp-1", (0.7, 0.2, 0.1)),
        StubModel("flat", "flat-1", (1 / 3, 1 / 3, 1 / 3)),
    ]
    model = EnsembleModel(bases, weights_dir=tmp_path)
    model.fit(home_win_corpus(make_match, 90), cutoff=KICKOFF_BASE)

    files = list(tmp_path.glob("ensemble_weights_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["schema_version"] == 1
    assert payload["model_version"] == "ens-1.0.0"
    assert payload["fallback"] is False
    assert sum(payload["weights"].values()) == pytest.approx(1.0)
    assert payload["cutoff"] == KICKOFF_BASE.isoformat()
    assert payload["validation"]["n"] >= MIN_VALIDATION
    assert set(payload["validation"]["per_model"]) == {"sharp-1", "flat-1"}
    for stats in payload["validation"]["per_model"].values():
        assert stats["log_loss"] > 0.0
        assert 0.0 <= stats["brier"] <= 2.0

    # a refit writes a NEW file — provenance is never overwritten
    model.fit(home_win_corpus(make_match, 90), cutoff=KICKOFF_BASE)
    assert len(list(tmp_path.glob("ensemble_weights_*.json"))) == 2


def test_default_bases_integration(make_match: MatchFactory) -> None:
    corpus = qualifier_corpus(make_match, 320)
    # give Dixon-Coles a WC-finals subset to fit on (its P0 design envelope)
    from dataclasses import replace

    corpus = [
        replace(m, stage="GROUP_STAGE") if i % 2 == 0 else m
        for i, m in enumerate(corpus)
    ]
    model = EnsembleModel()
    model.fit(corpus, cutoff=KICKOFF_BASE)

    assert sum(model.weights.values()) == pytest.approx(1.0)
    fixture = make_match(
        kickoff=KICKOFF_BASE + timedelta(days=1), home="Alba", away="Harbor"
    )
    p = model.predict(fixture)
    assert p.p_home + p.p_draw + p.p_away == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 < v < 1.0 for v in (p.p_home, p.p_draw, p.p_away))


def test_leakage_guards(make_match: MatchFactory) -> None:
    model = EnsembleModel([StubModel("s", "s-1", (0.5, 0.3, 0.2))])
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(make_match())
    model.fit(home_win_corpus(make_match, 40), cutoff=KICKOFF_BASE)
    with pytest.raises(LeakageError):
        model.predict(make_match(kickoff=KICKOFF_BASE))
    with pytest.raises(ValueError, match="no finished matches"):
        EnsembleModel([StubModel("s", "s-1", (0.5, 0.3, 0.2))]).fit(
            [], cutoff=KICKOFF_BASE
        )
