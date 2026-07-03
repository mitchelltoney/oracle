from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

from services.models.base import LeakageError
from services.models.features.build import build_training
from services.models.gbm import EVAL_FRACTION, MIN_TRAIN_ROWS, GbmModel
from tests.conftest import KICKOFF_BASE, MatchFactory
from tests.test_features import qualifier_corpus


def test_fit_predict_produces_valid_distribution(make_match: MatchFactory) -> None:
    corpus = qualifier_corpus(make_match, 300)
    model = GbmModel()
    model.fit(corpus, cutoff=KICKOFF_BASE)

    fixture = make_match(
        kickoff=KICKOFF_BASE + timedelta(days=1), home="Alba", away="Harbor"
    )
    p = model.predict(fixture)
    assert p.p_home + p.p_draw + p.p_away == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 < v < 1.0 for v in (p.p_home, p.p_draw, p.p_away))
    assert p.top_scorelines == []
    assert p.model_version == "gbm-1.0.1"
    # Alba dominates Harbor throughout the corpus; the model should notice
    assert p.p_home > p.p_away


def test_fit_requires_min_training_rows(make_match: MatchFactory) -> None:
    with pytest.raises(ValueError, match="at least"):
        GbmModel().fit(qualifier_corpus(make_match, 50), cutoff=KICKOFF_BASE)


def test_leakage_guards(make_match: MatchFactory) -> None:
    model = GbmModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(make_match())
    model.fit(qualifier_corpus(make_match, 300), cutoff=KICKOFF_BASE)
    with pytest.raises(LeakageError):
        model.predict(make_match(kickoff=KICKOFF_BASE))


def test_eval_set_is_the_chronological_tail(
    make_match: MatchFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Early stopping must hold out the LAST rows in time, never a random fold."""
    corpus = qualifier_corpus(make_match, 300)
    x, y, kept = build_training(corpus, cutoff=KICKOFF_BASE)

    captured: dict[str, Any] = {}

    class FakeDMatrix:
        def __init__(
            self, data: Any, label: Any = None, feature_names: Any = None
        ) -> None:
            self.data = np.asarray(data)
            self.label = None if label is None else np.asarray(label)

    class FakeBooster:
        best_iteration = 0

    def fake_train(params: Any, dtrain: Any, **kwargs: Any) -> FakeBooster:
        captured["dtrain"] = dtrain
        captured["deval"] = kwargs["evals"][0][0]
        return FakeBooster()

    monkeypatch.setattr(xgb, "DMatrix", FakeDMatrix)
    monkeypatch.setattr(xgb, "train", fake_train)

    GbmModel().fit(corpus, cutoff=KICKOFF_BASE)

    split = int(len(y) * (1.0 - EVAL_FRACTION))
    assert np.array_equal(captured["dtrain"].data, x[:split])
    assert np.array_equal(captured["deval"].data, x[split:])
    # the eval rows really are the latest kickoffs
    eval_kickoffs = [m.utc_kickoff for m in kept[split:]]
    assert min(eval_kickoffs) >= max(m.utc_kickoff for m in kept[:split])
    assert len(kept) >= MIN_TRAIN_ROWS
