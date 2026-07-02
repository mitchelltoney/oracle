"""Gradient-boosted match model (XGBoost) behind ``MatchModel``.

Trained on the deliberately small feature set from
``services.models.features.build`` (each feature's as-of-time source is
documented there). The early-stopping evaluation set is the CHRONOLOGICAL TAIL
of the training rows — never a random split, which would leak future matches
into training (hard rule 2). Hyperparameters are fixed and conservative
(shallow trees, heavy regularization — hard rule 6); the only data-driven
choice is the boosting-round count via the time-ordered holdout. Changing any
of them requires a new model version string (hard rule 1).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

import xgboost as xgb

from services.ingest import Match
from services.models.base import LeakageError, Prediction
from services.models.features.build import (
    FEATURE_NAMES,
    TeamState,
    build_training,
    featurize,
    state_at,
)

logger = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 200
EVAL_FRACTION = 0.15  # chronological tail used for early stopping
NUM_BOOST_ROUND = 400
EARLY_STOPPING_ROUNDS = 25

_PARAMS: dict[str, object] = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 3,
    "eta": 0.05,
    "min_child_weight": 10,
    "lambda": 5.0,
    "alpha": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 26,
    "verbosity": 0,
}


class GbmModel:
    name = "gbm"
    version = "gbm-1.0.0"

    def __init__(self) -> None:
        self._cutoff: datetime | None = None
        self._booster: xgb.Booster | None = None
        self._state: TeamState | None = None

    def fit(self, matches: Sequence[Match], cutoff: datetime) -> None:
        x, y, kept = build_training(matches, cutoff)
        if len(y) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"only {len(y)} training rows before cutoff; "
                f"gbm needs at least {MIN_TRAIN_ROWS}"
            )
        split = int(len(y) * (1.0 - EVAL_FRACTION))
        feature_names = list(FEATURE_NAMES)
        dtrain = xgb.DMatrix(x[:split], label=y[:split], feature_names=feature_names)
        deval = xgb.DMatrix(x[split:], label=y[split:], feature_names=feature_names)
        self._booster = xgb.train(
            _PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(deval, "eval")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        self._state = state_at(matches, cutoff)
        self._cutoff = cutoff
        logger.info(
            "fitted gbm on %d rows (eval tail %d), best_iteration=%d, cutoff %s",
            split,
            len(y) - split,
            self._booster.best_iteration,
            cutoff.isoformat(),
        )

    def predict(self, fixture: Match) -> Prediction:
        if self._cutoff is None or self._booster is None or self._state is None:
            raise RuntimeError("model is not fitted")
        if fixture.utc_kickoff <= self._cutoff:
            raise LeakageError(
                f"fixture {fixture.id} kicks off at {fixture.utc_kickoff.isoformat()}, "
                f"not after the fitted cutoff {self._cutoff.isoformat()}"
            )
        row = featurize(fixture, self._state).reshape(1, -1)
        dmat = xgb.DMatrix(row, feature_names=list(FEATURE_NAMES))
        probs = self._booster.predict(
            dmat, iteration_range=(0, self._booster.best_iteration + 1)
        )[0]
        return Prediction(
            fixture_id=fixture.id,
            p_home=float(probs[0]),
            p_draw=float(probs[1]),
            p_away=float(probs[2]),
            top_scorelines=[],  # the GBM has no scoreline distribution
            model=self.name,
            model_version=self.version,
        )
