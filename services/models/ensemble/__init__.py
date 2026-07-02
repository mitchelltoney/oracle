"""Ensemble ``MatchModel``: a weighted blend of the base models' probability vectors.

Weights are inverse-log-loss, fit on WALK-FORWARD validation predictions: the
chronologically last validation targets are split into contiguous folds, each
base is refit on matches strictly before the fold's first kickoff, and every
fold target is predicted out-of-sample — no training sample ever postdates its
target (hard rule 2). When the comparable validation history is too thin the
ensemble falls back to uniform weights and says so (hard rule 6).

Weight provenance is inspectable: when ``weights_dir`` is set (``make predict``
passes ``data/sim``), ``fit`` serializes the weights, fallback reason, and
per-model validation metrics to ``ensemble_weights_<UTC>.json``. That writer is
a sanctioned ``data/sim`` code path (hard rule 5); files are never overwritten.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from services.ingest import Match, MatchStatus
from services.models.base import LeakageError, MatchModel, Prediction
from services.models.dixon_coles import DixonColesModel
from services.models.elo import EloModel
from services.models.features import Tier, combine_corpora, importance_tier, with_name_ids
from services.models.gbm import GbmModel

logger = logging.getLogger(__name__)

VALIDATION_TARGETS = 60  # chronologically last non-friendly matches
FOLDS = 3
MIN_VALIDATION = 30  # below this, uniform weights (hard rule 6)
_EPS = 1e-15


def _default_bases() -> list[MatchModel]:
    return [DixonColesModel(), EloModel(), GbmModel()]


class EnsembleModel:
    name = "ensemble"
    version = "ens-1.0.0"

    def __init__(
        self,
        base_models: Sequence[MatchModel] | None = None,
        *,
        weights_dir: Path | None = None,
    ) -> None:
        self._bases: list[MatchModel] = (
            list(base_models) if base_models is not None else _default_bases()
        )
        self._weights_dir = weights_dir
        self._cutoff: datetime | None = None
        self._fitted: list[MatchModel] = []
        self._weights: dict[str, float] = {}  # keyed by base model_version
        self._fallback_reason: str | None = None
        self._validation_n = 0
        self._validation_stats: dict[str, dict[str, float]] = {}

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def fit(self, matches: Sequence[Match], cutoff: datetime) -> None:
        finished = [
            m
            for m in matches
            if m.status is MatchStatus.FINISHED
            and m.utc_kickoff < cutoff  # strict: hard rule 2
            and m.home_goals is not None
            and m.away_goals is not None
        ]
        if not finished:
            raise ValueError("no finished matches before cutoff to fit on")
        corpus = combine_corpora(finished, [])

        self._weights, self._fallback_reason = self._walk_forward_weights(corpus)

        # final refit of every base on the full pre-cutoff corpus
        fitted: list[MatchModel] = []
        for base in self._bases:
            try:
                base.fit(self._base_corpus(base, corpus), cutoff)
            except ValueError as exc:
                logger.warning("ensemble: dropping base %s: %s", base.name, exc)
                continue
            fitted.append(base)
        if not fitted:
            raise ValueError("no base model could be fitted before cutoff")
        self._fitted = fitted
        self._renormalize_to_fitted()
        self._cutoff = cutoff
        if self._weights_dir is not None:
            self._write_weights(cutoff)

    def predict(self, fixture: Match) -> Prediction:
        if self._cutoff is None:
            raise RuntimeError("model is not fitted")
        if fixture.utc_kickoff <= self._cutoff:
            raise LeakageError(
                f"fixture {fixture.id} kicks off at {fixture.utc_kickoff.isoformat()}, "
                f"not after the fitted cutoff {self._cutoff.isoformat()}"
            )
        rekeyed = with_name_ids(fixture)  # bases were fit on name-keyed corpora
        p_home = p_draw = p_away = 0.0
        for base in self._fitted:
            weight = self._weights[base.version]
            p = base.predict(rekeyed)
            p_home += weight * p.p_home
            p_draw += weight * p.p_draw
            p_away += weight * p.p_away
        total = p_home + p_draw + p_away
        return Prediction(
            fixture_id=fixture.id,
            p_home=p_home / total,
            p_draw=p_draw / total,
            p_away=p_away / total,
            top_scorelines=[],  # a blend of grids would misstate base disagreements
            model=self.name,
            model_version=self.version,
        )

    # -- weight fitting ----------------------------------------------------

    def _base_corpus(self, base: MatchModel, corpus: list[Match]) -> list[Match]:
        # Dixon-Coles keeps its P0 design envelope: World Cup finals matches only.
        if isinstance(base, DixonColesModel):
            return [m for m in corpus if importance_tier(m.stage) is Tier.WC_FINALS]
        return corpus

    def _walk_forward_weights(
        self, corpus: list[Match]
    ) -> tuple[dict[str, float], str | None]:
        pool = [m for m in corpus if importance_tier(m.stage) is not Tier.FRIENDLY]
        targets = pool[-min(VALIDATION_TARGETS, len(pool)) :]
        fold_size = math.ceil(len(targets) / FOLDS) if targets else 0
        losses: dict[str, list[tuple[float, float]]] = {
            b.version: [] for b in self._bases
        }  # version -> [(log_loss, brier)]
        scored = 0
        for i in range(0, len(targets), fold_size or 1):
            fold = targets[i : i + fold_size]
            if not fold:
                continue
            fold_cutoff = fold[0].utc_kickoff - timedelta(microseconds=1)
            fold_preds: dict[str, list[Prediction]] = {}
            try:
                for base in self._bases:
                    base.fit(self._base_corpus(base, corpus), fold_cutoff)
                    fold_preds[base.version] = [base.predict(m) for m in fold]
            except ValueError:
                continue  # a base can't fit this early: drop the fold for ALL bases
            scored += len(fold)
            for version, preds in fold_preds.items():
                for match, pred in zip(fold, preds, strict=True):
                    losses[version].append(_score(pred, match))

        self._validation_n = scored
        self._validation_stats = {
            version: {
                "log_loss": sum(ll for ll, _ in pairs) / len(pairs),
                "brier": sum(b for _, b in pairs) / len(pairs),
            }
            for version, pairs in losses.items()
            if pairs
        }
        uniform = {b.version: 1.0 / len(self._bases) for b in self._bases}
        if scored < MIN_VALIDATION:
            return uniform, (
                f"only {scored} comparable validation predictions "
                f"(need {MIN_VALIDATION}); using uniform weights"
            )
        inverse = {
            version: 1.0 / max(stats["log_loss"], _EPS)
            for version, stats in self._validation_stats.items()
        }
        total = sum(inverse.values())
        return {version: value / total for version, value in inverse.items()}, None

    def _renormalize_to_fitted(self) -> None:
        fitted_versions = {b.version for b in self._fitted}
        kept = {v: w for v, w in self._weights.items() if v in fitted_versions}
        missing = fitted_versions - kept.keys()
        for version in missing:  # base had no validation weight: give it nothing extra
            kept[version] = 0.0
        total = sum(kept.values())
        if total <= 0.0:
            kept = {v: 1.0 / len(fitted_versions) for v in fitted_versions}
            total = 1.0
        self._weights = {v: w / total for v, w in kept.items()}

    def _write_weights(self, cutoff: datetime) -> None:
        assert self._weights_dir is not None
        self._weights_dir.mkdir(parents=True, exist_ok=True)
        stem = f"ensemble_weights_{datetime.now(UTC):%Y%m%dT%H%M%S}Z"
        path = self._weights_dir / f"{stem}.json"
        suffix = 0
        while path.exists():  # provenance files are never overwritten
            suffix += 1
            path = self._weights_dir / f"{stem}_{suffix}.json"
        payload = {
            "schema_version": 1,
            "written_at": datetime.now(UTC).isoformat(),
            "cutoff": cutoff.isoformat(),
            "model_version": self.version,
            "weights": self._weights,
            "fallback": self._fallback_reason is not None,
            "fallback_reason": self._fallback_reason,
            "validation": {
                "n": self._validation_n,
                "folds": FOLDS,
                "per_model": self._validation_stats,
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("wrote ensemble weights to %s", path)


def _score(pred: Prediction, match: Match) -> tuple[float, float]:
    """(log_loss, brier) of one validation prediction against the known result."""
    assert match.home_goals is not None and match.away_goals is not None
    diff = match.home_goals - match.away_goals
    outcome = 0 if diff > 0 else 2 if diff < 0 else 1
    probs = (pred.p_home, pred.p_draw, pred.p_away)
    log_loss = -math.log(max(probs[outcome], _EPS))
    brier = sum(
        (p - (1.0 if i == outcome else 0.0)) ** 2 for i, p in enumerate(probs)
    )
    return log_loss, brier
