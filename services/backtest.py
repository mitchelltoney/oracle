"""Walk-forward backtest over held-out World Cup tournaments. Entry for ``make backtest``.

For each tournament: cutoff = one microsecond before its first finished kickoff;
every model trains only on matches strictly before that cutoff and predicts every
finished tournament match — no training sample can postdate its target (hard
rule 2). Predictions are scored IN MEMORY with the existing calibration metrics;
the append-only prediction log is never touched (hard rule 1).

Past tournaments' targets come from the ingested historical corpus (the
football-data.org free tier 403s on non-current seasons); only the live
``CURRENT_SEASON`` uses the API snapshot. Both corpora flow through the same
ingest snapshot abstraction (hard rule 3).

This module is the sanctioned writer of ``data/sim/backtest_report.json``
(hard rule 5). The report schema is stable so successive runs can be diffed:
``{schema_version, generated_at, corpus{...}, tournaments[...], overall[...],
unmatched_team_names[...]}``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from services.ingest import Match, MatchStatus, Snapshot, load_latest_snapshot
from services.metrics import score_predictions
from services.models.base import MatchModel
from services.models.dixon_coles import DixonColesModel
from services.models.elo import EloModel
from services.models.ensemble import EnsembleModel
from services.models.features import (
    Tier,
    combine_corpora,
    importance_tier,
    normalize_team,
    with_name_ids,
)
from services.models.gbm import GbmModel
from services.prediction_log import PredictionRecord

TOURNAMENTS = (2018, 2022, 2026)
CURRENT_SEASON = 2026  # the only season the API serves live; see module docstring

ModelsFactory = Callable[[], Sequence[MatchModel]]


def _season_targets(
    season: int, snapshot: Snapshot, history: Snapshot | None
) -> list[Match]:
    if season == CURRENT_SEASON:
        pool: Sequence[Match] = snapshot.matches
    else:
        pool = [
            m
            for m in (history.matches if history else [])
            if importance_tier(m.stage) is Tier.WC_FINALS
        ]
    return [
        m
        for m in pool
        if m.season == season
        and m.status is MatchStatus.FINISHED
        and m.home_goals is not None
        and m.away_goals is not None
    ]


def _default_models() -> Sequence[MatchModel]:
    return [DixonColesModel(), EloModel(), GbmModel(), EnsembleModel()]


def _model_corpus(model: MatchModel, corpus: list[Match]) -> list[Match]:
    # Dixon-Coles keeps its P0 design envelope: World Cup finals matches only
    # (the rekeyed historical CSV supplies pre-2018 World Cups).
    if isinstance(model, DixonColesModel):
        return [m for m in corpus if importance_tier(m.stage) is Tier.WC_FINALS]
    return corpus


def _record(pred_model: MatchModel, target: Match) -> PredictionRecord:
    prediction = pred_model.predict(with_name_ids(target))
    return PredictionRecord(
        fixture_id=target.id,
        home=target.home,
        away=target.away,
        kickoff_utc=target.utc_kickoff.isoformat(),
        model=prediction.model,
        model_version=prediction.model_version,
        probs={
            "home": prediction.p_home,
            "draw": prediction.p_draw,
            "away": prediction.p_away,
        },
        top_scorelines=[
            [float(h), float(a), p] for h, a, p in prediction.top_scorelines
        ],
        snapshot_as_of="backtest",  # in-memory only: never appended to the log
    )


def _table(reports: dict[str, Any]) -> str:
    lines = [f"{'model_version':<16} {'n':>5} {'brier':>8} {'log_loss':>9}"]
    for version in sorted(reports):
        r = reports[version]
        lines.append(f"{version:<16} {r.n:>5} {r.brier:>8.4f} {r.log_loss:>9.4f}")
    return "\n".join(lines)


def run(
    data_dir: Path = Path("data"), *, models_factory: ModelsFactory | None = None
) -> int:
    snapshots_dir = data_dir / "snapshots"
    try:
        snapshot = load_latest_snapshot(snapshots_dir)
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    history: Snapshot | None = None
    try:
        history = load_latest_snapshot(snapshots_dir, kind="history")
    except FileNotFoundError as exc:
        print(f"WARNING: {exc}; backtesting on the WC corpus only", file=sys.stderr)

    factory = models_factory or _default_models
    combined = combine_corpora(history.matches if history else [], snapshot.matches)
    version_to_name: dict[str, str] = {}
    all_records: list[PredictionRecord] = []
    all_targets: list[Match] = []
    unmatched: set[str] = set()
    tournaments_out: list[dict[str, Any]] = []

    for season in TOURNAMENTS:
        targets = _season_targets(season, snapshot, history)
        if not targets:
            print(f"WARNING: season {season}: no finished matches; skipping",
                  file=sys.stderr)
            continue
        cutoff = min(m.utc_kickoff for m in targets) - timedelta(microseconds=1)
        training = [m for m in combined if m.utc_kickoff < cutoff]
        if not training:
            print(f"WARNING: season {season}: no training data before "
                  f"{cutoff.isoformat()}; skipping", file=sys.stderr)
            continue

        records: list[PredictionRecord] = []
        trained_names: set[str] = set()
        for model in factory():
            try:
                model.fit(_model_corpus(model, training), cutoff)
            except ValueError as exc:
                print(f"WARNING: season {season}: skipping {model.name}: {exc}",
                      file=sys.stderr)
                continue
            version_to_name[model.version] = model.name
            records.extend(_record(model, target) for target in targets)
            if isinstance(model, EloModel):
                trained_names = set(model.ratings)

        if not records:
            print(f"WARNING: season {season}: no model could be fitted; skipping",
                  file=sys.stderr)
            continue
        if trained_names:
            for target in targets:
                for name in (target.home, target.away):
                    if normalize_team(name) not in trained_names:
                        unmatched.add(name)

        reports = score_predictions(records, targets)
        print(f"\n== World Cup {season}  ({len(targets)} matches, "
              f"cutoff {cutoff.isoformat()}) ==")
        print(_table(reports))
        tournaments_out.append(
            {
                "season": season,
                "cutoff": cutoff.isoformat(),
                "n_targets": len(targets),
                "models": [
                    {
                        "model": version_to_name[r.model_version],
                        "model_version": r.model_version,
                        "n": r.n,
                        "brier": r.brier,
                        "log_loss": r.log_loss,
                    }
                    for r in sorted(reports.values(), key=lambda r: r.model_version)
                ],
            }
        )
        all_records.extend(records)
        all_targets.extend(targets)

    if not tournaments_out:
        print("FATAL: no tournament could be backtested", file=sys.stderr)
        return 1

    overall = score_predictions(all_records, all_targets)
    print(f"\n== Overall  ({len({m.id for m in all_targets})} matches) ==")
    print(_table(overall))
    if unmatched:
        print(f"\nWARNING: {len(unmatched)} team names had no training history "
              f"(extend features.team_names.ALIASES?): {sorted(unmatched)}",
              file=sys.stderr)

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": {
            "wc_snapshot_as_of": snapshot.as_of_utc.isoformat(),
            "history_as_of": history.as_of_utc.isoformat() if history else None,
        },
        "tournaments": tournaments_out,
        "overall": [
            {
                "model": version_to_name[r.model_version],
                "model_version": r.model_version,
                "n": r.n,
                "brier": r.brier,
                "log_loss": r.log_loss,
            }
            for r in sorted(overall.values(), key=lambda r: r.model_version)
        ],
        "unmatched_team_names": sorted(unmatched),
    }
    out_dir = data_dir / "sim"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
