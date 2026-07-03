"""Monte Carlo the remaining bracket and write ``data/sim/latest.json``.

Entry point for ``make sim``. This module is the sanctioned writer of
``data/sim/latest.json`` (hard rule 5); the file is intentionally overwritten —
it is "latest" — and carries its full provenance (snapshot as-ofs, model
version, seed, pairing source) in the payload.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from services.ingest import Match, Snapshot, load_latest_snapshot
from services.models.base import MatchModel
from services.models.dixon_coles import DixonColesModel
from services.models.elo import EloModel
from services.models.ensemble import EnsembleModel
from services.models.features import combine_corpora
from services.models.gbm import GbmModel
from services.sim.bracket import build_bracket, fit_cutoff
from services.sim.engine import SimResult, simulate

MODEL_CHOICES = ("ensemble", "dc", "elo", "gbm")


def _build_model(
    choice: str,
    snapshot: Snapshot,
    combined: list[Match],
    cutoff: datetime,
    data_dir: Path,
) -> MatchModel:
    if choice == "dc":
        model: MatchModel = DixonColesModel()
        model.fit(snapshot.finished_before(cutoff), cutoff=cutoff)  # P0 corpus
        return model
    if choice == "elo":
        model = EloModel()
    elif choice == "gbm":
        model = GbmModel()
    else:
        model = EnsembleModel(weights_dir=data_dir / "sim")
    model.fit(combined, cutoff=cutoff)
    return model


def payload(result: SimResult, meta: dict[str, Any]) -> dict[str, Any]:
    round_keys = [f"reach_{name.lower()}" for name in result.rounds]
    teams: dict[str, dict[str, float]] = {}
    for idx, team in enumerate(result.teams):
        probs = {key: float(result.reach[r, idx]) for r, key in enumerate(round_keys)}
        probs["win"] = float(result.win[idx])
        teams[team.name] = probs
    return {
        "schema_version": 1,
        **meta,
        "rounds": list(result.rounds),
        "n_sims": result.n_sims,
        "pairing_source": result.pairing_source,
        "teams": teams,
    }


def run(argv: list[str] | None = None, data_dir: Path = Path("data")) -> int:
    parser = argparse.ArgumentParser(description="Monte Carlo the remaining bracket")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="ensemble")
    parser.add_argument("--n-sims", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args(argv)

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
        print(f"WARNING: {exc}; fitting on the WC corpus only", file=sys.stderr)

    now = datetime.now(UTC)
    cutoff = fit_cutoff(snapshot, now)  # before the earliest in-play kickoff
    combined = combine_corpora(history.matches if history else [], snapshot.matches)
    try:
        model = _build_model(args.model, snapshot, combined, cutoff, data_dir)
        bracket = build_bracket(snapshot, cutoff)
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    result = simulate(
        bracket, model, n_sims=args.n_sims, rng=np.random.default_rng(args.seed)
    )
    body = payload(
        result,
        {
            "generated_at": now.isoformat(),
            "snapshot_as_of": snapshot.as_of_utc.isoformat(),
            "history_as_of": history.as_of_utc.isoformat() if history else None,
            "model": model.name,
            "model_version": model.version,
            "seed": args.seed,
        },
    )

    out_dir = data_dir / "sim"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latest.json"
    out_path.write_text(json.dumps(body, indent=2), encoding="utf-8")

    favorites = sorted(
        body["teams"].items(), key=lambda item: item[1]["win"], reverse=True
    )[:8]
    print(f"simulated {result.n_sims} tournaments with {model.name} ({model.version})")
    for name, probs in favorites:
        print(f"  {name:<24} win {probs['win']:.3f}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
