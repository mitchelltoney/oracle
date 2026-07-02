"""Append-only prediction log (hard rule 1).

``PredictionLog`` is the ONLY code in the repo that opens the predictions file for
writing, and the only mode it ever uses is ``"a"``. There is no edit or delete API.
``written_at`` is stamped by ``append`` itself and appends are refused at/after
kickoff, so every logged row is provably pre-kickoff.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class PredictionRecord:
    fixture_id: int
    home: str
    away: str
    kickoff_utc: str  # ISO 8601, tz-aware
    model: str
    model_version: str
    probs: dict[str, float]  # {"home": .., "draw": .., "away": ..}
    top_scorelines: list[list[float]]  # [[home_goals, away_goals, prob], ..]
    snapshot_as_of: str  # provenance: as-of of the snapshot that fed the fit
    written_at: str = ""  # stamped by PredictionLog.append


class LateForKickoffError(RuntimeError):
    pass


class PredictionLog:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def append(
        self, record: PredictionRecord, *, now: datetime | None = None
    ) -> PredictionRecord:
        now = now or datetime.now(UTC)
        kickoff = datetime.fromisoformat(record.kickoff_utc)
        if now >= kickoff:
            raise LateForKickoffError(
                f"refusing to log fixture {record.fixture_id}: "
                f"now={now.isoformat()} is not before kickoff={record.kickoff_utc}"
            )
        stamped = replace(record, written_at=now.isoformat())
        line = json.dumps(asdict(stamped), sort_keys=True, separators=(",", ":"))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:  # append-only, never "w"
            f.write(line + "\n")
        return stamped

    def read_all(self) -> list[PredictionRecord]:
        if not self._path.exists():
            return []
        records: list[PredictionRecord] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(PredictionRecord(**json.loads(line)))
        return records

    def latest_per_fixture(
        self, model_version: str | None = None
    ) -> dict[tuple[int, str], PredictionRecord]:
        """Last logged row wins per (fixture_id, model_version)."""
        latest: dict[tuple[int, str], PredictionRecord] = {}
        for record in self.read_all():
            if model_version is not None and record.model_version != model_version:
                continue
            latest[(record.fixture_id, record.model_version)] = record
        return latest
