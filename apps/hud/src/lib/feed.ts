import type {
  CalibrationRow,
  FeedEvent,
  FeedSeverity,
  Fixture,
  HudState,
  PredictionRecord,
  SimPayload,
} from "./types";

/**
 * Terminal-feed diff engine: compares the previous committed state with the
 * freshly fetched draft and synthesizes events. Pure — the poller calls it
 * once per tick, before the single commit.
 */

export const PROB_MOVE_THRESHOLD = 0.03;
export const PROB_MOVE_ALERT_THRESHOLD = 0.1;
export const FEED_CAPACITY = 200;

let seq = 0;

function event(kind: FeedEvent["kind"], severity: FeedSeverity, text: string, at: number): FeedEvent {
  seq += 1;
  return { id: `${at}-${seq}`, at, kind, severity, text };
}

export interface DraftTick {
  fixtures: Fixture[];
  predictions: PredictionRecord[];
  sim: SimPayload | null;
  calibration: CalibrationRow[];
}

const pct = (p: number) => `${(p * 100).toFixed(1)}%`;

export function diffTick(
  prev: HudState,
  draft: DraftTick,
  now: number,
): FeedEvent[] {
  // First tick: no baseline to diff against — a single boot line instead of
  // fabricating a "new prediction" event per row already in the log.
  if (prev.tick === 0) {
    return [
      event(
        "boot",
        "info",
        `link established // ${draft.predictions.length} predictions, sim ${draft.sim?.model_version ?? "—"}`,
        now,
      ),
    ];
  }

  const events: FeedEvent[] = [];
  const prevByKey = new Map(
    prev.predictions.map((p) => [`${p.fixture_id}:${p.model_version}`, p]),
  );

  for (const record of draft.predictions) {
    const before = prevByKey.get(`${record.fixture_id}:${record.model_version}`);
    if (before === undefined) {
      events.push(
        event(
          "prediction.new",
          "info",
          `${record.model_version} logged ${record.home}–${record.away}: H ${pct(record.probs.home)} D ${pct(record.probs.draw)} A ${pct(record.probs.away)}`,
          now,
        ),
      );
      continue;
    }
    if (before.written_at === record.written_at) continue;
    const delta = Math.max(
      Math.abs(record.probs.home - before.probs.home),
      Math.abs(record.probs.draw - before.probs.draw),
      Math.abs(record.probs.away - before.probs.away),
    );
    if (delta >= PROB_MOVE_THRESHOLD) {
      events.push(
        event(
          "prediction.move",
          delta >= PROB_MOVE_ALERT_THRESHOLD ? "alert" : "notable",
          `${record.model_version} moved ${record.home}–${record.away} by ${pct(delta)} (H ${pct(before.probs.home)} → ${pct(record.probs.home)})`,
          now,
        ),
      );
    } else {
      events.push(
        event(
          "prediction.new",
          "info",
          `${record.model_version} re-logged ${record.home}–${record.away} (Δ ${pct(delta)})`,
          now,
        ),
      );
    }
  }

  if (prev.sim && draft.sim && prev.sim.generated_at !== draft.sim.generated_at) {
    const favorite = Object.entries(draft.sim.teams).sort(
      (a, b) => (b[1]["win"] ?? 0) - (a[1]["win"] ?? 0),
    )[0];
    const before = favorite ? (prev.sim.teams[favorite[0]]?.["win"] ?? 0) : 0;
    events.push(
      event(
        "sim.updated",
        "notable",
        favorite
          ? `bracket sim refreshed (${draft.sim.n_sims.toLocaleString()} runs) // favorite ${favorite[0]} ${pct(favorite[1]["win"] ?? 0)} (${pct(before)} prior)`
          : `bracket sim refreshed (${draft.sim.n_sims.toLocaleString()} runs)`,
        now,
      ),
    );
  }

  const draftFixtureIds = new Set(draft.fixtures.map((f) => f.id));
  for (const fixture of prev.fixtures) {
    if (
      !draftFixtureIds.has(fixture.id) &&
      new Date(fixture.kickoff_utc).getTime() < now
    ) {
      events.push(
        event(
          "fixture.resolved",
          "alert",
          `full time // ${fixture.home}–${fixture.away} (${fixture.stage}) resolved`,
          now,
        ),
      );
    }
  }
  const prevFixtureIds = new Set(prev.fixtures.map((f) => f.id));
  for (const fixture of draft.fixtures) {
    if (!prevFixtureIds.has(fixture.id) && prev.fixtures.length > 0) {
      events.push(
        event(
          "fixture.new",
          "info",
          `fixture scheduled // ${fixture.home}–${fixture.away} (${fixture.stage})`,
          now,
        ),
      );
    }
  }

  const prevCalibration = new Map(
    prev.calibration.map((c) => [c.model_version, c]),
  );
  for (const row of draft.calibration) {
    const before = prevCalibration.get(row.model_version);
    if (!before) continue;
    if (
      Math.abs(row.brier - before.brier) >= 1e-4 ||
      Math.abs(row.log_loss - before.log_loss) >= 1e-4
    ) {
      events.push(
        event(
          "calibration.change",
          "info",
          `${row.model_version} calibration → brier ${row.brier.toFixed(4)}, log-loss ${row.log_loss.toFixed(4)} (n=${row.n})`,
          now,
        ),
      );
    }
  }

  return events;
}

export function pushEvents(prevFeed: FeedEvent[], events: FeedEvent[]): FeedEvent[] {
  if (events.length === 0) return prevFeed;
  return [...[...events].reverse(), ...prevFeed].slice(0, FEED_CAPACITY);
}
