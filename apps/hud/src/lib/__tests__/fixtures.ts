import type { Fixture, PredictionRecord, SimPayload, SimTeam } from "../types";

/**
 * Shared sample payloads. The mini sim uses a 3-round, 8-team bracket —
 * deliberately NOT the production 5-round list, so anything hardcoding
 * round names fails these tests.
 */

export const MINI_ROUNDS = ["QUARTER_FINALS", "SEMI_FINALS", "FINAL"] as const;

function team(qf: number, sf: number, fin: number, win: number): SimTeam {
  return {
    reach_quarter_finals: qf,
    reach_semi_finals: sf,
    reach_final: fin,
    win,
  };
}

/**
 * State of play: QF1 Alpha–Bravo and QF2 Charlie–Delta are completed
 * (Alpha, Charlie advanced). QF3 Echo–Foxtrot and QF4 Golf–Hotel are
 * upcoming. One SF (Alpha–Charlie) is scheduled; the other is open.
 */
export function miniSim(overrides: Partial<SimPayload> = {}): SimPayload {
  return {
    schema_version: 1,
    generated_at: "2026-07-03T03:00:00+00:00",
    snapshot_as_of: "2026-07-03T02:58:00+00:00",
    history_as_of: "2026-07-03T02:59:00+00:00",
    model: "ensemble",
    model_version: "ens-1.0.1",
    seed: 26,
    rounds: [...MINI_ROUNDS],
    n_sims: 100_000,
    pairing_source: "mixed",
    teams: {
      Alpha: team(1, 1, 0.7, 0.4),
      Bravo: team(1, 0, 0, 0),
      Charlie: team(1, 1, 0.3, 0.1),
      Delta: team(1, 0, 0, 0),
      Echo: team(1, 0.55, 0.2, 0.1),
      Foxtrot: team(1, 0.45, 0.15, 0.08),
      Golf: team(1, 0.6, 0.4, 0.2),
      Hotel: team(1, 0.4, 0.25, 0.12),
    },
    ...overrides,
  };
}

export function fixture(overrides: Partial<Fixture> = {}): Fixture {
  return {
    id: 100,
    home: "Echo",
    away: "Foxtrot",
    kickoff_utc: "2026-07-05T18:00:00+00:00",
    stage: "QUARTER_FINALS",
    status: "TIMED",
    ...overrides,
  };
}

export function predRow(
  overrides: Partial<PredictionRecord> = {},
): PredictionRecord {
  return {
    fixture_id: 100,
    home: "Echo",
    away: "Foxtrot",
    kickoff_utc: "2026-07-05T18:00:00+00:00",
    model: "ensemble",
    model_version: "ens-1.0.1",
    probs: { home: 0.45, draw: 0.3, away: 0.25 },
    top_scorelines: [
      [1, 0, 0.12],
      [1, 1, 0.11],
      [0, 0, 0.09],
      [2, 0, 0.08],
      [2, 1, 0.07],
    ],
    snapshot_as_of: "2026-07-03T02:58:00+00:00",
    written_at: "2026-07-03T03:05:00+00:00",
    ...overrides,
  };
}

/** `now` for tests: between the completed QFs and the upcoming ones. */
export const TEST_NOW = new Date("2026-07-04T12:00:00+00:00");

/** Completed QF pairings, visible only through the prediction log. */
export function completedQfRows(): PredictionRecord[] {
  return [
    predRow({
      fixture_id: 1,
      home: "Alpha",
      away: "Bravo",
      kickoff_utc: "2026-07-01T18:00:00+00:00",
      model: "dixon_coles",
      model_version: "dc-1.0.0",
      probs: { home: 0.6, draw: 0.25, away: 0.15 },
    }),
    predRow({
      fixture_id: 2,
      home: "Charlie",
      away: "Delta",
      kickoff_utc: "2026-07-01T21:00:00+00:00",
      model: "dixon_coles",
      model_version: "dc-1.0.0",
      probs: { home: 0.5, draw: 0.3, away: 0.2 },
    }),
  ];
}
