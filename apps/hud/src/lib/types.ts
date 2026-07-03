/** Shapes served by services/api/app.py — keep in sync with that contract. */

export interface Fixture {
  id: number;
  home: string;
  away: string;
  kickoff_utc: string;
  stage: string;
  status: string;
}

export interface Probs {
  home: number;
  draw: number;
  away: number;
}

export interface PredictionRecord {
  fixture_id: number;
  home: string;
  away: string;
  kickoff_utc: string;
  model: string;
  model_version: string;
  probs: Probs;
  top_scorelines: [number, number, number][];
  snapshot_as_of: string;
  written_at: string;
}

export interface CalibrationRow {
  model_version: string;
  n: number;
  brier: number;
  log_loss: number;
}

/** Per-team survival vector: reach_<round> keys derived from `rounds`, plus `win`. */
export type SimTeam = Record<string, number>;

export interface SimPayload {
  schema_version: number;
  generated_at: string;
  snapshot_as_of: string;
  history_as_of: string;
  model: string;
  model_version: string;
  seed: number;
  rounds: string[];
  n_sims: number;
  pairing_source: string;
  teams: Record<string, SimTeam>;
}

export type EndpointName = "fixtures" | "predictions" | "calibration" | "sim";

export interface EndpointStatus {
  ok: boolean;
  error: string | null;
  lastOkAt: number | null;
}

export type FeedEventKind =
  | "boot"
  | "prediction.new"
  | "prediction.move"
  | "sim.updated"
  | "fixture.resolved"
  | "fixture.new"
  | "calibration.change"
  | "poll.error";

export type FeedSeverity = "info" | "notable" | "alert";

export interface FeedEvent {
  id: string;
  at: number;
  kind: FeedEventKind;
  severity: FeedSeverity;
  text: string;
}

export interface TimelinePoint {
  /** written_at of the logged prediction row */
  t: string;
  h: number;
  d: number;
  a: number;
}

export type TimelineMap = Readonly<
  Record<number, Readonly<Record<string, readonly TimelinePoint[]>>>
>;

export interface HudState {
  tick: number;
  lastTickAt: number | null;
  fixtures: Fixture[];
  predictions: PredictionRecord[];
  sim: SimPayload | null;
  calibration: CalibrationRow[];
  timeline: TimelineMap;
  feed: FeedEvent[];
  endpoints: Record<EndpointName, EndpointStatus>;
}
