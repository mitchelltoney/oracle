import type {
  CalibrationRow,
  Fixture,
  PredictionRecord,
  SimPayload,
} from "./types";

/**
 * Single typed client for the WC Oracle API. All HUD data access goes
 * through here — views never fetch directly.
 *
 * 404 means "data not generated yet" (no snapshot / no sim run), not an
 * error, so it maps to `empty`. The client never throws.
 */
export type ApiResult<T> =
  | { kind: "ok"; data: T }
  | { kind: "empty" }
  | { kind: "error"; message: string };

const BASE: string =
  (import.meta.env?.VITE_API_BASE as string | undefined) ?? "/api";

async function get<T>(path: string): Promise<ApiResult<T>> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { accept: "application/json" },
    });
  } catch (err) {
    return { kind: "error", message: err instanceof Error ? err.message : String(err) };
  }
  if (response.status === 404) return { kind: "empty" };
  if (!response.ok) {
    return { kind: "error", message: `GET ${path} → HTTP ${response.status}` };
  }
  try {
    return { kind: "ok", data: (await response.json()) as T };
  } catch {
    return { kind: "error", message: `GET ${path} → invalid JSON body` };
  }
}

export function fetchFixtures(): Promise<ApiResult<Fixture[]>> {
  return get<Fixture[]>("/fixtures");
}

export function fetchPredictions(
  modelVersion?: string,
): Promise<ApiResult<PredictionRecord[]>> {
  const query = modelVersion
    ? `?${new URLSearchParams({ model_version: modelVersion }).toString()}`
    : "";
  return get<PredictionRecord[]>(`/predictions${query}`);
}

export function fetchCalibration(): Promise<ApiResult<CalibrationRow[]>> {
  return get<CalibrationRow[]>("/calibration");
}

export function fetchSim(): Promise<ApiResult<SimPayload>> {
  return get<SimPayload>("/sim");
}

// TODO(api): the prediction log (data/predictions/predictions.jsonl) is
// append-only and already contains the full nightly history, but
// GET /predictions only exposes the latest row per (fixture_id,
// model_version). When a history endpoint exists (e.g.
// GET /predictions/history?fixture_id=), implement this for real and
// delete the client-side accumulation in lib/timeline.ts.
export function fetchPredictionHistory(
  _fixtureId: number,
): Promise<ApiResult<PredictionRecord[]>> {
  return Promise.resolve({ kind: "empty" });
}
