import {
  fetchCalibration,
  fetchFixtures,
  fetchPredictions,
  fetchSim,
  type ApiResult,
} from "./api";
import { diffTick, pushEvents, type DraftTick } from "./feed";
import type { Store } from "./store";
import { mergeTimeline, persistTimeline, type StorageLike } from "./timeline";
import type {
  EndpointName,
  EndpointStatus,
  FeedEvent,
  HudState,
} from "./types";

/**
 * One tick = fetch all endpoints, build the next state outside React, and
 * commit exactly once. Unchanged payloads keep their previous slice
 * reference so memoized selectors skip recomputation on no-op ticks.
 */

export const DEFAULT_POLL_INTERVAL_MS = 15_000;

interface PollerOptions {
  intervalMs?: number;
  storage?: StorageLike | null;
}

function resolveSlice<T>(
  result: ApiResult<T>,
  prevSlice: T,
  emptyValue: T,
  prevStatus: EndpointStatus,
  now: number,
): { slice: T; status: EndpointStatus; failed: boolean } {
  switch (result.kind) {
    case "ok":
      return {
        slice: result.data,
        status: { ok: true, error: null, lastOkAt: now },
        failed: false,
      };
    case "empty":
      return {
        slice: emptyValue,
        status: { ok: true, error: null, lastOkAt: now },
        failed: false,
      };
    case "error":
      // Keep showing the last good data; only the status strip degrades.
      return {
        slice: prevSlice,
        status: { ...prevStatus, ok: false, error: result.message },
        failed: true,
      };
  }
}

function samePredictions(
  prev: HudState["predictions"],
  next: HudState["predictions"],
): boolean {
  if (prev.length !== next.length) return false;
  return prev.every((p, i) => {
    const q = next[i];
    return (
      q !== undefined &&
      q.fixture_id === p.fixture_id &&
      q.model_version === p.model_version &&
      q.written_at === p.written_at
    );
  });
}

function sameFixtures(
  prev: HudState["fixtures"],
  next: HudState["fixtures"],
): boolean {
  if (prev.length !== next.length) return false;
  return prev.every((f, i) => {
    const g = next[i];
    return g !== undefined && g.id === f.id && g.status === f.status;
  });
}

function sameCalibration(
  prev: HudState["calibration"],
  next: HudState["calibration"],
): boolean {
  if (prev.length !== next.length) return false;
  return prev.every((c, i) => {
    const d = next[i];
    return (
      d !== undefined &&
      d.model_version === c.model_version &&
      d.n === c.n &&
      d.brier === c.brier &&
      d.log_loss === c.log_loss
    );
  });
}

export async function runTick(
  store: Store,
  storage: StorageLike | null,
  now: number = Date.now(),
): Promise<void> {
  const prev = store.getState();
  const [fixtures, predictions, calibration, sim] = await Promise.all([
    fetchFixtures(),
    fetchPredictions(),
    fetchCalibration(),
    fetchSim(),
  ]);

  const resolved = {
    fixtures: resolveSlice(fixtures, prev.fixtures, [], prev.endpoints.fixtures, now),
    predictions: resolveSlice(
      predictions,
      prev.predictions,
      [],
      prev.endpoints.predictions,
      now,
    ),
    calibration: resolveSlice(
      calibration,
      prev.calibration,
      [],
      prev.endpoints.calibration,
      now,
    ),
    sim: resolveSlice(sim, prev.sim, null, prev.endpoints.sim, now),
  };

  // Preserve slice identity when content is unchanged.
  const nextFixtures = sameFixtures(prev.fixtures, resolved.fixtures.slice)
    ? prev.fixtures
    : resolved.fixtures.slice;
  const nextPredictions = samePredictions(
    prev.predictions,
    resolved.predictions.slice,
  )
    ? prev.predictions
    : resolved.predictions.slice;
  const nextCalibration = sameCalibration(
    prev.calibration,
    resolved.calibration.slice,
  )
    ? prev.calibration
    : resolved.calibration.slice;
  const nextSim =
    prev.sim !== null &&
    resolved.sim.slice !== null &&
    prev.sim.generated_at === resolved.sim.slice.generated_at
      ? prev.sim
      : resolved.sim.slice;

  const draft: DraftTick = {
    fixtures: nextFixtures,
    predictions: nextPredictions,
    sim: nextSim,
    calibration: nextCalibration,
  };

  const errorEvents: FeedEvent[] = [];
  for (const name of ["fixtures", "predictions", "calibration", "sim"] as EndpointName[]) {
    const r = resolved[name];
    if (r.failed && prev.endpoints[name].ok) {
      errorEvents.push({
        id: `${now}-err-${name}`,
        at: now,
        kind: "poll.error",
        severity: "alert",
        text: `endpoint /${name} unreachable // ${r.status.error ?? "unknown error"}`,
      });
    }
  }

  const events = [...diffTick(prev, draft, now), ...errorEvents];
  const timeline = mergeTimeline(prev.timeline, nextPredictions);

  const next: HudState = {
    tick: prev.tick + 1,
    lastTickAt: now,
    fixtures: nextFixtures,
    predictions: nextPredictions,
    sim: nextSim,
    calibration: nextCalibration,
    timeline,
    feed: pushEvents(prev.feed, events),
    endpoints: {
      fixtures: resolved.fixtures.status,
      predictions: resolved.predictions.status,
      calibration: resolved.calibration.status,
      sim: resolved.sim.status,
    },
  };

  store.commitTick(next);
  if (storage && timeline !== prev.timeline) persistTimeline(storage, timeline);
}

export function startPoller(store: Store, options: PollerOptions = {}): () => void {
  const intervalMs = options.intervalMs ?? DEFAULT_POLL_INTERVAL_MS;
  const storage =
    options.storage !== undefined
      ? options.storage
      : typeof localStorage === "undefined"
        ? null
        : localStorage;

  let timer: ReturnType<typeof setInterval> | null = null;
  let disposed = false;
  let inFlight = false;

  const tick = () => {
    if (inFlight || disposed) return;
    inFlight = true;
    void runTick(store, storage).finally(() => {
      inFlight = false;
    });
  };

  const arm = () => {
    if (timer === null) timer = setInterval(tick, intervalMs);
  };
  const disarm = () => {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  };

  const onVisibility = () => {
    if (document.hidden) {
      disarm();
    } else {
      tick();
      arm();
    }
  };

  document.addEventListener("visibilitychange", onVisibility);
  tick();
  if (!document.hidden) arm();

  return () => {
    disposed = true;
    disarm();
    document.removeEventListener("visibilitychange", onVisibility);
  };
}
