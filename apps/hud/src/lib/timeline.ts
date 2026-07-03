import type { PredictionRecord, TimelineMap, TimelinePoint } from "./types";

/**
 * Client-side accumulation of prediction history.
 *
 * /predictions only exposes the latest row per (fixture, model_version),
 * so the HUD collects one point per observed `written_at` across poll
 * ticks and persists it to localStorage. History therefore only grows
 * while a tab was polling when a nightly run landed — the real fix is the
 * history endpoint noted in api.ts (TODO(api)).
 */

export const TIMELINE_STORAGE_KEY = "wc-oracle-hud:timeline:v1";
export const MAX_POINTS_PER_SERIES = 200;
const SCHEMA_VERSION = 1;

export type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">;

/** Merge latest rows into the map; returns `prev` unchanged if nothing is new. */
export function mergeTimeline(
  prev: TimelineMap,
  records: PredictionRecord[],
): TimelineMap {
  let next: Record<number, Record<string, readonly TimelinePoint[]>> | null = null;
  for (const record of records) {
    const series = prev[record.fixture_id]?.[record.model_version] ?? [];
    if (series.some((point) => point.t === record.written_at)) continue;
    next ??= { ...prev } as Record<number, Record<string, readonly TimelinePoint[]>>;
    const fixture = { ...(next[record.fixture_id] ?? {}) };
    const grown = [
      ...(fixture[record.model_version] ?? []),
      {
        t: record.written_at,
        h: record.probs.home,
        d: record.probs.draw,
        a: record.probs.away,
      },
    ].sort((a, b) => a.t.localeCompare(b.t));
    fixture[record.model_version] = grown.slice(-MAX_POINTS_PER_SERIES);
    next[record.fixture_id] = fixture;
  }
  return next ?? prev;
}

interface PersistedTimeline {
  schemaVersion: number;
  savedAt: string;
  points: TimelineMap;
}

export function loadTimeline(storage: StorageLike): TimelineMap {
  try {
    const raw = storage.getItem(TIMELINE_STORAGE_KEY);
    if (raw === null) return {};
    const parsed = JSON.parse(raw) as Partial<PersistedTimeline>;
    if (parsed.schemaVersion !== SCHEMA_VERSION || typeof parsed.points !== "object") {
      return {};
    }
    return parsed.points ?? {};
  } catch {
    return {};
  }
}

export function persistTimeline(storage: StorageLike, map: TimelineMap): void {
  const write = (points: TimelineMap) => {
    const body: PersistedTimeline = {
      schemaVersion: SCHEMA_VERSION,
      savedAt: new Date().toISOString(),
      points,
    };
    storage.setItem(TIMELINE_STORAGE_KEY, JSON.stringify(body));
  };
  try {
    write(map);
  } catch {
    // Quota: drop the fixture with the oldest activity and retry once,
    // then degrade silently to in-memory only.
    try {
      const pruned = pruneOldest(map);
      write(pruned);
    } catch {
      /* in-memory only */
    }
  }
}

function pruneOldest(map: TimelineMap): TimelineMap {
  const oldestActivity = (series: Readonly<Record<string, readonly TimelinePoint[]>>) =>
    Object.values(series)
      .flatMap((points) => points.map((p) => p.t))
      .sort()[0] ?? "";
  const entries = Object.entries(map).sort(
    (a, b) => oldestActivity(a[1]).localeCompare(oldestActivity(b[1])),
  );
  const keep = entries.slice(Math.ceil(entries.length / 2));
  return Object.fromEntries(keep);
}
