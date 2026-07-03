import { describe, expect, it, vi } from "vitest";

import {
  loadTimeline,
  MAX_POINTS_PER_SERIES,
  mergeTimeline,
  persistTimeline,
  TIMELINE_STORAGE_KEY,
  type StorageLike,
} from "../timeline";
import { predRow } from "./fixtures";
import type { TimelineMap } from "../types";

function memoryStorage(): StorageLike & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return {
    data,
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => {
      data.set(key, value);
    },
    removeItem: (key) => {
      data.delete(key);
    },
  };
}

describe("mergeTimeline", () => {
  it("adds one point per unseen written_at", () => {
    const merged = mergeTimeline({}, [
      predRow({ written_at: "2026-07-02T03:00:00+00:00" }),
    ]);
    expect(merged[100]?.["ens-1.0.1"]).toHaveLength(1);
    const again = mergeTimeline(merged, [
      predRow({ written_at: "2026-07-03T03:00:00+00:00" }),
    ]);
    expect(again[100]?.["ens-1.0.1"]).toHaveLength(2);
  });

  it("dedupes on written_at and returns the same reference when nothing is new", () => {
    const row = predRow();
    const merged = mergeTimeline({}, [row]);
    expect(mergeTimeline(merged, [row])).toBe(merged);
  });

  it("caps each series at MAX_POINTS_PER_SERIES, dropping the oldest", () => {
    let map: TimelineMap = {};
    for (let i = 0; i < MAX_POINTS_PER_SERIES + 5; i++) {
      map = mergeTimeline(map, [
        predRow({
          written_at: `2026-07-02T03:00:${String(i).padStart(2, "0")}.${String(i).padStart(3, "0")}+00:00`,
        }),
      ]);
    }
    const series = map[100]?.["ens-1.0.1"] ?? [];
    expect(series).toHaveLength(MAX_POINTS_PER_SERIES);
  });
});

describe("loadTimeline / persistTimeline", () => {
  it("round-trips through storage", () => {
    const storage = memoryStorage();
    const map = mergeTimeline({}, [predRow()]);
    persistTimeline(storage, map);
    expect(loadTimeline(storage)).toEqual(map);
  });

  it("discards garbage and wrong schema versions", () => {
    const storage = memoryStorage();
    storage.setItem(TIMELINE_STORAGE_KEY, "{not json");
    expect(loadTimeline(storage)).toEqual({});
    storage.setItem(
      TIMELINE_STORAGE_KEY,
      JSON.stringify({ schemaVersion: 999, points: { 1: {} } }),
    );
    expect(loadTimeline(storage)).toEqual({});
  });

  it("prunes and retries once on quota errors", () => {
    const storage = memoryStorage();
    const setItem = vi
      .fn<StorageLike["setItem"]>()
      .mockImplementationOnce(() => {
        throw new Error("QuotaExceededError");
      })
      .mockImplementation((key, value) => {
        storage.data.set(key, value);
      });
    const throwing: StorageLike = { ...storage, setItem };
    const map = mergeTimeline(
      mergeTimeline({}, [predRow({ fixture_id: 1, written_at: "2026-07-01T03:00:00+00:00" })]),
      [predRow({ fixture_id: 2, written_at: "2026-07-02T03:00:00+00:00" })],
    );
    persistTimeline(throwing, map);
    expect(setItem).toHaveBeenCalledTimes(2);
    const persisted = loadTimeline(storage);
    // The fixture with the oldest activity was pruned.
    expect(Object.keys(persisted)).toEqual(["2"]);
  });

  it("degrades silently when storage keeps failing", () => {
    const broken: StorageLike = {
      getItem: () => null,
      setItem: () => {
        throw new Error("QuotaExceededError");
      },
      removeItem: () => undefined,
    };
    expect(() => persistTimeline(broken, mergeTimeline({}, [predRow()]))).not.toThrow();
  });
});
