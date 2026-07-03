import { useSyncExternalStore } from "react";

import type { HudState } from "./types";

/**
 * Minimal external store. The whole state object is replaced exactly once
 * per poll tick via `commitTick` — the single batched update the
 * performance rules require. Nothing else mutates state.
 *
 * Selector contract: selectors passed to `useStore` must be referentially
 * stable across renders and return referentially stable output for the
 * same state. Non-trivial derivations belong in selectors.ts (memoized);
 * plain slice reads (`s => s.sim`) are safe because slices only change
 * identity at commit.
 */

export function initialState(): HudState {
  const endpoint = () => ({ ok: false, error: null, lastOkAt: null });
  return {
    tick: 0,
    lastTickAt: null,
    fixtures: [],
    predictions: [],
    sim: null,
    calibration: [],
    timeline: {},
    feed: [],
    endpoints: {
      fixtures: endpoint(),
      predictions: endpoint(),
      calibration: endpoint(),
      sim: endpoint(),
    },
  };
}

export interface Store {
  getState: () => HudState;
  commitTick: (next: HudState) => void;
  subscribe: (listener: () => void) => () => void;
}

export function createStore(initial: HudState = initialState()): Store {
  let state = initial;
  const listeners = new Set<() => void>();
  return {
    getState: () => state,
    commitTick: (next) => {
      state = next;
      for (const listener of listeners) listener();
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}

export const store: Store = createStore();

export function useStore<T>(selector: (s: HudState) => T): T {
  return useSyncExternalStore(store.subscribe, () =>
    selector(store.getState()),
  );
}
