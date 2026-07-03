import { describe, expect, it } from "vitest";

import {
  diffTick,
  FEED_CAPACITY,
  PROB_MOVE_THRESHOLD,
  pushEvents,
  type DraftTick,
} from "../feed";
import { initialState } from "../store";
import { fixture, miniSim, predRow } from "./fixtures";
import type { FeedEvent, HudState } from "../types";

const NOW = new Date("2026-07-04T12:00:00+00:00").getTime();

function committedState(draft: Partial<DraftTick>): HudState {
  return {
    ...initialState(),
    tick: 1,
    lastTickAt: NOW - 15_000,
    fixtures: draft.fixtures ?? [],
    predictions: draft.predictions ?? [],
    sim: draft.sim ?? null,
    calibration: draft.calibration ?? [],
  };
}

function draft(overrides: Partial<DraftTick> = {}): DraftTick {
  return {
    fixtures: [],
    predictions: [],
    sim: null,
    calibration: [],
    ...overrides,
  };
}

describe("diffTick", () => {
  it("suppresses per-row events on the first tick, emitting a single boot line", () => {
    const events = diffTick(
      initialState(),
      draft({ predictions: [predRow(), predRow({ fixture_id: 2 })], sim: miniSim() }),
      NOW,
    );
    expect(events).toHaveLength(1);
    expect(events[0]?.kind).toBe("boot");
  });

  it("emits prediction.move at the threshold but not below it", () => {
    const before = predRow({ probs: { home: 0.5, draw: 0.3, away: 0.2 } });
    const justBelow = predRow({
      written_at: "2026-07-04T03:00:00+00:00",
      probs: { home: 0.529, draw: 0.3, away: 0.171 },
    });
    const atThreshold = predRow({
      written_at: "2026-07-04T03:00:00+00:00",
      probs: { home: 0.5 + PROB_MOVE_THRESHOLD, draw: 0.3, away: 0.17 },
    });

    const prev = committedState({ predictions: [before] });
    const belowEvents = diffTick(prev, draft({ predictions: [justBelow] }), NOW);
    expect(belowEvents.some((e) => e.kind === "prediction.move")).toBe(false);
    const atEvents = diffTick(prev, draft({ predictions: [atThreshold] }), NOW);
    expect(atEvents.some((e) => e.kind === "prediction.move")).toBe(true);
  });

  it("is silent for an unchanged written_at", () => {
    const row = predRow();
    const events = diffTick(
      committedState({ predictions: [row] }),
      draft({ predictions: [row] }),
      NOW,
    );
    expect(events).toHaveLength(0);
  });

  it("reports sim refreshes with the new favorite", () => {
    const events = diffTick(
      committedState({ sim: miniSim() }),
      draft({ sim: miniSim({ generated_at: "2026-07-04T03:00:00+00:00" }) }),
      NOW,
    );
    const simEvent = events.find((e) => e.kind === "sim.updated");
    expect(simEvent?.text).toContain("Alpha");
  });

  it("flags fixtures that disappeared past kickoff as resolved", () => {
    const gone = fixture({ id: 7, kickoff_utc: "2026-07-04T09:00:00+00:00" });
    const events = diffTick(
      committedState({ fixtures: [gone] }),
      draft({ fixtures: [] }),
      NOW,
    );
    expect(events.some((e) => e.kind === "fixture.resolved")).toBe(true);
  });
});

describe("pushEvents", () => {
  it("keeps the newest events first and caps the buffer", () => {
    const mk = (i: number): FeedEvent => ({
      id: `e${i}`,
      at: i,
      kind: "boot",
      severity: "info",
      text: `event ${i}`,
    });
    let feed: FeedEvent[] = [];
    for (let i = 0; i < FEED_CAPACITY + 10; i++) feed = pushEvents(feed, [mk(i)]);
    expect(feed).toHaveLength(FEED_CAPACITY);
    expect(feed[0]?.id).toBe(`e${FEED_CAPACITY + 9}`);
  });

  it("returns the same reference when no events arrive", () => {
    const feed: FeedEvent[] = [];
    expect(pushEvents(feed, [])).toBe(feed);
  });
});
