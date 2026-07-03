import { describe, expect, it } from "vitest";

import {
  buildBracket,
  inferRound,
  orderRounds,
  reachKey,
  roundSizes,
  survivalOf,
  type BracketMatch,
} from "../bracket";
import {
  completedQfRows,
  fixture,
  miniSim,
  predRow,
  TEST_NOW,
} from "./fixtures";

describe("reachKey / survivalOf", () => {
  it("derives keys from round names, never hardcoded", () => {
    expect(reachKey("LAST_32")).toBe("reach_last_32");
    expect(reachKey("QUARTER_FINALS")).toBe("reach_quarter_finals");
  });

  it("builds a survival vector over sim.rounds plus win", () => {
    const survival = survivalOf(miniSim(), "Alpha");
    expect(survival).toEqual({
      QUARTER_FINALS: 1,
      SEMI_FINALS: 1,
      FINAL: 0.7,
      win: 0.4,
    });
  });

  it("returns null for a team missing from sim.teams", () => {
    expect(survivalOf(miniSim(), "Zulu")).toBeNull();
  });
});

describe("roundSizes", () => {
  it("derives sizes from the confirmed-entrant count, halving per round", () => {
    expect(roundSizes(miniSim())).toEqual({
      QUARTER_FINALS: 4,
      SEMI_FINALS: 2,
      FINAL: 1,
    });
  });
});

describe("inferRound", () => {
  it("finds the deepest round both teams are confirmed to reach", () => {
    const sim = miniSim();
    expect(inferRound(sim, "Alpha", "Bravo")).toBe("QUARTER_FINALS");
    expect(inferRound(sim, "Alpha", "Charlie")).toBe("SEMI_FINALS");
  });

  it("returns null when a team is unknown", () => {
    expect(inferRound(miniSim(), "Alpha", "Zulu")).toBeNull();
  });
});

describe("buildBracket", () => {
  const upcomingQfs = [
    fixture({ id: 3, home: "Echo", away: "Foxtrot" }),
    fixture({
      id: 4,
      home: "Golf",
      away: "Hotel",
      kickoff_utc: "2026-07-05T21:00:00+00:00",
    }),
  ];

  function build(
    overrides: Partial<Parameters<typeof buildBracket>[0]> = {},
  ) {
    return buildBracket({
      sim: miniSim(),
      fixtures: upcomingQfs,
      predictions: [...completedQfRows(), predRow({ fixture_id: 3 })],
      now: TEST_NOW,
      ...overrides,
    });
  }

  it("marks past-kickoff prediction-log pairings completed with inferred winners", () => {
    const tree = build();
    const qf = tree.matchesByRound["QUARTER_FINALS"] ?? [];
    const alphaBravo = qf.find((m) => m.fixtureId === 1);
    expect(alphaBravo?.status).toBe("completed");
    expect(alphaBravo?.winner).toBe("Alpha");
    const charlieDelta = qf.find((m) => m.fixtureId === 2);
    expect(charlieDelta?.winner).toBe("Charlie");
  });

  it("places upcoming fixtures by stage with ensemble probs attached", () => {
    const tree = build();
    const echoFoxtrot = tree.matchesByRound["QUARTER_FINALS"]?.find(
      (m) => m.fixtureId === 3,
    );
    expect(echoFoxtrot?.status).toBe("upcoming");
    expect(echoFoxtrot?.probs).toEqual({ home: 0.45, draw: 0.3, away: 0.25 });
  });

  it("falls back to the mean of model rows when no ensemble row exists", () => {
    const tree = build({
      predictions: [
        predRow({
          fixture_id: 3,
          model: "dixon_coles",
          model_version: "dc-1.0.0",
          probs: { home: 0.4, draw: 0.4, away: 0.2 },
        }),
        predRow({
          fixture_id: 3,
          model: "elo",
          model_version: "elo-1.0.0",
          probs: { home: 0.6, draw: 0.2, away: 0.2 },
        }),
      ],
    });
    const match = tree.matchesByRound["QUARTER_FINALS"]?.find(
      (m) => m.fixtureId === 3,
    );
    expect(match?.probs?.home).toBeCloseTo(0.5);
    expect(match?.probs?.draw).toBeCloseTo(0.3);
  });

  it("leaves probs null for an upcoming match with no predictions", () => {
    const tree = build({ predictions: completedQfRows() });
    const match = tree.matchesByRound["QUARTER_FINALS"]?.find(
      (m) => m.fixtureId === 4,
    );
    expect(match?.probs).toBeNull();
  });

  it("infers the round of stage-less prediction rows from reach vectors", () => {
    const tree = build();
    const qfIds = (tree.matchesByRound["QUARTER_FINALS"] ?? []).map(
      (m) => m.fixtureId,
    );
    expect(qfIds).toContain(1);
    expect(qfIds).toContain(2);
  });

  it("falls back to reach inference when a fixture stage is not a sim round", () => {
    const tree = build({
      fixtures: [
        fixture({ id: 3, home: "Echo", away: "Foxtrot", stage: "QF_WEIRD" }),
      ],
    });
    const match = tree.matchesByRound["QUARTER_FINALS"]?.find(
      (m) => m.fixtureId === 3,
    );
    expect(match).toBeDefined();
  });

  it("fills rounds to size with placeholders and labels the unambiguous SF", () => {
    // Only one SF slot-pair remains open and exactly two QFs (the upcoming
    // pair) are unassigned once Alpha–Charlie is scheduled.
    const sfFixture = fixture({
      id: 5,
      home: "Alpha",
      away: "Charlie",
      kickoff_utc: "2026-07-08T18:00:00+00:00",
      stage: "SEMI_FINALS",
    });
    const tree = build({ fixtures: [...upcomingQfs, sfFixture] });
    const sf = tree.matchesByRound["SEMI_FINALS"] ?? [];
    expect(sf).toHaveLength(2);
    const open = sf.find((m) => m.fixtureId === null);
    expect(open?.slots.map((s) => s.placeholder)).toEqual([
      "Winner of Echo–Foxtrot",
      "Winner of Golf–Hotel",
    ]);
    expect(open?.feeds.every((f) => f !== null)).toBe(true);
  });

  it("keeps ambiguous placeholders as TBD (no bracket-half guessing)", () => {
    const tree = build();
    const sf = tree.matchesByRound["SEMI_FINALS"] ?? [];
    // Two open SFs, four candidate QFs → ambiguous.
    for (const match of sf.filter((m) => m.fixtureId === null && m.feeds.every((f) => f === null))) {
      expect(match.slots.every((s) => s.placeholder === "TBD")).toBe(true);
    }
  });

  it("links feeds from named slots to the completed matches that produced them", () => {
    const sfFixture = fixture({
      id: 5,
      home: "Alpha",
      away: "Charlie",
      kickoff_utc: "2026-07-08T18:00:00+00:00",
      stage: "SEMI_FINALS",
    });
    const tree = build({ fixtures: [...upcomingQfs, sfFixture] });
    const sf = tree.matchesByRound["SEMI_FINALS"]?.find((m) => m.fixtureId === 5);
    const qf = tree.matchesByRound["QUARTER_FINALS"] ?? [];
    const alphaQf = qf.find((m) => m.winner === "Alpha");
    const charlieQf = qf.find((m) => m.winner === "Charlie");
    expect(sf?.feeds[0]).toBe(alphaQf?.id);
    expect(sf?.feeds[1]).toBe(charlieQf?.id);
  });

  it("handles a completed match the sim has not re-run for (winner null, no throw)", () => {
    const sim = miniSim();
    // Neither Echo nor Foxtrot has reach_semi_finals === 1 yet.
    const tree = build({
      fixtures: [],
      predictions: [
        predRow({ fixture_id: 3, kickoff_utc: "2026-07-01T18:00:00+00:00" }),
      ],
      sim,
    });
    const match = tree.matchesByRound["QUARTER_FINALS"]?.find(
      (m) => m.fixtureId === 3,
    );
    expect(match?.status).toBe("completed");
    expect(match?.winner).toBeNull();
  });

  it("renders slots for teams missing from sim.teams with null survival", () => {
    const tree = build({
      fixtures: [
        fixture({ id: 9, home: "Echo", away: "Zulu", stage: "QUARTER_FINALS" }),
      ],
    });
    const match = tree.matchesByRound["QUARTER_FINALS"]?.find(
      (m) => m.fixtureId === 9,
    );
    expect(match?.slots[1].survival).toBeNull();
  });

  it("builds a placeholder-only tree from sim alone", () => {
    const tree = build({ fixtures: [], predictions: [] });
    expect(tree.matchesByRound["QUARTER_FINALS"]).toHaveLength(4);
    expect(tree.matchesByRound["SEMI_FINALS"]).toHaveLength(2);
    expect(tree.matchesByRound["FINAL"]).toHaveLength(1);
    expect(
      Object.values(tree.matchesByRound)
        .flat()
        .every((m) => m.status === "unknown"),
    ).toBe(true);
  });

  it("assigns stable ids for equivalent inputs", () => {
    const a = build();
    const b = build();
    const ids = (tree: typeof a) =>
      tree.rounds.flatMap((r) => (tree.matchesByRound[r] ?? []).map((m) => m.id));
    expect(ids(a)).toEqual(ids(b));
  });

  it("names the champion once win probability resolves to 1", () => {
    const sim = miniSim();
    sim.teams["Alpha"] = {
      reach_quarter_finals: 1,
      reach_semi_finals: 1,
      reach_final: 1,
      win: 1,
    };
    const tree = build({ sim });
    expect(tree.champion).toBe("Alpha");
  });
});

describe("orderRounds", () => {
  it("sorts each round by mean feeder position so edges do not cross", () => {
    const rounds = ["A", "B"];
    const mk = (id: string, feeds: [string | null, string | null]): BracketMatch => ({
      id,
      round: "B",
      slots: [
        { kind: "tbd", team: null, placeholder: "TBD", survival: null },
        { kind: "tbd", team: null, placeholder: "TBD", survival: null },
      ],
      status: "unknown",
      winner: null,
      fixtureId: null,
      kickoffUtc: null,
      probs: null,
      feeds,
    });
    const matchesByRound: Record<string, BracketMatch[]> = {
      A: [mk("A:0", [null, null]), mk("A:1", [null, null]), mk("A:2", [null, null]), mk("A:3", [null, null])],
      B: [mk("B:0", ["A:2", "A:3"]), mk("B:1", ["A:0", "A:1"])],
    };
    orderRounds(rounds, matchesByRound);
    expect((matchesByRound["B"] ?? []).map((m) => m.id)).toEqual(["B:1", "B:0"]);
  });
});
