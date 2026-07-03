import { describe, expect, it } from "vitest";

import { buildConsensusRows } from "../consensus";
import { latestPerModel } from "../latest";
import { fixture, predRow } from "./fixtures";

/**
 * The live log holds superseded model_version generations of the same model
 * family (e.g. ens-1.0.0 alongside ens-1.0.1 after a version bump). Current-
 * view surfaces must only show the newest generation per family.
 */

function twoGenerationRows() {
  return [
    predRow({
      model: "ensemble",
      model_version: "ens-1.0.0",
      probs: { home: 0.9, draw: 0.05, away: 0.05 }, // stale generation
      written_at: "2026-07-02T06:00:00+00:00",
    }),
    predRow({
      model: "ensemble",
      model_version: "ens-1.0.1",
      probs: { home: 0.4, draw: 0.3, away: 0.3 },
      written_at: "2026-07-03T06:00:00+00:00",
    }),
    predRow({
      model: "elo",
      model_version: "elo-1.0.0",
      probs: { home: 0.5, draw: 0.25, away: 0.25 },
      written_at: "2026-07-02T06:00:00+00:00",
    }),
    predRow({
      model: "elo",
      model_version: "elo-1.0.1",
      probs: { home: 0.45, draw: 0.3, away: 0.25 },
      written_at: "2026-07-03T06:00:00+00:00",
    }),
    predRow({
      model: "dixon_coles",
      model_version: "dc-1.0.0",
      probs: { home: 0.55, draw: 0.25, away: 0.2 },
      written_at: "2026-07-03T06:00:00+00:00",
    }),
  ];
}

describe("latestPerModel", () => {
  it("keeps exactly one row per model family, the newest by written_at", () => {
    const latest = latestPerModel(twoGenerationRows());
    const byModel = new Map(latest.map((r) => [r.model, r.model_version]));
    expect(byModel).toEqual(
      new Map([
        ["ensemble", "ens-1.0.1"],
        ["elo", "elo-1.0.1"],
        ["dixon_coles", "dc-1.0.0"],
      ]),
    );
  });

  it("is a no-op when every family has a single generation", () => {
    const rows = [
      predRow({ model: "elo", model_version: "elo-1.0.1" }),
      predRow({ model: "gbm", model_version: "gbm-1.0.1" }),
    ];
    expect(latestPerModel(rows)).toHaveLength(2);
  });
});

describe("consensus with superseded generations in the log", () => {
  it("shows one column per family and excludes stale generations from the JSD", () => {
    const f = fixture({ id: 100 });
    const rows = buildConsensusRows([f], twoGenerationRows());
    const row = rows[0]!;
    expect(row.models.map((m) => m.model_version).sort()).toEqual([
      "dc-1.0.0",
      "elo-1.0.1",
      "ens-1.0.1",
    ]);
    // the stale ens-1.0.0 outlier (0.9 home) must not inflate disagreement:
    // recompute with it included and require the filtered JSD to be smaller
    const withStale = buildConsensusRows(
      [f],
      twoGenerationRows().filter((r) => r.model_version !== "ens-1.0.1"),
    )[0]!;
    expect(row.jsd).toBeLessThan(withStale.jsd);
  });
});
