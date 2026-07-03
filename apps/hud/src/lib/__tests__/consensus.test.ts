import { describe, expect, it } from "vitest";

import {
  buildConsensusRows,
  jensenShannon,
  jsdNormalized,
} from "../consensus";
import { fixture, predRow } from "./fixtures";
import type { Probs } from "../types";

const p = (home: number, draw: number, away: number): Probs => ({
  home,
  draw,
  away,
});

describe("jensenShannon", () => {
  it("is zero for identical distributions", () => {
    const dist = p(0.5, 0.3, 0.2);
    expect(jensenShannon([dist, dist, dist, dist])).toBeCloseTo(0, 12);
  });

  it("hits ln 2 for two point masses on different outcomes", () => {
    const jsd = jensenShannon([p(1, 0, 0), p(0, 0, 1)]);
    expect(jsd).toBeCloseTo(Math.log(2), 12);
    expect(jsdNormalized([p(1, 0, 0), p(0, 0, 1)])).toBeCloseTo(1, 12);
  });

  it("stays finite with zero components (0·ln 0 = 0)", () => {
    const jsd = jensenShannon([p(1, 0, 0), p(0.5, 0.5, 0), p(0, 0, 1)]);
    expect(Number.isFinite(jsd)).toBe(true);
    expect(jsd).toBeGreaterThan(0);
  });

  it("renormalizes drifted inputs", () => {
    const drifted = p(0.49, 0.29, 0.2); // sums to 0.98
    expect(jensenShannon([drifted, p(0.5, 0.3, 0.2)])).toBeLessThan(1e-4);
  });

  it("is invariant under model order permutation", () => {
    const dists = [p(0.6, 0.2, 0.2), p(0.3, 0.4, 0.3), p(0.1, 0.2, 0.7)];
    const reversed = [...dists].reverse();
    expect(jensenShannon(dists)).toBeCloseTo(jensenShannon(reversed), 12);
  });

  it("never exceeds ln 3 for 3-outcome distributions", () => {
    // Deterministic pseudo-random distributions (no Math.random in tests).
    let seed = 42;
    const next = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    for (let i = 0; i < 50; i++) {
      const dists = Array.from({ length: 4 }, () => {
        const [a, b, c] = [next(), next(), next()];
        const total = a + b + c;
        return p(a / total, b / total, c / total);
      });
      expect(jensenShannon(dists)).toBeLessThanOrEqual(Math.log(3) + 1e-9);
    }
  });

  it("returns 0 for fewer than two distributions", () => {
    expect(jensenShannon([])).toBe(0);
    expect(jensenShannon([p(0.5, 0.3, 0.2)])).toBe(0);
    expect(jsdNormalized([p(0.5, 0.3, 0.2)])).toBe(0);
  });

  it("grows with disagreement (monotonicity smoke test)", () => {
    const mild = jensenShannon([p(0.6, 0.2, 0.2), p(0.4, 0.3, 0.3)]);
    const hard = jensenShannon([p(0.9, 0.05, 0.05), p(0.1, 0.1, 0.8)]);
    expect(hard).toBeGreaterThan(mild);
  });
});

describe("buildConsensusRows", () => {
  it("groups model rows per fixture, sorted by disagreement descending", () => {
    const calm = fixture({ id: 1, home: "Alpha", away: "Bravo" });
    const split = fixture({ id: 2, home: "Charlie", away: "Delta" });
    const predictions = [
      predRow({ fixture_id: 1, model_version: "dc-1.0.0", probs: p(0.5, 0.3, 0.2) }),
      predRow({ fixture_id: 1, model_version: "elo-1.0.0", probs: p(0.5, 0.3, 0.2) }),
      predRow({ fixture_id: 2, model_version: "dc-1.0.0", probs: p(0.8, 0.1, 0.1) }),
      predRow({ fixture_id: 2, model_version: "elo-1.0.0", probs: p(0.2, 0.2, 0.6) }),
    ];
    const rows = buildConsensusRows([calm, split], predictions);
    expect(rows.map((r) => r.fixture.id)).toEqual([2, 1]);
    expect(rows[0]?.models).toHaveLength(2);
    expect(rows[0]?.jsdNorm).toBeGreaterThan(rows[1]?.jsdNorm ?? Number.NaN);
  });

  it("flags fixtures with fewer than two models as insufficient", () => {
    const rows = buildConsensusRows(
      [fixture({ id: 1 })],
      [predRow({ fixture_id: 1 })],
    );
    expect(rows[0]?.insufficient).toBe(true);
    expect(rows[0]?.jsd).toBe(0);
  });
});
