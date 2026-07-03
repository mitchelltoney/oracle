import type { Fixture, PredictionRecord, Probs } from "./types";

/**
 * Model-disagreement metric: generalized Jensen–Shannon divergence over the
 * n model distributions, JSD = H(mean of dists) − mean(H(dist)).
 *
 * Chosen because it is symmetric, order-invariant across models, defined
 * for zero probabilities (0·ln 0 = 0), and bounded — unlike pairwise KL,
 * which is asymmetric and blows up on zeros. The tight upper bound for
 * distributions over k outcomes is ln(min(n, k)); with 3 outcomes that is
 * ln 3, which `jsdNormalized` uses to map onto [0, 1] for display.
 */

function normalize(p: Probs): [number, number, number] {
  const total = p.home + p.draw + p.away;
  if (total <= 0) return [1 / 3, 1 / 3, 1 / 3];
  return [p.home / total, p.draw / total, p.away / total];
}

function entropy(dist: readonly number[]): number {
  let h = 0;
  for (const p of dist) if (p > 0) h -= p * Math.log(p);
  return h;
}

/** Generalized Jensen–Shannon divergence in nats. 0 for fewer than 2 distributions. */
export function jensenShannon(dists: readonly Probs[]): number {
  if (dists.length < 2) return 0;
  const normalized = dists.map(normalize);
  const mixture = [0, 1, 2].map(
    (i) => normalized.reduce((acc, d) => acc + (d[i] ?? 0), 0) / normalized.length,
  );
  const meanEntropy =
    normalized.reduce((acc, d) => acc + entropy(d), 0) / normalized.length;
  return Math.max(0, entropy(mixture) - meanEntropy);
}

/** JSD scaled to [0, 1] by its tight bound ln(min(nModels, 3)). */
export function jsdNormalized(dists: readonly Probs[]): number {
  if (dists.length < 2) return 0;
  return jensenShannon(dists) / Math.log(Math.min(dists.length, 3));
}

/** Above this normalized JSD, the consensus view treats the split as "loud". */
export const DISAGREEMENT_LOUD_THRESHOLD = 0.15;

export interface ConsensusModel {
  model: string;
  model_version: string;
  probs: Probs;
  written_at: string;
}

export interface ConsensusRow {
  fixture: Fixture;
  models: ConsensusModel[];
  jsd: number;
  jsdNorm: number;
  /** True when fewer than 2 models have logged a prediction. */
  insufficient: boolean;
}

export function buildConsensusRows(
  fixtures: Fixture[],
  predictions: PredictionRecord[],
): ConsensusRow[] {
  const rows = fixtures.map((fixture) => {
    const models = predictions
      .filter((p) => p.fixture_id === fixture.id)
      .map((p) => ({
        model: p.model,
        model_version: p.model_version,
        probs: p.probs,
        written_at: p.written_at,
      }))
      .sort((a, b) => a.model_version.localeCompare(b.model_version));
    const dists = models.map((m) => m.probs);
    return {
      fixture,
      models,
      jsd: jensenShannon(dists),
      jsdNorm: jsdNormalized(dists),
      insufficient: models.length < 2,
    };
  });
  return rows.sort((a, b) => b.jsdNorm - a.jsdNorm);
}
