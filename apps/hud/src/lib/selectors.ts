import { buildBracket, type BracketTree } from "./bracket";
import { buildConsensusRows, type ConsensusRow } from "./consensus";
import type { CalibrationRow, HudState, TimelinePoint } from "./types";

/**
 * All derived data is computed here, memoized per state object. The state
 * is replaced exactly once per tick, so a WeakMap keyed on it yields one
 * computation per selector per tick — nothing heavy runs in render.
 */

function memo1<R>(fn: (s: HudState) => R): (s: HudState) => R {
  const cache = new WeakMap<HudState, R>();
  return (s) => {
    if (cache.has(s)) return cache.get(s) as R;
    const value = fn(s);
    cache.set(s, value);
    return value;
  };
}

export const selectBracket = memo1((s: HudState): BracketTree | null =>
  s.sim === null
    ? null
    : buildBracket({
        sim: s.sim,
        fixtures: s.fixtures,
        predictions: s.predictions,
        now: new Date(s.lastTickAt ?? Date.now()),
      }),
);

export const selectConsensusRows = memo1((s: HudState): ConsensusRow[] =>
  buildConsensusRows(s.fixtures, s.predictions),
);

export interface RankedCalibrationRow extends CalibrationRow {
  rank: number;
}

export const selectCalibrationRanked = memo1(
  (s: HudState): RankedCalibrationRow[] =>
    [...s.calibration]
      .sort((a, b) => a.brier - b.brier || a.log_loss - b.log_loss)
      .map((row, index) => ({ ...row, rank: index + 1 })),
);

export interface TimelineFixtureOption {
  fixtureId: number;
  label: string;
  kickoff: string;
  pointCount: number;
}

export const selectTimelineFixtures = memo1(
  (s: HudState): TimelineFixtureOption[] => {
    const labels = new Map<number, { label: string; kickoff: string }>();
    for (const p of s.predictions) {
      labels.set(p.fixture_id, {
        label: `${p.home} – ${p.away}`,
        kickoff: p.kickoff_utc,
      });
    }
    for (const f of s.fixtures) {
      labels.set(f.id, { label: `${f.home} – ${f.away}`, kickoff: f.kickoff_utc });
    }
    const ids = new Set<number>([
      ...Object.keys(s.timeline).map(Number),
      ...labels.keys(),
    ]);
    return [...ids]
      .map((fixtureId) => {
        const meta = labels.get(fixtureId);
        const series = s.timeline[fixtureId] ?? {};
        const pointCount = Object.values(series).reduce(
          (acc, points) => acc + points.length,
          0,
        );
        return {
          fixtureId,
          label: meta?.label ?? `fixture ${fixtureId}`,
          kickoff: meta?.kickoff ?? "",
          pointCount,
        };
      })
      .sort((a, b) => b.kickoff.localeCompare(a.kickoff));
  },
);

export interface TimelineSeries {
  modelVersion: string;
  points: readonly TimelinePoint[];
}

/** Per-fixture series; memoized per (state, fixtureId). */
const timelineCache = new WeakMap<HudState, Map<number, TimelineSeries[]>>();

export function selectTimelineSeries(
  s: HudState,
  fixtureId: number,
): TimelineSeries[] {
  let perFixture = timelineCache.get(s);
  if (!perFixture) {
    perFixture = new Map();
    timelineCache.set(s, perFixture);
  }
  const cached = perFixture.get(fixtureId);
  if (cached) return cached;
  const series = Object.entries(s.timeline[fixtureId] ?? {})
    .map(([modelVersion, points]) => ({ modelVersion, points }))
    .sort((a, b) => a.modelVersion.localeCompare(b.modelVersion));
  perFixture.set(fixtureId, series);
  return series;
}
