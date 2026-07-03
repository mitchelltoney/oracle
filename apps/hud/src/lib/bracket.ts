import { latestPerModel } from "./latest";
import type { Fixture, PredictionRecord, Probs, SimPayload } from "./types";

/**
 * Client-side knockout-tree construction.
 *
 * /sim only publishes per-team per-round reach probabilities — no match
 * ids, no bracket edges. The tree is reconstructed from three sources:
 *   - sim.rounds + sim.teams (round list, survival vectors, 0/1 for
 *     resolved rounds → completed-match winners),
 *   - /fixtures (upcoming pairings, carries `stage`),
 *   - /predictions (past pairings; rows carry no stage, so their round is
 *     inferred from the reach vectors).
 * Future pairings the data cannot determine are rendered as placeholder
 * slots — bracket halves are NOT guessable from marginal reach probs, so
 * they are never faked.
 */

const EPS = 1e-9;

export interface TeamSlot {
  kind: "team" | "tbd";
  team: string | null;
  placeholder: string | null;
  /** round → reach prob (keys = sim.rounds), plus "win". Null if the team is absent from sim.teams. */
  survival: Record<string, number> | null;
}

export type MatchStatus = "completed" | "upcoming" | "unknown";

export interface BracketMatch {
  /** `${round}:${index}` in kickoff order — stable across ticks for equal inputs. */
  id: string;
  round: string;
  slots: [TeamSlot, TeamSlot];
  status: MatchStatus;
  winner: string | null;
  fixtureId: number | null;
  kickoffUtc: string | null;
  /** Ensemble home/draw/away split for upcoming matches (mean of models as fallback). */
  probs: Probs | null;
  /** Ids of previous-round matches feeding each slot. */
  feeds: [string | null, string | null];
}

export interface BracketTree {
  rounds: string[];
  matchesByRound: Record<string, BracketMatch[]>;
  champion: string | null;
  generatedAt: string;
}

export function reachKey(round: string): string {
  return `reach_${round.toLowerCase()}`;
}

export function survivalOf(
  sim: SimPayload,
  team: string,
): Record<string, number> | null {
  const entry = sim.teams[team];
  if (!entry) return null;
  const out: Record<string, number> = {};
  for (const round of sim.rounds) out[round] = entry[reachKey(round)] ?? 0;
  out["win"] = entry["win"] ?? 0;
  return out;
}

/** Matches per round, derived from how many teams entered the first round. */
export function roundSizes(sim: SimPayload): Record<string, number> {
  const sizes: Record<string, number> = {};
  const first = sim.rounds[0];
  if (first === undefined) return sizes;
  const entryKey = reachKey(first);
  const entrants = Object.values(sim.teams).filter(
    (t) => (t[entryKey] ?? 0) >= 1 - EPS,
  ).length;
  let n = Math.max(1, Math.round(entrants / 2));
  for (const round of sim.rounds) {
    sizes[round] = n;
    n = Math.max(1, Math.round(n / 2));
  }
  return sizes;
}

/** Deepest round both teams are confirmed (reach == 1) to have played. */
export function inferRound(
  sim: SimPayload,
  home: string,
  away: string,
): string | null {
  const h = sim.teams[home];
  const a = sim.teams[away];
  if (!h || !a) return null;
  let deepest: string | null = null;
  for (const round of sim.rounds) {
    const key = reachKey(round);
    if ((h[key] ?? 0) >= 1 - EPS && (a[key] ?? 0) >= 1 - EPS) deepest = round;
  }
  return deepest;
}

interface Pairing {
  fixtureId: number;
  home: string;
  away: string;
  kickoff: string;
  stage: string | null;
  completed: boolean;
}

function collectPairings(
  fixtures: Fixture[],
  predictions: PredictionRecord[],
  now: Date,
): Pairing[] {
  const byId = new Map<number, Pairing>();
  for (const f of fixtures) {
    byId.set(f.id, {
      fixtureId: f.id,
      home: f.home,
      away: f.away,
      kickoff: f.kickoff_utc,
      stage: f.stage,
      completed: false,
    });
  }
  for (const p of predictions) {
    if (byId.has(p.fixture_id)) continue;
    byId.set(p.fixture_id, {
      fixtureId: p.fixture_id,
      home: p.home,
      away: p.away,
      kickoff: p.kickoff_utc,
      stage: null,
      completed: new Date(p.kickoff_utc).getTime() < now.getTime(),
    });
  }
  return [...byId.values()];
}

function nextReachKey(sim: SimPayload, round: string): string {
  const idx = sim.rounds.indexOf(round);
  const next = sim.rounds[idx + 1];
  return next === undefined ? "win" : reachKey(next);
}

function inferWinner(sim: SimPayload, round: string, pairing: Pairing): string | null {
  const key = nextReachKey(sim, round);
  const advanced = [pairing.home, pairing.away].filter(
    (team) => (sim.teams[team]?.[key] ?? 0) >= 1 - EPS,
  );
  return advanced.length === 1 ? (advanced[0] ?? null) : null;
}

function ensembleProbs(
  fixtureId: number,
  predictions: PredictionRecord[],
): Probs | null {
  // newest generation per family only: with several ensemble generations in
  // the log, matching on a version prefix would pick whichever sorts first
  const rows = latestPerModel(
    predictions.filter((p) => p.fixture_id === fixtureId),
  );
  if (rows.length === 0) return null;
  const ens = rows.find((p) => p.model === "ensemble");
  if (ens) return ens.probs;
  const sum = rows.reduce(
    (acc, p) => ({
      home: acc.home + p.probs.home,
      draw: acc.draw + p.probs.draw,
      away: acc.away + p.probs.away,
    }),
    { home: 0, draw: 0, away: 0 },
  );
  return {
    home: sum.home / rows.length,
    draw: sum.draw / rows.length,
    away: sum.away / rows.length,
  };
}

function teamSlot(sim: SimPayload, team: string): TeamSlot {
  return { kind: "team", team, placeholder: null, survival: survivalOf(sim, team) };
}

function tbdSlot(placeholder: string): TeamSlot {
  return { kind: "tbd", team: null, placeholder, survival: null };
}

export function buildBracket(input: {
  sim: SimPayload;
  fixtures: Fixture[];
  predictions: PredictionRecord[];
  now: Date;
}): BracketTree {
  const { sim, fixtures, predictions, now } = input;
  const sizes = roundSizes(sim);
  const pairings = collectPairings(fixtures, predictions, now);

  const byRound: Record<string, Pairing[]> = {};
  for (const round of sim.rounds) byRound[round] = [];
  for (const pairing of pairings) {
    const round =
      pairing.stage !== null && sim.rounds.includes(pairing.stage)
        ? pairing.stage
        : inferRound(sim, pairing.home, pairing.away);
    if (round !== null) byRound[round]?.push(pairing);
  }

  const matchesByRound: Record<string, BracketMatch[]> = {};
  for (const round of sim.rounds) {
    const known = (byRound[round] ?? []).sort((a, b) =>
      a.kickoff.localeCompare(b.kickoff),
    );
    const matches: BracketMatch[] = known.map((pairing, index) => {
      const completed = pairing.completed;
      return {
        id: `${round}:${index}`,
        round,
        slots: [teamSlot(sim, pairing.home), teamSlot(sim, pairing.away)],
        status: completed ? "completed" : "upcoming",
        winner: completed ? inferWinner(sim, round, pairing) : null,
        fixtureId: pairing.fixtureId,
        kickoffUtc: pairing.kickoff,
        probs: completed ? null : ensembleProbs(pairing.fixtureId, predictions),
        feeds: [null, null],
      };
    });
    const size = sizes[round] ?? matches.length;
    for (let index = matches.length; index < size; index++) {
      matches.push({
        id: `${round}:${index}`,
        round,
        slots: [tbdSlot("TBD"), tbdSlot("TBD")],
        status: "unknown",
        winner: null,
        fixtureId: null,
        kickoffUtc: null,
        probs: null,
        feeds: [null, null],
      });
    }
    matchesByRound[round] = matches;
  }

  linkFeeds(sim, matchesByRound);
  labelPlaceholders(sim, matchesByRound);
  orderRounds(sim.rounds, matchesByRound);

  const champion =
    Object.entries(sim.teams).find(([, t]) => (t["win"] ?? 0) >= 1 - EPS)?.[0] ??
    null;

  return { rounds: [...sim.rounds], matchesByRound, champion, generatedAt: sim.generated_at };
}

/** Point each named slot at the previous-round match its team came through. */
function linkFeeds(
  sim: SimPayload,
  matchesByRound: Record<string, BracketMatch[]>,
): void {
  for (let i = 1; i < sim.rounds.length; i++) {
    const prev = matchesByRound[sim.rounds[i - 1] ?? ""] ?? [];
    const current = matchesByRound[sim.rounds[i] ?? ""] ?? [];
    for (const match of current) {
      match.slots.forEach((slot, slotIndex) => {
        if (slot.team === null) return;
        const feeder =
          prev.find((m) => m.winner === slot.team) ??
          prev.find(
            (m) =>
              m.status === "completed" &&
              m.slots.some((s) => s.team === slot.team),
          );
        if (feeder) match.feeds[slotIndex] = feeder.id;
      });
    }
  }
}

/**
 * Label a TBD match "Winner of X–Y" only when its feeders are unambiguous:
 * exactly one open slot-pair downstream and exactly two unassigned matches
 * upstream. Anything less determined stays "TBD" — bracket halves are not
 * inferable from the sim payload.
 */
function labelPlaceholders(
  sim: SimPayload,
  matchesByRound: Record<string, BracketMatch[]>,
): void {
  for (let i = 1; i < sim.rounds.length; i++) {
    const prev = matchesByRound[sim.rounds[i - 1] ?? ""] ?? [];
    const current = matchesByRound[sim.rounds[i] ?? ""] ?? [];
    const assigned = new Set(
      current.flatMap((m) => m.feeds.filter((f): f is string => f !== null)),
    );
    const unassigned = prev.filter((m) => !assigned.has(m.id));
    const openMatches = current.filter(
      (m) => m.slots[0].kind === "tbd" && m.slots[1].kind === "tbd",
    );
    if (openMatches.length === 1 && unassigned.length === 2) {
      const target = openMatches[0];
      if (!target) continue;
      unassigned.forEach((feeder, slotIndex) => {
        const [a, b] = feeder.slots;
        if (a.team !== null && b.team !== null) {
          target.slots[slotIndex === 0 ? 0 : 1] = tbdSlot(
            `Winner of ${a.team}–${b.team}`,
          );
        }
        target.feeds[slotIndex === 0 ? 0 : 1] = feeder.id;
      });
    }
  }
}

/** Stable-sort each round by mean feeder position so connector edges don't cross. */
export function orderRounds(
  rounds: string[],
  matchesByRound: Record<string, BracketMatch[]>,
): void {
  for (let i = 1; i < rounds.length; i++) {
    const prev = matchesByRound[rounds[i - 1] ?? ""] ?? [];
    const current = matchesByRound[rounds[i] ?? ""] ?? [];
    const position = new Map(prev.map((m, index) => [m.id, index]));
    const keyed = current.map((match, index) => {
      const feederPositions = match.feeds
        .filter((f): f is string => f !== null)
        .map((f) => position.get(f))
        .filter((p): p is number => p !== undefined);
      const key =
        feederPositions.length > 0
          ? feederPositions.reduce((a, b) => a + b, 0) / feederPositions.length
          : Number.POSITIVE_INFINITY;
      return { match, key, index };
    });
    keyed.sort((a, b) => a.key - b.key || a.index - b.index);
    matchesByRound[rounds[i] ?? ""] = keyed.map((k) => k.match);
  }
}
