import type { BracketMatch, TeamSlot } from "../lib/bracket";
import { kickoffLabel, pct, roundLabel } from "../lib/format";
import { selectBracket } from "../lib/selectors";
import { useStore } from "../lib/store";
import { ProbBar } from "../components/ProbBar";
import { Ticker } from "../components/Ticker";

function slotSurvival(slot: TeamSlot, match: BracketMatch, rounds: string[]): number | null {
  if (slot.survival === null) return null;
  // "Lit path" value: probability of surviving past this match, i.e.
  // reaching the next round (or winning the tournament, for the final).
  const idx = rounds.indexOf(match.round);
  const nextRound = rounds[idx + 1];
  return nextRound === undefined
    ? (slot.survival["win"] ?? 0)
    : (slot.survival[nextRound] ?? 0);
}

function Slot(props: {
  slot: TeamSlot;
  match: BracketMatch;
  rounds: string[];
}) {
  const { slot, match, rounds } = props;
  const survival = slotSurvival(slot, match, rounds);
  const isWinner = match.winner !== null && match.winner === slot.team;
  const isEliminated =
    match.status === "completed" && match.winner !== null && !isWinner;
  const modifier = isWinner
    ? " bracket-slot--winner"
    : isEliminated
      ? " bracket-slot--eliminated"
      : slot.kind === "tbd"
        ? " bracket-slot--tbd"
        : "";
  return (
    <div className={`bracket-slot${modifier}`}>
      <span className="bracket-slot__name">
        {slot.team ?? slot.placeholder ?? "TBD"}
      </span>
      {survival !== null && match.status !== "completed" && (
        <Ticker className="bracket-slot__prob" value={survival} format={pct} />
      )}
      {isWinner && <span className="bracket-slot__prob">ADV</span>}
    </div>
  );
}

function MatchCard(props: { match: BracketMatch; rounds: string[] }) {
  const { match, rounds } = props;
  return (
    <div className={`bracket-match bracket-match--${match.status}`}>
      <div className="bracket-match__meta">
        <span>
          {match.status === "completed"
            ? match.winner !== null
              ? "FULL TIME"
              : "FT — AWAITING SIM"
            : match.status === "upcoming"
              ? kickoffLabel(match.kickoffUtc)
              : "SLOT OPEN"}
        </span>
        {match.fixtureId !== null && <span>#{match.fixtureId}</span>}
      </div>
      <Slot slot={match.slots[0]} match={match} rounds={rounds} />
      <Slot slot={match.slots[1]} match={match} rounds={rounds} />
      {match.probs !== null && match.status === "upcoming" && (
        <ProbBar probs={match.probs} />
      )}
    </div>
  );
}

export function BracketView() {
  const bracket = useStore(selectBracket);
  const simStatus = useStore((s) => s.endpoints.sim);

  if (bracket === null) {
    return (
      <div className="empty-state">
        {simStatus.error !== null
          ? `sim endpoint unreachable // ${simStatus.error}`
          : "no bracket simulation yet — run "}
        {simStatus.error === null && <code>make sim</code>}
      </div>
    );
  }

  return (
    <div>
      <div className="bracket">
        {bracket.rounds.map((round) => (
          <div key={round} className="bracket-round">
            <div className="bracket-round__title">{roundLabel(round)}</div>
            <div className="bracket-round__list">
              {(bracket.matchesByRound[round] ?? []).map((match) => (
                <MatchCard key={match.id} match={match} rounds={bracket.rounds} />
              ))}
            </div>
          </div>
        ))}
      </div>
      {bracket.champion !== null && (
        <div className="bracket__champion">
          ★ CHAMPION // {bracket.champion.toUpperCase()} ★
        </div>
      )}
    </div>
  );
}
