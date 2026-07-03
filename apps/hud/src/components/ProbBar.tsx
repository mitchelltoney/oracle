import { pct } from "../lib/format";
import type { Probs } from "../lib/types";

/** Home / draw / away probability split as a segmented bar. */
export function ProbBar(props: { probs: Probs }) {
  const { home, draw, away } = props.probs;
  const segment = (
    kind: "home" | "draw" | "away",
    value: number,
  ) => (
    <div
      className={`probbar__seg probbar__seg--${kind}`}
      style={{ flexBasis: `${Math.max(0, value) * 100}%` }}
      title={`${kind} ${pct(value)}`}
    >
      {value >= 0.14 ? pct(value, 0) : ""}
    </div>
  );
  return (
    <div className="probbar">
      {segment("home", home)}
      {segment("draw", draw)}
      {segment("away", away)}
    </div>
  );
}
