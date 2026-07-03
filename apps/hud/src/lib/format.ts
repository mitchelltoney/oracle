export function pct(p: number, digits = 1): string {
  return `${(p * 100).toFixed(digits)}%`;
}

const ROUND_LABELS: Record<string, string> = {
  LAST_32: "ROUND OF 32",
  LAST_16: "ROUND OF 16",
  QUARTER_FINALS: "QUARTER-FINALS",
  SEMI_FINALS: "SEMI-FINALS",
  FINAL: "FINAL",
};

export function roundLabel(round: string): string {
  return ROUND_LABELS[round] ?? round.replaceAll("_", " ");
}

export function kickoffLabel(iso: string | null): string {
  if (iso === null) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });
}

export function clockLabel(at: number): string {
  return new Date(at).toISOString().slice(11, 19);
}

/** Stable display color per model family. */
const MODEL_COLORS: [prefix: string, color: string][] = [
  ["dc", "var(--model-dc)"],
  ["elo", "var(--model-elo)"],
  ["gbm", "var(--model-gbm)"],
  ["ens", "var(--model-ens)"],
];

export function modelColor(modelVersion: string): string {
  for (const [prefix, color] of MODEL_COLORS) {
    if (modelVersion.startsWith(prefix)) return color;
  }
  return "var(--accent)";
}
