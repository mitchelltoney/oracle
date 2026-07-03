import type { PredictionRecord } from "./types";

/**
 * The append-only log keeps every model_version generation (hard rule 1: a
 * model change bumps the version; old rows stay), so /predictions can return
 * several generations of the same model family per fixture — e.g. ens-1.0.0
 * AND ens-1.0.1. "Current view" surfaces (consensus columns, bracket
 * probability splits) must let only the newest generation per FAMILY speak:
 * group by `model` and keep the row with the latest written_at. History
 * surfaces (timeline traces, report-card rows) deliberately keep every
 * generation — the version boundary is real information there.
 */
export function latestPerModel(
  rows: readonly PredictionRecord[],
): PredictionRecord[] {
  const byModel = new Map<string, PredictionRecord>();
  for (const row of rows) {
    const current = byModel.get(row.model);
    // written_at is ISO 8601 UTC with a fixed layout: string compare is safe
    if (!current || row.written_at > current.written_at) {
      byModel.set(row.model, row);
    }
  }
  return [...byModel.values()];
}
