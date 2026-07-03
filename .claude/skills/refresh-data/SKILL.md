---
name: refresh-data
description: Full data cycle — ingest latest results and fixtures, snapshot, regenerate predictions for upcoming matches, re-run the bracket simulation. Use before matchdays or when data looks stale.
disable-model-invocation: true
---

Run the full data refresh cycle and verify each stage with evidence.

Steps:
1. `make ingest` — pull latest results, fixtures, and player availability through the provider abstraction. Confirm new snapshot files appeared in data/raw/ (show the ls with timestamps). If a provider fails, report which one and confirm the fallback/cache path was used — do not retry aggressively against a rate limit.
2. `make predict` — generate predictions for all upcoming fixtures across all models. Verify: new lines appended to data/predictions/ with UTC timestamps that are BEFORE each fixture's kickoff, and a model_version on every row. Paste one sample JSONL line.
3. `make sim` — re-run the bracket Monte Carlo. Confirm data/sim/latest.json was rewritten and show the top-5 tournament-winner probabilities as a sanity check.
4. Sanity checks before finishing:
   - No fixture in the past received a new prediction (that would be leakage into the log).
   - Prediction probabilities per match sum to ~1.0.
   - The number of remaining teams in the sim matches the real bracket state.
5. Summarize: fixtures updated, predictions logged (count), notable probability movements since last run.

If any stage fails, stop, report the failure with the actual error output, and do not run later stages against stale data.
