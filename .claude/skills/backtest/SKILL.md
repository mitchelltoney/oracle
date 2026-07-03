---
name: backtest
description: Run the historical calibration backtest for all models and report Brier score / log loss vs the previous run. Use after any change to model or feature code.
disable-model-invocation: true
---

Run the calibration backtest and report results. Arguments (optional): $ARGUMENTS may name specific models (e.g. "dixon_coles elo") — default is all models.

Steps:
1. Run `make backtest` (add MODEL=<name> filters if arguments were given). If the target doesn't exist yet, run the backtest entrypoint directly under services/models/ and say so.
2. Capture the output table: per model — Brier score, log loss, calibration slope/intercept if available, number of matches evaluated.
3. Compare against the most recent committed backtest report (data/sim/backtest_report.json or the last table in git history). Show a before/after delta per model.
4. Flag regressions: any model whose Brier score worsened by more than 0.005, or whose evaluated-match count changed unexpectedly (a count change usually means a data or leakage bug, not a modeling change).
5. Write the new report to data/sim/backtest_report.json.
6. Verdict line: IMPROVED / NEUTRAL / REGRESSED, with one sentence of interpretation. Do not spin a regression as acceptable without evidence.

Paste the actual command output as evidence — never summarize numbers you didn't see.
