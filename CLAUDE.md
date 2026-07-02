# WC Oracle — World Cup 2026 Prediction Dashboard

## What this is
A live prediction dashboard for the 2026 FIFA World Cup (tri-hosted: USA/Mexico/Canada, tournament in progress — knockout rounds as of July 2026). It ingests live match/player data, runs an ensemble of statistical + ML models, Monte-Carlo-simulates the remaining bracket, and renders everything in a futuristic Jarvis/Blue-Lock-style HUD (GSAP + Three.js).

The tournament is LIVE. The single most important near-term objective is a working ingestion → prediction → immutable-log pipeline, so every remaining match becomes calibration data. Polish comes after.

## Architecture
- `services/ingest/` — Python. Provider-abstracted data layer (football-data.org / API-Football / FBref adapters behind one interface). Aggressive caching, ETag/conditional requests, graceful degradation when a provider fails. All raw pulls snapshotted to `data/raw/` with timestamps.
- `services/models/` — Python. Each model is a self-contained module implementing a common `MatchModel` interface: `dixon_coles/`, `elo/`, `gbm/` (XGBoost on engineered features), `ensemble/`. Feature engineering lives in `services/models/features/` and reads only from versioned snapshots.
- `services/sim/` — Monte Carlo bracket simulator. Takes any `MatchModel`, simulates the remaining tournament N times (default 100k), handles knockout rules: extra time, penalty shootouts modeled as ~coin flips with small skill adjustment.
- `services/api/` — FastAPI. Serves predictions, model comparison, calibration metrics. SSE endpoint for live match updates (polling APIs upstream; do not pretend we have websockets from providers).
- `apps/hud/` — Vite + React + GSAP + Three.js frontend. All heavy computation stays server-side or in web workers; the UI thread only animates.
- `data/predictions/` — IMMUTABLE append-only prediction log (JSONL). See hard rules.

## Hard rules (never violate)
1. **Prediction log is append-only.** Every prediction is written with a UTC timestamp BEFORE kickoff and is never edited or deleted. Calibration metrics (Brier, log loss) are computed only from this log. If a model changes, new predictions get a new `model_version`; old rows stay.
2. **No data leakage.** Features for predicting match M may only use data with timestamps strictly before M's kickoff. Feature snapshots are versioned in `data/snapshots/`. Any new feature must state its as-of-time source.
3. **Provider abstraction.** No model or frontend code imports a data provider directly. Everything goes through the ingest interface.
4. **Never commit API keys.** Keys live in `.env` (gitignored). Fail loudly if missing.
5. **Do not modify files under `data/`** except through the ingest/logging code paths.
6. **Small-sample humility.** International football is low-data. Prefer regularized/shrunk estimates; any feature must be justified against overfitting, not just backtest wins.

## Conditions/location features (2026-specific, our differentiator)
- Altitude (Estadio Azteca ~2,240m; flag high-altitude venues)
- Heat/humidity by venue + kickoff local time; roofed/climate-controlled vs open-air
- Travel distance between a team's consecutive venues; rest-day differential
- Host-nation advantage (USA/MEX/CAN)
These are engineered features in `services/models/features/conditions.py`, each with a documented data source.

## Player layer
- Minutes-weighted club-season xG/xA aggregates per squad
- Availability: injuries, suspensions, yellow-card accumulation
- Fatigue proxy: minutes in trailing 14 days + club-season load
- Star-dependency index (share of team output from top 1–2 players)

## Commands
- `make dev` — run API + HUD dev servers
- `make ingest` — one-shot data refresh (snapshots + cache)
- `make predict` — generate + log predictions for upcoming fixtures (all models)
- `make sim` — re-run bracket Monte Carlo, write `data/sim/latest.json`
- `make backtest` — run models on historical tournaments, print calibration report
- `make test` — pytest (services) + vitest (hud)
- `make lint` — ruff + mypy (services), eslint + tsc (hud)

## Verification standards
- Show evidence, not assertions: paste test output, show the command and its result.
- Any statistical code change requires: unit tests, plus `make backtest` run with before/after Brier score.
- Frontend changes: confirm no jank — data updates must batch through a single state update per tick; GSAP timelines must be killed on unmount.
- After each meaningful change: `make lint && make test` must pass before the turn ends.

## Style
- Python 3.12, type-hinted, ruff-formatted. Pandas OK in features/backtests; hot paths (sim engine) use numpy vectorization — the 100k-run bracket sim must complete < 10s on an M5 Mac.
- TypeScript strict mode in `apps/hud/`.
- Keep diffs minimal. Do not refactor, reformat, or add docstrings to code adjacent to the task. No speculative abstractions.

## What NOT to do
- Don't add a database until flat files actually hurt. JSONL + parquet snapshots are fine for v1.
- Don't scrape aggressively; respect rate limits and cache. If a provider blocks us, degrade, don't hammer.
- Don't ship licensed team logos/badges in anything public-facing.
- Don't model penalty shootouts as strongly skill-determined.
- Don't start the HUD polish before the prediction pipeline logs real matches.

## Roadmap phases
1. **P0 (now):** ingest layer + Dixon-Coles implementation from scratch + immutable prediction log + minimal API. Every remaining knockout match gets logged predictions.
2. **P1:** Elo + GBM models, ensemble, bracket Monte Carlo, calibration dashboard endpoint.
3. **P2:** HUD — bracket view with survival probabilities, model-consensus panel, player cards, live win-probability chart.
4. **P3:** conditions/location feature study, in-match Bayesian updates, Three.js venue globe, report-card panel.
