# wc-oracle

Live prediction dashboard for the 2026 World Cup: ensemble models (Dixon-Coles / Elo / GBM),
Monte Carlo bracket simulation, immutable prediction log, and a Jarvis/Blue-Lock-style HUD.

See CLAUDE.md for architecture, hard rules, and roadmap. Predictions in `data/predictions/`
are append-only and written before kickoff — that log is the source of truth for calibration.
