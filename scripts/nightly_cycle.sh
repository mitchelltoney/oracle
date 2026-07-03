#!/bin/zsh
# Nightly WC Oracle data cycle: ingest -> predict -> sim, then commit ONLY the
# tracked data artifacts. Fail-fast (set -e): any stage failing aborts before
# the commit block, so partial state is never committed — consistent with the
# refresh-data skill. Rows appended before a later failure stay on disk; the
# next successful cycle commits them (the log is append-only, hard rule 1).
# Registered via ~/Library/LaunchAgents/com.wcoracle.nightly.plist.
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
cd "$(dirname "$0")/.."

log() { echo "[nightly $(date -u +%FT%TZ)] $*"; }

log "starting cycle"
make ingest
make predict
make sim

git add data/predictions data/sim
if git diff --cached --quiet; then
  log "no data changes to commit"
  exit 0
fi
n=$(git diff --cached --numstat -- data/predictions/predictions.jsonl | awk '{print $1+0}')
git commit -m "Nightly data cycle $(date -u +%F): ${n} predictions logged, sim updated"
git push origin main
log "cycle complete: ${n} predictions committed and pushed"
