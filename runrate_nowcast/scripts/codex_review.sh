#!/usr/bin/env bash
# Independent review of the dev branch by the Codex CLI. Run it on your own machine, where
# `codex` is installed and logged in. It works in its own clone (default ~/runrate-review),
# so your other checkouts are not touched. The review is saved under runrate_nowcast/reviews/,
# then committed and pushed to the dev branch so the Claude session can read it.
#
# Usage (from anywhere):  bash codex_review.sh [MODEL]
#   MODEL: the model name as Codex shows it (optional; Codex's default if omitted)
set -euo pipefail
BRANCH=claude/loving-hamilton-n0aajf
MODEL="${1:-}"
DIR="${REVIEW_DIR:-$HOME/runrate-review}"
[ -d "$DIR/.git" ] || git clone -q https://github.com/yb235/IterResearch.git "$DIR"
cd "$DIR"
git fetch -q origin main "$BRANCH"
git checkout -q -B "$BRANCH" "origin/$BRANCH"
SHA=$(git rev-parse --short HEAD)
OUT="runrate_nowcast/reviews/codex_$(date -u +%Y-%m-%dT%H%M)_${SHA}.md"
{
  echo "# Codex review of $BRANCH at $SHA (base origin/main)"
  echo
  echo "model: ${MODEL:-codex default}; run at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  codex exec -s read-only ${MODEL:+-m "$MODEL"} - < runrate_nowcast/reviews/REVIEW_BRIEF.md
} 2>&1 | tee "$OUT"
git add "$OUT"
git commit -q -m "Codex review of $SHA" && git push -q origin "$BRANCH"
echo "saved and pushed: $OUT"
