#!/usr/bin/env bash
# Commit state/ back to the repo (used by long-running workflows).
set -e
git config user.name "fno-alert-bot"
git config user.email "fno-alert-bot@users.noreply.github.com"
git add state/ 2>/dev/null || true
git diff --cached --quiet && exit 0
git commit -q -m "${1:-state} $(date -u +%F_%H%M)"
for i in 1 2 3; do git push -q && exit 0; git pull -q --rebase && continue; done
