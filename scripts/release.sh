#!/usr/bin/env bash
# scripts/release.sh — ship the next katgpt-core version. One command.
#
# What it does (from develop):
#   1. Finds the open release-plz PR (auto-created by CI on each develop push)
#   2. Auto-merges it into develop (merge commit, never squash)
#   3. Promotes develop → main (fast-forward)
#   4. CI auto-publishes katgpt-core to crates.io on the main push
#
# Usage:
#   ./scripts/release.sh             # ship it (from develop)
#   ./scripts/release.sh --publish   # just trigger the CI publish job (from main)
#
# Prerequisites (one-time):
#   brew install gh && gh auth login
#
# See README.md § "Releasing & Deploying" for the full flow.
set -euo pipefail

# ── Subcommand: --publish (manual CI publish trigger from main) ────────
if [[ "${1:-}" == "--publish" ]]; then
  BRANCH="$(git branch --show-current)"
  [[ "$BRANCH" == "main" ]] || { echo "error: --publish runs from main (on $BRANCH)" >&2; exit 1; }
  command -v gh >/dev/null 2>&1 || { echo "error: brew install gh" >&2; exit 1; }
  gh auth status >/dev/null 2>&1 || { echo "error: gh auth login" >&2; exit 1; }
  git push -u origin main
  echo "→ triggering release-plz release on main..."
  gh workflow run release-plz.yml --ref main -f command=release
  sleep 3
  RUN_ID="$(gh run list --workflow=release-plz.yml --branch=main --limit=1 --json databaseId --jq '.[0].databaseId')"
  [[ -n "$RUN_ID" ]] && gh run watch "$RUN_ID" --exit-status
  exit 0
fi

# ── Default: full ship flow (from develop) ────────────────────────────
BRANCH="$(git branch --show-current)"
[[ "$BRANCH" == "develop" ]] || { echo "error: run from develop (on $BRANCH)" >&2; exit 1; }

command -v gh >/dev/null 2>&1 || { echo "error: brew install gh" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "error: gh auth login" >&2; exit 1; }

# Find the open release-plz PR (branch starts with "release-plz-")
PR_JSON="$(gh pr list --state open --json number,title,headRefName \
  --jq '[.[] | select(.headRefName | startswith("release-plz"))] | .[0] // empty')"

if [[ -z "$PR_JSON" ]]; then
  echo "ℹ no open release PR. Commit a feat:/fix: on develop to trigger one." >&2
  exit 0
fi

PR_NUMBER="$(printf '%s' "$PR_JSON" | jq -r '.number')"
PR_TITLE="$(printf '%s' "$PR_JSON" | jq -r '.title')"

echo "→ found release PR #$PR_NUMBER: $PR_TITLE"

# Merge the PR into develop (merge commit — release-plz needs it for detection)
echo "→ merging PR #$PR_NUMBER into develop..."
gh pr merge "$PR_NUMBER" --merge --delete-branch

# Pull the merged develop
git pull origin develop

# Promote develop → main
echo "→ promoting develop → main..."
git checkout main
git pull origin main
git merge --ff-only develop
git push origin main

# Switch back to develop for continued work
git checkout develop

echo ""
echo "✓ shipped. CI is publishing katgpt-core to crates.io."
echo "  → https://github.com/katopz/katgpt-rs/actions"
echo "  → https://crates.io/crates/katgpt-core (live once CI finishes)"
