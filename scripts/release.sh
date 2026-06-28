#!/usr/bin/env bash
# scripts/release.sh — ship the next katgpt-core version. One command.
#
# What it does (from develop):
#   1. If no release PR is open, triggers release-plz to create one (and waits)
#   2. Merges the release PR into develop (merge commit, never squash)
#   3. Promotes develop → main (fast-forward)
#   4. Explicitly triggers the publish job on main (push alone does NOT fire it)
#   5. Waits for publish (cargo publish to crates.io + tag + GitHub release)
#
# The release-plz workflow itself ONLY fires on workflow_dispatch — pushes to
# develop/main do nothing. This script is the sole entry point for releases.
#
# Usage:
#   ./scripts/release.sh             # ship it (from develop)
#   ./scripts/release.sh --publish   # just trigger the CI publish job (from main)
#
# Prerequisites (one-time):
#   brew install gh && gh auth login
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

# Push any unpushed develop commits (so release-plz sees the latest)
git push origin develop 2>/dev/null || true

find_release_pr() {
  gh pr list --state open --json number,title,headRefName \
    --jq '[.[] | select(.headRefName | startswith("release-plz"))] | .[0] // empty'
}

# ── Step 1: ensure a release PR exists ─────────────────────────────────
PR_JSON="$(find_release_pr)"

if [[ -z "$PR_JSON" ]]; then
  echo "→ no open release PR. Triggering release-plz release-pr..."
  gh workflow run release-plz.yml --ref develop -f command=release-pr

  # Wait for the run to register, then watch it
  sleep 5
  RUN_ID="$(gh run list --workflow=release-plz.yml --branch=develop \
    --event=workflow_dispatch --limit=1 --json databaseId --jq '.[0].databaseId')"

  if [[ -n "$RUN_ID" ]]; then
    echo "→ waiting for release-pr job (run #$RUN_ID)..."
    gh run watch "$RUN_ID" --exit-status
  fi

  # Re-check for the PR
  PR_JSON="$(find_release_pr)"
fi

if [[ -z "$PR_JSON" ]] || [[ "$PR_JSON" == "null" ]]; then
  echo "ℹ nothing to release — no version-worthy changes since last release." >&2
  exit 0
fi

PR_NUMBER="$(printf '%s' "$PR_JSON" | jq -r '.number')"
PR_TITLE="$(printf '%s' "$PR_JSON" | jq -r '.title')"

echo "→ found release PR #$PR_NUMBER: $PR_TITLE"

# ── Step 2: merge the PR (merge commit, not squash) ───────────────────
echo "→ merging PR #$PR_NUMBER into develop..."
gh pr merge "$PR_NUMBER" --merge --delete-branch

# ── Step 3: pull the merged develop ───────────────────────────────────
git pull origin develop

# ── Step 4: promote develop → main ────────────────────────────────────
echo "→ promoting develop → main..."
git checkout main
git pull origin main
git merge --no-ff develop -m "release: promote develop to main"
git push origin main

# ── Step 5: trigger publish on main (push alone no longer auto-fires) ─
echo "→ triggering release-plz release on main..."
gh workflow run release-plz.yml --ref main -f command=release
sleep 3
RUN_ID="$(gh run list --workflow=release-plz.yml --branch=main \
  --event=workflow_dispatch --limit=1 --json databaseId --jq '.[0].databaseId')"
[[ -n "$RUN_ID" ]] && gh run watch "$RUN_ID" --exit-status

# Switch back to develop for continued work
git checkout develop

echo ""
echo "✓ shipped. CI is publishing katgpt-core to crates.io."
echo "  → https://github.com/katopz/katgpt-rs/actions"
echo "  → https://crates.io/crates/katgpt-core (live once CI finishes)"
