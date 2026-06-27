#!/usr/bin/env bash
# scripts/release.sh — manually trigger release-plz via GitHub Actions.
#
# This is a thin wrapper around `gh workflow run release-plz.yml`. The actual
# release logic (version bump, changelog, cargo publish, git tag, GitHub
# release) runs in CI so your local machine never needs cargo-registry creds.
#
# Usage:
#   ./scripts/release.sh             # default: release-pr (run on develop)
#   ./scripts/release.sh release-pr  # create / update the release PR
#   ./scripts/release.sh release     # publish unpublished katgpt-core versions
#
# Prerequisites (one-time):
#   brew install gh && gh auth login
#
# See README.md § "Releasing & Deploying" for the full dev→deploy flow.
set -euo pipefail

CMD="${1:-release-pr}"
case "$CMD" in
  release-pr|release) ;;
  *)
    cat >&2 <<'EOF'
usage: ./scripts/release.sh [release-pr|release]
  release-pr  create/update the "Prepare release" PR (must be on develop)
  release     publish unpublished katgpt-core versions (must be on main)
EOF
    exit 1
    ;;
esac

# ── Branch guard ────────────────────────────────────────────────────────
# release-pr accumulates changes on develop; release publishes from main.
BRANCH="$(git branch --show-current)"
case "$CMD" in
  release-pr) EXPECTED="develop" ;;
  release)    EXPECTED="main" ;;
esac

if [[ "$BRANCH" != "$EXPECTED" ]]; then
  echo "error: '$CMD' must run on '$EXPECTED' (currently on '$BRANCH')" >&2
  echo "hint:  git checkout $EXPECTED" >&2
  exit 1
fi

# ── Tooling guard ───────────────────────────────────────────────────────
if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not installed." >&2
  echo "hint:  brew install gh && gh auth login" >&2
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "error: not authenticated with GitHub." >&2
  echo "hint:  gh auth login" >&2
  exit 1
fi

# ── Push local commits (in case develop/main is ahead of origin) ────────
git push -u origin "$BRANCH"

# ── Trigger the workflow via workflow_dispatch ──────────────────────────
echo "→ triggering release-plz '$CMD' on '$BRANCH'..."
gh workflow run release-plz.yml --ref "$BRANCH" -f command="$CMD"

# ── Watch the run ───────────────────────────────────────────────────────
# Give the run a moment to register before we query it.
sleep 3
RUN_ID="$(gh run list \
  --workflow=release-plz.yml \
  --branch="$BRANCH" \
  --limit=1 \
  --json databaseId \
  --jq '.[0].databaseId')"

if [[ -z "$RUN_ID" ]]; then
  echo "⚠ could not locate the run; check the Actions tab manually." >&2
  exit 0
fi

echo "→ watching run $RUN_ID..."
gh run watch "$RUN_ID" --exit-status
echo "✓ done: $(gh run view "$RUN_ID" --json url --jq '.url')"
