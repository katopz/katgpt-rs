#!/bin/sh
# sync_commit.sh — doc-sync's mechanical commit gate.
#
# Makes the standing lesson ("the staged-index check must GATE the commit, not
# precede it") the ONLY commit path for this lane. The #42 incident class: a
# run-log commit on a shared checkout swept 17 files of stale sibling staged
# content (another box's index state left in the shared tree), reverting
# landed work. Prose did not stop it; this does:
#
#   1. REFUSES a non-empty index up front — stale sibling state is adjudicated
#      first, never swept (the incident's exact entry condition). Never
#      auto-clear: the staged content may be a live sibling's work.
#   2. Stages ONLY the named paths.
#   3. REFUSES if anything is staged beyond the named pathspecs (a sibling
#      `git add` landing concurrently between (1) and (3)).
#   4. Commits via the PARTIAL form `git commit -- <paths>` — a temporary
#      index built from HEAD + the named paths — so anything staged
#      concurrently between (3) and (4) stays in the real index, never the
#      commit, and the real index is never disturbed.
#
# Usage: sync_commit.sh <repo-dir> <commit-message> <path>...
#   Paths are git pathspecs relative to the repo root (directories allowed).
#   Name exactly the paths the run edited — never a directory you did not
#   touch.

set -eu

[ "$#" -ge 3 ] || {
  echo "usage: sync_commit.sh <repo-dir> <commit-message> <path>..." >&2
  exit 2
}

repo="$1"
msg="$2"
shift 2

[ -d "$repo/.git" ] || [ -f "$repo/.git" ] || {
  echo "sync_commit: not a git repo: $repo" >&2
  exit 2
}

# Gate 1 — the index must start empty.
stale=$(git -C "$repo" diff --cached --name-only)
if [ -n "$stale" ]; then
  echo "sync_commit: REFUSED — staged index not empty in $repo (stale sibling state; adjudicate before committing):" >&2
  echo "$stale" >&2
  exit 1
fi

git -C "$repo" add -- "$@"

# Gate 2 — everything staged must match the named pathspecs.
all_staged=$(git -C "$repo" diff --cached --name-only)
scoped=$(git -C "$repo" diff --cached --name-only -- "$@")
if [ "$all_staged" != "$scoped" ]; then
  echo "sync_commit: REFUSED — staged set exceeds the named paths (concurrent sibling staging?):" >&2
  echo "--- all staged:" >&2
  echo "$all_staged" >&2
  echo "--- within named paths:" >&2
  echo "$scoped" >&2
  exit 1
fi

# Partial commit — temporary index from HEAD + the named paths.
git -C "$repo" commit -m "$msg" -- "$@"

hash=$(git -C "$repo" rev-parse --short HEAD)
echo "sync_commit: $hash — named paths only; real index untouched."
