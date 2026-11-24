---
name: leftover-sweep
description: Sweep landed and leftover git branches, worktrees, mirrors and scratch after ticket landings - delete merged branches local and remote, tear down landed worktrees and target dirs, handle default-branch-blocked deletes, with live-lane safety checks and disk accounting. Use after landing a ticket branch, when a repo's branch or worktree list has grown noisy, when a lane stopped with its teardown undone, or when disk reclaim is wanted.
---

# Leftover sweep

Reduce the noise a fleet of parallel ticket lanes leaves behind: landed ticket
branches, worktrees that outlived their lanes, mirror dirs, merge scratch and
`/tmp/<plan>` target dirs. The sweep is read-classify-delete; every delete is
gated by a liveness check.

## What never gets touched

1. **Protected branches** — whatever the repo's own docs name as the
   integration branch (and any frozen promote lane). Only ticket-prefixed
   branches (`fix/*`, `feature/*`, `chore/*`, a lane's `sprint*/…`) are
   candidates.
2. **Live lanes** — a branch/worktree stays when ANY of these hold:
   - a claim line with an unexpired lease exists for its sprint/ticket;
   - a file under the worktree was modified TODAY by another agent;
   - a process is running inside it (`lsof +D <dir>`, or a heavy entry in the
     rig status);
   - the branch tip is fresh (hours, not days) AND unmerged.
   When in doubt, mail the claiming lane before deleting; never guess.
3. **Unmerged branches** — never deleted, never force-pushed. `git branch -d`
   refuses unmerged branches; trust that refusal.

## Survey (fetch first)

```sh
git -R <repo> fetch origin --quiet
git -R <repo> worktree list
git -R <repo> for-each-ref --format='%(refname:short)' refs/heads/   # minus protected
git -R <repo> ls-remote --heads origin | grep -E 'refs/heads/(fix|feature|chore)/'
```

Classify each candidate: `git merge-base --is-ancestor <branch>
origin/<integration> && echo MERGED`. Record tip + last-commit date beside
every verdict so the report explains itself.

## Delete

- Local merged branch: `git branch -d <branch>` (it refuses unmerged).
- Remote merged branch: `git merge-base --is-ancestor <branch>
  origin/<integration> && git push origin --delete <branch>` — check ancestry
  BEFORE every remote delete; a stale remote-tracking ref lies.
- Landed branch checked out in its own worktree: remove the worktree first
  (below), prune, then `git branch -d`.

## The default-branch block

`git push origin --delete <branch>` failing with *"refusing to delete the
current branch"* means the branch is the remote's DEFAULT. Diagnose:

```sh
git ls-remote --symref origin HEAD          # what the remote says its default is
```

Then read the repo's own branch law (AGENTS.md / README "Branch" section):

- If the docs name the integration branch as default but the remote default is
  a MERGED TICKET branch, the default is misconfigured — that is also why the
  delete was refused. Fixing it needs a GitHub admin token (`gh auth login`,
  or a PAT with repo admin):
  ```sh
  gh api repos/<org>/<repo> --method PATCH -f default_branch=<integration>
  git push origin --delete <branch>
  ```
- No working token on the box → **owner-gated, park it**: report the finding
  with the exact commands above and stop. Verify a candidate token first with
  one cheap GET (`curl -H "Authorization: token $TOK" .../repos/<org>/<repo>`
  → 401 = dead, do not use).
- Never change a default branch on the strength of a guess — the repo's own
  docs must name the target branch.

## Worktrees, mirrors, scratch

For a landed (merged) ticket's leftovers:

1. Preserve receipts first: copy the branch worktree's `.gate/receipt.json`
   (+ any verdict log tail) into the main clone's durable, gitignored gate
   dir before deleting anything.
2. Liveness: `lsof +D <dir>` must be empty; check the fleet status for
   contenders on this box.
3. `git worktree remove <wt>` — when untracked bulk (`target/`) makes it
   refuse, `rm -rf <dir>` then `git worktree prune` in the parent repo. For
   linked worktrees, prune EVERY parent repo that owns a member of the mirror.
4. Sweep the known scratch classes: `../<slug>-mirror` / `<plan>-mirror` dirs,
   `merge-*` worktrees, detached-HEAD merge worktrees of landed branches,
   `/tmp/<plan>*` target dirs and gate logs (after preserving anything the
   close-out record cites).
5. Disk accounting: `df -h /` before and after; quote the df DELTA, never du —
   du lies across CoW clones, and linked worktrees share the parent's
   `.git/objects` (deleting a worktree correctly leaves those).

## Report

A close-out table: deleted (branch/worktree/dir + size), kept (thing + owner
+ reason — live lease, fresh mtime, unmerged), owner-gated (thing + the exact
unblock command). The df delta lands in the Summary's cleanup line.
