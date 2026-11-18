# 1230 — Shared-checkout stale dirt: 9 regressive files block ff and would revert landed skill records

**Status:** OPEN — unowned, routed; the no-unique-content verification is COMPLETE (two independent adjudications), execution of the discard is blocked on the sibling-WIP rule (neither adjudicating lane may revert another session's tree state without an owner ask).

Found 2026-11-19 morning +07 by the i932 housekeeping lane (zed-fe7b1210@amo) during Protocol H; independently corroborated the same day by the plan-019 T4p lane (zed-eb3b7571@amo — the doc-sync census edit's author, who confirms their disk copy duplicates landed content `80111f549`).

## The finding

The shared amo checkout of this repo carries 9 UNCOMMITTED modified files whose disk content is OLDER than HEAD (stale-buffer / superseded-tool-output era; mtimes 2026-11-17 22:50 .. 2026-11-18 23:29, nothing touched since):

- `.agents/skills/boundary-guard/SKILL.md` — disk holds run-table window 511th→497th; HEAD holds 519th→505th (the 512th–519th landing wave through `80111f549`). Committing the dirt reverts EIGHT landed run rows and resurrects eight compacted ones.
- `.agents/skills/doc-sync/SKILL.md` — disk missing the landed 519-pairing / 518 / 513 run-log rows (the census-table edit already matches HEAD).
- `.agents/skills/research/vocab.md` + `.research/{055,062,117,505,603,648}_*.md` — disk missing later-landed PASS-Redirect / vocab lines (riir-train 696, Sigma-distill-spec, Cohen & Shapira, Blondel arXiv:2611.16736).

ALL NINE byte-verified 2026-11-19: **zero unique content** — every disk delta is a subset of landed history, recoverable via `git log -p` / `git log -S`.

## Why it is a landmine (live, not hypothetical)

- Any bulk commit (`git add -A` / `commit -a` / a sync lane sweeping staged state) reverts landed records — the `88e5f98` stale-buffer incident class, in the very file that documents it.
- **ff is now BLOCKED**: origin landed the 520th boundary row (`ce7235b6f`) inside `boundary-guard/SKILL.md` while the dirty copy sat — `git merge --ff-only origin/develop` refuses with "local changes would be overwritten" (measured 2026-11-19, this filing's lane). Every new upstream skill-row landing widens the divergence the eventual cleaner must unpick.
- The hazard has survived ~20h+ across multiple lanes' "dirty as-found, untouched" passes — each pass re-derives the adjudication instead of executing a recorded recipe. This file IS the recipe.

## Disposition recipe (verified, ready to execute)

- [ ] Owner ask OR hygiene-lane claim (the rule bars the two adjudicating lanes: "do not revert changes you did not make unless the user explicitly asks").
- [ ] `git fetch origin` fresh; confirm no NEW landings inside the 9 paths since this record (upstream moved twice while this was written — expect drift; re-diff).
- [ ] Re-run the 60-second no-unique verification: `git diff -- <the 9 paths>` — expect exactly the stale shapes documented above; anything else → STOP, re-adjudicate.
- [ ] `git checkout -- <the 9 paths>` (discards only stale duplicates of landed content).
- [ ] `git merge --ff-only origin/develop` (now unblocked; brings `34a5383a` + `ce7235b6f` + this issue's own landing).
- [ ] Run the owed run-table appends: the i932 lane's scoped riir-ai boundary run (512th-slot candidate, executed 2026-11-19 ~06:3x, exit 0, 89 edges) + this record's own doc-sync/boundary pairing rows — the table numbering has moved to 520th upstream, take the next free numbers at write time.

## Adjacent finding (adjudicated, NO action owed)

Orphaned local commit `f05d6fc98` (riir-refine legacy_ratchet 9→10, zed-a6237a4f@amo 11-18 22:20) — dual-landing twin of upstream `7eb3d047f` (same re-pin, the .issues/245 double-allocation). `git branch -a --contains f05d6fc98` is EMPTY: never published, superseded. Harmless object; GC collects. Recorded so no lane re-adjudicates it.
