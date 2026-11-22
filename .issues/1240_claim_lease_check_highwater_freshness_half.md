# Issue 1240 — claim_lease_check gains a highwater-freshness half: the pre-collision state is checkable at the moment lanes actually run a gate

**Status:** OPEN — filed 2026-11-22 (zed-amo-071num@amo, from the numbering-hygiene repair run). The repo layout: this is a `scripts/*.py` gate-adjacent tool, no cargo surface.
**Claimed:** sprint-071-idle@amo, 1795366755, expires 2026-11-24T16:59Z

## The pattern (four incidents in five days, every one reactive)

Each repair landed only after a sweep or a collision forced it:

1. seal-sprints `de2bc7a` — sprint!108 allocated TWICE the same day (`d0ccd82` t9_gate_snap, then `b6edcfd` verify_lanes reading the stale 107); `.issues/.highwater` sat at 107 through both.
2. seal-remake `55c21e96` — sprint!232 allocated (`8680ca68`), counter left at 231; caught by the numbering sweep's mmorpg-remake alias row.
3. katgpt-rs `88c360627` — the seal-sprints collision-channel ratchet 3→4 (the same 108 pair, the collision the stale counter produced).
4. katgpt-rs `f79df6081` — the fleet floors re-pin pass (9 ratchet breaches, 7 repos) that measured the accumulated drift.

The bump law exists (Issue 1205's one-line runbook: the bump rides the allocation commit) and `numbering_gate.py` catches the stale counter — but only when a lane happens to run it. `claim_lease_check.py` is the gate lanes DO run at claim/allocation time (AGENTS.md mandates it before implementing any filed issue).

## The ask (T1)

Extend `scripts/claim_lease_check.py` with a highwater-freshness half: when invoked with `--item NNN` (or `--lint <file>` under a numbered directory), read the enclosing directory's `.highwater` and compare against the directory's max allocated number (a cheap `ls` of the numbered dir — same walk the numbering gate already does). Verdicts:

- counter ≥ max → clean (silent);
- counter < max → **the pre-collision state** — print a LOUD warning line (`⚠ .issues/.highwater 107 < max 108 — the next allocation collides; bump the counter in the same commit as your allocation`) and, for `--item` mode, exit non-zero UNLESS `--allow-stale-highwater` is passed (an escape hatch for a lane that genuinely cannot bump — e.g. a read-only lease check on a foreign repo; the default stays refuse-loud because a claim is the moment before an allocation).

Design constraints:
- No new dependency, no cargo surface; pure stdlib like the rest of the script.
- The check rides the EXISTING invocation paths — never a separate script (the sweep-family lesson: a rule in one instrument is a rule that has not generalised).
- `--self-test` gains two fixture arms (counter-fresh = silent; counter-stale = the warning + the exit).
- katgpt-rs's own `scripts/docs_gate.sh` population floors must not regress (the script is already in the docs-gate reachability closure — no new file).

## Non-goals

No pre-commit hook (the fleet's repos share hooks unevenly; the claim-time check is where lanes already look). No change to `numbering_gate.py` itself (it stays the post-hoc authority).

## Evidence base

The four repairs above are the falsifiable premise: each collision/reset incident had a lane that ran `claim_lease_check.py` (or should have) hours before the sweep caught the drift. This check converts the catch from reactive (sweep, hours-days later) to alloc-time (the claim, minutes before the write).
