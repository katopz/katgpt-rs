# Issue 943: Issue-level claim lease before implementation (pre-work in-flight markers)

**Status:** Open — P2 multi-agent collision defense; rescoped at verdict round 1 (semantic-search v1 → claim lease; v2 deferred); sourced from the autoresearch rails distill (riir-refine/.research/272)
**Claimed:** zed-idle-943, 1792085272, expires 2026-10-17T17:27:52Z
**Date:** 2026-10-15
**Source:** this repo's Issue 825 + a private-tier same-day collision (fixture details in riir-refine/.research/272, private); the lease SHAPE comes from the riir-refine mining-queue claim markers (48 h expiry, push-as-arbiter) and CrocSwap/integer-mult-bounds' CLAIM packages @ `0fdd7d82` (Apache-2.0)

## Problem

Both paid multi-agent collisions in this workspace were SAME-ISSUE collisions, not semantic ones:

- **Issue 825 (this repo):** two sessions implemented ONE issue four hours apart; the second measured a "failure" against the first's in-flight state and recorded a false negative.
- **The private-lane collision (2026-10-15):** both boxes implemented the SAME sibling-repo issue the same day; one push landed, the other was rejected mid-flight after 800+ tests were already green. The yield per the claim-marker protocol was correct — but the protocol only ran AT PUSH TIME, after the duplicate work was done.

The gap is not vocabulary and not file sets: nothing marks an issue or plan as IN-FLIGHT before a second session starts on it. The mining queue has claim markers (claim-first, 48 h expiry); `.issues/` and `.plans/` files — the surfaces concurrent sessions actually allocate against — have none. Push-as-arbiter discovers collisions at push time, which is exactly one full implementation too late.

## Deliverables

- [ ] The lease convention: a session starting work on an `.issues/NNN_*.md` or
  `.plans/NNN_*.md` item appends `Claimed: <session>, <epoch>, expires <UTC>`
  under the Status line and commits + pushes BEFORE implementation starts
  (same shape as the mining queue's claim markers; the `Session: <name>,
  <epoch>` commit-body rule already exists — the lease reuses its vocabulary).
  Re-claiming an expired lease is a normal re-read of the file, not a ceremony.
- [ ] `scripts/claim_lease_check.py` — the session's FIRST step, cross-repo
  callable exactly like `dual_allocation_gate.py`, in the workspace's shell
  form (`git fetch origin && python3 ../katgpt-rs/scripts/claim_lease_check.py [--item NNN]`):
  the script itself NEVER fetches (instruments are observers, not ref
  writers) — freshness comes from the caller's fetch, and the script reports
  the age of `FETCH_HEAD` on every run so a stale fetch cannot read as a
  clean scan. It scans `.issues/` + `.plans/` for live (unexpired) leases;
  prints each live lease (file, session, epoch, expiry) and WARNs when the
  target item is live-leased by a different epoch. Exit 0 when clean or
  the lease is your own epoch; exit 1 only when the instrument cannot read
  (blind = loud, never a silent pass). It WARNS, never blocks — the push stays
  the arbiter (the private-lane yield proves post-hoc arbitration is correct
  behavior; the lease only moves discovery from push time to start time).
- [ ] Fixtures: this repo's Issue-825 pair must fire (a lease line on that
  issue would have warned the second session at start — assert against the
  recorded two-session account in this repo's AGENTS.md). The private-tier
  collision is the sibling repo's own fixture, tracked in the private distill
  note — not in this public file. An instrument whose own fixture does not
  fire is unproven (arm-reach discipline).
- [ ] Wire: one line in the riir-refine loop skill + one line beside the
  substrate-first gate in this repo's AGENTS.md ("run claim_lease_check
  before implementing any filed issue").
- [-] Deferred v2 — semantic term-overlap search (`claim_overlap_search.py`):
  ONLY on a genuinely semantic fixture — a collision with DIFFERENT issue
  numbers AND different vocabulary that the lease would have missed. Until
  one exists the class is hypothetical (the §1.55 "validates our design"
  trap); do not build it speculatively.

## Non-goals

- Leases are not locks: no hard blocking, no branch protection, no hook. A
  stale lease yields to a landed push exactly as today.
- No new state store: the lease line lives in the issue/plan file itself —
  no sidecar DB, no new ledger to drift.
- No semantic search in v1 (see the deferred v2 row).
