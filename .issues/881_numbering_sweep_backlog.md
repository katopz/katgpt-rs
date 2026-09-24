# Issue 881 — the numbering sweep's standing backlog: 3 repos over their ratchets

**Status:** OPEN — measured, deliberately NOT re-pinned.
**Filed:** 2026-09-24, from the citation-sweep campaign's neighbouring run.
**Instrument:** `scripts/numbering_drift_sweep.py` ·
pins `scripts/numbering_drift_floors.txt`.

## Why this is an issue and not a pin edit

Three repos breach their ratchets and each breach is a *measured event*, which
is what this file's own pin format says every bump must be. AGENTS.md's rule —
*"a pin file re-typed after every run is a diary, not a wall"* — forbids
absorbing them by re-typing the numbers, and re-typing is exactly what the
pressure at the end of a red run is. So they are recorded here, at their
measured values, with what is and is not known about each.

Measured 2026-09-24, after `scripts/fetch_contract_repos.py` (so every
staleness reading below is supported) and after `git merge --ff-only` on the
repos that were behind and tracked-clean:

| repo | column | pinned | measured | posture |
|---|---|---:|---:|---|
| riir-ai | `max_resets` | 9 | **26** | UNREAD — needs the per-reset walk |
| riir-shader | `max_hist` | 0 | **5** | READ, one systematic cause |
| mmorpg-editor | `max_resets` | 5 | **6** | READ-ONLY here per the owner rule |

Two more were resolved rather than pinned and are recorded so the next reader
does not re-derive them:

- **riir-reflex** — the stale `.plans` allocator was a REAL defect (counter at
  1 over a live `002_ane_lane_bench.md`, so the next allocation would have
  recycled 002) and was FIXED at riir-reflex `c9edb39`, not pinned. Its one
  historical collision WAS pinned, with the reason in the row.
- **riir-train** — its `.issues` stale-allocator row sits on an UNCOMMITTED
  file, so the sweep displays it and does not adjudicate it (Issue 797's
  split). Another session's in-flight edit; not ours to touch.

## T1 — riir-ai: read the 17 new resets before pinning anything

The row's pinned 9 name their specimens (`Issue 627→614 ×2`, `763→762`,
`Bench 641→640 ×2`, …) and were measured 2026-09-05. Nineteen days later the
walk reports **26**, so ~17 resets have landed since and **none has been
looked at**. A reset means a commit (or a merge resolution) landed a
`.highwater` BELOW its parents' max, and the re-climb **re-spends numbers** —
which is the mechanism that produces the historical collisions column two
tables over, so this is upstream of a defect class and not bookkeeping.

- [ ] Enumerate the resets with `--hist-rows` and diff them against the nine
  the row already names.
- [ ] For each new one, say whether it is a merge resolution taking the lower
  side (Issue 770's expected shape) or a genuine backwards write.
- [ ] Only then re-pin, at measured, with the specimens in the row — the shape
  every other row in that file uses.

⚠ Do NOT attribute this to the 2026-09-24 fast-forward of that checkout: it
brought **one** docs commit (`309893426a`), and one commit can add at most one
reset.

## T2 — riir-shader: five `.issues` collisions with one systematic cause

Measured rows: `030/031/032/033/034`, each held twice, and the pairs are
`NNN_<name>_port` against `NNN_<name>_queue` — `030_fog_height_port` ·
`030_fog_height_queue`, and the same shape at 031 (`parallax_uv`), 032
(`bloom_emissive`), 033 (`radial_blur`), 034 (`raging_sea`). That is not five
independent accidents; it is one convention that allocated a **pair** of
documents per number, five times.

- [ ] Confirm the reading in the repo (it was **4 commits behind origin** with
  a dirty tree at filing, so this box's view is not authoritative — Issue 798).
- [ ] Decide with that repo's owner whether the `_port`/`_queue` pairing is
  DELIBERATE (a number owning two documents by design, which is the
  `.benchmarks` family convention one directory over and would make this a
  SCOPE question, not a collision) or an accident to renumber.
- [ ] The answer determines the repair: a scope exclusion in the sweep, or a
  renumber plus a ratchet at measured. **Do not pin it at 5 before that
  question is answered** — a ratchet over an unread bucket is Issue 785's
  forbidden shape.

## T3 — mmorpg-editor: one new reset in a read-only repo

`max_resets` 5 → measured 6. That row already carries the Issue-798 lesson in
its own comment (it was first pinned from a checkout that sat behind origin and
had to be re-measured after a fast-forward), and the repo is READ-ONLY from
here by the owner rule. Lowest priority; confirm against origin and then either
bump with the specimen named or hand it to that repo's owner.

## What this issue does NOT claim

- It does not claim the three breaches share a cause. `max_resets` and
  `max_hist` are different columns measuring different events, and T2's is
  plausibly not a defect at all.
- It does not claim the pins are wrong. A ratchet pinned at a real past
  measurement and breached later is the instrument **working**; the finding is
  that nobody has read the delta.
- It does not touch the citation sweep, which PASSES workspace-wide as of
  `7303bf13` (23 CROSS rows → 0). The two sweeps are neighbours in the same
  file family and were measured in the same session; they are otherwise
  independent.
