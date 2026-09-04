# Issue 724: `.plans/` numbering collisions REGREW after a hand-sweep fixed 11 of them — nothing gates the allocator

**Status:** OPEN — T2 + T3 DONE 2026-09-04; **T1 (the untracked `075`) and T4 (the gate) remain.** The tracked collision is gone: `449` now belongs to Poincaré alone. Trigger was one untracked file; the search that followed found the pattern: **a hand-sweep resolved 11 `.plans/` collisions on 2026-07-15 (`f98f7b51`) and a new one landed 3 days later.** Fixed one at a time is how it stays invisible. The generalization first drafted here was WRONG and is corrected below — `.benchmarks/` must be excluded, on measured grounds.

---

## 1. The two live `.plans/` collisions

| number | files | state |
|---|---|---|
| **075** | `075_residual_attention_simd_audit.md` (tracked, `HEAD`, mtime 2026-08-02) · `075_riir_ai_m3_campaign_measured_distill.md` (**untracked**, written 2026-09-04 09:22) | caught **pre-commit** |
| **449** | `449_action_bridge_lean4_monotonicity_proof.md` (Plan dated 2026-06-23) · `449_poincare_latent_navigation_primitive.md` (Plan dated 2026-07-18) | **both tracked** — live in `HEAD` today |

Two unrelated documents each titled "Plan 449". Every cross-reference in this
workspace is by number, so `Plan 449` resolves to two things and a reader
following a citation cannot tell which.

**And 449 is the recurrence, not the original.** `f98f7b51` (2026-07-15) is
titled *"docs: resolve 11 `.plans/` numbering collisions (renumber per git-log
first-creation)"* — the sweep happened, it fixed eleven, and the Poincaré plan
dated **2026-07-18** collided three days later. A one-time cleanup with no gate
behind it buys three days.

## 2. Two allocators are BELOW their directory's max — the precursor state

| dir | `.highwater` | max file number | consequence |
|---|---:|---:|---|
| `.plans/` | **585** | **586** | the next allocation reads 585, uses 586 → collides with the existing 586 |
| `.benchmarks/` | **700** | **701** | same, at 701 |
| `.issues/` | 724 | 724 | correct |
| `.research/` | 531 | 531 | correct |
| `.proposals/` | 013 | 013 | correct |

A file claimed a number without writing the allocator back. This is not a
cosmetic mismatch — it is the loaded gun: `value + 1` is *already* taken. Both
are one-line fixes and are safe (recording that a number is allocated can only
ever move `.highwater` up).

## 3. CORRECTION — `.benchmarks/` is NOT part of this defect

This issue's first draft proposed a duplicate-number check over
`.plans/`/`.issues/`/`.docs/`/`.benchmarks/`. Measured, that would have fired
**~55 findings in `.benchmarks/` alone, all false**, and been the
cries-wolf instrument AGENTS.md warns gets ignored.

In `.benchmarks/` the leading number is the **owning plan/issue**, not a serial,
and a family per owner is the intended convention:

- `010_best_belief_floor_comparison` · `010_bom_floor_comparison` ·
  `010_report_the_floor_consolidated` · `010_sdar_arena` ·
  `010_sleep_time_floor_comparison` — five, all Issue 010 "Report the Floor".
- `294_ict_g1` · `294_ict_g2` · `294_ict_g3` · `294_ict_g10` ·
  `294_ict_goat_gates` · `294_ict_promotion` — six, all Plan 294's gates.

The directory is **mixed**: it also holds allocator-issued serials (`700`, `701`,
tracked by its own `.highwater`). So "duplicate number" is not a decidable
defect there without knowing which convention a given file follows, and the gate
must not guess.

Measured zero-duplicate directories — i.e. the ones that really are strict
serials, and the correct scope: **`.issues/`, `.research/`, `.proposals/`**, plus
`.plans/` (which is strict-serial by intent and is where both collisions are).

## Tasks

- [ ] **T1** — `075`: owner decides the repo first, because the number depends on
      it. The file's own vocabulary is **riir-clippy's** ("Batch 105", "the B60
      discipline", "named **corpus** entries", "measured-GOAT **distill**"), and
      riir-clippy is where rule-corpus mining lives per AGENTS.md; katgpt-rs
      ships modelless inference primitives. BOUNDARY.md's domain test arbitrates.
      Then renumber: **586** if it stays here, **076** if it moves (riir-clippy
      `.plans/.highwater` is 75, no `075_*` on disk).
- [x] **T2 — `449` RESOLVED 2026-09-04, and the citation grep INVERTED the
      precedent.** `f98f7b51`'s rule (*renumber per git-log first-creation*)
      would keep 449 on the 2026-06-23 ActionBridge plan and move Poincaré.
      Measured, that is the wrong direction:

      | | `Plan 449` mentions in its context | files citing its filename |
      |---|---:|---:|
      | Poincaré (2026-07-18) | **27** of 33 — README feature table, CHANGELOG ×4, `.docs/05_adaptation/`, a `katgpt-core` example | 5, incl. `.benchmarks/449_poincare_goat.md` + `.research/449_SeeSE3_*` sharing the number by the owner-number convention |
      | ActionBridge (2026-06-23) | **4** | 2 |

      Two reasons citation weight beats creation order here. It is 6 edits
      against ~34; and moving Poincaré would have rewritten **CHANGELOG**
      entries, which are historical records of what was said at the time and
      must not be edited to match a later renumber. **So ActionBridge moved:
      `.plans/449_action_bridge_lean4_monotonicity_proof.md` → `587_*`**, title
      updated with the rationale, all 6 references repointed
      (`tests/bridge_spec_match.rs` ×4, `.research/292_*` ×2),
      `.plans/.highwater` → 587. Residual ActionBridge-449 references: **0**
      (the one apparent hit, `README.md:2938`, is a *Poincaré* citation on a
      long line that also lists an unrelated `action_bridge` feature flag).
      **The precedent does not generalize past its own sweep** — record that
      before the next collision is resolved by rule instead of by measurement.
- [x] **T3 — DONE 2026-09-04** (`28c353a1`): `.plans/.highwater` 585 → 586 and
      `.benchmarks/.highwater` 700 → 701, both verified against a real file
      (`586_pot_scale_determinism_ternary_group.md`,
      `701_full_workspace_execution_pricing.md`). `.plans/` then went 586 → 587
      with T2's renumber.
- [x] **T4 — DONE 2026-09-04.** `scripts/numbering_gate.py` +
      `scripts/numbering_floors.txt`, wired into `scripts/docs_gate.sh`'s
      `CHECKS` (now 9/9 clean). Scope is the four serial-numbered dirs per §3;
      the pins file carries the measured reason `.benchmarks/` and `.docs/` are
      excluded, so a future widening has to argue with the evidence rather than
      rediscover it. Measured population: **995 numbered files over 4 dirs, 0
      tracked duplicates, 0 stale allocators, 1 untracked warning** (the `075`
      of T1). Verdicts: tracked duplicate, `.highwater` below max, per-dir
      population **floor** (every other verdict is a ceiling, so a regex
      regression would print a confident green over zero files), and untracked
      duplicates in their own **non-failing** class per
      `skill_repo_set_gate.py`'s precedent — they red the moment they are
      committed. **Canaried in five directions**, each restored: planted tracked
      duplicate → exit 1; `.highwater` lowered below max → exit 1; floor raised
      above measured → exit 1; pins file with no directories → exit **2** (an
      empty scope is refused, not passed over); broken regex → exit **2** via
      `selftest()` (untrustworthy instrument ≠ drift).

      **And the trigger was the real hazard, for the third time in this file.**
      `docs_gate.yml`'s `paths` filter globbed `.benchmarks/**` and `.docs/**`
      — *exactly the two directories this gate excludes* — and carried nothing
      for `.plans/`, `.issues/`, `.research/`, `.proposals/` or any
      `.highwater`. The gate could not have fired on the one push it exists
      for. Ten globs added to **both** hand-duplicated lists (GitHub Actions
      has no YAML anchors); verified the two lists are identical at 43 entries
      each and that the file still parses.

- [ ] **T4b** — nothing asserts that `docs_gate.yml`'s two `paths` lists stay
      in sync. The file's own comment says "keep the two lists in sync by
      hand", which is a known-unenforced invariant, and a PR-list that drifts
      behind the push-list is silent. A ~10-line check (parse both, compare as
      sets) belongs in `CHECKS`. Verified identical by hand this time — that is
      exactly the state that decays.

- [-] **T4-orig** — the axis: a numbering gate in `scripts/docs_gate.sh`'s `CHECKS`,
      scoped to `.plans/`, `.issues/`, `.research/`, `.proposals/` per §3, asserting
      (a) no duplicate `NNN_` prefix and (b) max ≤ `.highwater`. Requirements this
      repo's gate discipline imposes: a `selftest()` on every invocation exiting
      **2** (untrustworthy instrument ≠ drift); a **floor** on files scanned per
      directory, because every assertion here is a ceiling and a glob regression
      takes the population to zero and passes; and **untracked files reported
      separately without failing**, mirroring `skill_repo_set_gate.py`'s existing
      *untracked* class — a colleague's in-flight file is not a repo defect, but it
      must fail the moment it is committed. Canary all four directions before
      landing.
- [ ] **T5** — the `075` file's header reads `Date: 2026-09-05` on a file written
      **2026-09-04**. Trivial, but a plan whose own date is in the future is
      unusable as a timeline.

## References

- AGENTS.md §"Numbering Discipline" · the prior incident: `.issues/121` · the prior sweep that regrew: `f98f7b51` (2026-07-15)
- Found while closing a loose end flagged in a session handoff ("untracked in katgpt-rs: `.plans/075_riir_ai_m3_campaign_measured_distill.md` — not part of this task")
