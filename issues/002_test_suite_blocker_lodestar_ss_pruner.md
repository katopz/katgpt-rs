# Issue 002 — Test-suite blocker: refactor lost lodestar glue + ss_pruner cross-crate path

Status: **open** (BLOCKER for Proposal 003 Phase 0.5 loser-sweep)
Created: 2026-07-01
Related: `proposals/003_src_consolidation_master.md` (Phase 0.5 is gated on this)

## TL;DR

`cargo check --workspace` is clean, but `cargo test --workspace` FAILS to
compile in `katgpt-pruners` due to two pre-existing refactor bugs. This
**blocks** the loser-sweep re-bench (Proposal 003 Phase 0.5): the GOAT
gates cannot be re-run until tests compile.

One of the two bugs is exactly the "shame" scenario the user flagged:
**lodestar is a GOAT-passed DEFAULT-ON winner whose integration glue was
dropped during the speculative-crate extraction**, making it *look* broken
when it is actually a winner. Re-benching in this state would have
misclassified a winner as a loser.

## Bug 1 — `build_dd_tree_lodestar` lost in speculative-crate extraction

**Symptom:** `crates/katgpt-pruners/src/lodestar.rs` tests reference
`katgpt_speculative::dd_tree::{build_dd_tree_lodestar, extract_parent_tokens,
build_dd_tree_pruned}`. `extract_parent_tokens` and `build_dd_tree_pruned`
exist in `katgpt-speculative/src/dd_tree.rs`; **`build_dd_tree_lodestar` does
not.** 5 compile errors (E0432).

**Provenance (this is NOT a loser):**
- Plan 207 T6–T8 explicitly implemented `build_dd_tree_lodestar` with the
  (A) budget mask, (B) jump-ahead, (C) A* heap ordering.
- `.benchmarks/055_lodestar_overhead_goat.md` records **GOAT 5/5 PASS** —
  per-call ~4–8ns, default-0 path +4.3% (within noise), budget-mask path
  −86.7% (faster than baseline).
- Plan 207 status: "Promoted to DEFAULT-ON. ALL 15/15 TASKS COMPLETE."

**What happened:** when `dd_tree.rs` moved to the `katgpt-speculative` crate,
`build_dd_tree`, `build_dd_tree_pruned`, `build_dd_tree_screened`, and
`build_dd_tree_balanced` were ported. `build_dd_tree_lodestar` + its
`CompletionHorizon` trait integration were **dropped**. The `LodestarPruner`
stayed in `katgpt-pruners`, but its consumer (the tree builder) is gone.

**Fix options:**
1. **Restore** `build_dd_tree_lodestar` + `CompletionHorizon` trait into
   `katgpt-speculative/src/dd_tree.rs` (preferred — re-establishes the
   GOAT-passed integration). Source: Plan 207 + the deleted history.
2. **Exile** lodestar to `katgpt-deprecated` (WRONG — it's a verified winner;
   exile would be the "shame" misclassification).

→ Option 1. This is a restore, not a demote.

## Bug 2 — `ss_pruner.rs` references root `cumprodsum`

**Symptom:** `crates/katgpt-pruners/src/ss_pruner.rs:209`:
`crate::cumprodsum::influence(&decay_factors, 0, depth)` → E0433 "could not
find `cumprodsum` in the crate root". `cumprodsum` lives in ROOT
(`src/cumprodsum.rs`), not in `katgpt-pruners`.

**Root cause:** `ss_pruner` was moved to the pruners crate but its test still
references the root-crate path. Classic move-and-lose-the-dep bug — the exact
class Proposal 003 is designed to prevent by moving `cumprodsum` → `katgpt-core`.

**Fix options:**
1. **Short-term (unblock tests now):** make the test compute the expected
   value inline or add a `katgpt-core` re-export of `influence` and use
   `katgpt_core::cumprodsum::influence`.
2. **Long-term (Proposal 003 Phase 10):** move `cumprodsum` → `katgpt-core`
   so it's accessible to all crates. Then `ss_pruner` references
   `katgpt_core::cumprodsum::influence`.

→ Short-term fix to unblock the bench; long-term handled by Phase 10.

## Impact on Proposal 003

- **Phase 0.5 (loser-sweep) is BLOCKED** — cannot re-bench until tests
  compile. The whole point of the sweep is to avoid false-loser
  classification; running it against a broken test suite would defeat that.
- **lodestar is a confirmed winner** — GOAT 5/5, default-ON. Do NOT exile.
  Its `katgpt-pruners` home is correct; only the speculative-side glue is
  missing. After Bug 1 fix, lodestar tests pass and it stays default-ON.
- **This validates the user's concern** — "ensure loser is lose not bc of
  bug." Lodestar would have been a shame-misclassification without the
  re-bench discipline.

## Tasks

- [ ] **T1 (Bug 1):** restore `build_dd_tree_lodestar` + `CompletionHorizon`
      trait into `katgpt-speculative/src/dd_tree.rs`. Verify against Plan 207
      T6–T8 + Bench 055 numbers. `cargo test -p katgpt-pruners --lib` green.
- [ ] **T2 (Bug 2):** fix `ss_pruner.rs:209` path (short-term inline or
      katgpt-core re-export). `cargo test -p katgpt-pruners --lib` green.
- [ ] **T3:** `cargo test --workspace --lib` fully green (no compile errors).
- [ ] **T4:** re-run Bench 055 (lodestar GOAT) against the restored code —
      confirm 5/5 still PASS with the documented numbers.
- [ ] **T5:** unblock Proposal 003 Phase 0.5 (loser-sweep can now run).

## References

- Plan 207: `.plans/207_lodestar_completion_distance_pruning.md`
- Bench 055: `.benchmarks/055_lodestar_overhead_goat.md` (GOAT 5/5)
- Proposal 003: `proposals/003_src_consolidation_master.md` (Phase 0.5)

## TL;DR

Test suite won't compile (2 refactor bugs). Bug 1 is the dangerous one:
lodestar is a GOAT-5/5 default-ON winner whose `build_dd_tree_lodestar`
glue was dropped in the speculative-crate move — looks broken, actually a
winner. Exactly the false-loser risk the user flagged. Fix both before the
loser-sweep; lodestar stays, do not exile.
