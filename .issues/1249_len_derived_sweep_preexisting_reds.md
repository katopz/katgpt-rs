# Issue 1249: the len_derived drift sweep reads RED at a clean-tree posture on pre-existing items outside any single kernel fix

**Status:** Open — filed 2026-11-27 (zed-68667e1d@m3) while landing the i1245 residue fix (riir-infer `09d41700`); the sweep's riir-infer rows went strictly cleaner with that landing (binds 327→295, joined findings 2→0, eyes 19→17, the 2 geglu_f16 pins removed citing `09d41700`), and the remaining reds are PRE-EXISTING, none caused by it. Filed so the class is visible, not silently absorbed.

**Claimed:** zed-68667e1d@m3, 1795814641, expires 2026-11-29T21:30Z

## The findings (from `python3 scripts/len_derived_drift_sweep.py`, 2026-11-27T21:05Z, worktree = my own i1245 landing only)

1. **⛔ NEW PERSISTENT-UPSTREAM never pinned:** `riir-infer crates/riir-infer-gpu/src/sampling_cubecl.rs argmax_f32_last input x1 — not in len_derived_eyes_expected.txt`. Pre-existing: sampling_cubecl.rs last touched by the plan-626 lane (nothing fresh); the PERSISTENT-UPSTREAM caller cite is `ternary_deltanet_gpu_forward.rs:5613`. Needs an eyes row (after reading the field's creation, per the sweep's own note) or a verdict-half adjudication.
2. **⛔ CROSS-REPO VERDICT flips (4):** dropping riir-train flips `riir-infer .../elementwise_cubecl.rs:1823 x_handle` + `:1823 bias_handle` + `:1885 x_handle` + `gemv_cubecl.rs:1117 b_handle` (the `add_f32` launch) from GUARD-ONLY → UNRESOLVED. The sweep demands adjudication in `len_derived_stability_expected.txt` (conservative direction only) or a caller fix. riir-train was 22 commits behind its upstream at measurement — re-run after a fetch before adjudicating (the flip may be stale-checkout-shaped, Issue-798 class).
3. **✗ walk floor DELEGATION broken (×2):** the sweep carries no `min_rs_files` column and delegates to `orphaned_attr_drift_floors.txt`, which has NO non-zero row for `riir-ledger` (7 tracked .rs) or `seal-sprints` (1 tracked .rs) — both repos are registered contract repos, so the delegation wall fires. Remedy: add their floors rows (or register the delegation scope) in the floors files.

## Non-goals

- Not the i1245 landing's scope (that fix's rows are clean in the same run).
- UNRESOLVED=unpinned is by-design (Issue-785 backlog semantics) — only the four GUARD-ONLY flips above are named.

## Suggested resolution order

1. Fetch riir-train (+ riir-reflex, 2 behind) and re-run — stale-checkout items may clear themselves.
2. Pin or adjudicate the argmax eyes row.
3. Add the two walk-floor rows (riir-ledger, seal-sprints) to `orphaned_attr_drift_floors.txt`.
4. Adjudicate any surviving cross-repo flips into `len_derived_stability_expected.txt`.
