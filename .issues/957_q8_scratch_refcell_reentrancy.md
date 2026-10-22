# Issue 957: the int8-activation ternary matvecs panic `RefCell already borrowed` when two forwards share the rayon pool

**Status:** Closed 2026-10-23 — fixed in the commit that adds this file (shared `simd::with_thread_scratch`); record in HISTORY.md.

## Finding

`simd_ternary_q2_matvec_q8_parallel[_n]` (`ternary_q2_q8.rs`) and
`simd_ternary_group_matvec_q8_parallel` (`ternary_group_q8.rs`) quantize the
activations into a `thread_local! RefCell` scratch and run the row-parallel
rayon region INSIDE `with_borrow_mut`. A worker that waits in that region
work-steals other jobs; when the stolen job is another forward calling the
same matvec (two forwards on one pool — concurrent serving, an embedder beside
decode, parallel tests), it re-enters the borrowed cell on the same thread and
panics. Found from riir-infer Issue 1111 T3: adding one more concurrent
forward test to `tests/bonsai2_rotation_load.rs` turned the binary red ~1 run
in 100 (`RefCell already borrowed`, backtrace through
`with_q2_activations` → `on_rayon_worker`), different victim test each time,
never alone.

## Fix

One shared mechanism, both sites: `simd::with_thread_scratch(key, f)` moves
the scratch OUT of the cell for the call and back after it — no borrow spans
`f`. A re-entrant call sees the empty default and allocates its own (only on
actual re-entry); the steady state is still zero-allocation (moving a `Vec`
allocates nothing). Workspace sweep: the only other `thread_local!` + borrow
sites near a parallel region (riir-infer-gpu `weight_buffer_cache` STAGING,
`ATTN_TAP`) hold no borrow across one.

## Tasks

- [x] **T1** Regression arms `nested_parallel_calls_are_reentrant_and_bit_identical` in both modules (64 outer rayon tasks × the parallel matvec, 8 rounds, bit-identical to serial) — RED on the old code (both panic), GREEN after.
- [x] **T2** Shared helper, both call sites.
- [x] **T3** Downstream: riir-infer `bonsai2_rotation_load` 0/600 failures after (≈4/435 before).
