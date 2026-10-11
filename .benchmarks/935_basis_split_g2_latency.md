# Bench 935 — Plan 625 G2: `basis_split` split latency

**Status:** COMPLETE — 2026-10-12
**Feature:** `basis_split` (opt-in; NOT promoted — no wired consumer yet; promotion stays owner-gated per the standing law)
**Plan:** `.plans/625_basis_split_matched_pair_block_identification.md` T2.2 + T2.5 (the last two open rows)
**Box state:** `PROVENANCE: power=AC Power load=3.98 swap=0.00M canary=139.0us/best5 powermode=2(high)` — M3 Max, macOS 26.6.2, `scripts/bench_preflight.sh` **PASSED** immediately before the run (the two earlier refusals this session: load 12.95 and 8.78 — the staged-window discipline held; nothing was measured contaminated).

## What ran

`cargo bench -p katgpt-core --features basis_split,subspace_phase_gate --bench basis_split_goat g2_split_latency -- --save-baseline g2_quiet_2026_10_12` — criterion, `bench` profile `[optimized]` (the G2 `--release` mandate), exactly one case (the bench defines one):

| case | pipeline per iteration | verdict | measured |
|---|---|---|---|
| `basis_split/g2_split_latency/d64_n2048_full` | accumulate 2048 pair-differences → `finish()` → `split()` (TotalCovariance whitening, rank gate 10.0, d 64 → k 48, QL solver) → one held-out `project_invariant_into` | **PASS — 5.2× under the bar** | **1.923 ms** (slope 1,923,036 ns, 95% CI [1.9145, 1.9313] ms; mean 1.916 ms; 100 samples after a 3 s warmup) vs the **< 10 ms @ d=64/N=2048** bar from the plan |

Baseline saved as `g2_quiet_2026_10_12` (`target/criterion/basis_split_g2_split_latency/d64_n2048_full/`) so the next posture change A/Bs against a quiet-box reference instead of re-deriving one.

## Gates summary (this closes the plan's last rows)

- **G1** planted recovery — asserted in the bench at d∈{16,64,256} (d=256 executed green) + adversarial canaries assert `Unidentified` (previous session, `3ca8a1470`).
- **G2** latency — **this record: 1.923 ms, PASS.**
- **G3** no-regression — lib suite count floor holds (2157 passed at the feature posture, `3ca8a1470`).
- **G4** alloc-free — 0 allocations both solver postures (`3ca8a1470`).

All four gates green → the GOAT holds **as a primitive measurement**. `basis_split` stays **opt-in, default-off**: there is no wired consumer yet (reflex Issue 086 is the intended first consumer); promotion to default is owner-gated behind that wiring regardless of these numbers.

## Notes

- The T2.2 staging law worked as designed: two preflight REFUSALS (load 12.95, then 8.78 — sibling `cargo` sessions) were obeyed, and the run happened only after a PASSED preflight at load 3.98. The number above is a quiet-box number.
- The compile-readiness check (`--no-run`, 24.5 s warm) was done during the LOADED phase deliberately — compile load is not a measurement, and it made the window a pure measurement run (the bench itself finished in ~13 s of wall time).
- The `subspace_phase_gate` feature rides the invocation because `participation_ratio` is DRY-delegated to it (Phase 1's T0.2 reuse verdict); it is not otherwise load-bearing for the split path.
