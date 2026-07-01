# Issue 008: Subspace Phase-Gate G1 — one-sided Jacobi convergence failure on wide rank-deficient matrices

**Date:** 2026-07-02
**Severity:** 🔴 CRITICAL (invalidates a GOAT gate)
**Status:** ✅ **RESOLVED** (2026-07-02, same session)
**Discovered by:** Plan 301 Phase 3 validation session (commit ref: Phase 3 work)
**Plan:** [301_runtime_subspace_phase_gate_primitive.md](../.plans/301_runtime_subspace_phase_gate_primitive.md)
**Benchmark:** [301_subspace_phase_gate_g1.md](../.benchmarks/301_subspace_phase_gate_g1.md) (Phase 3 section)

---

## Summary

The Phase 2 G1 GOAT example (`crates/katgpt-core/examples/subspace_phase_gate_goat.rs`)
**FAILS on the committed `develop` HEAD**. The Phase 2 "G1 PASS" verdict recorded
in `.benchmarks/301_subspace_phase_gate_g1.md` is **STALE** — it was valid at
commit `e12dbda7` (2026-06-23) but a subsequent SVD refactor broke PCA-via-SVD
recovery for small N on wide matrices.

This is **NOT a modelless-quality problem with the plan's primitives** — the
plan's own correctness tests (`participation_ratio`, `numerical_rank`,
`phase_transition_gate`, and the square-matrix Jacobian SVD tests from Phase 3
T3.1–T3.3) all pass. It is a **numeric-convergence bug in the one-sided Jacobi
SVD** (`one_sided_jacobi_svd_into`) that only manifests on **wide
rank-deficient matrices** (rows ≪ cols).

## Reproduction

```bash
CARGO_TARGET_DIR=/tmp/iss008 cargo run --release -p katgpt-core \
  --example subspace_phase_gate_goat --features subspace_phase_gate
```

**Current (broken) output:**

```
N,d,mean_err,min_err,max_err,gate(N,d),pr_mean,nr99_mean
3,6,2.672580,...,false,0.578,0.7
6,6,2.914004,2.449486,3.464102,true,1.511,1.7   ← should be err≈0, pr≈5
7,6,2.575799,...,true,2.491,2.7
10,6,2.813396,...,true,1.842,2.0
50,6,0.000000,0.000000,0.000000,true,5.822,6.0   ← correct (large N)
200,6,0.000000,...,true,5.955,6.0                 ← correct
  T2.5 verdict: FAIL
  G1: FAIL — phase transition does NOT match theory.
```

**Expected** (per commit `e12dbda7`): err=0.000 at N≥6, pr≈5, G1 PASS.

## Root cause (analysis, not yet confirmed by fix)

The one-sided Jacobi SVD orthogonalizes column pairs of the m×n matrix. For the
G1 PCA path, the Jacobian is **N×D = N×48** (m=N rows, n=48 cols):

- **N=6** (6×48, rank 6): 48 columns in R^6 ⇒ 42 are linearly dependent. The
  Jacobi rotations churn on near-zero column pairs without converging the real
  6-dim structure. The `pr_mean=1.511` / `nr99=1.7` (vs expected ≈5/≈6) show
  the **spectrum itself is wrong**, not just the singular vectors — so this is
  a convergence failure, not a V-extraction/indexing bug.
- **N=50, 200** (tall-ish matrices, rank ≈ full): converges correctly (err=0).

So the bug is specific to the **m ≪ n, rank-deficient** regime. The likely
culprits:

1. **No column pivoting.** Columns are processed in index order; a near-zero
   column paired with another near-zero column produces `app≈0, aqq≈0`, and the
   per-pair test `apq.abs() <= tol * (app*aqq).sqrt()` degenerates (rhs ≈ 0),
   causing either spurious rotations on noise or premature skipping that
   prevents the real column structure from converging.
2. **Premature convergence break.** If a full sweep happens to apply no
   rotation (because every pair hit the degenerate tolerance), the `!rotated`
   break fires before the matrix is actually diagonalized.

## Suspect commits (post-benchmark SVD refactor)

The benchmark (`e12dbda7`, 2026-06-23) PASSES. The regression was introduced by
one of the three refactor commits on 2026-06-24/28:

| Commit | Date | Change | Likely culprit? |
|---|---|---|---|
| `77cb4268` | 06-24 | expose `thin_svd` + `SvdScratch` as public API | low (API surface) |
| `a08adc4a` | 06-24 | `thin_svd_into` SOA scratch-stored result (zero-alloc) | **medium** — changed result storage |
| `c775be2b` | 06-24 | eliminate heap allocs in Jacobian SVD | **high** — touched the Jacobi path |
| `6e9b22ac` | 06-28 | raise Jacobi SVD scratch cap k=16→64 | low (capacity only) |

Bisect: `git bisect` between `e12dbda7` (good) and HEAD on the example's G1
verdict will pinpoint the exact commit.

## Scope of impact

- **Plan 301 Phase 2 G1 gate: BROKEN.** The recorded PASS is stale.
- **Plan 301 Phase 5 promotion: BLOCKED.** A broken G1 voids the gate; even if
  the T3.4 latency gate is fixed via SIMD (Phase 4 T4.1), promotion to default
  requires G1 to genuinely pass.
- **riir-neuron-db Plan 002 (consolidation freeze gate):** consumes this
  primitive via the SVD. If it only ever factorizes **square or tall** matrices
  (shard `style_weights` is 8-dim, ambient ≤ 64), it may be unaffected — but
  this must be verified before Plan 002 ships. The HLA 8-dim case (Phase 3
  T3.1–T3.3) PASSES, so the square/small case is safe.
- **Plan 301 Phase 3 (this session): UNAFFECTED.** T3.1–T3.3 exercise the
  square R^8×8 path, which is correct. T3.4 is a latency gate, independent of
  correctness.

## Suggested fix path (executed)

1. **Bisect** to confirm which refactor commit introduced it.
   **Result:** `a08adc4a` (not `c775be2b` as suspected). Confirmed via
   worktree bisect: `77cb4268` PASSES, `a08adc4a` FAILS (+ panics at N=50
   due to the k≤16 cap, later masked by `6e9b22ac`).
2. **Root cause (different from the suggested column-norm pivoting):** the
   SOA refactor changed the extraction loop from `(0..n).map(norm)` (ALL
   columns) to `for i in 0..k { ... }` where `k = min(m,n)`. For wide
   matrices (m < n), this only scans the first m columns — but the non-zero
   singular values after Jacobi convergence can land in ANY of the n columns.
   The Jacobi rotations themselves were always correct; the extraction was
   reading the wrong columns.
3. **Actual fix applied (no column-norm pivoting needed):**
   - Restored the extraction scan to `0..n` (all columns), argsort all n,
     take top-k. This is the minimal fix that matches the pre-regression
     behavior.
   - Added a null-space deflation floor (`col_floor_sq = frob_sq * tol²`):
     skip pairs where BOTH columns are below the floor. Uses AND (not OR)
     so that a near-zero column paired with a signal column is still rotated.
     Prevents noise injection from null-space column pairs on wide matrices.
     Defensive — the extraction fix alone resolves G1, but the floor improves
     numerical stability and reduces wasted sweeps.
4. **Re-run the G1 example** — prints `G1: PASS` with err=0 at N≥6. ✅
5. **Re-verify** the existing tests — 665/665 pass (664 original + 1 new
   wide-matrix regression test). ✅ The tucker test (which uses `thin_svd_into`)
   initially regressed with an OR-based floor (too aggressive on full-rank
   matrices); fixed by switching to AND.

## Acceptance criteria

- [x] `cargo run --release -p katgpt-core --example subspace_phase_gate_goat --features subspace_phase_gate` prints `G1: PASS`.
- [x] N=6 row: `mean_err=0.000000`, `pr_mean` ≈ 4.3, `nr99_mean` ≈ 5.0.
- [x] All existing `subspace_phase_gate::tests` (17) still pass + 1 new wide-matrix regression test (18 total).
- [x] Phase 3 T3.1–T3.4 tests still pass (no square-matrix regression).
- [x] Update `.benchmarks/301_subspace_phase_gate_g1.md` Phase 2 section: replace "STALE" warning with re-verified PASS.
- [x] Update Plan 301 status line to reflect G1 re-verified.

## Priority

**Higher than the T3.4 latency / Phase 4 SIMD work.** A broken GOAT gate is a
correctness void; a slow-but-correct primitive is a perf gap. Fix the G1
regression first, then the SIMD latency, then promote.
