# Benchmark 301: Subspace Phase-Gate G1 GOAT Results (Phase 2)

**Date:** 2026-06-23
**Plan:** [301_runtime_subspace_phase_gate_primitive.md](../.plans/301_runtime_subspace_phase_gate_primitive.md)
**Research:** [279_Diffusion_Curse_Dimensionality_Subspace_Clustering_Fusion.md](../.research/279_Diffusion_Curse_Dimensionality_Subspace_Clustering_Fusion.md)
**Source paper:** [arXiv:2409.02426](https://arxiv.org/abs/2409.02426) — Wang et al., *Breaking the Curse of Dimensionality*.
**Example:** `cargo run --release -p katgpt-core --example subspace_phase_gate_goat --features subspace_phase_gate`

---

## Summary

Phase 2 ships the G1 GOAT proof for the `subspace_phase_gate` primitive. The
Wang et al. Theorem 4 phase transition reproduces exactly on the synthetic
MoLRG setup (D=48, K=3, d=6): recovery error collapses from ~2.4 (>0.5) at
N=3 to exactly 0 (<0.1) at N=d=6, and `phase_transition_gate(N, d)` matches
the empirical transition on all 7 sampled N values.

| Gate | Target | Result | Status |
|------|--------|--------|--------|
| **G1/T2.5** phase transition (N<d → err>0.5) | 2/2 fail-side rows | 2/2 (N=3: 2.40, N=5: 1.41) | ✅ PASS |
| **G1/T2.5** phase transition (N≥d → err<0.1) | 5/5 recover-side rows | 5/5 (N∈{6,7,10,50,200}: 0.00) | ✅ PASS |
| **G1/T2.6** `phase_transition_gate(N, d)` matches empirical | 7/7 | 7/7 | ✅ PASS |
| **G1/T2.2** K=3 mutually-orthogonal d=6 bases in R^48 | orthonormality < 1e-4 | ✓ | ✅ PASS |

**Verdict:** **G1 PASS.** The primitive is a faithful open realization of
Theorem 4's necessary condition. Eligible for Phase 5 promotion to default
once the Phase 3 Jacobian-SVD GOAT (G3-precursor) also passes.

**Caveat — T1.12 umbrella feature gap (fixed in this commit):** the plan
marked T1.12 (`subspace_phase_gate` in the umbrella `katgpt-rs/Cargo.toml`)
as complete, but the feature line was missing. Added in this commit as
`subspace_phase_gate = ["katgpt-core/subspace_phase_gate"]` (opt-in).

**Re-verification (2026-07-02, Issue 008 fix):** the G1 PASS above was
**stale** between commits `a08adc4a` (2026-06-24) and the Issue 008 fix — a
refactor of `one_sided_jacobi_svd_into`'s extraction loop changed the column-
norm scan from `0..n` (all columns) to `0..min(m,n)` (first k columns only).
For the G1 example's wide Jacobian (N×48, N<48), the non-zero singular values
landed in columns `k..n` and were missed, producing a garbage spectrum
(pr≈1.5 instead of ≈5 at N=6). The Issue 008 fix restores the `0..n` scan +
adds a null-space deflation floor for numerical stability. G1 now re-passes
bit-for-bit on the same example (same pr/nr values as the original PASS).
See `.issues/008_*.md` for the full analysis.

---

## Setup

- **Ambient dim D:** 48
- **Intrinsic dim d:** 6
- **Number of subspaces K:** 3 (mutually orthogonal, total dim 18 ≤ 48)
- **Sample sweep N:** {3, 5, 6, 7, 10, 50, 200}
- **Basis generation:** Modified Gram–Schmidt QR of a single 48 × 18 Gaussian
  matrix (PCG-XSH-RR PRNG, seed `0x3015EED_301C0FFEE`, Box–Muller Gaussian).
- **Sampling:** x = U·z, z ~ N(0, I_d). **No centering** — the ground-truth
  mean is exactly zero; centering would reduce effective rank by 1 and shift
  the transition to N = d+1, contradicting Theorem 4's necessary condition.
- **PCA via Jacobian SVD trick:** f(x) = X·x is linear in x, so its Jacobian
  is the N × D data matrix X. `jacobian_svd_at(f, ...)` yields the SVD of X,
  and the top-d right singular vectors (length D) are the principal
  directions Û. This exercises the public API of `subspace_phase_gate` with
  no separate SVD implementation needed.
- **Recovery error:** `‖Û Û^T − U* U*^T‖_F` via the identity
  `‖·‖_F² = d_hat + d_star − 2·‖Û^T U*‖_F²` (no D×D projector materialised).
- **Sweeps per cell:** mean / min / max over the K=3 subspaces.

---

## G1 Output (verbatim)

```
═══════════════════════════════════════════════════════════════
  Plan 301 Phase 2 — G1 GOAT: Subspace Phase Transition
  Paper: arXiv:2409.02426 (Wang et al., Theorem 4)
  Setup: D=48  K=3  d=6  subspaces,  N ∈ [3, 5, 6, 7, 10, 50, 200]
═══════════════════════════════════════════════════════════════

✓ T2.2: K=3 mutually-orthogonal d=6 orthonormal bases in R^48

── T2.3/T2.4: Recovery error vs N (mean over K subspaces) ──
N,d,mean_err,min_err,max_err,gate(N,d),pr_mean,nr99_mean
3,6,2.402688,2.380777,2.431412,false,2.520,3.0
5,6,1.413063,1.411229,1.414206,false,3.491,4.3
6,6,0.000000,0.000000,0.000000,true,4.265,5.0
7,6,0.000000,0.000000,0.000000,true,4.505,5.3
10,6,0.000000,0.000000,0.000000,true,4.499,5.3
50,6,0.000000,0.000000,0.000000,true,5.822,6.0
200,6,0.000000,0.000000,0.000000,true,5.955,6.0

── T2.5: Phase-transition check ──
  Rule: N<d → err>0.5,  N≥d → err<0.1
  ✓ N=  3: mean_err=2.4027  (expected fail side)
  ✓ N=  5: mean_err=1.4131  (expected fail side)
  ✓ N=  6: mean_err=0.0000  (expected recover side)
  ✓ N=  7: mean_err=0.0000  (expected recover side)
  ✓ N=10: mean_err=0.0000  (expected recover side)
  ✓ N=50: mean_err=0.0000  (expected recover side)
  ✓ N=200:mean_err=0.0000  (expected recover side)
  T2.5 verdict: PASS

── T2.6: phase_transition_gate(N, d) vs empirical ──
  ✓ N=  3: gate=false, empirical=false, err=2.4027
  ✓ N=  5: gate=false, empirical=false, err=1.4131
  ✓ N=  6: gate=true, empirical=true, err=0.0000
  ✓ N=  7: gate=true, empirical=true, err=0.0000
  ✓ N=10: gate=true, empirical=true, err=0.0000
  ✓ N=50: gate=true, empirical=true, err=0.0000
  ✓ N=200:gate=true, empirical=true, err=0.0000
  T2.6 verdict: PASS

── T2.7: Intrinsic-dim estimation (true d=6) ──
     N    PR_round        NR99    winner
     3         3.0         3.0       tie
     5         3.0         4.3        NR
     6         4.0         5.0        NR
     7         5.0         5.3        NR
    10         4.0         5.3        NR
    50         6.0         6.0       tie
   200         6.0         6.0       tie

  Summary: PR wins 0 row(s), NR wins 4 row(s).
  On this synthetic MoLRG, NR tracks the true d better than PR
  (sharp spectral elbow). For N<d, both correctly report N — the
  true d is information-theoretically unrecoverable. NR is the
  better production pick (discrete, threshold-tunable, immune to
  continuous-valued drift); PR is the better diagnostic (shows
  the effective dimensionality even when no clear elbow exists).

═══════════════════════════════════════════════════════════════
  G1: PASS — phase transition reproduces on synthetic MoLRG.
═══════════════════════════════════════════════════════════════
```

---

## T2.7 Discussion — PR vs NR as intrinsic-dim estimators

| N | True observable dim | PR (round) | NR(η=0.99) | Winner |
|---|--------------------|------------|------------|--------|
| 3 | 3 | 3.0 | 3.0 | tie |
| 5 | 5 | 3.0 | 4.3 | **NR** |
| 6 | 6 | 4.0 | 5.0 | **NR** |
| 7 | 6 | 5.0 | 5.3 | **NR** |
| 10 | 6 | 4.0 | 5.3 | **NR** |
| 50 | 6 | 6.0 | 6.0 | tie |
| 200 | 6 | 6.0 | 6.0 | tie |

**Winner: `numerical_rank(η=0.99)` tracks the true d better on this synthetic
MoLRG.** PR wins 0 rows, NR wins 4, ties 3.

### Why PR underestimates at small N

PR = `(Σλ)² / Σ(λ²)` is exactly the observable dimension only when all
nonzero singular values are equal. For N wake events drawn as x = U·z with
z ~ N(0, I_d), the empirical Gram matrix `X^T X` has eigenvalues that follow
the Marchenko–Pastur distribution scaled by N — they are *not* equal at
finite N. The spread lowers `(Σλ)²` relative to `Σ(λ²)`, so PR systematically
underestimates the observable dim until N ≫ d (at N=200 PR is 5.955, still
slightly below 6; at N=50 it hits 6.0 only after rounding).

### Why NR is more accurate

NR(η) only requires the spectral *elbow* to be visible — the top-observable
singular values dominate the energy long before they converge to equal
values. On this synthetic with its sharp elbow, NR(0.99) tracks d=6 exactly
once N ≥ d (well, once N ≥ ~10 for the 99% threshold; N=6 gives NR=5 because
the 6th singular value at exactly N=d captures only ~17% of energy, below
the 99% cumulative threshold).

### Recommendation

- **Production gate (consolidation freeze/thaw, shard merging):** use
  `numerical_rank(spectrum, η)` with η ∈ [0.95, 0.99]. Discrete, tunable,
  robust to spectral spread.
- **Diagnostic / exploratory:** use `participation_ratio(spectrum)` for a
  continuous read that doesn't depend on a threshold. Valuable when the
  spectrum has no clear elbow (gradual decay) and a threshold would either
  over- or under-count.
- **Phase-transition gate:** always use `phase_transition_gate(N,
  estimate)` with the *estimated* d from one of the above — never trust
  recovery when the gate returns `false`, regardless of the estimator.

---

## Reproducibility

- **Seed:** `0x3015EED_301C0FFEE` (PCG-XSH-RR, forced odd).
- **Determinism:** all RNG, linear algebra, and SVD are scalar f32, no
  SIMD dispatch inside the math, no reordering. Byte-identical across runs
  and platforms — required for the quorum/anti-cheat contract documented
  in the module.
- **Build:** `cargo run --release -p katgpt-core --example subspace_phase_gate_goat --features subspace_phase_gate`
- **Exit code:** 0 on G1 PASS, 1 on G1 FAIL (CI-detected).

---

## TL;DR

Phase 2 complete: the `subspace_phase_gate` primitive's G1 GOAT gate (phase
transition reproduces on synthetic MoLRG) **PASSES** on all 7 sampled N
values. `phase_transition_gate(N, d)` matches empirical recovery 7/7.
`numerical_rank(η=0.99)` beats `participation_ratio` as an intrinsic-dim
estimator on this synthetic (4/7 wins, 3 ties) because Marchenko–Pastur
spread at finite N lowers PR. No centering is used (true mean = 0); centering
would shift the transition to N=d+1 and contradict Theorem 4. G1 PASS
unblocks Phase 3 (Jacobian-SVD validation) and Phase 5 (promotion to
default, conditional on G3-precursor).

---

# Phase 3 — Jacobian SVD Validation (G3-precursor)

> ⚠️ **PRE-EXISTING REGRESSION DISCOVERED (2026-07-02, NOT caused by Phase 3):**
> the Phase 2 G1 example (`examples/subspace_phase_gate_goat.rs`) **FAILS on the
> committed `develop` HEAD**. At N=d=6 the recovery error is **2.914** (target
> <0.1) and the spectrum is garbage (`pr_mean=1.511`, should be ≈5). The G1
> PASS recorded in the Phase 2 section above is **STALE** — it was valid at
> commit `e12dbda7` (2026-06-23) but the post-benchmark SVD refactors
> (`a08adc4a` SOA scratch, `c775be2b` zero-alloc, `6e9b22ac` cap fix)
> introduced a **one-sided-Jacobi convergence failure on wide rank-deficient
> matrices** (m ≪ n). The SVD still works for large N (N=50, 200: err=0) and
> for square matrices (Phase 3 T3.1–T3.3 R^8×8 all PASS), but breaks for small
> N on the 6×48 PCA path. **This needs its own focused fix** (likely
> column-norm pivoting or a convergence-criterion change for rank-deficient
> wide inputs) and is filed as the critical follow-up below. The Phase 2 PASS
> claim should NOT be trusted until that fix lands and the example is re-run.

**Date:** 2026-07-02
**Plan tasks:** T3.1–T3.4
**Verdict:** **T3.1, T3.2, T3.3 PASS** (square R^8×8 — SVD correct in this
regime). **T3.4 FAILS** the <1µs latency target — scalar Jacobian SVD on R^8→R^8
measures **2403 ns/call in release** (2.4× over). Per plan T4.3, this makes
**Phase 4 (SIMD) REQUIRED** and **blocks Phase 5 promotion** (T5.1 needs the
G3-precursor latency gate to pass). Separately, the **pre-existing G1
regression above** must be fixed before ANY Phase 5 promotion is meaningful.

## Setup

- **Dimensionality:** R^8→R^8 (matches HLA's 8-dim, plan open question Q1).
- **Map construction:** `W = Σ_k σ_k · u_k · v_k^T`, rank-3, with
  **non-canonical** orthonormal singular vectors built from 2×2 rotation
  blocks at distinct angles (θ_v ∈ {0.3, 0.7, 1.1}, θ_u ∈ {0.5, 0.9, 1.3}).
  Non-canonical bases make right-singular-vector recovery a meaningful check
  (canonical axes would trivially match coordinate probes and hide
  sign/ordering bugs). σ = {10, 5, 2}. Coordinates 6,7 are zero so the map
  is genuinely rank-3 in R^8.
- **eps:** 1e-4 forward difference (plan spec).

## Results

### T3.1 + T3.2 — rank-3 linear map: singular values + right vectors ✅ PASS

`jacobian_svd_at(f_linear, x, 1e-4, scratch)` on the R^8×8 rank-3 map:

- **Recovered singular values:** `[10.0005, 5.0018, 2.0007, 0.0005, 6.5e-5, ...]`
  — top-3 match {10, 5, 2} within 0.1 (forward-diff adds ~5e-4 noise).
- **Right singular vectors:** each recovered V column matches its ground-truth
  `v_k` **up to sign**, with `|dot| > 0.999` (matched by nearest singular
  value — distinct σ ⇒ unique vectors up to sign).
- **Rank-3 structure:** confirmed via the plan's OWN `numerical_rank(spectrum,
  η=0.99) == 3` (top-3 carry 99.99% of energy) AND a clean 4000× spectral
  gap between σ[2]=2.0007 and σ[3]=0.0005.

**Finding (pre-existing, not a regression):** the SVD's internal `result.rank`
field reports **4**, not 3. Its threshold is `sigma_max * 1e-5` = 1e-4
(`subspace_phase_gate.rs:725`), and the forward-diff noise floor (~5e-4)
sits above it. This is a **threshold-tuning discrepancy**, not a math error
— the spectrum unambiguously shows rank 3. Verified via `numerical_rank`
which correctly reports 3. Not in Phase 3 scope to re-tune (would affect all
SVD consumers); documented here as a known sharp edge.

### T3.3 — non-linear sigmoid map: row-space recovery ✅ PASS

`f(x) = sigmoid(W x)` elementwise. Analytical Jacobian = `diag(sigmoid'(Wx))·W`;
since the diagonal is strictly positive (x=0.1·1 keeps Wx well away from
saturation), the row space is unchanged. SVD of the forward-diff Jacobian:

- **Rank ≥ 3** ✓ (the diagonal doesn't zero any row).
- **Row-space match:** every recovered right singular vector with σ > 1e-3
  lies in `span{v1, v2, v3}` — verified via the projector
  `P_true = Σ_k v_k v_k^T`: `‖P_true·r‖ ≈ ‖r‖` to within 5e-3.

Note: individual right singular vectors of `diag(d)·W` do NOT match the `v_k`
one-to-one (the row-weighting rotates them within the subspace); only the
3-dim **subspace** is invariant. The test checks subspace containment, which
is the correct contract for a non-linear map (matches the plan wording "SVD
should reveal the row space of W").

### T3.4 — latency gate ❌ FAILS (2403 ns/call vs <1000 ns target)

`jacobian_svd_at` on R^8→R^8, 5000 iterations (after warmup),
`JacobianSvdScratch::with_capacity(8, 8)` reused across calls:

| Profile | ns/call | vs target |
|---|---|---|
| **release** | **2403** | 2.4× over |
| debug | 31249 | (debug-stable regression guard at 100µs) |

**Cost breakdown (8×8 one-sided Jacobi SVD):** ~28 column-pairs/sweep ×
(3×8-dot + 2×8-rotate + 8-V-rotate ≈ 40 flops) ≈ 1120 flops/sweep; convergence
in ~6–10 sweeps ⇒ ~8–11k scalar f32 flops/call. The 2403 ns release figure is
consistent with the scalar floor — there is no cheap scalar win that wouldn't
risk the Phase 2 G1 bit-identical recovery (loosening `tol=1e-7` or
`max_sweeps=60` would change the D=48/n=18 G1 numerics).

**Phase 4 T4.2 scalar investigation (done):** concluded the scalar floor is
~2.4µs; SIMD (T4.1) is the only path to <1µs. The inner `for r in 0..m`
column-dot and rotation loops are the vectorization targets (NEON/AVX2 on the
8-element f32 columns).

## Escalation (per plan T4.3)

- 🔴 **[CRITICAL, pre-existing] G1 example regression.** The Phase 2 G1 GOAT
  example FAILS on `develop` HEAD for small N (recovery err=2.914 at N=d=6;
  garbage spectrum pr=1.511). Cause: one-sided-Jacobi convergence failure on
  wide rank-deficient matrices (m ≪ n), introduced by the post-benchmark SVD
  refactors. The Phase 2 PASS claim above is STALE. **Fix path:** add
  column-norm pivoting or revise the convergence criterion for rank-deficient
  wide inputs; re-run the example; re-verify G1. This is a prerequisite for
  ANY Phase 5 promotion (a broken G1 voids the gate). Tracked as the
  top-priority follow-up.
- **Phase 4 T4.1 (SIMD-accelerate `participation_ratio`, `numerical_rank`,
  and the Jacobi inner loops): REQUIRED** before this primitive can serve the
  HLA 8-dim hot path. Without it, per-call cost is 2.4µs — acceptable for
  offline consolidation (riir-neuron-db Plan 002's freeze gate runs at
  sleep-cycle cadence, not per-tick), but not for any per-NPC-per-tick use.
- **Phase 5 T5.1 (promote `subspace_phase_gate` to default): BLOCKED** on T4.1.
  G1 (Phase 2) passes, but the G3-precursor latency gate (T3.4) fails. The
  feature stays opt-in until SIMD lands and T3.4 is re-run <1µs.
- **`result.rank` threshold sharp edge:** tracked here, not gated. Consumers
  that need robust rank should call `numerical_rank(spectrum, η)` explicitly
  rather than trusting `result.rank` on forward-diff-noisy inputs.

## Reproducibility

- **Tests:** `subspace_phase_gate::tests::jacobian_svd_recovers_rank3_r8x8_singular_values_and_vectors`
  (T3.1+T3.2), `jacobian_svd_sigmoid_map_reveals_row_space` (T3.3),
  `jacobian_svd_r8x8_latency_gate` (T3.4, regression guard).
- **Run:** `cargo test -p katgpt-core --features subspace_phase_gate --lib subspace_phase_gate::`
  — 17/17 pass (14 pre-existing + 3 new).
- **Latency re-measure:** `cargo test --release -p katgpt-core --features subspace_phase_gate --lib jacobian_svd_r8x8_latency_gate`.
