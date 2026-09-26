# Issue 899: Second-moment, null-normalized drift alignment (the Plan 610 redesign)

**Status:** Open — filed 2026-09-26 from [Bench 900](../.benchmarks/900_arm_drift_alignment_goat.md) (Plan 610 GOAT FAIL). The bars below are pre-registered; no code has been written for this issue yet.
**Class:** poc (redesign of an opt-in primitive against its own failed gate)
**Parent:** [Plan 610](../.plans/610_arm_drift_alignment.md) · [Research 591](../.research/591_Trajectory_Aligned_Curiosity.md) · riir-ai `.research/389` (guide; P2 is refuted at the current design)

## Problem

Bench 900 measured two structural defects in the shipped `TrajectoryAlignedCuriosity`, which uses the first-moment mean pull and an axis-aligned preconditioner:

1. **The first moment is blind to spread families.** `m = Σ_j p_j g_j` barely moves when mass shifts onto a zero-centroid family (`±e_i` pairs cancel). That shift then reads as drift away from the coherent family, and `|·|` credits the coherent family. Measured: when S is the better family, aligned loses by 104–132 cycles against every other arm, with 20 of 32 runs censored.
2. **An axis-aligned preconditioner cannot separate drift signal from pool geometry.**
   - Preconditioner off: a cluster-density prior (pure-noise AUC 1.000).
   - Preconditioner on: kills the prior (0.395) and the planted signal with it (held-out AUC 0.602).
   - κ = 0: 0.172.
   - The latent basis has no meaning, so per-coordinate scaling is the wrong null model.

## Hypothesis (pinned before code)

Score arm `k` in ARM space through the pool's **squared-cosine kernel**, z-scored against the **i.i.d.-arm-noise null**:

```text
d_j   = fast_ema(p_j) − slow_ema(p_j)          the incumbent's own per-arm derivative (no pull vector)
v_j   ← EMA of (p_j(t) − p_j(t−1))²            per-arm step-noise variance
C_kj  = ⟨ĝ_k, ĝ_j⟩²                            fixed pool kernel, built once (n² floats)
z_k   = (C·d)_k / √(Σ_j C_kj² · v_j + ε)       null-normalized alignment
r̃_k   = sigmoid(β · |z_k|)
```

- **Squared cosines add instead of cancelling.** Mass moving onto `±e1` raises `(C·d)` at both arms, which addresses defect 1. `C·d` is `ĝ_kᵀ (dM) ĝ_k` for the second moment `M = Σ p_j g_j g_jᵀ`, so this is the second-moment drift read along each arm's axis.
- **The null normalization is basis-free.** Under i.i.d. arm noise, `z_k` has unit variance for every arm whatever the pool geometry, which addresses defect 2.
- **Cost:** `O(n)` per candidate after an `O(n²)` matrix-vector product per observation. At 16 arms that is 256 FMAs, comparable to the current `O(n·dim)` pull. At 64 arms it is 4096, the same as the current pull at dim 64.
- **Off-pool candidates** (perturbation on) need `⟨ĝ_k, g_j⟩²` for all `j`, which costs `O(n·dim)` per candidate. Pool-indexed candidates use `C` directly.

## Pre-registration amendment (2026-09-26, before any run)

- **β_z = 1** for the z-score form. `z` is in null-standard-deviation units and `fast_sigmoid` returns exactly 1.0 above 40, so β = 4 would tie every arm with `|z| > 10` and blind the rank-AUC. At β_z = 1 a 2σ excursion maps to `sigmoid(2) ≈ 0.88`, the same point the first-moment form reaches at `|cos| = 0.5` with β = 4.
- **Priorities are simplex-normalized** (`p / Σp`) before the kernel. `renormalize_priorities` rescales to max = 1, and every max change would otherwise inject a common-mode drift proportional to `p_j`.
- **`v_j`** uses the existing `DEFAULT_SCALE_ALPHA = 0.05`. The first observation only primes it.
- **Pool-indexed candidates** read `z` at their arm, since the reward goes to that arm. Off-pool candidates compute the kernel on the fly.
- **Implementation:** a `DriftSummary` strategy (`FirstMomentDrift`, `SecondMomentDrift`) behind one `TrajectoryAlignedCuriosity<S>` sampler.

## Pre-registered bars

Same fixture as Bench 900 (dim 16, F 8 at `e0 + 0.2·N`, S at `±e1..±e4`, 16/32 seeds). The shipped first-moment form stays as the comparison arm.

- **G1:** mean AUC(F vs S) ≥ 0.8 **and** held-out AUC ≥ 0.8.
- **Negative control:** pure-noise `|mean AUC − 0.5| ≤ 0.15`.
- **G3 forward** (F better): beats MatchedUniform **and** GlobalNorm, with paired 95% CIs excluding 0.
- **G3 reversed** (S better, promoted from characterization to a gate): not significantly worse than MatchedUniform, i.e. paired Δcycles CI lower bound ≤ 0.
- **G4:** 0 allocations warm; `sample_candidates` ≤ 2× `DerivativeCuriosity::sample_candidates` (interleaved, release).

**Honest-null clause:** if the redesign fails G1 or the reversed-reward gate, record it in Research 591 and retire guide 389 P2 until a new mechanism exists. Do not re-tune κ, β or the EMA rates after seeing results.

## Tasks

- [ ] T1 — `SecondMomentAlignment` (or a `DriftSummary` strategy on `TrajectoryAlignedCuriosity`, whichever keeps one sampler): fixed per-arm state, `C` built at construction, zero-alloc observe and score. Unit tests: `±e_i` visibility; unit null variance on i.i.d. noise.
- [ ] T2 — run the Bench 900 fixture against the bars above as a new arm in `tests/plan_610_arm_drift_alignment_goat.rs`, keeping the first-moment rows as the comparison.
- [ ] T3 — record the result: Bench 901, Research 591 status, guide 389 P2. Promote the redesign only if every bar passes; otherwise apply the honest-null clause.
