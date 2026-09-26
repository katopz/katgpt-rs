# Bench 901 — second-moment, null-normalized drift alignment (Issue 899)

**Status:** GOAT **FAIL** on 2 of 6 pre-registered bars (G1 held-out 0.766 < 0.8; G4 2.07× > 2.0). It passes the other four: G1 F-vs-S, the negative control, G3 forward and G3 reversed. It is the only drift summary that stays sound in the loop in both directions. It is now `TrajectoryAlignedCuriosity`'s default summary, and the feature stays opt-in.

**Date:** 2026-09-26 · **Issue:** 899 ([HISTORY.md](../HISTORY.md)) · **Parent:** [Plan 610](../.plans/610_arm_drift_alignment.md), [Bench 900](900_arm_drift_alignment_goat.md) · **Test:** `tests/plan_610_arm_drift_alignment_goat.rs` (`issue_899_*`, `characterization_first_moment_preconditioner_off_loop`)

**Box state:** M3 Max on AC power, 87% memory free, loadavg **18.7** / 15.9 / 10.7 at the final run (sibling sessions building). Release profile, `--test-threads=1`. G4 is an interleaved median-of-ratios, so load cancels within each pair. It measured 2.07–2.08× in four runs across load levels (rounds 1.94–2.19).

**Pre-registration:** bars at `099b9d4a6` (amendment: β_z = 1, simplex shares); v2 at `b1615393d`, committed before any v2 run.

## What was measured

```text
s_j = p_j / Σp,  d_j = fast_ema(s_j) − slow_ema(s_j)   (kernel warm-started on the first observation)
v_j ← EMA_0.05 (s_j(t) − s_j(t−1))²,  C_kj = cos²(g_k, g_j)
z_k = (C·d)_k / √(Σ_j (C_kj − m_k)² v_j),  m_k = s·C_k       (v2: simplex-centered null)
r̃_k = sigmoid(|z_k|)
```

The fixture, seeds and bars are identical to Bench 900.

## Three variants, all reported

The rows are in the order they were measured. Each variant was pre-registered or disclosed before its run.

| Variant | G1 AUC F vs S | G1 held-out | Noise AUC | G3 fwd Δ vs own uniform | G3 rev Δ vs own uniform | G4 |
|---|---|---|---|---|---|---|
| v1 as pre-registered (i.i.d. null, zero-init kernel) | 0.963 | 0.926 | **0.949** FAIL | −82.1 [−110.1, −54.1] | −48.5 [−79.8, −17.2] | 2.02× |
| v1 + warm start (defect repair) | 0.816 | **0.633** FAIL | **0.186** FAIL | −66.2 [−102.2, −30.2] | −68.2 [−87.7, −48.6] | 1.74× |
| **v2** simplex-centered null (final) | **0.883** | **0.766** FAIL | **0.559** PASS | **−79.4** [−106.9, −51.9] PASS | **−62.6** [−92.5, −32.6] PASS | **2.07×** FAIL |

v2 against the other arms (32 paired seeds, cycles until the better family holds 0.75 of the priority mass):
- **Forward** (F better): Second 281.1, GlobalNorm 409.6, SecondUniform 360.5, ExtrinsicOnly 471.9.
- **Reversed** (S better): Second 372.8, GlobalNorm 440.9, SecondUniform 435.3, ExtrinsicOnly 468.4.
- Every paired CI excludes 0, in both directions.

**Allocations:** 0 over 1000 warm `cycle_aligned` calls. The `n×n` kernels are built once, at construction.

## What each step taught

1. **The v1 G1 pass came mostly from the zero-init transient.** The kernel's EMAs start at 0, and the slow EMA (α = 0.03) is still 5% unsettled at step 100, so every arm reads a common-mode upward drift during the read window. `C·d` reads that drift in proportion to each row's sum, which is larger in the dense F cluster. That inflated G1 and failed the noise control (0.949). Warm-starting both EMAs on the first observation is a defect repair, not a tunable. The prediction stated before the run was that noise AUC would fall toward 0.5. It fell past it, to 0.186, which led to finding 2.
2. **Simplex shares need a simplex null.** `Σ d = 0`, so the shares are negatively correlated. The i.i.d. denominator `√Σ C² v` over-states the null variance of dense rows. The correct null centers each kernel row by its share-weighted mean. Because `Σ d = 0`, that leaves the numerator exactly unchanged; only the denominator moves. v2 brings noise AUC to 0.559.
3. **The G3 pathology Bench 900 found is fixed.** Every v2 variant wins against its own matched-uniform bonus in **both** directions. The first moment loses by more than 100 cycles when the better family has a zero pull centroid. The second moment sees mass moving onto `±e_i` pairs, because their squared cosines add.
4. **Where it still fails.**
   - **G1 held-out, 0.766:** held-out F arms whose own share only shrinks through renormalization are ranked above S arms 77% of the time, not the 80% required. The shrink itself reads as a genuine negative `z` on S, and `|·|` credits it.
   - **G4, 2.07×:** four SIMD row dots per scored arm. A fused single-pass scalar loop measured slower (2.47×) and was reverted.

## Post-hoc characterization: preconditioner off (Plan 610's first moment, κ → ∞)

A floor that dominates every coordinate makes `û = d/‖d‖` exactly. With the transient repaired, this form passes G1 (1.000 / 1.000) and the noise control (0.525). In the loop it is the strongest momentum amplifier measured:

| Direction | Cycles | Δ vs MatchedUniform | Δ vs GlobalNorm |
|---|---|---|---|
| Forward | 116.1 | −246.0 [−283.5, −208.5] | −293.5 [−332.2, −254.8] |
| Reversed | 597.6 (almost always censored) | +131.5 [+98.2, +164.7] | +156.7 [+123.0, +190.5] |

So the first moment's blindness to spread families is structural, not a preconditioner artifact. The fixture's planted-truth G1 cannot see it, and the reversed loop can. **A planted-truth mechanism gate is not sufficient for this class of primitive; the loop in both directions is.**

## Forking-path disclosure

Three variants were run on one fixture. v2's gains could be partly fixture-fit. Its motivation, the simplex null, is derivable without the data, and it was committed before its run, but the fixture is the one that suggested it. No fourth variant was run, and β, the EMA rates and the kernel were never re-tuned.

## Decision

- **Not promoted:** 2 of 6 bars fail, and `arm_drift_alignment` stays default-off.
- **Default summary** is now `SecondMomentDrift`: it wins against the first moment on the noise control and the reversed loop, and it is sign-consistent. `FirstMomentDrift` stays as the comparison arm.
- **Guide 389 P2** (game-side fusion) stays retired under the honest-null clause. Reopen it only with a mechanism that passes G1 held-out, or with an owner decision that the loop gates, not planted-truth G1, are the right promotion criterion for this class. The measurements above argue for that; this bench does not assume it.

Reproduce: `cargo test --release --features arm_drift_alignment --test plan_610_arm_drift_alignment_goat -- --nocapture --test-threads=1`
