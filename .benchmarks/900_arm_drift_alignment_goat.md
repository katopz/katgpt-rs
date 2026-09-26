# Bench 900 — `arm_drift_alignment` GOAT gate (Plan 610)

**Status:** GOAT **FAIL** — the feature stays opt-in. G1 missed both pre-registered bars. G3 passed as pre-registered, but a post-hoc control shows the win is geometry-specific. The redesign is filed as Issue 899.

**Date:** 2026-09-26 · **Plan:** [610](../.plans/610_arm_drift_alignment.md) · **Research:** [591](../.research/591_Trajectory_Aligned_Curiosity.md) · **Guide:** riir-ai `.research/389` · **Test:** `tests/plan_610_arm_drift_alignment_goat.rs`

**Box state:** M3 Max, AC power (100%, charged), loadavg 5.0/6.0/6.2, 85% memory free. A sibling session's Zed `cargo test` was compiling in another target dir throughout. Release profile, `--test-threads=1`. Every timing is an interleaved median-of-ratios (`tests/common/ab_timing.rs`), so load drift cancels within each pair.

**Pre-registration:** the fixture and every bar were committed in Plan 610 at `9451fd5ba` before any gate ran. One harness defect was corrected after the first run, and it is recorded below.

## What was measured

`r̃_k = sigmoid(β·|⟨ĝ_k, û⟩|)`, where `û` is the preconditioned drift of the **priority-weighted mean pull** `m = Σ_j p_j g_j`:
- The drift is `d = fast_ema(m) − slow_ema(m)` from the 10:1 temporal derivative kernel.
- Each coordinate is divided by its drift scale `s_j` (EMA of `|d_j|`) plus a relative floor `κ·max s`, with `κ = 0.1`.

Plan 610's design correction explains why the drift is taken in latent space: Research 591's arm-space drift cannot be dotted against latent directions.

Fixture: dim 16, 16 arms.
- **F** is 8 arms at `e0 + 0.2·N(0,I)`.
- **S** is 8 arms at `±e1..±e4`, orthogonal to the drift axis with zero centroid.
- 16 seeds for G1 and G2, 32 paired seeds for G3.

## Results

| Gate | Readout | Bar | Result |
|---|---|---|---|
| G1 planted drift | AUC(F vs S), mean / min | ≥ 0.8 | **0.783** / 0.594 — FAIL |
| G1 planted drift | held-out AUC (`F[4..8]` vs S) | ≥ 0.8 | **0.602** / 0.250 — FAIL |
| G1 negative control | pure-noise AUC(F vs S) | \|m − 0.5\| ≤ 0.15 | 0.395 (sd 0.173) — PASS |
| G1 scale invariance, κ = 0 | max \|û − û'\| | ≤ 1e-4 | 2.09e-7 — PASS |
| G1 scale invariance, κ = 0.1 | max \|ΔAUC\| / min Kendall τ | ≤ 0.02 / ≥ 0.9 | 0.156 / 0.672 — FAIL |
| G2 incumbent `DerivativeCuriosity` | AUC | — | 0.500 (tied by construction; trivial) |
| G2 own-arm drift `\|fast_j − slow_j\|` | held-out AUC | < 0.8 | 0.496 — PASS (the incumbent family fails) |
| G3 loop A/B | cycles to F ≥ 0.75 mass: Aligned / GlobalNorm / MatchedUniform / ExtrinsicOnly | A beats B and C, CI excl. 0 | **229.8** / 409.6 / 348.9 / 471.9 — PASS |
| G3 paired Δcycles | Aligned − GlobalNorm / − MatchedUniform / − ExtrinsicOnly | — | −179.8 [−213.4, −146.2] / −119.1 [−145.8, −92.3] / −242.1 [−276.6, −207.6] |
| G3 secondary | extrinsic reward over 300 cycles, Aligned − MatchedUniform | — | +8.8 [+7.2, +10.3] |
| G3 pin | aligned sampling vs bare `PoolConjecturer`, same seed | bit-identical | PASS (scoring is a pure side channel) |
| G4 allocations | 1000 warm `cycle_aligned` | 0 | 0 — PASS |
| G4 cost | `sample_candidates` aligned / incumbent (16 arms × dim 16) | ≤ 2.0× | **1.57×** (97.3 → 153.2 ns; rounds 1.32–1.72) — PASS |
| G4 plan's literal bar | observe only, aligned / incumbent | ≤ 2.0× | 4.38× (12.1 → 52.3 ns) — FAIL, as the design correction predicted (`O(n·dim)` pull vs `O(n)`) |

### Characterization rows (not gates; the post-hoc ones are labeled)

| Row | Result |
|---|---|
| κ = 0 (pure per-coordinate normalization), G1 held-out AUC | 0.172 — worse than chance |
| Preconditioner OFF (`û = d/‖d‖`), G1 AUC / held-out | 1.000 / 1.000 (post-hoc) |
| Preconditioner OFF, pure-noise AUC | **1.000**: it favors F with no drift at all |
| S with non-zero centroid (`+e1..+e8`), G1 AUC | 0.727, since `\|·\|` credits the anti-drift family by design |
| **Reversed reward** (S is the better family), cycles to S ≥ 0.75 | Aligned **572.6** (20 of 32 censored at 600) vs 440.9 / 464.6 / 468.4. Aligned − each: **+131.7 / +108.0 / +104.2** cycles, all CIs exclude 0 (post-hoc) |

## Verdict: three structural findings

1. **Within the preconditioned form, no κ passes both G1 and the negative control.**
   - With the preconditioner off, the score is a cluster-density prior. It scores F at 1.0 with no drift at all, because 8 arms on `e0` make every fluctuation of `m` largest along `e0`.
   - With the preconditioner on, that prior is gone (noise AUC 0.395), but the planted signal goes with it. The drivers' mean direction carries noise components within one decade of the `e0` drift. The floor at κ = 0.1 does not suppress them, so they are inflated to parity with the drift axis: `û` spreads over 16 coordinates, and held-out F arms score like S arms.
   - At κ = 0 it is worse (0.172). An axis-aligned preconditioner in a latent basis with no meaning cannot separate drift signal from pool geometry.
2. **The mean pull is blind to drift toward a spread family.** Moving mass onto `±e_i` pairs moves `m` only by shrinking F's share. That reads as drift away from F, and `|·|` credits F. So aligned reinforces the wrong family and loses by more than 100 cycles when S is the better family. The G3 PASS is momentum toward a **coherent** cluster, not a general acceleration.
3. **The incumbent's Solver-free cycle allocated on every cycle.** `DerivativeCuriosity::cycle_curiosity` built `Candidate::new(Direction::zeros(dim), …)` as a `resize` default even when the resize was a no-op: one heap Vec per cycle. The same shape was copied into `cycle_aligned` and caught by G4. Both now use the existing `ScratchBuffers::ensure_len`, as `CgspLoop::cycle` always did. G4 asserts 0 allocations over 1000 warm cycles for both. The incumbent's G5 now reads 132.7 ns/cycle; there is no before/after timing A/B, so no speedup is claimed.

**Demotion clause applied:** the feature stays opt-in and nothing was re-tuned. The G3 win is recorded together with its reversal. Guide 389's P2 game-side fusion is **refuted at the current design**.

## Harness correction (after the first run)

The first G3 run reported `cycles-to-acquire = 1.0` for every arm, which is impossible from a 0.5 starting share. `renormalize_priorities` rescales so the max is 1, not so the sum is 1, and the readout was summing raw priorities. The pre-registered readout is "F holds ≥ 0.75 of the priority **mass**", so the instrument now divides by the total. The bar is unchanged. Every G3 figure above comes from the corrected readout.

## What the test asserts

The test asserts the measured verdicts as pins, so it is green today and reds when a verdict changes:
- G1 remains FAIL and κ = 0.1 invariance remains FAIL.
- G3 remains PASS, and the reversed-reward sign stays positive.
- The preconditioner-off prior stays above 0.95.

The gates that pass are hard asserts: the negative control, κ = 0 invariance, G2, both G4 alloc rows, and the G4 cost bar in release.

## Next: Issue 899

The hypothesis to test is a **second-moment, null-normalized** score. Take the priority drift through the pool's squared-cosine kernel, `(C·d)_k` with `C_kj = ⟨ĝ_k, ĝ_j⟩²`. Then z-score it against the i.i.d.-arm-noise null, `√(Σ_j C_kj² v_j)`, where `v_j` is each arm's noise variance.
- `C` sees drift toward `±e_i` pairs, whose squared cosines add instead of cancelling (finding 2).
- The null normalization is what the per-coordinate preconditioner was meant to do, but it is basis-free, which addresses finding 1.

Reproduce: `cargo test --release --features arm_drift_alignment --test plan_610_arm_drift_alignment_goat -- --nocapture --test-threads=1`
