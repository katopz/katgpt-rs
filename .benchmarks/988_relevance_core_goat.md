# Bench 988: relevance_core (ACE) GOAT gate — planted nonlinear head, G0–G6

**Status:** DONE — G0–G6 PASS (synthetic, disclosed). Plan 640 Phase 1; Research 683 §8 bars, pre-registered in the plan before the first run.

## Provenance

- Box: m3 (owner laptop, AC). No latency claim is made: the gate is an accuracy/basis table, deterministic (fixed seeds, f64 sequential folds), so no preflight bar applies.
- Tree: katgpt-rs develop (`basis_split` row-layout repair `9cae0610f` + Phase 0 `2d50c0f06`) + this commit.
- Command: `cargo test --release -p katgpt-core --features relevance_core --test bench_640_relevance_core_poc -- --nocapture` (dev profile also green, 1.80 s).

## Fixture

D=32, planted rank R=3 (unit-variance coordinates), C=4 classes. A decoy direction is orthogonal to `Q_p` with variance 25, plus isotropic noise σ=0.3. The frozen head is `f_c(h) = Σ_j A[c,j]·tanh(1.5·q_j·h)`. It reads exactly `span(Q_p)`, and its Jacobians vary per sample (both fixture refusals armed: Jacobian relative variance > 0.01, energy outside the planted span < 1e-6). Labels are `argmax f(h)`. `S` comes from 128 samples of central finite differences (ε=1e-5) through `accumulate_sens`. ACE runs on the **population** Cov: the pre-registered G6 constraint (decoy uncorrelated with planted coordinates *in Cov*) holds exactly only there. The sample-Cov rows are disclosed below.

## Results (k = R = 3, every arm scored through the same frozen head after `project_through_into`)

| arm | labels | acc | contrast vs random | sim vs planted (f64) |
|---|---|---|---|---|
| ACE | 0 | **1.0000** | **+0.4658** | **1.0000** |
| ridge probe (R557) | 1024 | 0.8677 | +0.3335 | 0.5389 |
| ridge probe | 32 | 0.6265 | +0.0923 | 0.0553 |
| ridge probe | 8 | 0.5190 | −0.0151 | 0.2293 |
| PCA | 0 | 0.7944 | +0.2603 | 0.6664 |
| random (seeded) | — | 0.5342 | — | — |
| residual (complement of ACE) | — | 0.3853 | — | — |

full = 1.0000 · chance (majority class) = 0.4209 · ACE r-sweep: k=1 0.6523, k=2 0.7979, k=3..5 1.0000.

| gate | bar | measured | verdict |
|---|---|---|---|
| G0 | BLAKE3 ×2 over the f64 (Q, energies), two independent fixture builds | identical | PASS |
| G1 | sim ≥ 0.90 · aligned ≥ 0.9·full · residual ≤ chance+0.05 | 1.0000 · 1.0000 · 0.3853 ≤ 0.4709 | PASS |
| G2 (falsifier) | ACE contrast ≥ probe(1024) − 0.05 | +0.4658 vs +0.3335 | PASS |
| G3 | S = I ⇒ ACE ≡ PCA, sim ≥ 0.99 | 1.000002 at k=1 and k=R+1 | PASS |
| G4 | extract (warm scratch) + eval zero-alloc; fit allocs counted | 0 · 0 · fit with scratch build = 13 | PASS |
| G5 (strict gain) | ACE contrast ≥ probe(8) + 0.05 | +0.4658 vs −0.0151 | PASS |
| G6 | PCA top-1 on the decoy ≥ 0.9 · ACE sim ≥ 0.90 | 0.9999 · 1.0000 | PASS |

## Disclosed findings (not gated)

1. **Estimation sensitivity to Cov along high-variance directions.** On the SAMPLE Cov, ACE's mapped-back `Q = orth(Cov^{1/2} V)` inherits the sample's O(1/√N) decoy↔planted cross-covariance, amplified by the decoy amplitude: N=4096 gives acc 0.8711, sim 0.9939, decoy leak 1.70e-2; N=65536 gives acc 0.9819, sim 0.9998, decoy leak 5.47e-4. Even at N=4096 ACE still clears G2 parity (+0.3369 vs +0.3335) and G5 by a wide margin. **Consumer rule:** budget the Cov sample count against the largest variance ratio (leak ∝ a_dec/√N), or shrink the cross-block before extraction.
2. **G3 is only well-posed at a spectral gap.** The first draft compared at k=R, which cuts inside the degenerate planted eigenspace (three eigenvalues 1+σ²), where any basis is arbitrary (sim 0.72). The bar was re-run at the gapped boundaries k=1 and k=R+1 on the SAME Cov PCA reads. This repairs the control's specification; the bar value (0.99) is unchanged.
3. **The probe incumbent is linear against a nonlinear head.** Part of the ACE > probe(1024) margin is a property of the fixture: the probe's basis similarity is 0.54. G2's bar is parity, so the margin over parity is not claimed as a gain. The claimed gain is G5 (zero labels vs 8).

## Verdict

GOAT gate PASS on the planted fixture. Per the Feature Flag Discipline: opt-in remains, and promotion needs a wired consumer passing its own gate. The first candidate is riir-infer Issue 1237 (census).
