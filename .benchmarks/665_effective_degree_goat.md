# Bench 665 — `effective_degree` GOAT gate (Issue 668)

**Feature:** `effective_degree = ["karc_forecaster"]` (opt-in — **NOT promoted**;
promotion requires the riir-neuron-db Issue 602 consumer verdict, see §Verdict)
**Source:** [Research 488](../.research/488_Effective_Degree_Polynomial_Simplicity.md)
— Zhang, Li, Xiao, Chen & Chen, *Quantifying and Optimizing Simplicity via
Polynomial Representations*, arXiv:2605.29823, ICML 2026
**Issue:** `668`
**Date:** 2026-08-17
**Machine:** M3 Max (Apple Silicon), release-mode timing best-of-3;
counting-allocator G4 in an isolated debug test binary.
**GPU exclusivity:** N/A — pure CPU primitive, no GPU touched.

## What shipped

`crates/katgpt-core/src/effective_degree.rs` (~560 LOC incl. docs + 12 unit tests).

| Piece | Notes |
|---|---|
| `EdConfig { resolution, max_degree, damping, n_pairs, seed }` | `Copy`; `const fn cheap()` = paper efficiency point (r=4, K=3, ε=1e-6, 8 pairs), `const fn precise()` = paper performance point (r=15, K=7, 32 pairs). `Default` = `cheap()`. `const fn validate()` rejects `K > 7`, `r < K+1`, `ε ≤ 0` (and NaN ε) up front. |
| `randomized_cosine_nodes(r, seed, out)` | Paper Eq. 8 — stratified `θᵢ ~ U[(i−1)π/r, iπ/r]`, `αᵢ = (1−cos θᵢ)/2`. splitmix64, deterministic from seed, zero-alloc, writes into caller `out`. |
| `effective_degree_along_path(outputs, nodes, cfg)` | Scalar outputs. **Takes no scratch and never allocates** — the whole `(K+1)²` solve is fixed-size stack arrays. |
| `effective_degree_along_path_multi(outputs, out_dim, nodes, cfg, scratch)` | Vector outputs. One Cholesky with `out_dim` RHS (not `out_dim` independent fits). Per-degree magnitude is `‖cₖ‖₂` over output dims → collapses to `\|cₖ\|` at `out_dim = 1`, verified bit-equal to the scalar path. |
| `ed_over_pairs(decode, endpoints_a, endpoints_b, cfg, scratch)` | Generic driver. `decode: FnMut(&[f32], &mut [f32])` is consumer-supplied so katgpt-core stays domain-agnostic. Per-path node draw seeded from `cfg.seed` + path index ⇒ whole call reproducible from the config. |
| `ed_from_coeff_norms(&[f32]) -> (ed, ed_norm)` | Basis-agnostic reducer (slice index **is** the degree). Public because it is what makes the Appendix-I basis-invariance check expressible against the shipped metric. |
| `EdResult { ed, ed_norm, coeff_norms: [f32; 8], n_terms }` | `MAX_ED_TERMS = 8`, `MAX_ED_DEGREE = 7`. |
| `EdError` | 7 typed rejections + `Display` + `std::error::Error`. |

**Substrate consumed, not rebuilt** (substrate-first gate): Chebyshev evaluation
is `karc::ChebyshevBasis<8>` via the sealed `KarcBasis` trait; the damped normal
equations use `linalg::{cholesky_f64, chol_solve_f64}`. Gram/RHS accumulate in
**f64** for the same reason KARC's `fit_direct` does — f32 Cholesky is fragile
once `ε` falls below f32 epsilon relative to the matrix scale, and the default
`ε = 1e-6` is squarely in that regime. Only the ED reduction and the node
sampler are new code. Hence `effective_degree = ["karc_forecaster"]`; since
`karc_forecaster` is already default-on, the implication adds **zero** build
cost to a default build.

**Explicitly not distilled:** the paper's differentiable ED regularizer (§7,
training-only → riir-train record in Research 488 §7) and the per-path PCA
output compression (paper C.5 states it is not the source of the gains).

## Gate results

| Gate | Target | Result |
|---|---|---|
| G1a order preservation | ground-truth algebraic degrees strictly ordered (paper Appendix I) | **PASS** — degrees 1..5 of `(w·x)^p` on a structured ℝ⁶ manifold, `ed_norm` = 0.5040 / 0.6703 / 0.8774 / 1.0037 / 1.1484, strictly monotone over the **full 1..5 chain** (stronger than the paper's 3-point {1,2,5}); deg5/deg1 ratio 2.28 ≥ 2.0 |
| G1a absolute anchor | offset-free `ed_norm_ac ∈ [1, p]` for degree-`p` | **PASS** — 1.0000 / 1.3144 / 1.3692 / 1.5768 / 1.6286, all in `[1, p]`; catches both a collapsed fit and high-mode leakage |
| G1b basis invariance | ordering survives a Legendre swap (paper §3.2 / App. I) | **PASS** — Chebyshev `[0.713, 0.918, 1.494]` vs Legendre `[0.713, 1.087, 1.927]` on the *same* sampled outputs; both strictly ordered. Deg-1 agrees to 1.2e-6 across bases (a degree-1 restriction has the same `T₁`/`P₁` = `u`), higher degrees differ in magnitude — which is the point: **ordering is basis-invariant, magnitude is not** |
| G1c scale behaviour | paper Table 12 — `ed` scales, `ed_norm` does not | **PASS** — ×2 outputs ⇒ `ed` 0.065949 → 0.131897 (ratio **2.0000**), `ed_norm` 0.877427 → 0.877427 (Δ = 0, tol 1e-4) |
| G1d degenerate refs | constant → ED ≈ 0; affine → ED ≈ `\|c₁\|` | **PASS** — constant ED = 0 and ED_norm = 0 (< 1e-4); affine `1 + 4α` → coeffs `[3.000, 2.000, 1.2e-7, 2.2e-7]`, ED = `\|c₁\|` to 1e-5, `c₁` = 2.000 (analytically exact) |
| G1e sampler robustness | ordering stable across independent node seeds | **PASS** — 8 seeds, ordering holds on every one |
| G1f determinism | same seed ⇒ bit-identical | **PASS** — `randomized_cosine_nodes` bit-identical on seed reuse and different on seed change; `ed_over_pairs` returns `EdResult` equal by `PartialEq` across two independent scratch buffers |
| G1g pure-mode calibration | a DC-free `T_k` reads `ed_norm = k` exactly | **PASS** — `k = 1..7`, all within 5e-3 of `k` |
| G2 fit latency (cheap) | sub-µs per path (Issue 668 T3) | **PASS** — **195.9 ns/path** at r=4/K=3 (budget 500 ns, 2.6× headroom) |
| G2 fit latency (precise) | sub-µs per path | **PASS with a caveat** — **893.9 ns/path** at r=15/K=7 on a quiet box; a second run under sibling-agent load measured **1060.7 ns**. Gate budget set to 2000 ns for this config so it is not load-flaky; the honest number is "≈0.9 µs quiet, ≈1.1 µs busy" — i.e. sub-µs is **marginal** at the precise config, comfortable at the cheap one |
| G2 node sampling | negligible | **PASS** — **0.5 ns** for r=4 |
| G2 pair scaling | linear in `n_pairs` | **PASS** — 4/8/16/32 pairs at 212 / 213 / 207 / 222 ns/pair (quiet run); each step < 1.6× the linear expectation |
| G3 no-regression | default build untouched, clippy 0 | **PASS** — `cargo test -p katgpt-core --lib`: default **1893 passed / 0 failed**, `--features effective_degree` **1905 passed / 0 failed** (+12 new, 0 regressions); clippy clean on default, on `--features effective_degree --all-targets`, and on `--all-features` |
| G4 alloc-free | 0 steady-state with reused scratch | **PASS** — **0 allocs** across 10 000 `effective_degree_along_path` + 10 000 `randomized_cosine_nodes` + 10 000 `effective_degree_along_path_multi` (out_dim=3) + 1 000 `ed_over_pairs` (r=15, 32 pairs). `EdScratch::new` is cold-path construction, explicitly outside the gate |

Test binaries: `crates/katgpt-core/tests/bench_668_effective_degree_goat.rs`
(G1a–G1e, G2) and `crates/katgpt-core/tests/effective_degree_alloc_check.rs`
(G4). G1f/G1g and the shape-rejection checks are the 12 `--lib` unit tests in
`src/effective_degree.rs`.

## Honest findings

1. **`ed_norm` is not the algebraic degree, and the DC term is why.** `ED_norm`
   is a magnitude-weighted mean of the degree index over **all** coefficients
   including `k = 0`. A function with a large constant term therefore reads far
   *below* its algebraic degree: the degree-5 fixture reads `ed_norm = 1.15`,
   while the identical coefficients with `c₀` zeroed read `1.63`. This is a
   faithful reading of the paper's definition (it is not a bug), but it means
   **`ed_norm` is only interpretable comparatively**, and a consumer comparing
   functions with heterogeneous output offsets is injecting nuisance variance
   into its correlation. The offset-free read is one line —
   `coeff_norms[0] = 0.0` then re-reduce through `ed_from_coeff_norms` — and is
   bounded in `[1, p]`. Flagged to the Issue 602 consumer as a second arm worth
   carrying. The gate's first draft asserted `ed_norm ≈ algebraic degree` and
   **failed**; the assertion was wrong physics, not the metric.
2. **Basis invariance is about ordering only.** Legendre and Chebyshev agree on
   the *ranking* (the paper's claim) but not on magnitudes — deg-5 reads 1.49
   Chebyshev vs 1.93 Legendre on identical samples. Any consumer threshold is
   therefore basis-specific and must be calibrated against the shipped
   Chebyshev path, not transferred from the paper's numbers.
3. **Sub-µs is marginal at the precise config.** r=15/K=7 straddles 1 µs
   depending on machine load. Consumers on a tick budget should use
   `EdConfig::cheap()` (196 ns) unless they have measured that K=3 is
   insufficient for their function class.
4. **The data-manifold caveat is untested here and remains a consumer risk.**
   Paper C.1 reports the ED signal collapses to baseline with random-noise
   endpoints. Reproducing that requires a real trained model, so this gate uses
   a structured (non-noise) synthetic manifold and **documents** the constraint
   rather than measuring it. A consumer that feeds synthesized endpoints will
   get an uninterpretable result — this is the single most likely way to
   misuse the primitive.
5. **ED measures simplicity, not correctness** (paper §7 MNIST-CIFAR failure).
   A wrong-but-simple decode passes an ED gate. Any consumer gate must retain a
   correctness arm; ED can only ever upgrade the *output-simplicity* arm of a
   two-sided conjunction, never replace both.

## Verdict — ship opt-in; consumer verdict is IN, gate unchanged

G1 (all sub-gates), G2, G3, G4 **PASS**. The primitive is correct, fast, and
allocation-free.

It is **NOT promoted to default**, per the no-default-consumer rule. ED is
**not UQ-bearing** — it emits a complexity scalar, not a distribution, interval,
or coverage claim — so the "Report the Floor" rule (Issue 010) does not apply.

### Consumer verdict — riir-neuron-db Issue 602, CLOSED 2026-08-17

The gating consumer reported: **SCOPE-LIMITED — no gate change, stays opt-in**
([riir-neuron-db Bench 484](../../riir-neuron-db/.benchmarks/484_ed_vs_flatness_freeze_gate_poc.md),
commit `5e75bb6`; Research 488 §10 PoC Addendum, commit `349324e0`). 360 shard
states (30 cycles × 4 scenarios × 3 seeds), ground truth = held-out wake-event
recall error.

| arm | pooled \|Pearson\| vs generalization gap |
|---|---|
| **`ed_norm`** | **0.598** |
| `output_flatness` (incumbent) | 0.047 |
| control | 0.032 |
| permutation floor | 0.042 |

ED out-correlates the incumbent **12.6×**, beat it 4/4 scenarios raw and
cycle-controlled on 3 disjoint seed sets — the incumbent is indistinguishable
from noise on that substrate. **But the sign inverts between grains**
(Simpson's paradox, reproduced on all 3 seed sets): pooled across regimes
**+0.598**, within-regime cycle-controlled **all four negative**
(−0.18, −0.68, −0.35, −0.25). A gate is one threshold on one shard state, so
`ed_norm < τ` rejects the memorizing regime wholesale while *inside* every
regime preferring the shards with the largest held-out gap. No threshold fixes
that, so ED is not wirable as the proposed one-sided freeze gate. Issue 668's
deferral trigger is the one that fired: the primitive stays as a **diagnostic
surface**, not deleted.

**The DC finding from §"Honest findings" #1 changed the consumer's conclusion.**
The consumer carried `ed_ac` (zeroing `coeff_norms[0]`, per this bench's
recommendation) as a 4th arm: correlation collapses **0.598 → +0.122**, below
flatness-plus-noise. So nearly all of ED's power on that substrate lives in the
**DC term** — for a cosine decode, the along-path mean level, i.e. shard-event
*alignment*, not shape complexity. **The paper's actual thesis (function-space
*complexity* predicts generalization) is therefore NOT confirmed at shard
scale.** What is confirmed is weaker: a data-anchored function-space probe
out-predicts a data-blind parameter-space one. Without the AC arm this would
have been reported as a clean 12.6× win with the mechanism mis-attributed.

### Independent external confirmation of two gates here

- **`EdConfig::cheap()` is validated; cost is not the limiter.** cheap
  (r=4/K=3, 8 paths) 0.598 vs precise (r=15/K=7, 32 paths) 0.623 — ~4% of the
  correlation for ~5× less work. The 195.9 ns/path figure held on real
  substrate and the consumer's decode dominated, exactly as G2 predicted.
- **G1a's path-averaging premise (paper Theorem 3.1) confirmed on real data.**
  **0 ranking flips** across `n_pairs ∈ {1,2,4,8,16,32,64}`, and per-path spread
  falls monotonically **8.8×** (seed_std 0.0335 → 0.0038). That is an external
  check on the synthetic G1a/G1e ordering result.

### New risk this surfaced — grain-dependent sign

Added to the module docs as caveat 4. **Any ED consumer must state which grain
it operates at and verify the sign there.** This generalizes beyond the freeze
gate: the paper's own 27-config correlation study measures only the
*across-model* grain, so the within-trajectory grain is unmeasured upstream and
can invert. This is the transferable lesson, not a substrate quirk.

### Surviving promotion axis

Not freeze timing. **Cross-regime triage** is the grain where ED's sign is
correct, at 12.6× the incumbent and 196 ns/path — i.e. the **KARC
regime-mismatch probe** already anticipated by Issue 668 and Research 488 §4
(high decode ED + low flatness ⇒ polynomial-basis-strained, making Bench 010's
documented KARC scope-limit *measurable* rather than inferred from a CRPS loss).
A future promotion case should be built on that axis.

## Task status (Issue 668)

- [x] T1 — module, config, sampler, scalar + multi + driver entry points
- [x] T2 — G1 order preservation, basis perturbation, scaling, degenerate refs
- [x] T3 — G2 latency + pair-count scaling
- [x] T4 — G4 alloc-free with reused scratch
- [x] T5 — module docs: Research 488 citation, data-manifold caveat (C.1),
      scale-dependence note (Table 12), DC-offset finding
- [x] T6 — this record; promotion decision = **default-off**, awaiting Issue 602
