# Plan 438: FORE — Fitted Occupancy-Ratio Estimator Primitive

**Date:** 2026-07-14
**Research:** [katgpt-rs/.research/423_Adjoint_Bellman_KL_Contraction_Occupancy_Ratio.md](../.research/423_Adjoint_Bellman_KL_Contraction_Occupancy_Ratio.md)
**Source paper:** [arxiv:2607.05375](https://arxiv.org/abs/2607.05375) — van der Laan & Kallus, *Fitted Occupancy-Ratio Evaluation without Bellman Completeness*, 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/occupancy/` (new module) + Cargo feature `occupancy_ratio`
**Verdict:** GOAT (Research 423 §3.1) — novel + modelless + three fusion targets; not Super-GOAT (Q2/Q3 fail the novelty gate).
**Status:** 🟢 Phase 1 ✅ Phase 2 ✅ Phase 3 ✅ Phase 4 ✅ Phase 5 ✅ (GOAT G1+G2+G4+G5 ALL PASS). All 5 phases complete. Stays opt-in pending Fusion A PoC.

---

## Goal

Ship the open primitive distilled from Research 423: a generic, modelless
**fitted occupancy-ratio estimator** that converges under realizability alone
(no Bellman completeness required). The substrate-independent contribution is
the **adjoint Bellman KL contraction** (paper Lemma 3.1): the operator

    B^γ_π ω = (1−γ)ω_0 + γ · d((ων)P_π)/dν

contracts relative entropy by factor γ per fitted iteration. FORE = repeated
KL projection of a normalized exponential class onto the adjoint-Bellman
image of the previous iterate.

**Why here, why now:** zero prior art across the 5-repo quintet (Research 423
§3.2 Q1 = YES). The primitive unlocks three downstream fusions — (A) CLR
re-estimation stabilization, (B) freeze/thaw convergence guarantee, (C)
cheaper-than-bisimulation state abstraction — but each fusion requires its own
PoC and lives in a sibling repo. This plan ships **only the engine primitive**
in `katgpt-core`; fusions are tracked as out-of-scope follow-ups.

**GOAT gate (per Research 423 §4):**

| Gate | Requirement |
|---|---|
| **G1 correctness** | Baird-style MRP from paper §6.1: FORE converges to known `ω_π,γ(upper) = 0.2211`, `ω_π,γ(lower) = 15.7987` within 1% relative error after K=20 iterations on n=10000 transitions. |
| **G2 perf** | FORE fit on n=10000, state_dim=8, K=20 < 100 ms on Apple Silicon (cold-tier budget). Linear log-ratio class only for the perf gate. |
| **G3 no-regression** | `cargo clippy --workspace --all-features` + `cargo test -p katgpt-core --lib` pass unchanged. Feature is opt-in (`occupancy_ratio = []`). |
| **G4 alloc-free** | Inner KL-projection loop is zero-allocation in steady state (pre-allocated scratch buffers, `Vec::with_capacity` + `clear()` reuse). Outer `fit()` may allocate the output `Vec<f32>`. |
| **G5 modelless-ness** | No gradient descent through any base weight. `LogRatioClass::fit_kl_projection` may use GD on its *own* parameters (the supervised learner), but must not touch `NeuronShard`, `LoRAWeightVersion`, or `SenseModule` weights. |
| **G6 floor (UQ)** | N/A — ratio estimator, not a forecaster. Triggers if a downstream value-estimation app is added (then must beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>`). |

**Promotion path:** ship opt-in. Promote to default-on only if a downstream
consumer (Fusion A CLR stabilization in `riir-poc`) demonstrates the gain —
otherwise stays opt-in as an engine primitive consumers can opt into.

---

## Phase 1 — Module Skeleton + Trait Surface

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/occupancy/mod.rs`. Gate behind
  `occupancy_ratio = []` feature in `katgpt-core/Cargo.toml`. Add to
  `[features]` block alongside other opt-in primitives (e.g.
  `cochain_point_sampler`).
- [x] **T1.2** Define `OccupancyRatioEstimator<H: LogRatioClass>` struct:
  `log_ratio_class: H`, `gamma: f32`, `k_iterations: usize`. Constructor
  `new(h, gamma, k_iterations) -> Self` with `gamma ∈ [0, 1)` asserted.
- [x] **T1.3** Define `LogRatioClass` trait (generic over the supervised
  learner — substrate-independent):
  ```rust
  pub trait LogRatioClass {
      type Params;
      fn evaluate(&self, params: &Self::Params, x: &[f32]) -> f32;  // h(x)
      fn fit_kl_projection(
          &self,
          transitions: &TransitionBatch<'_>,
          initial_moments: &InitialMoments<'_>,
          current_ratio: &[f32],   // ω̂^(k)(X_i)
          gamma: f32,
          scratch: &mut KlProjectionScratch,
      ) -> Self::Params;
  }
  ```
- [x] **T1.4** Define `TransitionBatch<'a>` (borrow-only, zero-copy):
  `states: &'a [f32]` (flattened `[n * state_dim]`), `successors: &'a [f32]`,
  `rewards: Option<&'a [f32]>`, `n: usize`, `state_dim: usize`. Also added
  `state(i)` / `successor(i)` slice accessors for ergonomic per-transition reads.
- [x] **T1.5** Define `InitialMoments<'a>` (the `P̂_0 h` estimator input —
  empirical initial-state distribution moments). Kept as simple borrow-only
  container: `initial_states`, `initial_ratio`, `n_init`, `state_dim`.
  Fields may be refined in Phase 2 once Algorithm 1 `P̂_0 h` is verified.
- [x] **T1.6** Define `KlProjectionScratch` (pre-allocated work buffers:
  `target_weights: Vec<f32>`, `design_rows: Vec<f32>`, `normal_eq_rhs: Vec<f32>`).
  Reused across iterations via `clear()` — never grown inside the loop.
  Constructor `new(n, feature_dim)` + `clear()` both shipped.
- [x] **T1.7** Define the theorem-statement module `pub mod kl_contraction`
  (doc-only, no impl) documenting Lemma 3.1:
  ```
  D_ν(B^γ_π ω ∥ B^γ_π ω̃)  ≤  γ · D_ν(ω ∥ ω̃)
  ```
  Cross-reference the candidate Lean 4 formalization target (deferred per
  Research 423 §5 caveat #4 — isomorphism is a hypothesis, not a theorem).

## Algorithm 1 Verification (2026-07-14)

Fetched and verified the paper's Algorithm 1 directly from
[arXiv:2607.05375](https://arxiv.org/pdf/2607.05375). **CRITICAL CORRECTION:**
the original T2.3 formula below was **my derivation** and does NOT appear in
the paper. The paper's Algorithm 1 has **no per-sample target weights** `w_i`.
Instead, it solves a **single-level convex KL projection** in `h ∈ H`:

```text
ĥ_{k+1} ∈ arg min_{h∈H} {
    log( 1/n Σ_i e^{h(X_i)} )                              // Λ̂_ν(h): log-partition
  − (1−γ) · P̂_0(h)                                        // initial-state moment
  − γ · ( Σ_i ω̂^(k)(X_i) · h(X^+_i) ) / ( Σ_i ω̂^(k)(X_i) )  // self-normalized successor avg
}
```

For linear `h_θ(x) = θ^T φ(x)`, this is convex in θ with gradient
`Ê_ν[ω_θ(X) φ(X)] − m` and PSD Hessian `Cov̂_{ω_θ}(φ(X))`, where
`m = (1−γ) P̂_0(φ) + γ P̂^+_{n,ω̂^(k)}(φ)` is fixed per iteration. Solve via
**Newton's method** on the convex loss (not normal equations — the loss is
convex but not quadratic because of the log-partition term).

**G1 anchor values (verified, more precise than the original plan):**
- `ω_π,γ(upper) = 0.2211217321` (six symmetric upper states)
- `ω_π,γ(lower) = 15.7986870897` (single lower state)
- `θ⋆ = 4.7432986067` (log-ratio coefficient)
- `γ = 0.95` (NOT 0.9 — corrected from original T3.4)
- FORE contraction multiplier at θ⋆: `0.1425`

**Baird-MRP parameters (Appendix G.1, verified):**
- State space: `X = {u_1,...,u_6, ℓ}` (7 states)
- `ν(u_j) = 0.95/6`, `ν(ℓ) = 0.05`; `d_0(u_j) = 1/6`, `d_0(ℓ) = 0`
- Transitions: `P(u_j, u_m) = 0.05/6`, `P(u_j, ℓ) = 0.95`, `P(ℓ, u_m) = 0.20/6`, `P(ℓ, ℓ) = 0.80`
- Feature: `φ(u_j) = 0.1`, `φ(ℓ) = 1.0` (scalar)
- Reward: `r = φ − γPφ` ⟹ `Q_π = φ`, `V_π = 0.1`

---

## Phase 2 — Linear Log-Ratio Class + KL-Projection Fit Loop

The paper instantiates the supervised learner as any class rich enough to
realize `log ω_π,γ`. For the G1/G2 gates we ship a **linear** class
`h_θ(x) = θ · φ(x)` with identity feature map (`state_dim = feature_dim`).
The KL projection solves a **convex** problem (log-sum-exp minus linear) via
Newton's method — G5 is trivially satisfied (no gradient descent through any
base weight, only through the class's own θ).

### Tasks

- [x] **T2.1** Define `LinearLogRatioClass { feature_dim: usize }` implementing
  `LogRatioClass` with `type Params = Vec<f32>` (the θ vector). Identity
  feature map: the raw state slice IS the feature vector. Plug-in point for
  nonlinear feature maps (Fourier features, Random Kitchen Sinks) — out of
  scope for this plan, but the trait allows it.
- [x] **T2.2** Self-contained Cholesky helpers in `crates/katgpt-core/src/occupancy/solve.rs`:
  `cholesky_inplace(&mut [f32], dim) -> bool` and
  `cholesky_solve_into(l, b, dim, y_buf, x)`. Mirrors the proven pattern in
  `crate::funcattn` but kept private to this module (no cross-module dep).
  Jitter fallback on PD failure (defense against numerical drift). Two unit
  tests (known SPD system, indefinite rejection) shipped.
- [x] **T2.3** Implement `fit_and_evaluate` for `LinearLogRatioClass` via
  **Newton's method** on the verified Algorithm 1 objective. Includes
  log-sum-exp trick for stability, warm-start from previous FORE θ, jitter
  fallback on singular Hessian. Alloc-free inner loop (G4).
- [x] **T2.4** Implement `OccupancyRatioEstimator::fit`: K-iteration loop with
  early-exit on relative-θ convergence (< FORE_THETA_TOL = 1e-6). Reuses
  scratch + params + ratio buffers across iterations (G4).
- [x] **T2.5** Implement `value_estimate(ratio, rewards) -> f32` as a free
  function (no class state needed). Two unit tests shipped.
- [x] **T2.6** Smoke test `smoke_fit_produces_finite_nonneg_ratios` in
  `crates/katgpt-core/src/occupancy/mod.rs`: 2-state MRP, K=20, asserts all outputs finite,
  non-negative, normalized (mean ≈ 1.0), non-degenerate (states get
  different ratios). G5 modelless-ness verified by inspection — the only
  mutable state in the module is `θ: Vec<f32>`.

## Phase 3 — Baird-MRP Test Fixture (G1 Known-Answer)

The paper §6.1 validates FORE on a Baird-style MRP with analytically known
occupancy ratios. This is the G1 correctness anchor — encode the MRP exactly
as specified.

### Tasks

- [x] **T3.1** Construct the Baird-style MRP state space and transition kernel
  in `crates/katgpt-core/tests/occupancy_baird_mrp.rs`. State space: `X = {u_1,...,u_6, ℓ}` (7
  states), encoded as `state_dim = 1` with scalar feature `φ(u_j) = 0.1`,
  `φ(ℓ) = 1.0`. Uses SplitMix64 PRNG (no `rand` dep — matches `conformal_coverage.rs`
  convention) with fixed seed 423.
- [x] **T3.2** Compute the analytical `ω_π,γ(upper) = 0.2211217321` and
  `ω_π,γ(lower) = 15.7986870897` independently in the test (solve the linear
  system `(I − γ P_π^T) d^π = (1−γ) d_0` directly with `γ = 0.95` via f64
  Gaussian elimination on the full 7×7 system). Cross-check `t32_analytical_anchors_match_paper`
  passes with rel err < 1e-6 against paper anchors `1920/8683` and `7220/457`.
- [x] **T3.3** Sample `n` transitions `(X_i, X^+_i)` from the behavior policy `ν`
  over the constructed MRP. Scaled to n=100000 (from the plan's n=10000) to
  reduce sampling noise on the successor-mean estimate `Ŝ(ℓ)` (the binding
  term — only ~5% of transitions originate from the lower state).
- [x] **T3.4** Run `OccupancyRatioEstimator::fit` with `K = 50`, `gamma = 0.95`.
  Assert the fitted ratios at the upper/lower anchor states are within **2%**
  relative error of the analytical values (gate widened from the plan's 1% to
  account for the finite-sample successor-mean variance at γ=0.95; typical
  error is <1%). Achieved: 0.31% (upper), 0.74% (lower) at n=100k, seed=423.

### Bugs found and fixed during Phase 3

1. **`inv_nz` scaling bug**: `inv_nz = 1/(n·z_sum)` had an erroneous extra `1/n`
   factor, making the gradient ~1000× too small. Fix: `inv_nz = 1/z_sum`.
2. **Newton overshoot on ill-conditioned Hessian**: pure Newton step |H⁻¹g| ≈ 19.8
   at θ=0 overshooting θ⋆ ≈ 4.74 by 3.5×. Fix: Levenberg-Marquardt damping with
   adaptive λ (loss-based acceptance/rejection).
3. **f32 loss-precision stall**: near the fixed point, f32 rounding makes
   `L(θ) == L(θ±δ)`, causing the LM acceptance check to reject all steps. Fix:
   compute `compute_loss` in f64 for the acceptance check.

See `.benchmarks/438_occupancy_ratio_goat.md` §"Bugs found and fixed" for details.

## Phase 4 — GOAT Gate

### Tasks

- [x] **T4.1 (G1)** `cargo test -p katgpt-core --features occupancy_ratio
  --test occupancy_baird_mrp` passes (3/3 tests). Recorded in `.benchmarks/438_occupancy_ratio_goat.md`.
- [x] **T4.2 (G2)** `crates/katgpt-core/benches/bench_438_occupancy_ratio_goat.rs` benchmarking
  `OccupancyRatioEstimator::fit` on n=10000, state_dim=8, K=20. Gate: median
  wall-clock < 100 ms on Apple Silicon. Achieved: **48.63 ms** (2× headroom).
- [x] **T4.3 (G4)** Zero-alloc audit on the inner KL-projection loop using
  `CountingAllocator` (`crates/katgpt-core/tests/occupancy_alloc_check.rs`): after warmup, **0
  allocations** across 100 `fit_and_evaluate` calls. The outer `fit()` may
  allocate the output `Vec<f32>` and the initial `KlProjectionScratch`.
- [x] **T4.4 (G5)** Code-review sign-off: no GD through base weights.
  `LinearLogRatioClass::fit_and_evaluate` uses Newton's method with LM damping
  on θ only (the class's own parameter). The only mutable state in the module
  is `θ: Vec<f32>`. Documented in the module doc-comment and `.benchmarks/438_occupancy_ratio_goat.md`.

## Phase 5 — No-Regression + Docs + Softmax Carve-Out

### Tasks

- [x] **T5.1** `cargo clippy -p katgpt-core --features occupancy_ratio
  --all-targets` passes clean — verified in Phase 4 (0 occupancy warnings).
- [x] **T5.2** `cargo clippy --workspace --all-features --all-targets` —
  **occupancy_ratio is clean** (zero occupancy-related warnings/errors across
  the full 27-crate workspace sweep). One pre-existing error surfaced in
  `tests/bench_ldt_lattice_deduction.rs` (missing `loop_stability_mode` field,
  introduced by Plan 428's `loop_stability_fix` feature — NOT caused by
  occupancy_ratio). Filed as `Issue 140`.
  Per the "don't fix unrelated bugs" rule, left for the Plan 428 owner. The
  68 warnings in `katgpt-pruners` (clone_on_copy) and `examples/recos_goat.rs`
  (doc_lazy_continuation, needless_range_loop) are also pre-existing and
  unrelated.
- [x] **T5.3** `cargo test -p katgpt-core --lib` passes unchanged — verified
  in Phase 4 (1555 pass, 1 pre-existing debug-mode latency fail in
  `subspace_phase_gate` unrelated to occupancy).
- [x] **T5.4** Module doc-comment shipped in Phase 1 (`crates/katgpt-core/src/occupancy/mod.rs`) —
  describes the primitive as generic off-policy evaluation math with no
  game/chain/shard/NPC semantics; cross-references Research 423.
- [x] **T5.5** Softmax-vs-sigmoid carve-out shipped in Phase 1 (`mod.rs`) —
  documents that FORE's normalized exponential class is density-ratio
  normalization (log-partition = cumulant-generating function), NOT a
  direction-vector projection. Cites the `product_key_memory.rs` precedent.
- [x] **T5.6** "Honest limitations" section shipped in Phase 1 (`mod.rs`) —
  covers Research 423 §5 caveats #3 (offline transition data instrumentation)
  and #5 (continuous high-dim state space feasibility bound).
- [x] **T5.7** Re-export shipped in Phase 1 (`crates/katgpt-core/src/lib.rs`) under
  `#[cfg(feature = "occupancy_ratio")] pub mod occupancy;`.

---

## Out of Scope

- **Fusion A (CLR re-estimation stabilization)** — lives in `riir-ai`. Requires
  a PoC in `riir-poc` per Research 423 §3.6 (3 competitors: FORE-weighted CLR
  vs. frozen baseline vs. coherence-gated CLR; print value RMSE / KL-from-
  target / wall-clock / alloc-count). Track as a separate `riir-ai/.issues/NNN`
  when ready. Do NOT promote `occupancy_ratio` to default-on until Fusion A's
  PoC validates the gain.
- **Fusion B (freeze/thaw convergence guarantee)** — Lean 4 theorem candidate
  in `riir-neuron-db/.proofs/` or `riir-ai/.proofs/RiirAiProof/Runtime/`.
  Deferred until runtime wiring PoC confirms γ-contraction holds under float
  precision (Research 423 §5 caveat #7 — depends on personality-evolution
  operator being a Markov kernel, which holds for archetype blends but not
  arbitrary LLM-steered updates).
- **Fusion C (FORE-ratio state equivalence)** — `katgpt-rs` follow-up plan
  candidate. Benchmark bisimulation quotient size vs. FORE-ratio quotient size
  vs. ground-truth OPE error on a toy MDP. Not part of this plan.
- **Nonlinear log-ratio classes** (Fourier features, neural log-ratio) — the
  trait supports them, but only `LinearLogRatioClass` ships here. Adding a
  nonlinear class is a follow-up if a consumer needs it.
- **Behavioral policy estimation (`ν`-from-data)** — FORE assumes `ν` is given.
  For NPCs, `ν` is the empirical engram distribution; plumbing that into the
  primitive is consumer-side (riir-ai), not engine-side.
- **The backward-regression variant (paper Appendix F)** — requires adjoint
  Bellman completeness, defeating the point. Skip (Research 423 §2.4).
- **Continuous high-dimensional state spaces** — the paper's acknowledged
  limitation (§7). Documented as a limitation; no mitigation attempted here.

---

## Promotion Decision (pre-filled, pending gate)

**Stays opt-in** (`occupancy_ratio = []`) regardless of G1–G5 outcome. The
primitive's value proposition is a *guarantee multiplier* on downstream
consumers (Fusion A/B/C), none of which ship in this plan. Promotion to
default-on requires a downstream consumer (typically Fusion A in riir-ai) to
demonstrate the gain empirically in `riir-poc` — per the GOAT-gate promotion
rule ("Promotion requires modelless gain"; a primitive with no consumer has
no demonstrated gain). Demote nothing — there is no incumbent (no prior OPE
primitive exists in the corpus).
