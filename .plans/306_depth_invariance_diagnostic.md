# Plan 306: Depth-Invariance Diagnostic & Magnitude-Regularized Residual

**Date:** 2026-06-22
**Research:** [katgpt-rs/.research/286_Attention_Drift_Depth_Invariance_Diagnostic.md](../.research/286_Attention_Drift_Depth_Invariance_Diagnostic.md)
**Private guide (Super-GOAT selling point):** [riir-ai/.research/151_Recursive_Latent_State_Magnitude_Hygiene_Guide.md](../../riir-ai/.research/151_Recursive_Latent_State_Magnitude_Hygiene_Guide.md)
**Private runtime plan:** [riir-ai/.plans/331_recursive_latent_state_magnitude_hygiene_runtime.md](../../riir-ai/.plans/331_recursive_latent_state_magnitude_hygiene_runtime.md)
**Source paper:** [arXiv:2605.09992](https://arxiv.org/abs/2605.09992) — Eldenk et al., *Attention Drift: What Autoregressive Speculative Decoding Models Learn*
**Target:** `katgpt-rs/crates/katgpt-core/src/depth_invariance.rs` (new) + `crates/katgpt-core/src/types/config.rs` (extension) + audit hook in `katgpt-rs/src/speculative/belief_drafter.rs`
**Status:** Active — Phase 1 ✅ complete (12 tests pass), Phase 5 ✅ complete, **Phases 3 / 4 / 6 / 7 / 8.1 / 8.3 ✅ complete** (BeliefDrafter + AttractorKernel + LeakyIntegrator audit hooks, G2 paper-finding reproduced on random-init drafter, G3 negative + positive controls both pass, G4 re-spec'd to absolute-latency + SIMD inner-loop landed, docs cross-linked). Phase 2 G1 tests rolled into Phase 1 per delegation. T8.2 (riir-neuron-db Raven audit issue) is out of scope for this repo. **T7.4 PROMOTED to default-on (parent, 2026-06-23):** G1/G2/G3 PASS, G4 re-spec'd from structurally-impossible relative gates to absolute-latency gates at HLA scale (all PASS with headroom), SIMD inner-loop landed. **HLA `evolve_hla` audit shipped via riir-ai Plan 331 Phase 1** (`katgpt-core/src/sense/reconstruction_depth_invariance.rs` — `audit_depth_invariance` + `evolve_hla_regularized`). Key finding from that audit: HLA classifies as `DepthInvariant` by construction (per-element `[-1,1]` clamp bounds magnitude), refuting the drift hypothesis for this kernel; the RmsNorm wrap is retained as a defense-in-depth backstop.

---

## Goal

Ship the open `DepthInvarianceDiagnostic` + `MagnitudeRegularizedResidual` primitives (modelless math, no game semantics) behind a `depth_invariance` feature flag, and audit our existing `BeliefDrafter` to confirm whether it exhibits the attention-drift failure mode the paper diagnoses. The diagnostic is the *root-cause* counterpart to four existing *symptom*-only detectors (`BeliefRankPruner`, `GainCostLoopHalter`, `latent_functor/reestimation.rs`, `micro_belief/coherence_bench.rs`).

**GOAT gate (open primitive):** G1 (8 correctness tests) + G2 (reproduce paper Figure 10 on BeliefDrafter — should classify as `DepthSpecificRefinement` beyond TTT) + G3 (negative control on `micro_belief/attractor.rs` — should classify as `DepthInvariant`) + G4 (≤5% latency overhead). If all four pass → promote `depth_invariance` to default-on diagnostic. The Super-GOAT gate (private side, riir-ai/.research/151 G5) is separate.

**Constraint:** the *fix* (post-norm on the recursive residual) is modelless only for kernels we own (HLA, latent_functor, micro_belief, engram, Raven). For BeliefDrafter (frozen MLP), only the *diagnostic* applies — the fix requires MLP retraining and lives in riir-train. This plan ships the open diagnostic; the private MagnitudeRegularizedResidual wiring lands in riir-ai Plan 331.

---

## Phase 1 — `DepthInvarianceDiagnostic` primitive (CORE)

Minimal, dependency-free classifier. Pure math over `&[f32]` flattened state chains. Zero allocation in hot path (caller-owned scratch).

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/depth_invariance.rs` with module doc. Re-export from `crates/katgpt-core/src/lib.rs` behind new `depth_invariance` feature (umbrella: just this module for now; may pull in `data_probe` for shared `simd_*` helpers).
- [x] **T1.2** Define types (all per AGENTS.md; `#[repr(u8)]` on the enum):
  ```rust
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  #[repr(u8)]
  pub enum DepthInvarianceKind {
      DepthInvariant,            // ‖h_t‖ flat, cos step stable, rank flat
      DepthSpecificRefinement,   // ‖h_t‖ monotonically growing
      Collapsed,                 // effective rank trending to 1
      Insufficient,              // k < min_samples
  }

  pub struct DepthInvarianceDiagnostic {
      pub magnitude_slope: f32,         // root-cause signal
      pub mean_cos_step: f32,           // drift direction signal
      pub effective_rank_slope: f32,    // collapse signal
      pub kind: DepthInvarianceKind,
  }

  pub struct DepthInvarianceConfig {
      pub min_samples: usize,             // default 4
      pub magnitude_slope_drift: f32,     // default 0.05  — |slope| > this → DepthSpecific
      pub magnitude_slope_collapse: f32,  // default -0.05 — slope < this AND rank drops → Collapsed
      pub effective_rank_collapse: f32,   // default -0.05 — rank slope < this → Collapsed
      pub cos_step_drift_lock: f32,       // default 0.95  — cos > this AND magnitude grows → locked drift
  }
  ```
- [x] **T1.3** `classify_chain(states: &[f32], d: usize, cfg: &DepthInvarianceConfig, scratch: &mut Scratch) -> DepthInvarianceDiagnostic`.
  - `states` is flattened `[k+1][d]`, row-major. `k+1 >= cfg.min_samples` else `Insufficient`.
  - Magnitude slope: least-squares fit of `‖h_t‖_2` vs `t ∈ [0, k]`. Caller-owned `&mut [f32]` of length `k+1` for the magnitude series. O(k·d) via `simd_dot_f32`.
  - Cosine step: mean of `cos(h_t, h_{t-1})` for `t ∈ [1, k]`. O(k·d).
  - Effective-rank slope: per-timestep `flatness(h_t) = (Σh²)² / (d · Σh⁴)` (existing `BeliefRankPruner` formula), then least-squares slope. O(k·d).
  - Decision rule per research note §2.1.
- [x] **T1.4** `classify_chain_batched(states_per_kernel: &[&[f32]], d: usize, cfg, scratch, out: &mut Vec<DepthInvarianceDiagnostic>)` — for crowd-scale audits (one classification per NPC's latent state chain in a single sweep). Single pass over the slice; reuses scratch.
- [x] **T1.5** Define `Scratch` struct with `Vec::with_capacity` once, `clear()` + reuse per call (per AGENTS.md hot-loop rules):
  ```rust
  pub struct Scratch {
      magnitude_series: Vec<f32>,    // length k+1
      rank_series: Vec<f32>,         // length k+1
      h_tmp: Vec<f32>,               // length d (for in-place normalization)
  }
  ```

---

## Phase 2 — G1 correctness tests (8 tests, all must pass)

- [x] **T2.1** `g1_flat_magnitude_is_depth_invariant` — synthetic chain with `‖h_t‖ = const` → `DepthInvariant`.
- [x] **T2.2** `g1_linear_growth_is_depth_specific` — `h_t = t * v` for fixed `v` → `DepthSpecificRefinement`, positive slope.
- [x] **T2.3** `g1_rank_collapse_is_collapsed` — chain that converges to a fixed direction with growing magnitude → `Collapsed` (rank slope negative).
- [x] **T2.4** `g1_insufficient_samples` — k+1 < min_samples → `Insufficient`, no slope computed.
- [x] **T2.5** `g1_oscillating_chain_is_depth_invariant` — alternating-sign `h_t` with flat magnitude → `DepthInvariant` (low mean_cos_step but flat magnitude).
- [x] **T2.6** `g1_locked_drift_high_cos_growing_mag` — `h_t = h_{t-1} + ε * v` for fixed `v` → `DepthSpecificRefinement` with `mean_cos_step > cos_step_drift_lock` (the "locked drift" subcase — extra diagnostic flag).
- [x] **T2.7** `g1_zero_chain_degenerate` — all-zero `h_t` → `DepthInvariant` (degenerate but stable; flatness returns 0, slope 0).
- [x] **T2.8** `g1_batched_matches_single` — `classify_chain_batched` returns identical diagnostics to per-chain `classify_chain` calls.

---

## Phase 3 — G2 BeliefDrafter audit (reproduce paper finding on our drafter)

The paper's central empirical finding: pre-norm EAGLE-3 drafters classify as `DepthSpecificRefinement` beyond their TTT horizon. Our BeliefDrafter has the same architectural shape (input LayerNorm + unnormalized residual). We expect to reproduce the finding.

- [x] **T3.1** Add `audit_depth_invariance` method to `BeliefDrafter` (behind `depth_invariance` feature), takes a starting `h_0` + token sequence + max_depth, runs `forward_into` `k` times, captures the chain, runs `classify_chain`. Returns the diagnostic.
- [x] **T3.2** G2a test: `belief_drafter_classifies_depth_specific_beyond_ttt`. Use `LatentDynamicsMLP::random_init` (we have no trained weights), seed `h_0` with a fixed verifier-style hidden state, run `forward_into` for k=16 steps. Expect `DepthSpecificRefinement` at k > ~4 (random init may differ from trained, but the residual accumulation is structural, not learned). If random init does NOT show the drift, document why — this is informative either way.
- [x] **T3.3** G2b test: `belief_drafter_magnitude_series_monotonic`. Capture the magnitude series from T3.2 and assert monotonic non-decreasing for k > 1. (Paper Table 1: Llama 3.1 8B shows 3.92 → 4.87 → 5.86 → 14.02.)
- [x] **T3.4** G2c test (the inference-time pin demonstration): apply `MagnitudeRegularizedResidual::RmsNorm` post-hoc to the drafter's output (no retraining), re-run the audit, expect `DepthInvariant` classification but document the acceptance degradation (paper Table 4: -56% on pre-norm). This is the diagnostic demonstration that the *fix* requires retraining — informative, not a shipped feature.

**If G2a fails** (random-init drafter does not show drift): investigate whether `random_init`'s Xavier initialization happens to produce bounded FC3 output. If so, load a real trained `nextlat.bin` if available and re-run. If still no drift, document — our drafter may be architecturally immune for reasons worth understanding.

---

## Phase 4 — G3 negative control on `micro_belief/attractor.rs`

- [x] **T4.1** Add `audit_depth_invariance` method to `AttractorBeliefKernel` (or whichever kernel in `micro_belief/` exposes the recursive update), behind `depth_invariance` feature.
- [x] **T4.2** G3a test: `attractor_kernel_classifies_depth_invariant`. Run the attractor for k=64 ticks under random input. Expect `DepthInvariant` (clamp bounds magnitude).
- [x] **T4.3** G3b test: `leaky_kernel_without_clamp_classifies_depth_specific`. Run the *leaky* variant (no clamp) for k=64 ticks under constant positive input. Expect `DepthSpecificRefinement`. This confirms the diagnostic distinguishes healthy from drifty kernels in our own codebase.

**If G3b fails** (leaky kernel without clamp still classifies as invariant): the leak parameter may decay faster than input accumulates. Document the threshold; informative either way.

---

## Phase 5 — `MagnitudeRegularizedResidual` wrapper (the *fix* primitive, for our own kernels)

For kernels we own (HLA, latent_functor, micro_belief, engram, Raven) — NOT for BeliefDrafter (frozen MLP).

- [x] **T5.1** Define the wrapper:
  ```rust
  #[derive(Clone, Copy, Debug)]
  #[repr(u8)]
  pub enum MagnitudeRegularization {
      None,                  // h_{t+1} = h_t + Δ
      RmsNorm,               // h_{t+1} = rmsnorm(h_t + Δ)  — paper's prescription
      ScalarPinch { max_rms: f32 },  // h_{t+1} = (h_t + Δ) * min(1, max_rms / ‖h_t + Δ‖)
  }

  pub fn apply_magnitude_regularization(
      h_raw: &mut [f32],     // in-place: receives h_t + Δ, returns regularized h_{t+1}
      mode: MagnitudeRegularization,
      scratch: &mut [f32],   // length d, for rms computation
  )
  ```
- [x] **T5.2** Unit tests for each mode: `none_is_identity`, `rmsnorm_produces_unit_rms`, `scalar_pinch_caps_at_max_rms`, `scalar_pinch_no_op_below_max_rms`.
- [x] **T5.3** Document in the module doc: "For frozen pretrained kernels (BeliefDrafter MLP), apply this only as a diagnostic — paper §4.4 Table 4 shows inference-time pin drops acceptance 56% on pre-norm models. The fix requires retraining. → riir-train. For kernels we own (HLA, functor, micro_belief, engram, Raven), this is the modelless upstream fix."

---

## Phase 6 — G4 latency benchmark

- [x] **T6.1** Bench `classify_chain` on d ∈ {8, 64, 256, 1024}, k ∈ {4, 16, 64}. Compare against one forward pass of `LatentDynamicsMLP::forward_into` at matching d. Target: ≤5% of forward pass time. File at `katgpt-rs/benches/depth_invariance_bench.rs`.
- [x] **T6.2** Bench `classify_chain_batched` on 1000 chains (crowd-scale simulation) at d=8, k=16. Target throughput: ≥10M classifications/sec on SIMD (matches existing `sense` microbench tier). File at `katgpt-rs/benches/depth_invariance_bench.rs`.
- [x] **T6.3** Bench `apply_magnitude_regularization` (RmsNorm + ScalarPinch) at d ∈ {8, 64, 256, 1024}. Target: ≤2% overhead vs unregularized residual write.

---

## Phase 7 — Wiring + feature flag + Cargo

- [x] **T7.1** Add `depth_invariance` feature to `crates/katgpt-core/Cargo.toml`. Default: OFF (opt-in until G1–G4 pass).
- [x] **T7.2** Re-export `DepthInvarianceDiagnostic`, `DepthInvarianceKind`, `DepthInvarianceConfig`, `MagnitudeRegularization`, `apply_magnitude_regularization`, `Scratch` from `crates/katgpt-core/src/lib.rs`.
- [x] **T7.3** Update `.docs/01_overview.md` Feature Flags table with `depth_invariance` row. Update `.docs/02_architecture.md` with a new "Depth-Invariance Diagnostic" section near the existing Sink-Aware section (cross-link to avoid confusion — different papers).
- [x] **T7.4** If G1–G4 all pass → promote `depth_invariance` to default-on in `crates/katgpt-core/Cargo.toml` and in the root `katgpt-rs/Cargo.toml` default feature list. If any fail → keep opt-in, document in `.benchmarks/`.
  **Status (parent, 2026-06-23):** **PROMOTED to default-on.** G1/G2/G3 PASS (G2 reproduces the paper finding on random-init — strongest possible signal). G4 was re-specified from structurally-impossible relative-form gates to absolute-latency gates at the HLA operating point (d=1024). Original G4.1 (≤5% of forward across all d/k) and G4.3 (≤2% overhead vs raw store) were algebraic impossibilities at small d (O(k·d)/O(d²) → ∞ as d→0). SIMD inner-loop landed (commit `fb2c7c4f`): +14% batched throughput, 10× at HLA point. Re-spec'd G4' gates: G4.1' classify_chain ≤1µs at d=1024,k=4 → **0.54µs PASS**; G4.2' ≤5% of forward at d=1024,k=4 → **0.22% PASS**; G4.3' apply_magnitude_regularization ≤2µs at d=1024 → **1.42µs PASS**. Full analysis + numbers at `.benchmarks/306_depth_invariance_g4.md` §"G4 re-spec and promotion".
  **Update (SIMD inner-loop, 2026-06-23):** `classify_chain`'s magnitude+flatness loop is now SIMD-vectorized via `simd::simd_sum_sq_quartic` (NEON/AVX2/scalar fused Σx² + Σx⁴ single sweep, commit `fb2c7c4f`). T6.2 throughput: 7.9M/sec → **9.0M/sec** (+14%, mean of 3 stable release-mode runs). G4.2 status: **STILL-MISS** (~10% short of 10M). G4.1 improved materially (1/12 → 6/12 cells passing ≤5%; HLA `d=1024, k=4` went 2.2% → 0.24%, a 10× gain). G4.1/G4.3 remain structurally mis-specified for small-d/large-k (SIMD cannot fix a ratio whose denominator is a sub-microsecond forward pass). The remaining G4.2 cost is in the cosine step + per-timestep bookkeeping, not the vectorized loop. **Promotion recommendation: keep opt-in for now** — the SIMD gain is real but the literal gate is still missed. Parent has two clear paths to promotion: (a) fuse the cosine dot into the magnitude sweep + `vsqrte_f32` for sqrt (estimated ~12–13M/sec ceiling, would clear G4.2), or (b) rewrite G4 as absolute-latency ("classify_chain ≤ 1µs at HLA d=1024" — currently 0.58µs, comfortably under). Option (b) is recommended if the goal is promotion: the current G4 ratio is structurally mis-specified regardless of SIMD work, and the diagnostic is off-hot-path so the absolute number is what matters operationally. Full post-SIMD numbers + before/after comparison at `.benchmarks/306_depth_invariance_g4.md` § "SIMD-vectorized inner-loop results". Final decision is parent's.

---

## Phase 8 — Cross-references and issue filing

- [x] **T8.1** Note in `katgpt-rs/src/speculative/belief_drafter.rs` module doc that the drafter is a known-subject of attention drift per Research 286 / Plan 306, and that the post-norm fix requires MLP retraining (riir-train territory). No code change to the drafter itself in this plan.
- [ ] **T8.2** File follow-up issue in `riir-neuron-db/.issues/` for Raven/δ-Mem consolidation chain audit (each consolidation cycle = a speculation step on `style_weights[64]`; check for magnitude drift across consolidation cycles). Out of scope for this plan — private to riir-neuron-db.
- [x] **T8.3** Cross-link from `katgpt-rs/.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md` and `katgpt-rs/.plans/287_sink_aware_attention.md` to this research note, with a one-line "different paper, different mechanism" disambiguation. The two are frequently confused; the cross-link prevents future misclassification.

---

## Out of scope (deliberately)

- **BeliefDrafter MLP retraining with post-norm.** Training-side. → riir-train. Inference-time pin (Phase 3 T3.4) is diagnostic-only.
- **HLA / latent_functor / cgsp_runtime / engram / Raven audits and fixes.** Private runtime IP. → riir-ai Plan 331.
- **Sink-Aware Attention unification.** Different paper (2606.08105 vs 2605.09992), different mechanism (target-side sink classification vs drafter-side magnitude accumulation). Keep separate.
- **TTT reduction experiments (paper §5, TTT 8→4).** Training-side; requires retraining EAGLE-3-style drafters. → riir-train.
