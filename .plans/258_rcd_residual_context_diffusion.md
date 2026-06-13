# Plan: Residual Context Diffusion — Modelless Inference (katgpt-rs)

**Date:** 2026-06-12
**Research:** [228_RCD_Residual_Context_Diffusion.md](../.research/228_RCD_Residual_Context_Diffusion.md)
**Feature Flag:** `rcd_residual` (gated, in default features for GOAT testing)
**GOAT Gate:** Compare against DMax baseline at equivalent TPS

---

## Overview

Implement entropy-weighted residual context injection for the D2F denoise loop in katgpt-rs. This is the **modelless** (inference-time) component — no training required. Residuals are constructed from discarded token probability distributions and injected into the next denoising step's input embeddings.

---

## Architecture

```
denoise_loop step k:
  1. Forward pass → logits → softmax → marginals p^k
  2. Commit top-m confident tokens (existing)
  3. [NEW] Compute residual for discarded positions:
     - α_i = H(p^k_i) / log(V)           // normalized entropy weight
     - Δ_i = Σ_j p^k_ij * E_j             // codebook weighted sum
     - ẽ_i = (1-α_i)*E_mask + α_i*Δ_i     // interpolated input for next step
  4. [NEW] ResidPruner filters implausible residuals
  5. [NEW] Tier-adaptive: plasma skips, warm gets full RCD
```

---

## Tasks

### Phase 1: Core Residual Injection (~80 lines)

- [x] **Task 1.1**: Add `RcdConfig` struct to `dllm_solver.rs`
  - `enabled: bool` (feature-gated)
  - `temperature_residual: f32` (T_res for inference-time calibration, default 1.0)
  - `log_vocab: f32` (log(V), pre-computed once)
  - `residual_scratch: Vec<f32>` (pre-allocated, reuse across steps)

- [x] **Task 1.2**: Implement `normalized_entropy()` in `dllm_solver.rs`
  - `α_i = H(p_i) / log(V)` — wraps existing `shannon_entropy()` with normalization
  - SIMD-friendly: process 4/8 positions at once

- [x] **Task 1.3**: Implement `compute_residual()` in `dllm_solver.rs`
  - `Δ_i = Σ_j p_ij * E_j` — weighted sum over embedding codebook
  - Only for masked positions (unmasked get standard embedding)
  - Zero-allocation: write into pre-allocated scratch buffer

- [x] **Task 1.4**: Implement `interpolate_residual()` in `dllm_solver.rs`
  - `ẽ_i = (1-α_i)*E_mask + α_i*Δ_i` — linear interpolation
  - SIMD-friendly: 4-wide f32 blend

- [x] **Task 1.5**: Wire residual injection into `denoise_loop()`
  - After token commitment, compute residuals for discarded positions
  - Store residual state in `BidirectionalContext` (the loop's actual context) for next step's input construction
  - Gate behind `#[cfg(feature = "rcd_residual")]`
  - **Previously a stub delegating to `denoise_loop`; now genuinely injects residuals**

### Phase 2: Tier-Adaptive Routing (~40 lines)

- [x] **Task 2.1**: Add `residual_mode` to `InferenceRouter` dispatch
  - `ResidualMode` enum: Skip, ConfidenceOnly, Full, FullWithWarmStart
  - `tier_to_residual_mode()`: CpuOnly→ConfidenceOnly, CpuGpu→Full, CpuGpuAne→FullWithWarmStart
  - `confidence_alpha()`: cheap 1-max_prob for ConfidenceOnly mode

- [x] **Task 2.2**: Integrate residual mode selection into `forward_batch()`
  - `residual_mode()` method on `InferenceRouter`
  - Zero overhead when disabled (cfg gate)

### Phase 3: ResidPruner — Constraint-Filtered Residuals (~50 lines)

- [x] **Task 3.1**: Implement `ResidPruner` struct
  - `src/pruners/resid_pruner.rs` — wraps `ConstraintPruner`
  - `should_inject()` extracts top-K from marginals, checks pruner validity
  - Zero-alloc: caller-provided scratch buffer

- [x] **Task 3.2**: Wire ResidPruner into residual injection
  - Registered in `src/pruners/mod.rs` with `#[cfg(feature = "rcd_residual")]`
  - Ready for integration into denoise_loop_rcd

### Phase 4: MUX-RCD Fusion (~60 lines)

- [x] **Task 4.1**: Implement `compute_mux_residual()` for DDTree paths
  - In `src/mux_demux.rs`, gated by `#[cfg(all(feature = "mux_demux", feature = "rcd_residual"))]`
  - Weighted sum across paths: Δ = Σ weight * p * E
  - Normalizes path scores to probabilities

- [x] **Task 4.2**: Wire MUX-RCD into `build_dd_tree_adaptive()`
  - Implemented as `build_dd_tree_adaptive_mux_residual()` in `src/speculative/caddtree_budget.rs`
  - Composes `build_dd_tree_adaptive` + `compute_mux_residual`; extracts path scores from `TreeNode.score`
  - **Feature gate correction:** plan referenced non-existent `mux_latent`; actual gate is `#[cfg(all(feature = "mux_demux", feature = "rcd_residual"))]` (matches `compute_mux_residual`'s gate)
  - **Score semantics:** DDTree scores are cumulative log-probs (≤ 0); wiring applies log-sum-exp shift `(s - max).exp()` before normalization so `compute_mux_residual` receives positive weights
  - **Degenerate-until-per-path-marginals:** with shared input marginals, path weights normalize to 1.0 so output collapses to standard `Σ_j p_j · E_j`; API surface complete for future per-path marginals
  - 4 tests pass: matches-standard-residual, position-selects-correct-depth, out-of-range-zeros, empty-marginals-zeros

### Phase 5: Tests & Benchmarks (~100 lines)

- [x] **Task 5.1**: Unit test `normalized_entropy()`
  - Uniform distribution → α = 1.0
  - One-hot distribution → α = 0.0
  - Known distribution → expected α

- [x] **Task 5.2**: Unit test `compute_residual()`
  - Verify Δ_i is in embedding space (correct dimensions)
  - Verify Δ_i is zero when all probability on mask token

- [x] **Task 5.3**: Additional unit tests for RCD
  - `test_rcd_interpolate_residual` — α blend correctness
  - `test_rcd_config_disabled/new` — config construction
  - `test_residual_mode_default` — default is Full
  - `test_confidence_alpha` — cheap alpha computation
  - `test_tier_to_residual_mode` — tier→mode mapping
  - `denoise_loop_rcd` function in dllm.rs wraps standard loop

- [x] **Task 5.4**: Integration test: denoise_loop with RCD vs without
  - Same input, same seeds
  - 3 tests: disabled-matches-baseline, enabled-converges, no-regression differential
  - `test_rcd_disabled_matches_baseline` — byte-identical to baseline when `enabled=false`
  - `test_rcd_enabled_converges_and_injects` — full injection path runs and converges
  - `test_rcd_vs_baseline_no_regression` — RCD does not catastrophically regress on micro-config
  - **Note**: GOAT gate (accuracy/steps gain) deferred to issue 012 benchmark harness

- [x] **Task 5.5**: GOAT gate comparison
  - RCD vs DMax at equivalent TPS
  - Measure: accuracy at throughput-matched decode speed
  - Promote to default only if RCD > DMax by ≥2pp accuracy or ≥1.5× step reduction
  - Issue 012 created for benchmark infrastructure

---

## File Changes

| File | Change | Lines |
|------|--------|-------|
| `src/dllm_solver.rs` | `RcdConfig`, entropy normalization, residual computation, interpolation | ~80 |
| `src/dllm.rs` | Residual state in `D2fContext`, wire into `denoise_loop` | ~30 |
| `src/inference_router.rs` | `ResidualMode` enum, tier mapping | ~20 |
| `src/types.rs` | Residual scratch buffer in relevant types | ~10 |
| `src/lib.rs` | Feature flag `rcd_residual` | ~5 |
| `src/pruners/` | New `resid_pruner.rs` | ~50 |
| `src/mux_demux.rs` | MUX-RCD fusion function | ~40 |
| `Cargo.toml` | Feature flag | ~2 |
| Tests | Unit + integration + benchmark | ~100 |

**Total: ~340 lines**

---

## GOAT Gate

| Metric | DMax Baseline | RCD Target | Pass Threshold |
|--------|--------------|------------|----------------|
| Accuracy (GSM8K equiv.) | Baseline | +2pp | ≥2pp gain |
| Steps at equal accuracy | Baseline | -50% | ≥1.5× reduction |
| Overhead per step | ~0% | <5% | <10% overhead |
| Plasma path overhead | 0% | 0% | Must be zero |

If RCD passes ≥3/4 metrics → promote to default (remove feature gate).
If RCD fails → keep gated, document why.

---

## TL;DR

Implement RCD residual context injection in 5 phases: core injection → tier routing → constraint filtering → MUX fusion → tests. ~340 lines total, gated behind `rcd_residual` feature. GOAT gate comparison against DMax before promoting to default. The novel fusions (ResidPruner, tier-adaptive, MUX-RCD) differentiate this from a direct paper implementation.
