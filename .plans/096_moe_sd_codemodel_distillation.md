# Plan 096: MoE+SD Co-Design Model Distillation

> **Parent**: Research 59 (MoE + Speculative Decoding Co-Design)
> **Depends**: Plan 022 (Sparse MLP) ✅, Plan 026 (Inference Budget) ✅, Plan 055 (MTP Drafter) ✅
> **Scope**: Raven slot overlap diagnostic, LeviathanVerifier Amdahl cost model, delta sparse matmul exploration
> **Feature Gate**: `spec_cost_model` (D2 only, opt-in diagnostic)

## Tasks

- [x] T1: Raven slot routing overlap diagnostic (D1) — `RoutingOverlapSnapshot` type added behind `domain_latent` feature, field on `DraftResult`
- [x] T2: LeviathanVerifier Amdahl cost model with `spec_cost_model` feature gate (D2) — `SpecCostSnapshot` type, `spec_cost_model` feature in `Cargo.toml`, field on `DraftResult`, exported from `mod.rs`
- [x] T3: GOAT benchmark — 5/5 proofs pass: snapshot construction, Amdahl prediction accuracy, Leviathan infrastructure, f_sparse consistency, cost model error bound (`tests/bench_051_moe_sd_codemodel_goat.rs`)
- [x] T4: Conditional — delta sparse matmul (D3) — SKIPPED (T3 was infrastructure-only, no real overlap measurement; condition >30% not evaluated)

## Objective

Distill the three applicable insights from Cohere's MoE+Speculative Decoding analysis into our non-MoE stack:

1. **D1 (Raven Overlap Metric)**: Measure temporal correlation in Raven RSM slot routing across K+1 consecutive tokens. This is our closest analog to Cohere's 38% expert overlap. If Raven shows similar locality, it validates our O(1) slot memory design.

2. **D2 (Amdahl Cost Model)**: Instrument `LeviathanVerifier` with Amdahl decomposition to predict optimal K (draft length) for a given config. Feature-gated behind `spec_cost_model`. Enables data-driven K selection instead of hardcoded defaults.

3. **D3 (Delta Sparse Matmul)**: Conditional — only pursue if D1 shows >30% neuron overlap across consecutive tokens. Would enhance `sparse_matmul` to process only delta neurons during verification, exploiting temporal locality.

**Honest scope**: We have no MoE architecture. These are **analogous** optimizations, not direct transfers. The value is in validating/exploiting temporal locality in our existing sparse activation patterns.

## Background: Why This Matters

Cohere proved three things about MoE + speculative decoding:

1. **Temporal routing correlation** — adjacent tokens route to overlapping experts (38% at step 1 vs 6.25% uniform). This makes verification cheaper than naive (2.55× top-k instead of 3.2-3.6×).

2. **Amdahl decomposition** — target forward pass splits into scaling part (f=0.30 expert loading) and fixed part (1-f=0.70 attention/norms/launches). Verification cost: `f × unique_ratio + (1-f)`.

3. **Co-design principle** — sparsity level and inference parameters should be co-optimized for target workload.

Our mapping:

```
Cohere MoE expert routing  →  Our Raven RSM slot routing (top-k sigmoid)
Cohere sparse activation   →  Our sparse_matmul (ReLU zero-skipping)
Cohere Leviathan verify    →  Our LeviathanVerifier (exact same algorithm)
Cohere batch-size regimes  →  Our domain inference budget (Plan 026)
```

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                  Plan 096 Distillation Stack                    │
│                                                                 │
│  T1: Raven Overlap (diagnostic, always-on under domain_latent) │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LeviathanVerifier::speculate()                          │  │
│  │    for each of K+1 tokens:                               │  │
│  │      forward_raven() → record which slots were selected  │  │
│  │    unique_slots / total_slots → overlap_ratio            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  T2: Amdahl Cost Model (feature: spec_cost_model)              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  SpecCostSnapshot                                        │  │
│  │    f_sparse: f64     // fraction of time in sparse MLP   │  │
│  │    f_fixed: f64      // fraction in attention/norms/samp │  │
│  │    unique_ratio: f64 // unique neurons / per-token avg   │  │
│  │    predicted_ratio: f64  // f_sparse * unique + f_fixed  │  │
│  │    actual_ratio: f64     // measured T(K+1)/T(1)         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  T4: Delta Sparse (conditional on T3 overlap > 30%)           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  sparse_matmul_delta()                                   │  │
│  │    Track active neuron set across tokens                  │  │
│  │    For token N>1: only compute neurons not in set         │  │
│  │    Accumulate shared + delta outputs                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  T3: GOAT Benchmark                                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  bench/051_moe_sd_codemodel.goat.md                      │  │
│  │    - Raven overlap ratio (step 1-4) for draft/bpe_draft  │  │
│  │    - Amdahl f_sparse, unique_ratio for K=3,5,7           │  │
│  │    - Predicted vs actual verification cost                │  │
│  │    - PASS if overlap > 20% OR cost model error < 15%     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## T1: Raven Slot Routing Overlap Diagnostic

### Location

`katgpt-rs/src/speculative/verifier.rs` — inside `LeviathanVerifier::speculate()`

### Implementation

```rust
/// Diagnostic: Raven slot routing overlap across K+1 tokens.
/// Analogous to Cohere's "expert overlap" metric.
/// Only collected when `domain_latent` feature is active (Raven in use).
#[derive(Debug, Default)]
pub struct RoutingOverlapSnapshot {
    /// Per-step overlap ratio: shared slots / top_k
    pub step_overlap: Vec<f64>,
    /// Total unique slots across all K+1 tokens
    pub unique_slots: usize,
    /// top_k (slots selected per token)
    pub top_k: usize,
    /// Number of tokens in verification batch
    pub n_tokens: usize,
}
```

Add `routing_overlap: Option<RoutingOverlapSnapshot>` field to `DraftResult` (behind `domain_latent` feature).

### Measurement

During `LeviathanVerifier::speculate()`, for each target forward pass:
1. Record which Raven slots were activated (top-k indices)
2. Compute overlap: `|slots[i] ∩ slots[i-1]| / k` for each step
3. Compute unique: `|∪ slots[0..K+1]|`
4. Store in `DraftResult::routing_overlap`

### Feature Gate

Uses existing `domain_latent` feature (which gates Raven). No new feature gate needed.

## T2: Amdahl Cost Model

### Location

`katgpt-rs/src/speculative/verifier.rs` — new `SpecCostModel` struct

### Feature Gate

`spec_cost_model` — opt-in diagnostic. Add to `Cargo.toml` features, NOT in default set.

```toml
[features]
spec_cost_model = []  # Amdahl cost model for LeviathanVerifier (Research 59, Plan 096)
```

### Implementation

```rust
/// Amdahl decomposition of speculative verification cost.
/// Based on Cohere's analysis: T(K+1)/T(1) = f_sparse × unique_ratio + (1-f_sparse)
#[cfg(feature = "spec_cost_model")]
#[derive(Debug)]
pub struct SpecCostSnapshot {
    /// Fraction of forward pass in sparse MLP operations
    pub f_sparse: f64,
    /// Fraction in fixed costs (attention, norms, sampling, kernel overhead)
    pub f_fixed: f64,
    /// Ratio of unique active neurons across K+1 tokens vs single token
    pub unique_ratio: f64,
    /// Amdahl prediction: f_sparse × unique_ratio + f_fixed
    pub predicted_ratio: f64,
    /// Wall-clock measurement: T(K+1) / T(1) in nanoseconds
    pub actual_ratio: f64,
    /// Draft length K used
    pub k: usize,
}
```

### Measurement

Instrument `LeviathanVerifier::speculate()` with `Instant::now()` timestamps:

1. Time single-token decode (baseline): `t_decode`
2. Time K+1 verification pass: `t_verify`
3. For each layer, time the MLP portion (sparse or dense): accumulate `t_mlp`
4. `f_sparse = t_mlp / t_verify`
5. `f_fixed = 1.0 - f_sparse`
6. `actual_ratio = t_verify / t_decode`
7. `predicted_ratio = f_sparse × unique_ratio + f_fixed`

Add `cost_snapshot: Option<SpecCostSnapshot>` to `DraftResult` (behind `spec_cost_model`).

### Integration with Domain Inference Budget

The cost model output informs `InferenceOverrides::draft_lookahead` (Plan 026):
- If `actual_ratio < 1.5` → verification is cheap → can increase K
- If `actual_ratio > 2.5` → verification is expensive → decrease K
- If `predicted_ratio` overestimates by >20% → overhead amortization is strong → safe to increase K

## T3: GOAT Benchmark

### Location

`katgpt-rs/.benchmarks/051_moe_sd_codemodel_goat.md`

### Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Raven slot overlap (step 1) | > 20% | Below this, no locality to exploit (Cohere measured 38%) |
| Amdahl cost model error | < 15% | If prediction error > 15%, model needs refinement |
| `f_sparse` consistency | < 10% variance across runs | Cost model must be stable |

**PASS**: Any 2 of 3 criteria met.
**FAIL**: < 2 criteria → stop before T4.

### Benchmark Config

```
Config::draft()     — embd=4, heads=2, mlp=16, K=3
Config::bpe_draft() — embd=16, heads=2, mlp=64, K=5
```

200 iterations each, release build, report mean ± std.

## T4: Delta Sparse Matmul (Conditional)

### Gate Condition

Only implement if T3 shows Raven slot overlap > 30% at step 1.

### Location

`katgpt-rs/crates/katgpt-core/src/types.rs` — new function `sparse_matmul_delta()`

### Implementation Sketch

```rust
/// Delta sparse matmul: only compute newly active neurons not already in `prev_active` set.
/// Exploits temporal locality in ReLU activation patterns (analogous to MoE expert overlap).
#[cfg(feature = "sparse_mlp")]
#[inline(always)]
pub fn sparse_matmul_delta(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
    prev_active_indices: &[usize],
    prev_active_count: usize,
    delta_indices: &mut [usize],
    delta_values: &mut [f32],
    shared_output: &[f32],  // output from shared neurons (already computed)
) -> (usize, usize) {
    // 1. Find alive neurons in current input
    // 2. Split into shared (in prev_active) and delta (new)
    // 3. Compute only delta neurons
    // 4. output = shared_output (from prev) + delta_output (newly computed)
    // Returns (shared_count, delta_count)
}
```

### Integration

Used by `LeviathanVerifier` during the verification loop (tokens 1..K+1 after token 0):
1. Token 0: standard `sparse_matmul`, record active indices
2. Token 1..K: `sparse_matmul_delta`, only compute new neurons
3. Accumulate: output = shared contribution + delta contribution

### Feature Gate

Enhances existing `sparse_mlp`, no new gate. But only active when `sparse_mlp` + LeviathanVerifier is used.

## Module Structure

```
katgpt-rs/
├── crates/katgpt-core/src/
│   └── types.rs                    # sparse_matmul_delta() (T4)
├── src/
│   ├── speculative/
│   │   ├── verifier.rs             # RoutingOverlapSnapshot (T1), SpecCostSnapshot (T2)
│   │   └── types.rs                # RoutingOverlapSnapshot + SpecCostSnapshot in DraftResult
│   └── benchmark.rs                # bench_routing_overlap() (T3)
├── .benchmarks/
│   └── 051_moe_sd_codemodel_goat.md  # GOAT results
├── .research/
│   └── 59_MoE_Speculative_Decoding_CoDesign.md  # Parent research
└── .plans/
    └── 096_moe_sd_codemodel_distillation.md  # This plan
```

## Feature Gate Summary

| Gate | Scope | Default | Plan |
|------|-------|---------|------|
| `domain_latent` | T1 (Raven overlap) | ✅ Default-on | 038 |
| `spec_cost_model` | T2 (Amdahl cost model) | ❌ Opt-in | **096** |
| `sparse_mlp` | T4 (Delta sparse) | ✅ Default-on | 022 |

## References

- Cohere blog: https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding
- MoESD paper: https://arxiv.org/pdf/2505.19645
- MagicDec: https://arxiv.org/abs/2408.11049
- Expert routing correlation: https://arxiv.org/abs/2505.16056
- Our Research 02 (Speculative Decoding): `.research/002_Fast Inference from Transformers via Speculative Decoding.md`
- Our Research 06 (Raven RSM): `.research/006_Raven_Routing_Slot_Memories.md`
- Our Research 08 (Sparse MLP): `.research/008_Sakana_TwELL_Sparse_MLP.md`
- Our Research 09 (EMO): `.research/009_EMO_Emergent_Modularity.md` — same verdict: no MoE for us