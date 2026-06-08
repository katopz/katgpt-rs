# Plan 216: SubstrateGate — Inference-Time Capability Substrate Routing

**Research**: R191 (Prism Capability Substrate Extraction)
**Status**: IN PROGRESS
**Feature Gate**: `substrate_gate` (off by default until GOAT proof)
**Depends On**: Plan 022 (Sparse MLP), Plan 087 (CNA Steering)

---

## Overview

Implement Prism-inspired capability substrate routing at inference time. Pre-computed per-capability MLP channel masks intersect with ReLU activation masks for dual sparsity. DDTree branches route through different capability substrates. Recovery scoring extends `ScreeningPruner`.

---

## Architecture

```
ForwardContext (transformer.rs)
    │
    ├── ReLU activation mask (sparse_mlp, existing)
    │       active_indices / active_values
    │
    ├── [NEW] Capability substrate mask (substrate_gate)
    │       SubstrateMask (packed bitmask per capability)
    │       SubstrateRouter (classify input → select mask)
    │       ∩ intersection with ReLU mask
    │
    ├── DDTree branch routing
    │       Each branch can use different SubstrateMask
    │       Score: logprob × recovery × constraint_validity
    │
    └── SubstrateScreeningPruner (extends ScreeningPruner)
            Uses recovery under mask as relevance signal
```

---

## Tasks

### Phase 1: Core Types & Infrastructure

- [ ] T1: Define `SubstrateMask` type in `katgpt-core/src/types.rs`
  - Packed bitmask (`Vec<u64>`) over `[layers × d_ff]` MLP channels
  - Per-layer active counts
  - Recovery score field
  - BLAKE3 hash for provenance
  - `serde` Serialize/Deserialize for `.mask` file loading

- [ ] T2: Define `SubstrateRouter` trait in `katgpt-core/src/traits.rs`
  - `select_mask(tokens, config) -> Option<&SubstrateMask>`
  - `register_mask(capability, mask)`
  - Default impl: `NoSubstrateRouter` (returns None, falls back to full MLP)

- [ ] T3: Add `SubstrateMask` field to `Config` in `types.rs`
  - `substrate_masks: Vec<SubstrateMask>` — loaded at model init
  - `substrate_threshold: f32` — minimum recovery score to use mask
  - Validation: mask dimensions match model architecture

### Phase 2: Dual Sparsity Execution

- [ ] T4: Implement mask intersection in `transformer.rs` `forward_base()`
  - After ReLU, before w2 down-projection
  - `active ∩ substrate` bitwise AND
  - Update `active_indices` / `active_values` with intersection
  - Zero runtime cost when `substrate_gate` feature disabled (`#[cfg]`)

- [ ] T5: Implement `SubstrateRouter` integration in `ForwardContext`
  - `router: Box<dyn SubstrateRouter>` — set during context init
  - Per-token mask selection based on token context
  - Cache selected mask for sequence (don't re-classify every token)

### Phase 3: DDTree Integration

- [ ] T6: Extend DDTree branch scoring with substrate recovery
  - Each branch can specify a capability name
  - Branch score = logprob × sigmoid(recovery) × constraint_validity
  - Sigmoid (not softmax) per project conventions

- [ ] T7: Implement substrate-aware branch expansion
  - When expanding DDTree, offer substrate-specific branches
  - E.g., for Python→Rust: "stdlib" branch, "async" branch, "error_handling" branch
  - Each branch uses its SubstrateMask for the forward pass

### Phase 4: ScreeningPruner Extension

- [ ] T8: Implement `SubstrateScreeningPruner`
  - `relevance(token, context) -> f32`
  - Uses mask's activation concentration as signal
  - Sigmoid-gated output
  - Integrates with existing `ScreeningPruner` trait

### Phase 5: Mask Loading & Export

- [ ] T9: Implement `.mask` file loader
  - Parse `SubstrateMaskFile` (bincode or JSON)
  - Validate dimensions against loaded model
  - Register masks with `SubstrateRouter`
  - Error handling: missing file → fall back to full MLP (no crash)

- [ ] T10: Define `.mask` file format spec (shared with riir-ai)
  - Version field for forward compatibility
  - Per-layer packed bitmasks
  - Recovery score, capability name, model ID
  - BLAKE3 adapter hash for provenance
  - Documented in `.docs/` for cross-project consumption

### Phase 6: CPU/GPU Auto-Route

- [ ] T11: CPU path — sparse index-packed matmul with dual mask
  - Reuse existing `simd_sparse_matmul_rows` with intersection mask
  - Benchmark: single-capability vs full MLP on CPU

- [ ] T12: GPU path — batched multi-substrate matmul
  - When `n_branches × substrate_size > threshold` → batch on GPU
  - Different masks per batch element → gather/scatter
  - Benchmark: multi-branch substrate routing on GPU

- [ ] T13: Auto-route heuristic
  - Threshold: if `substrate_active_ratio > 0.4` → dense path (mask overhead > savings)
  - If `n_branches > 4 && gpu_available` → GPU batch
  - Else → CPU sparse

### Phase 7: Tests & Examples

- [ ] T14: Unit tests for `SubstrateMask`
  - Bitmask operations (set, get, intersection)
  - Serialization round-trip
  - Dimension validation

- [ ] T15: Integration test — before/after with CNA-discovered mask
  - Run CNA discovery on test model → extract mask
  - Forward pass with mask vs without
  - Assert: output difference < threshold
  - Assert: FLOPs reduced (count active channels)

- [ ] T16: Example — capability-routed speculative decoding
  - Load model with 2+ capability masks
  - Run DDTree with substrate routing
  - Show before/after: tokens/sec, acceptance rate, output quality
  - Expected: acceptance rate ↑ 5%+, FLOPs ↓ 10%+

- [ ] T17: Example — CNA mask export to SubstrateGate
  - Run CNA discovery → save as `.mask` file
  - Load in SubstrateGate → run inference
  - Show recovery measurement

### Phase 8: GOAT Proof

- [ ] T18: GOAT benchmark — accuracy
  - Run full test suite with and without `substrate_gate`
  - Gate G1: accuracy ≥ 98% of baseline

- [ ] T19: GOAT benchmark — throughput
  - Measure tokens/sec with and without `substrate_gate`
  - Gate G2: throughput ≥ 100% of baseline (no hurt)

- [ ] T20: GOAT benchmark — FLOPs reduction
  - Count active MLP channels per token with and without mask
  - Gate G3: FLOPs ≤ 60% of baseline for single-capability tasks

- [ ] T21: GOAT benchmark — CNA mask quality
  - Compare CNA-discovered mask recovery vs Prism-style ReLP mask
  - Gate G4: CNA mask recovery ≥ 50% of Prism recovery

- [ ] T22: GOAT benchmark — DDTree substrate routing
  - Compare acceptance rate with and without substrate routing
  - Gate G5: acceptance rate improvement ≥ 5%

- [ ] T23: GOAT benchmark — zero overhead when disabled
  - Run all tests with feature disabled
  - Gate G6: zero codegen when feature disabled
  - Gate G7: all existing tests pass with/without

### Phase 9: Feature Gate & Default

- [ ] T24: Add `substrate_gate` feature to `katgpt-core/src/lib.rs` and `Cargo.toml`
  - Dependencies: `sparse_mlp`, `cna_steering`
  - All code behind `#[cfg(feature = "substrate_gate")]`
  - Off by default until GOAT proof

- [ ] T25: If all GOAT gates pass (T18-T23) → change to default-on
  - Add to default features in `Cargo.toml`
  - Update `01_overview.md` feature table
  - Update `07_adaptation.md` technique table

---

## Feature Gate

```
[features]
substrate_gate = ["sparse_mlp", "cna_steering"]
```

Default: **off** until GOAT proof. If G1-G7 all pass → **default-on**.

---

## Dependencies

| Dependency | Plan | Status |
|-----------|------|--------|
| Sparse MLP (TwELL) | Plan 022 | ✅ Default-on |
| CNA Steering | Plan 087 | ✅ Default-on, GOAT proved |
| DDTree infrastructure | Existing | ✅ Working |
| ScreeningPruner trait | Existing | ✅ Working |
| ConstraintPruner trait | Existing | ✅ Working |

---

## Performance Expectations

| Metric | Baseline (no mask) | With SubstrateGate | Change |
|--------|-------------------|-------------------|--------|
| MLP FLOPs per token | 100% | 10-40% | **-60% to -90%** |
| Total decode FLOPs | 100% | 60-90% | **-10% to -40%** |
| Throughput (tokens/sec) | baseline | ≥ baseline | **no hurt** |
| Accuracy | baseline | ≥ 98% baseline | **no hurt** |
| DDTree acceptance rate | baseline | +5-15% | **gain** |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| CNA masks not sufficient (low recovery) | Fall back to full MLP; feature only activates when mask has sufficient recovery |
| Mask intersection overhead > savings | Runtime threshold: skip mask when active_ratio > 0.4 |
| GPU multi-mask batching complex | Start with CPU-only path; GPU path is Phase 6 optimization |
| Model-specific masks don't generalize | Each mask is model+version tagged; validate on load |

---

## TL;DR

SubstrateGate implements Prism's capability extraction at inference time: pre-computed per-capability MLP masks intersected with ReLU sparsity for dual sparsity, DDTree branches routed through different substrates, recovery scoring as screening signal. 9 phases, 25 tasks, GOAT-gated with 7 criteria, default-on if proven.
