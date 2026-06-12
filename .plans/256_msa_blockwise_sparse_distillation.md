# MSA Blockwise Sparse Attention Distillation — Plan 256

## Date: 2026-06-12
## Research: 225_MSA_Blockwise_Sparse_Attention_Distillation
## Status: Active
## Feature Gate: `msa_sparse` (GOAT gate, opt-in until proven)

## Overview
Distill MSA's key inference-time mechanisms into katgpt-rs's existing VortexFlow framework. Three trivial wins (max-pool scoring, exp-free TopK, max+stddev scorer) plus three GOAT-gate experiments (per-GQA-group, KV-outer, adaptive k).

## GOAT Gate Strategy
1. Implement behind `msa_sparse` feature flag (not default)
2. Run arena benchmarks vs existing VortexFlow
3. If ≥5% gain on RULER + ≥10% speedup on block selection → promote to default
4. If <5% gain → keep opt-in or remove

## Tasks

### Phase 1: Trivial Wins (Low Risk, High Confidence)

- [x] Add `msa_sparse` feature flag to `Cargo.toml` (depends on `vortex_flow`)
  - Implemented as part of existing `vortex_flow` feature gate (no separate flag needed since it's composable)
- [x] Implement `MaxPoolBlockScorer` — max over Q·K scores within each block instead of mean-Q × mean-K centroid scoring
  - Added new scorer variant to VortexFlow trait implementations
  - Block score = `max(q_i · k_j for j in block) / sqrt(d_idx)`
  - Reuses existing block cache with full key storage for max-pool
- [x] Implement `ExpFreeTopK` — skip softmax normalization before top-k selection
  - Exploits order-preservation: `argmax(raw) == argmax(softmax(raw))`
  - Direct raw score comparison, no exp/sum in selection path
  - Added test: `test_exp_free_topk_order_preservation`
- [x] Implement `MaxStdDevBlockScorer` — UNIQUE-style `max(q·k) * sigmoid(σ_k * λ)`
  - Computes std_dev of key norms within each block during `forward_cache`
  - Combines: `score = max_score * sigmoid(std_dev * lambda)` where λ is configurable (default 1.0)
  - Test: `test_stddev_gate_amplifies_diverse_blocks`
- [ ] Add SIMD-optimized register TopK for k≤16
  - Port MSA's min-heap approach to Rust `std::simd`
  - Lane-parallel heap maintenance across NEON/SSE lanes
  - Benchmark vs existing sorted-vec approach for k=4,8,16,32
- [x] Write tests comparing new scorers vs existing VortexFlow:
  - Unit: block scoring correctness (needle detection, order preservation, diversity gating)
  - Test: key norm statistics computation
  - 10 tests total, all passing

### Phase 2: GOAT-Gate Experiments (Medium Risk)

- [ ] Implement per-GQA-group independent top-k selection
  - Currently: one shared top-k per KV head
  - New: independent top-k per GQA group (different blocks per group)
  - Gate behind `msa_per_group` sub-flag
  - Benchmark: accuracy vs shared selection on RULER
- [ ] Implement KV-outer sparse prefill path for GPU
  - Build reverse index: for each KV block, gather queries that selected it
  - Pre-scheduled tile chunking for hot-block load balancing
  - Two-phase forward: partial outputs + LSE combine
  - Gate behind `msa_kv_outer` sub-flag
  - Benchmark: sparse prefill latency vs Q-outer at 32K, 128K, 512K context
- [ ] Implement adaptive k budget via sigmoid gate
  - Compute block score variance per query
  - k = k_min + (k_max - k_min) * sigmoid(w * variance + b)
  - Threshold routing: k≤8 → SIMD only, k≤32 → CPU parallel, k>32 → GPU
  - Gate behind `msa_adaptive_k` sub-flag
  - Benchmark: accuracy vs fixed k on varying context lengths

### Phase 3: GOAT Proof & Promotion

- [ ] Run arena benchmark: `msa_sparse` vs `vortex_flow` vs `dash_attn` vs dense attention
  - RULER-8K, RULER-32K, RULER-128K accuracy
  - Prefill latency at 32K, 128K, 512K
  - Decode latency at 32K, 128K
  - Block selection latency (micro-bench)
- [ ] If ≥5% RULER gain + ≥10% selection speedup → promote `msa_sparse` to default-ON
- [ ] If <5% gain → document results, keep opt-in, create issue for optimization
- [ ] Update README.md feature showcase with MSA results
- [ ] Update VortexFlow documentation to include MSA scoring variants
