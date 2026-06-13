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
  - Separate GOAT gate in Cargo.toml: `msa_sparse = ["vortex_flow"]`
  - MSA scorers and types gated behind `msa_sparse` in mod.rs and vortex_flow.rs
  - Compiles clean with and without the flag
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
- [x] Add SIMD-optimized register TopK for k≤16
  - Register-based sorted top-k with SIMD threshold filter
  - NEON path: 4-wide batch threshold comparison + SIMD-parallel insertion search
  - AVX2 path: 8-wide threshold comparison + SIMD-parallel insertion search (x86_64)
  - Scalar fallback for non-SIMD platforms
  - Benchmark: `tests/bench_256_simd_topk.rs` — k=4,8,16 vs scalar, n=64..1024
  - NEON: 4-wide batch comparison via vcgtq_f32 + vmaxvq_u32 fast-path, NEON-parallel insertion search
  - AVX2: 8-wide batch comparison via _mm256_cmp_ps + movemask, AVX2-parallel insertion search
  - Scalar fallback for other targets (binary search + shift)
  - k=1 fast-path via simd_argmax_f32, k>16 falls back to selection sort
  - 17 tests pass including correctness parity across k=1,2,4,8,16,17
- [x] Write tests comparing new scorers vs existing VortexFlow:
  - Unit: block scoring correctness (needle detection, order preservation, diversity gating)
  - Test: key norm statistics computation
  - 10 tests total, all passing

### Phase 2: GOAT-Gate Experiments (Medium Risk)

- [x] Implement per-GQA-group independent top-k selection
  - Currently: one shared top-k per KV head
  - New: independent top-k per GQA group (different blocks per group)
  - Gate behind `msa_per_group` sub-flag
  - [x] `PerGroupTopKRouter` struct + VortexFlow impl in `block_topk.rs`
  - [x] `VortexFlowConfig::MsaPerGroup` variant in `vortex_flow.rs`
  - [x] `VortexRouter::MsaPerGroup` + `VortexRouterCache::MsaPerGroup` variants
  - [x] All match arms in `n_blocks()`, `from_config()`, `forward_cache()`, `forward_indexer()`, `cache_new()`
  - [x] Export in `mod.rs`
  - [x] 4 tests: different-blocks-per-group, total-leq-topk, each-group-gets-block, backward-compat-n_groups=1
  - [x] Benchmark vs shared selection on RULER
    - `tests/bench_256_per_group.goat.rs` — GOAT FAIL: coverage 1.003× (need ≥1.5×), latency 0.983× (PASS)
- [x] Implement KV-outer sparse prefill path for GPU
  - [x] Build reverse index: for each KV block, gather queries that selected it (`KvOuterIndex`)
  - [x] Hot-block sorting by query count for cache locality
  - [x] Two-phase forward: partial outputs + LSE combine (`KvOuterPrefill::prefill_sparse`)
  - [x] Gate behind `msa_kv_outer` sub-flag
  - [x] 5 tests: index build, hot blocks, single-block dense parity, needle-in-haystack, LSE numerical stability
  - [x] Benchmark: sparse prefill latency vs Q-outer at 32K, 128K, 512K context
    - `tests/bench_256_kv_outer.goat.rs` — GOAT FAIL at 128K+ (1.14×, need ≥1.5×); wins at 32K (2.02×); numerical equivalence 3.64e-6
- [x] Implement adaptive k budget via sigmoid gate
  - Compute block score variance per query
  - k = k_min + (k_max - k_min) * sigmoid(w * variance + b)
  - Threshold routing: k≤8 → SIMD only, k≤32 → CPU parallel, k>32 → GPU
  - Gate behind `msa_adaptive_k` sub-flag
  - `AdaptiveKConfig` with builder pattern (`with_params(w, b)`)
  - `compute_adaptive_k()` — 4-way unrolled variance + sigmoid gate
  - `AdaptiveKRouter<R: VortexFlow>` — wraps inner router, reads scratch scores for variance, truncates decision to adaptive k
  - 9 tests: high/low variance, bounds, edge cases, BlockTopK integration, bias extremes
  - [x] Benchmark: accuracy vs fixed k on varying context lengths
    - `tests/bench_256_adaptive_k.goat.rs` — GOAT FAIL: recall 0.629 (need ≥0.90), compute savings 37.1% (PASS)

### Phase 3: GOAT Proof & Promotion

- [ ] Run arena benchmark: `msa_sparse` vs `vortex_flow` vs `dash_attn` vs dense attention
  - RULER-8K, RULER-32K, RULER-128K accuracy
  - Prefill latency at 32K, 128K, 512K
  - Decode latency at 32K, 128K
  - Block selection latency (micro-bench)
  - **Deferred → Issue 014**: requires trained model weights + RULER dataset; not feasible in modelless inference mode. The 3 Phase 2 micro-benchmarks below serve as modelless GOAT proxies and their failures predict the arena would also fail.
- [ ] If ≥5% RULER gain + ≥10% selection speedup → promote `msa_sparse` to default-ON
  - **SKIP**: all 3 Phase 2 micro-benchmarks FAILED their GOAT gates (see verdict below)
- [x] If <5% gain → document results, keep opt-in, create issue for optimization
  - **DONE**: results documented below; Issue 014 created for arena benchmark infrastructure
- [x] Update README.md feature showcase with MSA results
  - Added "MSA Sparse Attention Family" subsection under VortexFlow showcase with GOAT results table
- [x] Update VortexFlow documentation to include MSA scoring variants
  - Updated `src/dash_attn/vortex_flow.rs` module doc + `VortexFlowConfig` variant docs

### GOAT Verdict: ❌ FAIL — `msa_sparse` stays opt-in

All three Phase 2 micro-benchmarks (modelless RULER proxies) **FAILED** their GOAT gates:

| Benchmark | Metric | Result | Threshold | Verdict |
|-----------|--------|--------|-----------|---------|
| Per-group | Coverage ratio | 1.003× | ≥ 1.5× | ❌ FAIL |
| Per-group | Latency ratio | 0.983× | ≤ 2.0× | ✅ PASS |
| KV-outer | Speedup @ 32K | 2.02× | ≥ 1.5× | ✅ PASS |
| KV-outer | Speedup @ 128K | 1.14× | ≥ 1.5× | ❌ FAIL |
| KV-outer | Speedup @ 512K | 0.83× | ≥ 1.5× | ❌ FAIL |
| Adaptive-k | Compute savings | 37.1% | ≥ 25% | ✅ PASS |
| Adaptive-k | Recall ratio | 0.629 | ≥ 0.90 | ❌ FAIL |

**Root causes:**
1. **Per-group coverage saturates**: with diverse needle queries, both shared and per-group routers already cover ~all reachable blocks; the union ratio hovers at ~1.0. Per-group's structural diversification (forcing each partition to contribute) isn't visible in cross-query union. Per-group DOES win on latency at top_k=32 (0.40–0.52× due to smaller argtopk per partition).
2. **KV-outer block sharing drops with context**: reverse-index amortization only helps when many queries share blocks. With fixed n_queries=256 and top_k=32, avg queries/block = 256*32/n_blocks. As n_blocks grows (512K context = 8192 blocks), block sharing drops to ~1 query/block, and index-building overhead dominates. KV-outer wins at 32K (high sharing) but loses at 512K.
3. **Adaptive-k recall is mathematically bounded**: recall normalized by fixed k is bounded by k_adaptive/k_fixed ≈ 20.14/32 = 0.63. The two GOAT criteria (≥25% savings → avg k ≤ 24, AND ≥90% recall → requires avg k ≥ 28.8) are in direct tension. A precision/weighted-recall metric would better reflect selection quality.

**Numerical equivalence**: KV-outer output matches Q-outer baseline within 3.64e-6 max diff at 32K (math is correct).

**Recommendation**: Keep `msa_sparse` (and sub-features `msa_per_group`, `msa_kv_outer`, `msa_adaptive_k`) as opt-in. The infrastructure is correct and numerically validated. Each technique has a narrow regime where it wins (per-group: high top_k latency; KV-outer: short context with high block sharing; adaptive-k: compute-constrained decode). Full RULER arena evaluation deferred to Issue 014 (needs model weights + dataset).
