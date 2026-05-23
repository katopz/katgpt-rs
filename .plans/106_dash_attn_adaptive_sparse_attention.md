# Plan 106: DashAttention — Adaptive Sparse Hierarchical Attention

**Branch:** `develop/feature/106_dash_attn_adaptive_sparse`
**Depends on:** Plan 044 (PFlash), Plan 070 (SP-KV)
**Research:** 68 (DashAttention)
**Feature Gate:** `dash_attn` (off by default) — both `microgpt-rs` and `microgpt-core`
**Goal:** Replace PFlash's fixed-budget top-k block selection with α-entmax adaptive routing. Add learned chunk summaries via `head_cls` vectors. Benchmark adaptive vs fixed sparsity.

---

## Tasks

### Phase 1: α-entmax Kernel (CPU)
- [ ] **T1**: Create `src/dash_attn/mod.rs` — module index, re-exports, feature gate `#[cfg(feature = "dash_attn")]`
- [ ] **T2**: Implement `entmax_1p5()` in `src/dash_attn/entmax.rs` — α=1.5 special case: `p_i = max(0, 0.5*s_i - τ)²`. Two-pass threshold finding (sort + cumulative sum). Returns sparse weights and threshold τ. Unit tests: known inputs → exact zeros, sum=1.0, non-negative
- [ ] **T3**: Implement `entmax_support()` — extract active indices from entmax weights. Returns `Vec<usize>` of positions where weight > 0
- [ ] **T4**: Implement `entmax_gqa_aggregate()` — average entmax probabilities across query heads in same GQA group. Input: `[n_query_heads][n_chunks]`, output: `[n_kv_heads][n_chunks]`. Zeros propagate (non-dispersive)
- [ ] **T5**: Add `DashAttnConfig` to `microgpt-core/src/types.rs` — `chunk_size: usize` (64), `alpha: f32` (1.5), `scaling_factor: f32` (1.0), `sigma: f32` (1e6), `estimate_diagonal: bool` (true)
- [ ] **T6**: Register `#[cfg(feature = "dash_attn")]` gate in `Cargo.toml` features + `pub mod dash_attn` in `lib.rs`

### Phase 2: Learned Chunk Summaries
- [ ] **T7**: Create `src/dash_attn/chunk_summary.rs` — `ChunkSummaryQuery` struct with `head_cls: Vec<f32>` (shape `[n_kv_head, head_dim]`)
- [ ] **T8**: Implement `summarize_chunk()` — local SDPA: `k̄_c = softmax(q̄ · K_chunk / √d) · K_chunk`. At zero-init: returns mean pooling. After training: weighted attention
- [ ] **T9**: Implement `ChunkSummaryCache` — stores completed chunk summaries `[n_chunks, n_kv_head, head_dim]`. Append-only during decode. `allocate()`, `append()`, `view()` methods
- [ ] **T10**: Unit tests: verify zero-init → mean pooling equivalence, trained → non-uniform weighting

### Phase 3: Entmax Block Routing
- [ ] **T11**: Create `src/dash_attn/routing.rs` — `score_blocks_entmax()` function
- [ ] **T12**: Implement routing pipeline: (1) compute chunk logits `z = q · k̄ / √d`, (2) apply scaling factor γ, (3) α-entmax routing, (4) GQA aggregation, (5) extract support + compute routing bias `d_{i,j} = (log w - μ) / σ`
- [ ] **T13**: Implement `compute_routing_bias()` — produces additive attention bias for Stage 2. Returns (active_indices, bias_per_chunk). μ = mean of log weights on support
- [ ] **T14**: Unit tests: verify adaptive support (variable number of active chunks per query), routing bias correctness, GQA aggregation preserves sparsity

### Phase 4: Forward Integration
- [ ] **T15**: Create `src/dash_attn/forward.rs` — `forward_dash_attn_prefill()` for prefill mode
- [ ] **T16**: Implement prefill flow: (1) chunk summarization over K, (2) entmax routing, (3) sparse attention on active chunks with routing bias, (4) store chunk summaries to cache
- [ ] **T17**: Implement `forward_dash_attn_decode()` — reuse cached chunk summaries, only score against cached summaries + current diagonal chunk
- [ ] **T18**: Add `AttentionMode::DashAttn` variant to `microgpt-core/src/types.rs` — dispatches to `forward_dash_attn_prefill` / `forward_dash_attn_decode`
- [ ] **T19**: Wire into `transformer.rs` forward dispatch — `match config.attention_mode { DashAttn => ... }`

### Phase 5: PFlash Integration (Drop-in Replacement)
- [ ] **T20**: Create `block_select_entmax()` alternative to existing `block_select()` in `speculative/prefill.rs`
- [ ] **T21**: Benchmark: PFlash top-k vs PFlash entmax at same sparsity target — measure (a) chunk selection quality (NIAH retrieval), (b) selection time (µs), (c) adaptive support variance (min/max/mean active chunks)
- [ ] **T22**: Benchmark: learned chunk summary vs mean-K scoring — measure NIAH retrieval quality with synthetic data

### Phase 6: Benchmarks & GOAT Proof
- [ ] **T23**: `bench_dash_attn_routing()` — Compare top-k (fixed 8 blocks) vs entmax (adaptive 1-16 blocks) across: (a) NIAH needle position sweep, (b) multi-needle retrieval, (c) random noise queries. Report: accuracy, average active blocks, min/max active blocks
- [ ] **T24**: `bench_dash_attn_vs_pflash()` — End-to-end: PFlash standard prefill vs PFlash + entmax routing. Measure: TTFT, compression ratio, NIAH retrieval at various context lengths
- [ ] **T25**: `bench_entmax_overhead()` — Measure α-entmax threshold finding time for n_chunks ∈ {64, 128, 256, 512}. Should be <50µs for 256 chunks
- [ ] **T26**: GOAT proof test: entmax routing selects more chunks for hard queries, fewer for easy ones. Synthetic test with known difficulty labels. Assert: average active blocks for hard queries > 2× easy queries

### Phase 7: Documentation & Polish
- [ ] **T27**: Update `README.md` — add DashAttention section after PFlash/SP-KV with: adaptive routing table, composability pipeline, feature flag
- [ ] **T28**: Update `Cargo.toml` feature flags section in README
- [ ] **T29**: Fix all clippy warnings: `cargo clippy --features dash_attn --fix --allow-dirty`
- [ ] **T30**: Commit with message `feat(dash_attn): adaptive sparse hierarchical attention via α-entmax routing`

---

## Architecture

```text
src/dash_attn/                    — Feature-gated module: #[cfg(feature = "dash_attn")]
├── mod.rs                        — Module index, re-exports
├── entmax.rs                     — entmax_1p5(), entmax_support(), entmax_gqa_aggregate()
├── chunk_summary.rs              — ChunkSummaryQuery, ChunkSummaryCache
├── routing.rs                    — score_blocks_entmax(), compute_routing_bias()
└── forward.rs                    — forward_dash_attn_prefill(), forward_dash_attn_decode()

src/speculative/
└── prefill.rs                    — block_select_entmax() alternative (T20)

microgpt-core/src/types.rs
├── DashAttnConfig struct         — chunk_size, alpha, scaling_factor, sigma, estimate_diagonal
└── AttentionMode::DashAttn       — new variant

src/transformer.rs
└── ForwardContext                 — dispatch DashAttn variant

src/benchmark.rs                  — Phase 6 benchmarks
```

---

## Key Design Decisions

1. **α=1.5 only** — Quadratic operations (no exp/log), closed-form threshold in 2 passes. α=1.25→1.5 annealing is training-time only; we do inference at fixed α=1.5.

2. **Feature gate `dash_attn`** — All new code behind feature flag. Default off. Zero cost when disabled. Composable with `sp_kv` and `turboquant`/`spectralquant`.

3. **CPU-first** — α-entmax routing on CPU is O(n_chunks) per query, trivially fast for n_chunks ≤ 512. GPU kernel work goes to riir-ai Plan 106+.

4. **Drop-in for PFlash** — `block_select_entmax()` has same signature as `block_select()` — can swap without changing pipeline. Returns variable-length Vec<usize> instead of fixed-size.

5. **Zero-init fallback** — `ChunkSummaryQuery::head_cls` starts at zero → mean pooling. Models trained without DashAttention work correctly (no training required for inference benefit).

6. **Prior strength σ=10^6** — Paper shows weak prior is sufficient. For inference, we use σ=10^6 which makes the routing bias near-zero, relying on the adaptive support for sparsity.

7. **Not for training** — We implement inference only. riir-ai/riir-burner handles training with differentiable routing.

---

## Expected Outcomes

### Success Criteria

1. ✅ `entmax_1p5()` produces valid probability distribution (sum=1, non-negative, exact zeros)
2. ✅ Adaptive support: different queries select different numbers of chunks
3. ✅ GQA aggregation preserves sparsity (non-dispersive)
4. ✅ Learned chunk summary at zero-init equals mean pooling
5. ✅ `forward_dash_attn_prefill()` compiles and runs without panics
6. ✅ PFlash+entmax ≥ PFlash+top-k at same average sparsity (NIAH retrieval)
7. ✅ α-entmax overhead < 50µs for 256 chunks (trivial vs attention cost)

### What This Proves

- ✅ Adaptive sparsity via α-entmax is feasible in Rust
- ✅ Variable budget allocation improves quality vs fixed top-k
- ✅ Learned chunk summaries are backward-compatible (zero-init)
- ✅ Composable with existing SP-KV, TurboQuant, SpectralQuant

### What This Does NOT Prove

- ❌ Quality improvement on real trained models (requires training, riir-ai scope)
- ❌ GPU speedup (requires WGSL kernels, riir-ai scope)
- ❌ Better than NSA/InfLLMv2 on full benchmarks (requires 8B model + RULER eval)

---

## Relationship to Existing Plans

| Plan | Relationship |
|------|-------------|
| Plan 044 (PFlash) | `block_select_entmax()` replaces `block_select()` — same pipeline, adaptive budget |
| Plan 070 (SP-KV) | Composable: entmax routing (block-level) + SP-KV utility (token-level) |
| Plan 057 (HLA) | Orthogonal: HLA replaces attention entirely, DashAttn improves sparse selection |
| Plan 097 (Delta Routing) | Orthogonal: different axes (spatial vs depth) |
| Plan 099 (OCTOPUS) | Composable: entmax selects blocks, OCTOPUS compresses selected KV |
| Plan 102 (TileRT) | DashAttn can use TileRT's execution pipeline for kernel scheduling |

---

## Risks

1. **Entmax threshold convergence** — For pathological score distributions, bisection may need many iterations. Mitigation: α=1.5 has at most 2 passes; fallback to hard thresholding.

2. **Variable-length support breaks downstream** — PFlash expects fixed k blocks; changing to variable may break benchmarking harness. Mitigation: `block_select_entmax()` returns `Vec<usize>` and max allocation estimate.

3. **No quality guarantee without training** — Learned chunk summaries only help after training. Zero-init fallback is safe but provides no improvement over mean pooling. Mitigation: benchmark both modes, document zero-init behavior.

4. **Feature gate combinatorics** — `dash_attn` + `sp_kv` + `turboquant` + `spectralquant` all need to compile together. Mitigation: CI builds with `--all-features`.

5. **riir-ai GPU kernel dependency** — Full speedup requires WGSL kernels. Mitigation: Phase 1-5 are CPU-only, GPU work is separate riir-ai plan.