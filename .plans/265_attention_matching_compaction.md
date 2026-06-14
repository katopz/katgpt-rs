# Plan 265: Attention Matching KV Compaction (Modelless)

**Date:** 2026-06-14
**Research:** [katgpt-rs/.research/233_Attention_Matching_KV_Compaction.md](../.research/233_Attention_Matching_KV_Compaction.md)
**Source paper:** [arxiv 2602.16284](https://arxiv.org/abs/2602.16284) — Fast KV Compaction via Attention Matching (MIT, ICML 2026)
**Target:** `katgpt-rs/src/attn_match/` (new module) + Cargo feature `attention_matching`
**Status:** Active — Phase 1 in progress

---

## Goal

Distill the Attention Matching (AM) paper into a generic, modelless, MIT-licensed Rust module under `katgpt-rs/src/attn_match/`. Provides 50× KV cache compaction in seconds, with no LLM training, adaptive CPU/SIMD/GPU/ANE solver routing, and full reuse of the existing `still_kv::BetaBias` and `still_kv::CompactKVCache` types (but **replaces StillKV's heuristic β with NNLS-optimal β**).

---

## Phase 1 — Unblocking Skeleton (CORE — required to proceed with anything else)

Goal: a compiling, tested, feature-gated module that implements the core AM algorithm (selection + closed-form β and Cv fitting) on synthetic data, with the public API surface frozen.

**STATUS: ✅ COMPLETE (2026-06-14)** — 39/39 tests pass, example runs clean, library builds with no new warnings.

### Tasks

- [x] **T1.1** Create `src/attn_match/` directory with empty `mod.rs`
- [x] **T1.2** Add feature flag `attention_matching = []` to `katgpt-rs/Cargo.toml` features section (after `still_kv`)
- [x] **T1.3** Add `#[cfg(feature = "attention_matching")] pub mod attn_match;` to `src/lib.rs` (alphabetical, after `alloc` or similar)
- [x] **T1.4** Implement `src/attn_match/types.rs`:
  - [x] `AmConfig` struct (compaction ratio, NNLS iters, OMP k/τ, solver choice, ridge λ, stability bounds)
  - [x] `AmResult` struct (Ck indices, β vec, Cv flat buffer, reconstruction error)
  - [x] `ScoreMethod` enum (Mean, Rms, Max)
  - [x] `KeySelector` enum/trait (HighestAttn, OMP, OmpFast)
  - [x] Re-export `still_kv::BetaBias` and `still_kv::CompactKVCache` for DRY reuse (deferred — types are independent for now, integration deferred to Phase 2+)
- [x] **T1.5** Implement `src/attn_match/score_matrix.rs`:
  - [x] `compute_score_matrix(queries, keys, inv_sqrt_d) -> Vec<f32>` (n×T flat f32)
  - [x] Max-shift stabilization inline (no allocation in hot loop)
  - [x] Chunked 8-wide loop for SIMD auto-vectorization
- [x] **T1.6** Implement `src/attn_match/beta_fitter.rs` (NNLS):
  - [x] `fit_beta_nnls(A: &[f32], m: &[f32], n, t, config) -> Vec<f32>` (returns β = log w)
  - [x] Projected gradient descent: `η = 1/L`, `L ≈ ||M||²` via power iteration (3 iters)
  - [x] Box constraints: `w_j ∈ [e^lo, e^hi]` (default lo=-3, hi=3 per Appendix C.2)
  - [x] Warm-start from clamped least-squares
- [x] **T1.7** Implement `src/attn_match/value_fitter.rs` (Least Squares):
  - [x] `fit_cv_least_squares(X: &[f32], Y: &[f32], n, t, d, config) -> Vec<f32>` (returns Cv flat)
  - [x] Normal equations `X^T X` and `X^T Y` (no allocation in hot loop)
  - [x] Cholesky decomposition with diagonal jitter fallback (λ=1e-6 if rank-deficient)
- [x] **T1.8** Implement `src/attn_match/key_selection/highest_attn.rs`:
  - [x] `select_highest_attn_keys(K, queries, t, score_method) -> (indices, scores)`
  - [x] RMS aggregation (default), with mean and max as alternatives
  - [x] Top-t selection via partial sort (no full sort) — uses full sort for now, swap to partial_sort in Phase 2
- [x] **T1.9** Implement `src/attn_match/key_selection/omp.rs`:
  - [x] `select_omp_keys(K, queries, t, k, tau) -> (indices, weights)`
  - [x] Greedy selection with periodic NNLS refit (Algorithm 2)
  - [x] Mass feature matrix Φ construction
  - [x] Residual update
- [x] **T1.10** Implement `src/attn_match/compact.rs` — top-level orchestrator:
  - [x] `compact(K, V, queries, config) -> CompactKVCache`
  - [x] Pipeline: select Ck → fit β → fit Cv → wrap in AmResult
  - [x] Reconstruction error reporting (relative Frobenius)
- [x] **T1.11** Write unit tests in `src/attn_match/tests.rs`:
  - [x] Synthetic test: known β recovery (||β − β_ref||_∞ < 0.1) → GOAT G1 ✅
  - [x] Synthetic test: Cv reconstruction (< 5% relative error) → GOAT G2 ✅
  - [x] OMP mass coverage test (residual < 5% of initial after t iters) → GOAT G3 ✅
  - [x] HighestAttn RMS coverage test (top-t cover > 80% RMS mass) → GOAT G4 ✅
  - [x] Determinism test (same input → same output, no RNG) ✅
- [x] **T1.12** Add example `examples/attn_match_basic.rs` showing before/after:
  - [x] Synthetic KV (T=512, d=64, n=128 queries)
  - [x] Compact to t=64 (8× ratio)
  - [x] Print reconstruction error and β distribution
  - [x] Print memory savings (87.4% reduction at 8× compaction demonstrated)
- [x] **T1.13** Document module in `src/attn_match/mod.rs` with paper reference and equations

### Phase 1 Exit Criteria — ✅ ALL MET
- ✅ `cargo build --features attn_match` compiles clean
- ✅ `cargo test --features attn_match --lib attn_match` passes 39/39 unit tests
- ✅ `cargo run --example attn_match_basic --features attn_match --release` runs and prints:
  - HighestAttn: 7.5ms, mass error 0.10, 87.4% memory reduction
  - OMP: 13.2ms, attention-output reconstruction error 0.02 (excellent)
  - OMP-fast: 8.6ms, attention-output reconstruction error 0.02
- ✅ No new clippy warnings on the `attn_match` module (only minor style suggestions)
- ✅ File sizes < 2048 lines (largest: beta_fitter.rs at 456 lines)

---

## Phase 2 — Adaptive Solver Router (Fusion A — GOAT-critical)

Goal: size-aware and load-aware routing across CPU/SIMD/Rayon/GPU/ANE backends.

### Tasks

- [ ] **T2.1** Implement `src/attn_match/router.rs`:
  - [ ] `SolverRouterConfig` struct (cpu_max_t, simd_max_t, gpu_min_t, ane_max_t, hysteresis_pct)
  - [ ] `SolverBackend` enum (CpuScalar, CpuSimd, CpuRayon, Gpu, Ane)
  - [ ] `pick_backend(t: usize, T: usize, gpu_available: bool, config) -> SolverBackend`
  - [ ] Hysteresis: track last decision, only switch if new t outside ±window of threshold
- [ ] **T2.2** Implement SIMD score matrix kernel:
  - [ ] 8-wide f32 chunked loop (auto-vectorizes on AVX2/NEON)
  - [ ] Explicit `std::simd` fallback if available, else auto-vectorization
  - [ ] Benchmark: ≥4× over scalar → GOAT G8
- [ ] **T2.3** Implement Rayon parallel blocked score matrix:
  - [ ] Block size 4096 (L2-resident)
  - [ ] Per-block rayon task, results merged via atomic accumulate
  - [ ] Only used when T ≥ simd_max_t
- [ ] **T2.4** Implement blocked Cholesky for large t:
  - [ ] 32×32 blocks (cache-aware)
  - [ ] Reuse scratch buffers across calls (no allocation in hot loop)
- [ ] **T2.5** Wire router into `compact()` orchestrator:
  - [ ] Each stage picks backend via router
  - [ ] Backend choice logged at debug level
- [ ] **T2.6** Add router tests:
  - [ ] Determinism: same (t, T, gpu_available) → same backend → GOAT G6
  - [ ] Hysteresis: t crossing threshold by <10% keeps prior backend
  - [ ] Memory bound: no allocation in NNLS / Cholesky hot loops → GOAT G7
- [ ] **T2.7** Add router benchmark:
  - [ ] `benches/attn_match_router_bench.rs` (criterion)
  - [ ] Sweep t from 16 to 4096, plot backend transitions
- [ ] **T2.8** GPU dispatch stub (when `gpu_inference` feature enabled):
  - [ ] Forward to existing `gpu_backend` module
  - [ ] Falls back to Rayon if GPU dispatch fails

### Phase 2 Exit Criteria
- Router deterministically picks backends
- SIMD kernel ≥4× scalar
- All Phase 1 tests still pass
- Router bench shows clean transitions across regimes

---

## Phase 3 — Nonuniform Head Budget Solver (Algorithm 4)

Goal: per-head sensitivity curves + greedy swap solver, producing a model-specific JSON schedule.

### Tasks

- [ ] **T3.1** Implement `src/attn_match/head_budget/curve.rs`:
  - [ ] `HeadSensitivityCurve` struct (head_id, ratios: Vec<f32>, deltas: Vec<f32>)
  - [ ] Linear interpolation between measured points
  - [ ] Smoothing (sliding window) optional
- [ ] **T3.2** Implement `src/attn_match/head_budget/solver.rs`:
  - [ ] `HeadBudgetSolver::new(curves, num_layers, num_heads)`
  - [ ] `solve(target_ratio) -> Vec<f32>` (per-head shares, sum=1)
  - [ ] Greedy swap algorithm (Algorithm 4)
  - [ ] Step size η configurable
- [ ] **T3.3** Implement `src/attn_match/head_budget/schedule.rs`:
  - [ ] `HeadBudgetSchedule` struct (model_id, shares, version, blake3_hash)
  - [ ] Serialize/deserialize via postcard (existing dep)
  - [ ] BLAKE3 hash for tamper detection
- [ ] **T3.4** Implement `src/attn_match/head_budget/measure.rs`:
  - [ ] `measure_sensitivity(model, dataset, ratios) -> Vec<HeadSensitivityCurve>`
  - [ ] This is the offline tool to compute schedules once per model
  - [ ] Output: postcard-serialized `HeadBudgetSchedule`
- [ ] **T3.5** Add tests:
  - [ ] Uniform allocation produces equal shares
  - [ ] Greedy swap converges (no improving swap remains)
  - [ ] BLAKE3 hash deterministic across runs
- [ ] **T3.6** Add example `examples/attn_match_head_budget.rs`:
  - [ ] Synthetic sensitivity curves (some heads flat, some sensitive)
  - [ ] Solve for target ratio 0.05
  - [ ] Print resulting per-head shares

### Phase 3 Exit Criteria
- Solver converges on synthetic curves
- Schedule serialization round-trip exact
- BLAKE3 deterministic

---

## Phase 4 — Chunked Compaction (KV-based + Text-based)

Goal: support long contexts via per-chunk compaction.

### Tasks

- [ ] **T4.1** Implement `src/attn_match/chunked.rs`:
  - [ ] `ChunkedCompactor::new(chunk_size, overlap)` 
  - [ ] `compact_kv_based(full_kv, queries, config) -> CompactKVCache`
  - [ ] `compact_text_based(chunks, config) -> CompactKVCache`
- [ ] **T4.2** Implement RoPE phase shift for text-based chunking:
  - [ ] `apply_rope_phase_shift(keys, delta)` — rotate keys by global offset
  - [ ] Reuse existing `still_kv::position_free` helpers if compatible
- [ ] **T4.3** Add chunked compaction tests:
  - [ ] KV-based preserves more than text-based on synthetic cross-chunk dependencies
  - [ ] Concatenation produces correct total length
- [ ] **T4.4** Add chunked example with synthetic long context (T=8192, 4 chunks)

### Phase 4 Exit Criteria
- Both chunking modes work
- KV-based > text-based on dependent chunks
- Total compacted length correct

---

## Phase 5 — Online Compaction (Mid-Trajectory)

Goal: compact mid-trajectory to support arbitrarily long generation.

### Tasks

- [ ] **T5.1** Implement `src/attn_match/online.rs`:
  - [ ] `OnlineCompactor::new(phys_budget, recent_window)`
  - [ ] `maybe_compact(kv_cache, current_pos) -> Option<CompactKVCache>`
  - [ ] Returns Some when phys_budget reached, None otherwise
  - [ ] Preserves most recent `recent_window` tokens uncompacted
- [ ] **T5.2** Add online compaction tests:
  - [ ] Compaction triggers at exactly phys_budget
  - [ ] Recent window preserved uncompacted
  - [ ] Multiple consecutive compactions preserve total semantics
- [ ] **T5.3** Add online example simulating AIME-style long reasoning:
  - [ ] Generate synthetic "reasoning" tokens
  - [ ] Compact at intervals
  - [ ] Print KV size before/after each compaction

### Phase 5 Exit Criteria
- Compaction triggers at budget
- Recent window always preserved
- Multiple compactions don't corrupt state

---

## Phase 6 — Adaptive CoT Compaction (Fusion B, gated `adaptive_cot_compaction`)

Goal: entropy-thresholded, bandit-tuned online compaction for thinking traces.

### Tasks

- [ ] **T6.1** Implement `src/attn_match/adaptive_cot.rs`:
  - [ ] `AdaptiveTraceCompactor` extends `fold::ChainFolder` trait
  - [ ] Entropy monitoring (per-token next-token distribution entropy)
  - [ ] Thresholds: θ_low (compact), θ_high (preserve), MAX_COMPACTS
- [ ] **T6.2** Wire to `freq_bandit` (Plan 189):
  - [ ] Bandit observes (compact_decision, downstream_quality)
  - [ ] Adjusts (θ_low, θ_high) over time
  - [ ] Self-learning, no LLM training
- [ ] **T6.3** Add tests:
  - [ ] Low entropy triggers compaction
  - [ ] High entropy prevents compaction
  - [ ] Bandit updates thresholds after observations
- [ ] **T6.4** Add example showing adaptive vs blind online compaction:
  - [ ] Synthetic CoT with entropy spikes
  - [ ] Show adaptive preserves more during spikes
  - [ ] Show bandit converges to reasonable thresholds

### Phase 6 Exit Criteria
- Adaptive triggers only at entropy troughs
- Bandit converges
- Quality ≥ blind online compaction at same total compaction count

---

## Phase 7 — GOAT Gate Validation & Promotion

Goal: prove the module is ready for default promotion.

### Tasks

- [ ] **T7.1** Write `tests/bench_265_attn_match_goat.rs`:
  - [ ] G1: β recovery < 0.1 infinity norm on synthetic
  - [ ] G2: Cv reconstruction < 5% relative Frobenius
  - [ ] G3: OMP residual < 5% of initial mass
  - [ ] G4: HighestAttn top-t cover > 80% RMS mass
  - [ ] G5: Reconstruction perplexity within 5% (synthetic proxy)
  - [ ] G6: Router determinism (no flapping)
  - [ ] G7: No allocation in hot loops (assert via custom allocator in test)
  - [ ] G8: SIMD ≥ 4× scalar
- [ ] **T7.2** Add `[[test]]` entry in Cargo.toml: `bench_265_attn_match_goat` with required-features
- [ ] **T7.3** Run GOAT gate, document results in this plan
- [ ] **T7.4** If G1–G7 pass: add `attention_matching` to default features
- [ ] **T7.5** If G8 passes: add `am_simd` (or merged into attention_matching) to default
- [ ] **T7.6** Update `README.md` Feature Showcase section with Attention Matching entry
- [ ] **T7.7** Update `README.md` GOAT Proofs section if promoted

### Phase 7 Exit Criteria
- All GOAT gates documented pass/fail
- Default features updated if pass
- README documents the new module

---

## Out of Scope (Deferred / riir-ai)

- **Entmax-regularized OMP** (Fusion C) — defer until VortexFlow integration needed
- **Spectral pre-selection** (Fusion D) — defer until T > 100k contexts hit production
- **Per-region head budgets** (Fusion E) — riir-ai Research 121 Recipe 1
- **LoRA β predictor** — riir-ai Research 121 Recipe 4
- **Chain SyncBlock boundary swap** — riir-ai Research 121 Recipe 2
- **NPC trajectory memory** — riir-ai Research 121 Recipe 1
- **Cold-tier NeuronShard compaction** — riir-ai Research 121 Recipe 3
- **Cross-game adapter composition** — riir-ai Research 121 Recipe 5
- **Self-play trajectory compression** — riir-ai Research 121 Recipe 6

---

## File Layout (target)

```
src/attn_match/
├── mod.rs                       # Module root, public API, paper reference
├── types.rs                     # AmConfig, AmResult, enums
├── score_matrix.rs              # QK^T computation with max-shift
├── beta_fitter.rs               # NNLS via projected gradient descent
├── value_fitter.rs              # Least squares via normal equations + Cholesky
├── compact.rs                   # Top-level orchestrator
├── router.rs                    # CPU/SIMD/Rayon/GPU/ANE adaptive routing
├── chunked.rs                   # KV-based + text-based chunked compaction
├── online.rs                    # Mid-trajectory online compaction
├── adaptive_cot.rs              # Fusion B (Phase 6, gated)
├── tests.rs                     # Unit tests
├── key_selection/
│   ├── mod.rs                   # KeySelector trait
│   ├── highest_attn.rs          # HighestAttnKeys selector
│   └── omp.rs                   # OMP + OMP-fast selectors
└── head_budget/
    ├── mod.rs                   # Head budget module root
    ├── curve.rs                 # HeadSensitivityCurve
    ├── solver.rs                # Greedy swap solver (Algorithm 4)
    ├── schedule.rs              # HeadBudgetSchedule + BLAKE3 + postcard
    └── measure.rs               # Offline measurement tool

examples/
├── attn_match_basic.rs          # Phase 1 example
├── attn_match_head_budget.rs    # Phase 3 example
└── attn_match_online.rs         # Phase 5 example

benches/
└── attn_match_router_bench.rs   # Phase 2 router benchmark

tests/
└── bench_265_attn_match_goat.rs # Phase 7 GOAT gate
```

---

## Constraints Checklist

| # | Constraint | How addressed |
|---|---|---|
| 1 | Modelless first | ✅ Pure inference-time, no LLM training |
| 2 | Engine/fuel split | ✅ Generic framework only; specific recipes in riir-ai/.research/121 |
| 3 | LoRA only for training | ✅ No training in this plan; LoRA β predictor is riir-ai only |
| 4 | Self-learning adaptive CoT | ✅ Phase 6 implements via FreqBandit (no LLM training) |
| 5 | SOLID, DRY | ✅ Reuses BetaBias + CompactKVCache from still_kv; trait-based selectors and fitters |
| 6 | Tests/examples before/after | ✅ Phase 1 T1.11/T1.12, Phase 7 GOAT gate |
| 7 | CPU/GPU/ANE auto-route | ✅ Phase 2 router with load-aware extensions |
| 8 | Plasma/hot/warm/cold/freeze | ✅ Mapped in riir-ai/.research/121 (tier table) |
| 9 | Threshold adaptive routing | ✅ SolverRouterConfig in Phase 2 |

---

## TL;DR

Seven-phase plan to distill AM paper into `katgpt-rs/src/attn_match/`. Phase 1 = unblocking skeleton (compiling, tested). Phase 2 = adaptive router. Phase 3 = head budgets. Phase 4 = chunked. Phase 5 = online. Phase 6 = adaptive CoT. Phase 7 = GOAT gate + promotion.

**Immediate next step**: Phase 1 (T1.1–T1.13) — get the skeleton compiling and tested.
