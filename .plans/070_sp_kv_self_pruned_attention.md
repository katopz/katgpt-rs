# Plan 070: SP-KV Self-Pruned Key-Value Attention

**Source**: [Self-Pruned Key-Value Attention: Learning When to Write by Predicting Future Utility](https://arxiv.org/pdf/2605.14037) (Meta FAIR, May 2026)

**Goal**: Implement learned sparse-write KV cache — predict future utility per KV pair, conditionally write only useful entries. Fill the decode-time sparse-write gap (we have sparse-read via PFlash, sparse-compress via TurboQuant, but no selective write during decode).

**Key Insight**: SP-KV adds exactly **one additive bias term** to existing `attention_head()`: `gate_bias = log(u_s)` during training, `0 | -inf` during inference. The utility predictor is a 2-layer MLP (`h → u ∈ (0,1)`) per KV head. No auxiliary loss — trains via next-token prediction only.

## Architecture Map

```
Existing                          SP-KV Addition
─────────────────────────────     ─────────────────────────────────
attention_head()                  attention_head_gated() + log(u) bias
KVCache (unconditional write)     SpKvCache (conditional write if u ≥ τ)
AttentionScorer (prefill-time)    UtilityPredictor (decode-time, every step)
FlashPrefillConfig (window+sinks) SpKvConfig (window + threshold τ)
forward_base()                    forward_sp_kv() dispatch variant
─                                 forward_sp_kv_tq() (SP-KV + TurboQuant)
```

## Phase Layout

- **Phase 1**: Core mechanism (types, utility predictor, gated attention, sparse-write cache)
- **Phase 2**: Forward pass integration + training pipeline (soft gating, TAHG)
- **Phase 3**: TurboQuant fusion + GPU kernel (riir-ai)
- **Phase 4**: Benchmarks, NAS probe, documentation

---

## Tasks

### Phase 1: Core Mechanism
- [x] **T1**: Create `src/sp_kv/mod.rs` — module index, re-exports ✅
- [x] **T2**: Create `src/sp_kv/types.rs` — `SpKvConfig`, `SpKvGateMode`, `SpKvCache`, `SpKvLayerCache`, `UtilityPredictorWeights`, `SpKvPredictors`, `GateBiasBuffer` ✅
- [x] **T3**: Implement `UtilityPredictor` — 2-layer MLP: `d_model → hidden → n_kv_heads`, sigmoid output, `predict()` and `predict_single_head()`, soft/hard/tahg gate bias helpers, `UtilityAggregation` enum ✅
- [x] **T4**: Implement `attention_head_gated()` — copy `attention_head()`, add `gate_bias: Option<&[f32]>` param, add `gate_bias[t]` to score before softmax ✅
- [x] **T5**: Implement `SpKvCache` — sparse-write KV cache: `SpKvLayerCache` with `write_gated()` / `write_unconditional()`, per-position `retained` bitfield + `retained_count`, density tracking ✅
- [x] **T6**: Add `AttentionMode::SpKv` variant to `types.rs` + `sp_kv` feature flag in `Cargo.toml` + `pub mod sp_kv` in `lib.rs` ✅

### Phase 2: Forward Pass + Training
- [x] **T7**: Implement `forward_sp_kv()` — mirrors `forward_base()` with: (1) utility prediction after RMSNorm before QKV, (2) conditional KV write via `SpKvLayerCache::write_gated()`, (3) gate-biased attention via `attention_head_gated()`, (4) `SpKvForwardContext` for zero-alloc gate bias buffer + predictor scratch ✅
- [x] **T8**: Implement soft-gating training mode — `GateBiasBuffer::build_soft()`: bias = `log(u + ε)`, preserves gradient flow, predictor bias=+5 init (σ(5)≈0.993) ✅
- [x] **T9**: Implement TAHG (Threshold-Aware Hard Gating) — `SpKvGateMode::Tahg`, `GateBiasBuffer::build_tahg()`, `SpKvPredictors::freeze()`, annealing via `SpKvConfig::gate_mode_at_step()` ✅
- [x] **T10**: Add SP-KV fields to `Config`: `sp_kv_window` (128), `sp_kv_threshold` (0.5), `sp_kv_predictor_hidden` (0=auto), `sp_kv_predictor_lr_mult` (5.0) + `InferenceOverrides::sp_kv_threshold` ✅
- [x] **T11**: Wire `forward_sp_kv()` — `AttentionMode::SpKv` variant added, `SpKvForwardContext` for dispatch, full feature-gated module behind `sp_kv` flag ✅

### Phase 3: Integration
- [ ] **T12**: Implement `forward_sp_kv_tq()` — SP-KV selective write + TurboQuant quantize what's kept (two-stage compression: selective write + lossy quant)
- [ ] **T13**: Add `sp_kv` feature flag to `Cargo.toml`
- [ ] **T14**: Create `riir-ai/crates/riir-gpu/src/kernels/attention_score_sp_kv.wgsl` — attention scoring with gate bias uniform buffer
- [ ] **T15**: Create `riir-ai/crates/riir-gpu/src/forward_sp_kv.rs` — GPU dispatch for SP-KV forward pass

### Phase 4: Benchmarks + Documentation
- [ ] **T16**: Benchmark: baseline `attention_head()` vs `attention_head_gated()` — measure gate bias overhead (expect <1%)
- [ ] **T17**: Benchmark: KV cache memory — full KV vs SP-KV at τ={0.3, 0.5, 0.7, 0.9} — measure density ratio
- [ ] **T18**: Benchmark: decode latency — full KV vs SP-KV sparse decode at batch=1 and batch=16
- [ ] **T19**: Test: palindrome reversal — verify SP-KV can learn long-range dependencies that sliding window cannot (paper Appendix G)
- [ ] **T20**: Test: utility predictor gradient flow — verify `log(u)` gate preserves gradients, frozen predictor in TAHG phase
- [ ] **T21**: Update README.md — add SP-KV section with architecture diagram
- [ ] **T22**: Create `.docs/14_sp_kv_research.md` — full research distillation
- [ ] **T23**: Commit with message `feat(sp_kv): self-pruned key-value attention (Plan 070)`

---

## SpKvConfig Design

```rust
pub struct SpKvConfig {
    /// Local sliding window always retained (default: 128).
    pub window: usize,
    /// Gate threshold τ for hard gating at inference (default: 0.5).
    pub threshold: f32,
    /// Utility predictor hidden dimension (default: d_model / 4).
    pub predictor_hidden: usize,
    /// Utility predictor learning rate multiplier (default: 5.0).
    pub predictor_lr_mult: f32,
    /// Initial bias for utility predictor (default: 5.0, σ(5)≈0.993 = fully open).
    pub predictor_init_bias: f32,
    /// TAHG annealing steps (default: 500).
    pub tahg_anneal_steps: usize,
    /// TAHG starts at this fraction of training (default: 0.75).
    pub tahg_start_fraction: f32,
}
```

## SpKvCache Design

```rust
pub struct SpKvLayerCache {
    /// Standard KV storage (sparse — only retained positions filled).
    pub key: Vec<f32>,       // [block_size, kv_dim]
    pub value: Vec<f32>,     // [block_size, kv_dim]
    /// Per-position gate utility scores (for training gradient flow).
    pub utilities: Vec<f32>, // [block_size, n_kv_heads]
    /// Bitfield: which positions have retained KV entries.
    pub retained: Vec<bool>, // [block_size]
    /// Number of retained positions (for density computation).
    pub retained_count: usize,
}

pub struct SpKvCache {
    pub layers: Vec<SpKvLayerCache>,
    pub config: SpKvConfig,
}
```

## UtilityPredictor Design

```rust
pub struct UtilityPredictorWeights {
    /// First layer: [hidden, d_model]
    pub w1: Vec<f32>,
    /// First layer bias: [hidden]
    pub b1: Vec<f32>,
    /// Second layer: [n_kv_heads, hidden]
    pub w2: Vec<f32>,
    /// Second layer bias: [n_kv_heads] — init to +5.0 for fully-open start
    pub b2: Vec<f32>,
}

impl UtilityPredictorWeights {
    /// Predict utility per KV head from hidden state.
    /// Returns [n_kv_heads] values in (0, 1).
    pub fn predict(&self, h: &[f32], d_model: usize, n_kv_heads: usize, hidden: usize) -> Vec<f32>
    // MLP: hidden = SiLU(w1 * h + b1), utilities = sigmoid(w2 * hidden + b2)
}
```

## attention_head_gated Insertion Point

In `attention_head()` at line ~358 (Pass 1: Q·K scores):

```rust
// Existing:
let score = dot * scale;

// SP-KV addition (when gate_bias is Some):
let score = dot * scale + gate_bias[t]; // gate_bias[t] = log(u_t) or 0.0 or -inf
```

One line change. Gate bias is precomputed before attention loop:
- Training: `gate_bias[s] = log(u_s + ε)` for all s in [0, t_n)
- Inference: `gate_bias[s] = if in_window(t,s) { 0.0 } else if u_s >= τ { 0.0 } else { -inf }`

## Relationship to Existing Mechanisms

| Mechanism | Compression Axis | When | Orthogonal? |
|-----------|-----------------|------|-------------|
| **TurboQuant** | Precision (f32→2-4 bit) | Always | ✅ Combine: SP-KV selects, TQ quantizes what's kept |
| **PFlash** | Prefill tokens (block-sparse) | Prefill only | ✅ Different phase |
| **Raven RSM** | Fixed slots (O(1) memory) | Always | ❌ Alternative paradigm (replaces KV) |
| **HLA/AHLA** | No KV at all (streaming) | Always | ❌ Alternative paradigm (replaces KV) |
| **SP-KV** | Selective write (sparse entries) | Decode | New axis |

## Expected Results (from paper, 8.1B model)

- Gate density: ~30% retained on standard tasks, ~17% on RULER
- NLL degradation: +0.08% at τ=0.5, +0.46% at τ=0.7
- Decode speedup: 2.1×–4.6× at batch=16
- NIAH perfect retrieval at only 5-7% density
- Scaling: follows same power law as full attention (R² > 0.999)

## References

- Paper: arXiv:2605.14037 (Meta FAIR, 2026)
- Baseline comparisons: StreamingLLM, H2O, KVZap, ExpectedAttention
- Our TurboQuant: arXiv:2504.19874
- Our PFlash: `src/speculative/prefill.rs`
- Our Raven: `src/transformer.rs:RavenKVCache`
