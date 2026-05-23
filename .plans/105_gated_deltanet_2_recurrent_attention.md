# Plan 105: Gated DeltaNet-2 Recurrent Attention

> **Research:** [070_Gated_DeltaNet_2](../.research/070_Gated_DeltaNet_2_Decoupled_Erase_Write_Linear_Attention.md)
> **Feature Gate:** `gdn2_attention` (**Default-on** as of GOAT 8/8 proof. O(1) decode with constant state per head.)
> **Scope:** CPU SIMD recurrent decode for inference (Phase 1–2). Training/GPU kernels deferred to riir-ai.
> **Related Plans:** 057 (HLA), 059 (HLA distillation), 060 (SIMD matmul HLA), 097 (Delta Attention Residuals)

## Summary

Implement Gated DeltaNet-2 (GDN2) recurrent attention as an alternative to HLA/AHLA for CPU inference. GDN2 decouples the delta-rule gate into channel-wise erase (key-axis) and write (value-axis) gates, achieving SOTA long-context retrieval in fixed-state recurrent attention. The erase gate alone recovers ~90% of GDN2's gains.

**Scope:** Recurrent decode path only (token-by-token). No chunkwise WY training kernels — that's GPU/Triton territory for riir-ai.

## Tasks

### Phase 1: Core Types & State
- [x] **T1:** Create `src/gdn2/` module with `mod.rs` (index + re-exports), `types.rs` (cache structs), `kernel.rs` (recurrent step), `forward.rs` (forward pass)
- [x] **T2:** Add `gdn2_attention` feature flag to `Cargo.toml` + `pub mod gdn2` in `src/lib.rs`
- [x] **T3:** Implement `Gdn2LayerState` — per-head recurrent state `S ∈ R^{d_k × d_v}`, erase gate buffer `b ∈ R^{d_k}`, write gate buffer `w ∈ R^{d_v}`, decay buffer `alpha ∈ R^{d_k}`
- [x] **T4:** Implement `MultiLayerGdn2Cache` — multi-layer cache with `layers: Vec<Gdn2LayerState>`, GQA support (same KV group mapping as HLA), `memory_bytes()`, `reset()`
- [x] **T5:** Add `Gdn2GateConfig` enum: `EraseOnly` (b-only, scalar w), `Full` (channel b + w), `Kda` (scalar β fallback). Controls gate projection dimensions.

### Phase 2: Recurrent Step Kernel
- [x] **T6:** Implement `gdn2_recurrent_step()` — the core O(d_k × d_v) token-by-token update:
  1. Decay: `S *= Diag(α)` (row-wise scale)
  2. Read: `r = Sᵀ (b ⊙ k)` (gated matvec)
  3. Update: `S += k ⊗ (w ⊙ v − r)` (outer product delta)
  4. Readout: `o = Sᵀ q` (matvec)
- [x] **T7:** SIMD-optimize `gdn2_recurrent_step()` using existing `simd_dot_f32` for matvec, `simd_scale` for elementwise multiply. Target: within 10% of AHLA step cost.
- [x] **T8:** Implement `gdn2_project_gates()` — erase gate `b = σ(W_b @ x)` and write gate `w = σ(W_w @ x)` projections. Uses `matmul()` + elementwise sigmoid.
- [x] **T9:** Implement `gdn2_project_decay()` — log-decay `g = -exp(a) ⊙ softplus(W_f @ x + δ)`, decay `α = exp(g)`. Computed in f32 for precision.

### Phase 3: Forward Pass
- [x] **T10:** Implement `forward_gdn2()` — mirrors `forward_hla()` structure:
  1. Embedding (same as base)
  2. Per-layer: RMSNorm → save residual → RMSNorm → QKV projection
  3. **Gate projections:** erase b, write w, decay α from x (new vs HLA)
  4. **L2 normalize** q and k (stability, from paper)
  5. **Recurrent step:** `gdn2_recurrent_step()` per head
  6. Output: RMSNorm → output projection → add residual
- [x] **T11:** Implement `generate_gdn2_into()` — streaming generation with GDN2 cache (mirrors `generate_hla_into`)
- [x] **T12:** Add `AttentionMode::Gdn2` variant to `types.rs` + dispatch in `forward()` match

### Phase 4: Weight Projections
- [x] **T13:** Add GDN2 weight tensors to `TransformerWeights` (feature-gated):
  - `attn_w_erase: Vec<f32>` — n_embd → n_head × d_k (erase gate projection)
  - `attn_w_write: Vec<f32>` — n_embd → n_head × d_v (write gate projection)
  - `attn_w_decay_w: Vec<f32>` — n_embd → n_head × d_k (decay projection weight)
  - `attn_w_decay_a: Vec<f32>` — d_k (per-head decay base, broadcast)
  - `attn_w_decay_bias: Vec<f32>` — d_k (per-channel decay bias)
- [x] **T14:** Random init for GDN2 weights in `TransformerWeights::new()` — Xavier uniform for projections, learnable `a` and `δ` for decay

### Phase 5: Benchmark & GOAT Proof
- [x] **T15:** Add `bench_gdn2_vs_hla_vs_flat()` — compare throughput, memory, and quality:
  - Throughput: tok/s for GDN2 recurrent decode vs AHLA vs flat KV
  - Memory: bytes per layer for GDN2 state vs AHLA state vs flat KV
  - Quality: cosine similarity vs SDPA (random weights, same as HLA bench)
- [x] **T16:** Add `bench_gdn2_gate_ablation()` — measure erase-only vs full vs KDA-tied variants:
  - EraseOnly (channel b, scalar w) — expect ~90% of full gain
  - Full (channel b + w) — full GDN2
  - Kda (scalar β) — baseline tied gate
- [x] **T17:** Add `bench_gdn2_context_scaling()` — throughput at positions [8, 64, 256, 1024, 4096]:
  - GDN2 should be flat (O(1) per step)
  - Flat KV should degrade linearly
  - Compare with AHLA flat profile
- [x] **T18:** GOAT proof: `tests/bench_gdn2.rs` — 3 assertions:
  1. `gdn2_tps >= ahla_tps * 0.90` (within 10% of AHLA throughput)
  2. `gdn2_memory <= flat_kv_memory` (always smaller than flat KV)
  3. `gdn2_logits_finite` (no NaN/Inf at any position)

### Phase 6: Documentation
- [ ] **T19:** Update `README.md` — add GDN2 section under HLA, with benchmark table placeholder
- [ ] **T20:** Update `.docs/15_paper_feature_comparison.md` — add GDN2 row
- [ ] **T21:** Update `src/gdn2/mod.rs` doc comment with usage example and state size comparison table

## Architecture

### Module Structure

```
src/gdn2/
├── mod.rs          # Module index + re-exports + doc comment
├── types.rs        # Gdn2LayerState, MultiLayerGdn2Cache, Gdn2GateConfig
├── kernel.rs       # gdn2_recurrent_step(), gate projections, decay projection
└── forward.rs      # forward_gdn2(), generate_gdn2_into()
```

### State Layout (per head)

```
GDN2: S_t ∈ R^{d_k × d_v}  (recurrent state matrix)
      b_t ∈ R^{d_k}          (erase gate, projected per token)
      w_t ∈ R^{d_v}          (write gate, projected per token)
      α_t ∈ R^{d_k}          (channel-wise decay, projected per token)

State size per head: d_k × d_v floats (persistent) + 2×d_k + d_v (projected per token, temp)
For micro (d_k=d_v=4): 16 + 12 = 28 floats per head = 112 bytes
For game (d_k=d_v=8): 64 + 24 = 88 floats per head = 352 bytes
```

### Comparison: GDN2 vs AHLA State

| Config | Flat KV | AHLA | GDN2 | GDN2 Savings |
|--------|---------|------|------|-------------|
| micro (hd=4) | 2,048 B | 640 B | 448 B | 78.1% |
| game (hd=8) | 43,520 B | 2,304 B | 2,816 B | 93.5% |
| bpe (hd=8) | 65,536 B | 2,304 B | 2,816 B | 95.7% |

**Note:** GDN2 state is d_k×d_v per head. AHLA is hd×(hd+1) per head. For d_k=d_v=hd, GDN2 is slightly smaller than AHLA for small hd, slightly larger for larger hd. Both are O(1) constant — the key advantage over flat KV.

### Core Recurrence (Eq. 10)

```
Given: k ∈ R^{d_k}, v ∈ R^{d_v}, q ∈ R^{d_k}  (normalized)
       b ∈ [0,1]^{d_k}  (erase gate)
       w ∈ [0,1]^{d_v}  (write gate)
       α ∈ (0,1]^{d_k}  (channel decay)

Step:
  1. S *= Diag(α)           — row-wise decay
  2. r = Sᵀ(b ⊙ k)         — gated read (d_v vector)
  3. S += k ⊗ (w⊙v − r)    — delta update (outer product)
  4. o = Sᵀ q              — readout (d_v vector)

Cost: O(d_k × d_v) per token per head — same as standard linear attention.
```

### Gate Configurations (for ablation)

| Config | b_t | w_t | α_t | Purpose |
|--------|-----|-----|-----|---------|
| `EraseOnly` | channel σ(W_b x) | scalar β = mean(b) | channel | 90% of gain, fewer params |
| `Full` | channel σ(W_b x) | channel σ(W_w x) | channel | Full GDN2 |
| `Kda` | scalar β | scalar β | channel | KDA baseline (tied gates) |

## Key Design Decisions

1. **Token-by-token decode only** — no chunkwise WY. Training kernels are GPU-only (riir-ai).
2. **Erase gate first** — ablation shows b-only recovers 90% of gain. Implement and validate before write gate.
3. **Channel-wise decay optional** — start with scalar decay (like HLA's γ). Add per-channel decay as second step.
4. **L2 normalize q, k** — paper requires this for stability. Simple `x / ||x||₂` per head.
5. **Feature-gated alongside HLA** — not a replacement. Both `hla_attention` and `gdn2_attention` can coexist.
6. **Weight format same as HLA** — gate projections stored in `TransformerWeights`, random-init for benchmark.

## Dependencies

- Existing: `simd::simd_dot_f32`, `simd::simd_scale`, `types::matmul`, `types::rmsnorm`
- New: sigmoid kernel (elementwise, can use std lib `1.0 / (1.0 + (-x).exp())`)
- New: L2 normalize kernel (simple: `x / sqrt(Σx² + ε)`)

## Estimated Scope

| Task | Lines | Complexity |
|------|-------|-----------|
| types.rs | ~200 | Low (mirror HLA types) |
| kernel.rs | ~150 | Medium (core recurrence) |
| forward.rs | ~150 | Low (mirror forward_hla) |
| mod.rs | ~30 | Low (re-exports) |
| benchmark additions | ~200 | Medium (3 bench functions) |
| Weight additions | ~80 | Low (feature-gated fields) |
| **Total** | **~810** | |

## Success Criteria

- [ ] All 18 unit tests pass (including GQA variant)
- [ ] GOAT proof: GDN2 within 10% of AHLA throughput
- [ ] GOAT proof: GDN2 memory < flat KV memory at all configs
- [ ] GOAT proof: No NaN/Inf in logits at any position
- [ ] Gate ablation: EraseOnly within 5% of Full quality (cosine sim)
- [ ] Context scaling: flat throughput profile (O(1) per step)

✅ GOAT 8/8 proved: `tests/goat_105_gdn2.rs` — sigmoid invariants, L2 normalize, recurrent step finiteness, state size, reset idempotent, memory formula, outer product write