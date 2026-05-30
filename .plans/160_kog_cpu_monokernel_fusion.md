# Plan 160: Kog CPU Monokernel Fusion — RMSNorm Folding + QKV Interleaving

**Date:** 2026-05-30
**Status:** Planned
**Research:** R139 (Kog Monokernel CPU Conceptual Mapping)
**Cross-ref:** riir-ai P171 (GPU decode fusion — identical optimizations on GPU path)
**Feature Gate:** `kog_cpu_fusion` (opt-in, GOAT proof required before default-ON)

---

## Context

Kog AI achieves 3,000 tok/s via monokernel optimizations (Research 139). Two patterns transfer to our CPU SIMD inference stack:

1. **RMSNorm gamma folding** — fold norm weights into projection weights at init time, eliminating per-token gamma loads
2. **QKV weight interleaving** — repack separate Q/K/V weight matrices into cache-friendly execution order

Our current `rmsnorm()` has **no gamma parameter** — it's just `x *= inv_rms`. Real models (Gemma 2) use `y[i] = x[i] * inv_rms * gamma[i]`. When gamma exists, we can fold it into the weight matrix that follows, saving a per-norm memory read pass.

At micro config (n_embd=16) this saves ~6.6 KB/token. At Gemma 2 scale (n_embd=2304) it saves ~960 KB/token. At 30,000 CCU, that's 28.8 GB/s of memory bandwidth saved — material.

**Why do this for research spirit:**
- Keeps katgpt-rs identical to riir-ai Plan 171 (Phase 3: T10-T11)
- The MBU diagnostic framework applies to CPU too — we can't measure if we don't instrument
- Even if micro gains are noise, the code patterns and GOAT proof framework are reusable

---

## Optimization Alignment

Per `.contexts/optimization.md`:
- ✅ "Profile first" — we have measured MBU from riir-ai (2.1% GPU, CPU likely lower)
- ✅ "Pre-compute values that don't change across samples" — gamma is constant, fold at init
- ✅ "Pre-allocate output arrays upfront" — interleaved weights replace existing buffers, same total size
- ✅ "Batch API: amortize overhead" — folding eliminates 4 weight loads per layer entirely

---

## Tasks

### Phase 1: RMSNorm Gamma Infrastructure (T1-T3)

- [ ] T1: Add norm gamma weights to `LayerWeights`
  - Add `attn_norm_gamma: Vec<f32>` — pre-attention RMSNorm gamma `[n_embd]`
  - Add `mlp_norm_gamma: Vec<f32>` — pre-MLP RMSNorm gamma `[n_embd]`
  - Add `post_norm_gamma: Vec<f32>` — post-attention RMSNorm gamma `[n_embd]` (Gemma 2 post-norm)
  - Add `final_norm_gamma: Vec<f32>` — final RMSNorm gamma `[n_embd]`
  - Init with `1.0` (identity) in `TransformerWeights::new()` — zero behavioral change
  - **Scope:** `src/transformer.rs` `LayerWeights` struct + `TransformerWeights::new()`

- [ ] T2: Add gamma parameter to `rmsnorm()` and `rmsnorm_with_eps()`
  - New signature: `fn rmsnorm_with_gamma(x: &mut [f32], gamma: &[f32])`
  - Implementation: `x[i] *= inv_rms * gamma[i]` (fused multiply, no separate pass)
  - Keep existing `rmsnorm()` as `rmsnorm_with_gamma(x, &[1.0; 0])` or just unchanged for backward compat
  - **Scope:** `crates/katgpt-core/src/types.rs`

- [ ] T3: Wire gamma into `forward_base()` and `forward_coda()`
  - Replace `rmsnorm(&mut ctx.x)` with `rmsnorm_with_gamma(&mut ctx.x, &layer_weights.attn_norm_gamma)`
  - Same for MLP norm, post-norm (Gemma 2), final norm
  - **GOAT gate:** only active under `#[cfg(feature = "kog_cpu_fusion")]`
  - Without feature: existing `rmsnorm()` called (gamma=identity, behavior unchanged)

### Phase 2: RMSNorm Gamma Folding (T4-T5)

- [ ] T4: Implement `fold_gamma_into_weight()` in `TransformerWeights`
  - For each projection preceded by RMSNorm: `weight[row * n_embd + col] *= gamma[col]`
  - Applies to: `attn_wq`, `attn_wk`, `attn_wv` (after pre-attention norm), `mlp_w1` (after MLP norm)
  - Sets gamma to `1.0` after folding (runtime rmsnorm becomes identity-multiply)
  - **Scope:** `impl TransformerWeights { fn fold_gamma(&mut self, config: &Config) }`
  - Call after `new()` or after loading weights from file (future GGUF path)

- [ ] T5: GOAT proof — folded weights produce identical forward pass output
  - Generate random weights with non-trivial gamma (not all 1.0)
  - Run `forward_base` with unfolded gamma → capture logits for 16 tokens
  - Fold gamma, run `forward_base` again → capture logits
  - Assert bit-identical (or within 1e-6 FP tolerance)
  - Same for `forward_coda` path
  - **Scope:** `src/transformer.rs` `#[cfg(test)] mod tests`

### Phase 3: QKV Weight Interleaving (T6-T8)

- [ ] T6: Add interleaved QKV weight storage to `LayerWeights`
  - Add `attn_qkv_fused: Option<Vec<f32>>` — `[n_embd + 2*kv_dim, n_embd]` interleaved
  - Layout: group by head — `[Q_head0, K_head0, V_head0, Q_head1, K_head1, V_head1, ...]`
  - GQA: K/V groups shared, so interleaving is per-head-group, not per-head
  - None by default — only populated when `kog_cpu_fusion` is enabled and interleave is called
  - **Scope:** `src/transformer.rs` `LayerWeights`

- [ ] T7: Implement `interleave_qkv()` in `TransformerWeights`
  - Repack `attn_wq` + `attn_wk` + `attn_wv` into `attn_qkv_fused`
  - Row order: for each head group, contiguous Q rows, K rows, V rows
  - Preserves original weights (fused is an additional buffer)
  - **Scope:** `impl TransformerWeights { fn interleave_qkv(&mut self, config: &Config) }`

- [ ] T8: Wire fused QKV into `forward_base()` and `forward_coda()`
  - When `attn_qkv_fused` is Some: single matmul into fused buffer, then split Q/K/V slices
  - When None: existing separate `matmul()` calls (backward compat)
  - Cache locality win: single sequential weight read instead of 3 scattered reads
  - **GOAT gate:** `#[cfg(feature = "kog_cpu_fusion")]`

### Phase 4: MBU Diagnostic + GOAT Proof (T9-T11)

- [ ] T9: Add MBU (Memory Bandwidth Utilization) diagnostic
  - Track bytes read from weight buffers per forward pass
  - Compare against theoretical peak memory bandwidth (CPU-specific)
  - Print MBU % in benchmark output
  - **Scope:** `src/benchmark/` or new `src/mbu.rs`

- [ ] T10: GOAT proof — QKV interleaving produces identical attention output
  - Generate random weights, run forward with separate Q/K/V
  - Interleave, run forward with fused QKV
  - Assert attention outputs bit-identical (within FP tolerance)
  - **Scope:** `src/transformer.rs` `#[cfg(test)] mod tests`

- [ ] T11: GOAT proof — full pipeline (gamma fold + QKV interleave) end-to-end
  - 128-token greedy generation with baseline weights
  - Fold gamma + interleave QKV, generate 128 tokens
  - Assert all tokens identical
  - **Scope:** integration test

### Phase 5: Feature Gate + Benchmark (T12-T14)

- [ ] T12: Feature gate `kog_cpu_fusion` in `Cargo.toml`
  - Opt-in (not in default features yet)
  - Gates: T2 (gamma rmsnorm), T3 (forward wiring), T8 (fused QKV path)
  - Does NOT gate T1 (weight fields always present) or T4/T7 (init methods always available)

- [ ] T13: Benchmark — measure per-layer time before vs after optimization
  - Config::micro() — expect negligible change (model fits in L1)
  - Config::gemma2_2b() — expect measurable MBU improvement
  - Report: tok/s, ms/token, MBU %

- [ ] T14: If GOAT passes with no regression — promote `kog_cpu_fusion` to default-ON

---

## Technical Notes

### RMSNorm Gamma Folding Math

Original forward:
```
x = rmsnorm(x)              // x[i] *= inv_rms  (no gamma currently)
q = matmul(wq, x)           // q[i] = Σ wq[i,j] * x[j]
```

With gamma (proper RMSNorm):
```
x = rmsnorm_with_gamma(x, gamma)  // x[i] *= inv_rms * gamma[i]
q = matmul(wq, x)                 // q[i] = Σ wq[i,j] * (x[j] * inv_rms * gamma[j])
```

After folding gamma into wq:
```
wq_folded[i,j] = wq[i,j] * gamma[j]    // done once at init
x = rmsnorm(x)                          // x[i] *= inv_rms  (no gamma needed)
q = matmul(wq_folded, x)                // q[i] = Σ (wq[i,j]*gamma[j]) * (x[j]*inv_rms)
                                        //      = Σ wq[i,j] * gamma[j] * x[j] * inv_rms  ✅ identical
```

**Key insight:** gamma is element-wise and static, so `wq * gamma` is valid because `(a*b)*c = a*(b*c)` for scalar multiplication.

### QKV Interleaving Layout

Current (separate matrices, scattered reads):
```
wq: [n_embd × n_embd]    ← read 1
wk: [kv_dim × n_embd]    ← read 2 (different memory region)
wv: [kv_dim × n_embd]    ← read 3 (different memory region)
```

Interleaved (single matrix, sequential read):
```
qkv_fused: [(n_embd + 2*kv_dim) × n_embd]
Row 0:     Q_head0_dim0   ─┐
Row 1:     Q_head0_dim1    │ head 0 Q
...                        │
Row hd-1:  Q_head0_dimN   ─┘
Row hd:    K_head0_dim0   ─┐
...                        │ head 0 K (GQA: shared group)
Row 2hd-1: K_head0_dimN   ─┘
Row 2hd:   V_head0_dim0   ─┐
...                        │ head 0 V (GQA: shared group)
Row 3hd-1: V_head0_dimN   ─┘
Row 3hd:   Q_head1_dim0   ─┐
...                        │ head 1 Q
```

Benefit: when computing head h, Q/K/V weight rows are contiguous in memory → better L1/L2 cache utilization.

### Memory Budget Impact

| Config | Before (per layer) | After (per layer) | Saved |
|--------|-------------------|-------------------|-------|
| micro (n_embd=16, 1 layer) | ~2.1 KB gamma reads | 0 KB | ~2.1 KB |
| Gemma 2 (n_embd=2304, 26 layers) | ~960 KB gamma reads | 0 KB | ~960 KB |

QKV interleaving: same total weight bytes, better cache locality (reduced cache miss rate, not reduced bandwidth).

---

## Scope & Limits

- **CPU SIMD only** — no GPU, no wgpu, no Metal
- **Both forward paths** — `forward_base` and `forward_coda` must both be updated
- **All configs** — micro, gqa_draft, gemma2_2b
- **No model retraining** — inference-only optimization
- **LoRA compatible** — folding happens before LoRA application, LoRA path unchanged
- **Feature-gated** — zero impact when `kog_cpu_fusion` is off

## Dependencies

- Plan 103 (CODA fused kernels) ✅ — forward_coda already has delayed RMS infrastructure
- Plan 087 (Gemma 2 config) ✅ — `Config::gemma2_2b()` already exists with proper RMSNorm settings
- riir-ai Plan 171 Phase 3 (T10-T11) — same optimizations on GPU side, keep patterns identical
