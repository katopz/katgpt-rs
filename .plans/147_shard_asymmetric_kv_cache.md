# Plan 147: Shard-Inspired Asymmetric Codec KV Cache Compression

> **Origin:** Research 109 (Shard — Drop-In 10× KV Cache)
**Status:** ✅ COMPLETE — DRAFT (high-value research, pending GOAT proof validation)
> **GOAT Pillar:** Infrastructure (supports all 4 pillars via memory efficiency)
> **Feature Gate:** `shard_kv` — opt-in, not in `default` or `full`
> **Related Plans:** 123 (Asymmetric KV benchmarks), 077 (SpectralQuant), 101 (Hybrid OCT+PQ)
> **Related Research:** 109 (Shard), 81 (Asymmetric K/V), 39 (SpectralQuant), 63 (OCTOPUS), 20 (TurboQuant)
> **Depends on:** `spectralquant` (eigenbasis), `turboquant` (Lloyd-Max codebook, rotation)
> **Blocks:** Nothing

---

## Task Index

- [x] Phase 1: RoPE-Removal Enhancement (T1–T6) — P1
- [x] Phase 2: Full Asymmetric Codec (T7–T16) — P2
- [x] Phase 3: Fused GPU Attention (T17–T21) — P3, riir-ai

---

## Verdict

Shard's asymmetric codec design (PCA on no-RoPE K + Hadamard+VQ on V) achieves 10–11× compression, roughly doubling our best result. The RoPE-removal insight is the single highest-value contribution — it can enhance our existing SpectralQuant with a drop-in change. The fused compressed attention is a novel capability for GPU paths.

**Priority: P1 for RoPE-removal enhancement to SpectralQuant, P2 for full asymmetric codec, P3 for fused GPU attention.**

---

## Motivation

Our current KV compression stack applies the *same codec* to both K and V (OCTOPUS triplet or SpectralQuant eigenbasis), only varying bit widths. Research 81 proved V compression is quality-free, but Shard demonstrates a deeper principle: K and V have fundamentally different *structural* properties requiring different *methods*:

- **K** is low-rank after RoPE removal → PCA is optimal
- **V** has flat spectral structure → rotation + VQ is optimal
- Applying the same codec to both leaves compression on the table

---

## Architecture

### Phase 1: RoPE-Removal Enhancement to SpectralQuant (P1, ~2 days)

Drop-in change to existing `spectralquant/`:

```
Before: raw K (with RoPE) → eigenbasis → water-fill → quantize
After:  raw K → undo RoPE → eigenbasis → water-fill → quantize
                                                ↑ 2× more variance captured
```

**Files to modify:**
- `src/spectralquant/spectral.rs` — add `undo_rope()` before eigendecomposition
- `src/spectralquant/spectral_rotation.rs` — store RoPE angles in calibration struct
- `src/spectralquant/forward.rs` — reapply RoPE after dequant (or use Shard's Δ-identity)

**GOAT proof targets:**
1. Eigenvalue concentration improves: d_eff drops (more variance in fewer components)
2. Cosine similarity at same bit budget improves ≥2%
3. Compression ratio at same quality target improves ≥1.5×

**No new feature gate** — enhancement to existing `spectral_quant` feature.

### Phase 2: Full Asymmetric Codec (P2, ~4 days)

New `ShardKVCache` that applies different methods to K and V:

```
K path (prefill):
  1. undo_rope(keys, positions, inv_freq)
  2. per-layer SVD (reuse spectralquant/)
  3. DP bit allocation with drop penalty (new — adapt from rope.py)
  4. int4 coefficient packing

V path (prefill):
  1. hadamard(values)
  2. k-means VQ on groups of 4 channels, 256-entry codebook
  3. pack indices

Decode streaming:
  1. Lloyd-Max 8-bit (reuse turboquant/codebook.rs)
  2. Bit-exact lossless path

Sink + window:
  1. First 4 tokens: FP16 (attention sinks)
  2. Last 64 tokens: FP16 (recency window)
```

**New files:**
- `src/shard_kv/mod.rs` — module index
- `src/shard_kv/types.rs` — `ShardConfig`, `ShardLayer`, `ShardCalibration`
- `src/shard_kv/rope.rs` — RoPE undo/reapply utilities
- `src/shard_kv/k_pca.rs` — PCA compression path for K
- `src/shard_kv/v_vq.rs` — Hadamard + k-means VQ for V
- `src/shard_kv/dp_bits.rs` — DP bit allocation with drop penalty
- `src/shard_kv/kv_cache.rs` — `ShardKVCache` struct
- `src/shard_kv/forward.rs` — dequant + attention paths
- `src/shard_kv/sink_window.rs` — attention sink + recency window management

**Feature gate:** `shard_kv` (opt-in, requires `spectral_quant` + `turboquant`)

**GOAT proof targets:**
1. Compression ≥ 8× at d=128, 8K context equivalent
2. Cosine similarity K ≥ 0.995, V ≥ 0.98 at recommended config
3. NIAH-equivalent: attention score correlation ≥ 0.998 vs FP16
4. 8-bit decode streaming: bit-exact match on 150+ tokens
5. Sink protection: NIAH does not collapse without sink+window

### Phase 3: Fused Compressed Attention — GPU (P3, ~5 days, riir-ai)

GPU kernel that computes attention scores directly on int4 PCA coefficients:

```
Q·K_score = Σ_i [(a_i·c_i + b_i·d_i)·cos(θ_i·Δ) + (b_i·c_i - a_i·d_i)·sin(θ_i·Δ)]
```

Where a,b are no-RoPE Q halves, c,d are reconstructed from int4 PCA coefficients.

**Location:** `riir-ai/crates/riir-gpu/src/shard_kv/` (if approved)
**Feature gate:** `shard_fused` (riir-ai, opt-in, **secret** — competitive advantage)

This phase stays in riir-ai because:
- GPU kernel implementation is private IP
- Fused attention gives real-time game AI inference a memory-bandwidth advantage
- Per the GOAT pillars matrix, game-specific GPU optimizations are secondary moat

**Note:** Only proceed if Phase 2 proves ≥8× compression on our model configs. GPU kernel effort is wasted if CPU compression doesn't validate first.

---

## Task Breakdown

### - [x] Phase 1: RoPE-Removal Enhancement (P1)

| Task | Description | Est. | Depends |
|------|-------------|------|---------|
| T1 | Add `undo_rope()` to `spectralquant/spectral.rs` | 2h | — |
| T2 | Extend `SpectralQuantCalibration` to store RoPE parameters | 1h | — |
| T3 | Modify `spectral_rotation.rs` to operate on no-RoPE basis | 2h | T1, T2 |
| T4 | Update `forward.rs` to reapply RoPE after dequant (or use Δ-identity) | 3h | T3 |
| T5 | GOAT benchmark: d_eff improvement, cosine sim, compression ratio | 2h | T4 |
| T6 | Validate: no quality regression on existing SQ benchmarks | 2h | T5 |

**Total: ~12 hours (2 days)**

### - [x] Phase 2: Full Asymmetric Codec (P2)

| Task | Description | Est. | Depends |
|------|-------------|------|---------|
| T7 | Create `shard_kv/` module structure + types | 2h | — |
| T8 | Port `undo_rope`/`reapply_rope` to Rust | 2h | T7 |
| T9 | Implement DP bit allocation with 4× drop penalty | 3h | T7 |
| T10 | Port k-means VQ for V path (groups of 4, 256 codebook) | 4h | T7 |
| T11 | Implement `ShardKVCache` struct (store, dequant, sink/window) | 6h | T8-T10 |
| T12 | Implement `forward.rs` (dequant K, dequant V, attention) | 4h | T11 |
| T13 | Implement 8-bit Lloyd-Max decode streaming | 2h | T11 (reuse TQ) |
| T14 | GOAT benchmark: compression, cosine sim, attention correlation | 3h | T12, T13 |
| T15 | Sink/window ablation: measure quality with/without protection | 2h | T14 |
| T16 | Cross-method benchmark: Shard vs SQ vs OCT vs TQ vs HybridOctPQ | 3h | T14 |

**Total: ~31 hours (4 days)**

### - [x] Phase 3: Fused GPU Attention (P3, riir-ai)

| Task | Description | Est. | Depends |
|------|-------------|------|---------|
| T17 | Derive and verify per-pair Δ RoPE identity for our model dimensions | 3h | Phase 2 |
| T18 | WGSL kernel: int4 coefficient unpack + rank-r inner product | 4h | T17 |
| T19 | WGSL kernel: Hadamard past weighted sum for V | 2h | T17 |
| T20 | Integrate into riir-gpu attention pipeline | 4h | T18, T19 |
| T21 | GPU benchmark: memory bandwidth, decode throughput vs FP16 | 3h | T20 |

**Total: ~16 hours (2-3 days)**

---

## Feature Gate

```toml
# katgpt-rs/Cargo.toml
[features]
shard_kv = ["spectral_quant", "turboquant"]  # Asymmetric codec (PCA-K + VQ-V)

# riir-ai/crates/riir-gpu/Cargo.toml (future, Phase 3)
[features]
shard_fused = []  # Fused compressed attention GPU kernel (secret)
```

**Why `shard_kv` is opt-in (not default):**
1. Decode throughput is 0.4–0.5× FP16 — unacceptable for latency-sensitive paths
2. K-means VQ adds complexity over OCTOPUS's data-oblivious approach
3. PCA during prefill adds latency for short prompts
4. Best suited for memory-bound workloads (long context, batch inference)

**Why `shard_fused` is secret (riir-ai only):**
1. GPU fused attention on compressed K is a competitive advantage for real-time game AI
2. The wgpu kernel implementation is private IP
3. Per GOAT pillars: game-specific GPU optimizations support secondary moat
4. Public API only exposes compression results, not the kernel internals

---

## GOAT Proof Specification

### Phase 1 GOAT (RoPE-Removal Enhancement)

| ID | Proof | Threshold | Method |
|----|-------|-----------|--------|
| G1 | d_eff reduction | d_eff(no-RoPE) < d_eff(raw) × 0.7 | Eigenvalue analysis |
| G2 | Cosine improvement | cos(no-RoPE) ≥ cos(raw) + 0.02 at 3-bit | Random K reconstruction |
| G3 | Compression improvement | ratio(no-RoPE) ≥ ratio(raw) × 1.3 | Same quality target |
| G4 | No regression | All existing SQ GOAT proofs still pass | Existing benchmarks |

### Phase 2 GOAT (Full Asymmetric Codec)

| ID | Proof | Threshold | Method |
|----|-------|-----------|--------|
| G5 | Compression | ≥ 8× at d=128, 512 tokens | Byte counting |
| G6 | K cosine sim | ≥ 0.995 at recommended config | Dequant vs original |
| G7 | V cosine sim | ≥ 0.98 at recommended config | Dequant vs original |
| G8 | Attention correlation | ≥ 0.998 vs FP16 reference | Score correlation |
| G9 | 8-bit streaming | 100% match on 150 tokens | Bit-exact comparison |
| G10 | Sink ablation | NIAH-equivalent collapses without sinks | Ablation test |
| G11 | Cross-method | Shard ≥ best competitor at same quality | Head-to-head |

---

## Honest Assessment

### What Shard Gets Right
1. The RoPE-removal insight is genuinely novel and high-value
2. Asymmetric codec design (PCA-K + VQ-V) is the right architecture
3. Attention sink protection is critical and easy to implement
4. 8-bit lossless decode streaming is a clean result

### What We Should Be Cautious About
1. **Decode throughput (0.4× FP16)** — Shard trades speed for memory. Our stack is faster.
2. **K-means VQ complexity** — adds data-dependent training during prefill. Our OCTOPUS is data-oblivious.
3. **PCA rank tuning** — rank=192 is Llama-specific. Our model may need different rank.
4. **Large model focus** — Shard targets Llama-3.1-8B (32 layers, 8 KV heads, d=128). Our micro config (head_dim=4) may not benefit from PCA at all.
5. **Throughput vs capacity trade-off** — For batch inference (our main use case), capacity wins. For latency-sensitive game AI, our existing stack may be better.

### What We Should Not Do
1. Don't replace Hybrid OCT+PQ as default — it's faster and data-oblivious
2. Don't implement k-means VQ for V if OCTOPUS already works well — complexity without clear win
3. Don't build GPU fused attention until CPU path validates compression claims
4. Don't add Shard to `full` feature set — it's niche (long context, memory-bound)

---

## Dependencies

```
Phase 1:
  spectralquant/ (existing)
    spectral.rs ← add undo_rope()
    spectral_rotation.rs ← no-RoPE basis
    forward.rs ← reapply RoPE

Phase 2:
  + spectralquant/ (from Phase 1)
  + turboquant/codebook.rs (Lloyd-Max reuse)
  + turboquant/rotation.rs (Hadamard reuse)
  + NEW: shard_kv/ module

Phase 3 (riir-ai):
  + riir-gpu/ attention pipeline
  + Phase 2 validated on CPU
```

---

## References

- Research 109: Shard — Drop-In 10× KV Cache Compression
- Shard blog: https://krishgarg.com/shard
- Shard code: `.raw/shard/`
- MMO GOAT Pillars Decision Matrix: `riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md`
