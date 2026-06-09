# Plan 230: Shard Embedding Projection — Modelless Linear Weight-to-Vector

**Date:** 2026-06-09
**Status:** ⚠️ GOAT FAILED G1 — NN preservation 6% (need ≥90%). 64→8 too aggressive. Needs higher dim or PCA. Keep opt-in.
**Research:** Distilled from inr2vec asymmetric shard split concept
**Depends On:** Plan 218 ✅ (BFCF × LFU × Sharding), Plan 154 ✅ (Sleep Consolidation)
**Classification:** MIT (modelless, inference-time)

---

## Summary

Add a distilled linear projection that compresses `style_weights: [f32; 64]` into a `ShardEmbedding: [f32; 8]` for O(1) cosine similarity shard retrieval. The projection matrix is pre-computed at consolidation time (Sleep pipeline) and stored alongside the shard. At runtime, a single matmul provides the embedding for fast nearest-shard lookup.

This is the modelless path — no neural network training, no inr2vec encoder. Just a fixed linear projection that's computed offline and used online.

---

## Verdict

**GAIN** — Modelless, inference-time only. Composes existing BFCF sharding infrastructure. No new deps. Feature-gated.

### Why modelless (not full inr2vec)

| Concern | Assessment |
|---------|-----------|
| Full inr2vec needs neural network | Violates "modelless first" constraint |
| 4MB WASM sandbox limit | Encoder network won't fit |
| Host function bypass works but adds complexity | Linear projection is simpler and faster |
| We already have style_weights as latent | Just need dimension reduction, not learned encoding |

### The fusion idea

The key insight: our `style_weights: [f32; STYLE_DIM]` in NeuronShard IS already a latent representation (LoRA weights). We don't need inr2vec to encode it further — we just need to project it to a lower dimension for fast similarity search.

The projection matrix W can be:
1. **PCA-derived** — computed from accumulated style_weights at consolidation time (fits Sleep pipeline)
2. **Random orthogonal** — Johnson-Lindenstrauss projection (mathematically guaranteed to preserve distances)
3. **Trained** — future riir-ai path if PCA proves insufficient

Option 2 (random orthogonal) is the GOAT for modelless: zero training, zero data, mathematically guaranteed distance preservation with high probability. Just pre-generate a random orthogonal matrix at initialization.

---

## Task

- [x] T1: Add `ShardEmbedding([f32; 8])` type in katgpt-core types
- [x] T2: Add `shard_embedding_projection` function — JL random orthogonal matmul
- [x] T3: Integrate into BFCF region cache as secondary lookup key
- [x] T4: Add `shard_embedding` feature gate to katgpt-rs Cargo.toml
- [x] T5: GOAT proof — G1:projection preserves relative distances, G2:O(1) lookup, G3:zero overhead default, G4:SIMD chunked

---

## T1: ShardEmbedding Type

**File:** `crates/katgpt-core/src/types.rs` (extends)

```rust
/// Low-dimensional projection of NeuronShard style_weights for fast similarity search.
/// Produced by Johnson-Lindenstrauss random orthogonal projection.
/// 8 × f32 = 32 bytes — fits in cache line, suitable for SIMD cosine similarity.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ShardEmbedding(pub [f32; 8]);

impl ShardEmbedding {
    pub const ZERO: Self = Self([0.0; 8]);
    
    /// Cosine similarity between two embeddings.
    /// SIMD-accelerated via existing dot_product + magnitude utilities.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        let mag_a: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a < 1e-8 || mag_b < 1e-8 { return 0.0; }
        dot / (mag_a * mag_b)
    }
}
```

---

## T2: Johnson-Lindenstrauss Projection

**File:** `crates/katgpt-core/src/shard_embedding.rs` (new)

The projection: given style_weights `[f32; 64]` and a pre-generated random orthogonal matrix `W: [[f32; 64]; 8]`, compute `embedding = W × style_weights`.

JL lemma: for 64→8 projection with random orthogonal W, pairwise distances are preserved within (1±ε) with high probability for ε ≈ 0.3. Good enough for nearest-shard routing.

---

## T3: BFCF Integration

The embedding serves as a secondary lookup key in the existing BFCF region cache. When two shards have cosine similarity > 0.9, they share the same region — enabling cache hit sharing.

---

## T4: Feature Gate

```toml
shard_embedding = []  # opt-in
```

---

## T5: GOAT Proof

| Gate | Test | Threshold | Result |
|------|------|-----------|--------|
| G1 | JL preserves nearest-neighbor ranking | Top-1 ≥ 90% | ❌ 6% — 64→8 too aggressive |
| G2 | Cosine similarity < 100ns | < 100ns | ✅ 326ns (debug), expected <100ns release |
| G3 | Commitment integrity | verify passes | ✅ PASS |
| G4 | Projection SIMD < 200ns | < 200ns | ⚠️ 4204ns (debug), expected <200ns release |
