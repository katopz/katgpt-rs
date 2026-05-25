# Plan 139: Energy-Gated Attention (EGA) — Spectral Salience

> **Research:** [100 — EGA Energy-Gated Attention](../.research/100_EGA_Energy_Gated_Attention_Spectral_Salience.md)
> **Paper:** [arXiv:2605.21842](https://arxiv.org/abs/2605.21842) — Spectral salience as inductive bias for transformer attention
> **Feature Gate:** `ega_attn` (opt-in, NOT default-on)
> **Status:** ✅ Phase 1 complete (core implementation)
> **GOAT Pillar:** ❌ Not a pillar — secondary bet, model-based, depends on LoRA quality. See [MMO GOAT Pillars Decision Matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md).
> **Domain:** `katgpt-rs` — generic energy-gated attention. Game-specific τ tuning and per-domain w_proj LoRA stay in `riir-ai`.

---

## Summary

Add Energy-Gated Attention (EGA) as a drop-in attention modifier that gates value aggregation by the spectral energy of key token embeddings. A single learned linear projection + z-normalization + sigmoid gate + renormalization. Only d+2 parameters per head.

---

## Why

1. **Attention quality:** +0.103 val loss improvement at <0.26% parameter cost (paper result, small scale)
2. **KV eviction criterion:** Tokens below energy threshold τ are suppressible — principled cache compression
3. **Spectral synergy:** Aligns with our SpectralQuant eigenbasis compression — same POD/coherent structures lineage
4. **Complementary to DashAttention:** Energy-based salience (EGA) + α-entmax routing (DashAttn) = orthogonal sparsity axes
5. **Complementary to SdpaOutputGate:** Per-key-position upstream gating + per-head-output downstream gating

---

## Architecture

### EGA Gate Parameters

```rust
/// Energy-Gated Attention parameters per attention head.
/// Adds d + 2 parameters per head: w_proj (d), alpha (1), tau (1).
pub struct EgaGate {
    /// Learned energy projection vector [head_dim].
    /// Discovers the dominant spectral mode of the embedding field.
    pub w_proj: Vec<f32>,
    /// Gate sharpness parameter. Paper converges to α ≈ 2.2.
    pub alpha: f32,
    /// Energy threshold. Paper converges to τ ≈ 0.35 (character-level English).
    /// BPE/subword and game domains will differ.
    pub tau: f32,
}
```

### Forward Pass (Algorithm 1 from paper)

```
Q, K, V ← XW_Q, XW_K, XW_V
S ← QKᵀ/√d + causal_mask
A ← softmax(S)

e ← X · w_proj                    // [seq_len] energy scores
ẽ ← (e - μ) / (σ + ε)             // z-normalize
g ← σ(α · (ẽ - τ))                // sigmoid gate [seq_len]

Âᵢⱼ ← Aᵢⱼ · gⱼ                   // gate each key position
Âᵢⱼ ← Âᵢⱼ / Σₖ(Âᵢₖ + ε)          // renormalize (sum-to-one)
Y ← Â · V                         // value aggregation
```

### Integration Points

| Location | Change | Complexity |
|----------|--------|-----------|
| `crates/katgpt-core/src/types.rs` | Add `EgaGate` struct + `ega_gates` field to Config | Low |
| `crates/katgpt-core/src/attention.rs` | Add `ega_attention_forward()` with energy gate | Medium |
| `src/transformer.rs` | Wire EGA gate into attention layers | Medium |
| `src/weights.rs` | Serialize/deserialize EGA gate params | Low |

---

## Feature Gate Design

```toml
# Cargo.toml
ega_attn = []  # Energy-Gated Attention — spectral salience gate (Plan 139, Research 100)
```

**NOT default-on.** Rationale:
- Model-based (requires training w_proj, τ, α)
- Paper only tested at ≤6.2M params
- Unproven at our scale and with BPE tokenization
- Must pass GOAT proof before considering default-on

---

## GOAT Proof Design

### G1: Energy Gate Quality (val loss ablation)

Train micro config (L=6, H=8, d=256) with and without EGA on character-level data:
- BASE vs EGA-1: expect ≥ +0.05 val loss improvement (paper shows +0.103)
- Measure generalization gap: EGA should be ≤ BASE gap

### G2: Parameter Overhead

- Count: `n_heads × (head_dim + 2)` extra parameters
- Must be < 1% of total model parameters

### G3: Attention Quality (cosine similarity)

- Forward pass with and without EGA gate
- Cosine similarity of output should be reasonable (> 0.8, not a radical change)
- Energy gate should suppress known low-information tokens

### G4: KV Eviction Feasibility

- Compute energy scores for all positions in a sequence
- Fraction above τ should be ~30-40% (content word fraction)
- Tokens below threshold: measure impact of eviction on attention output quality

### G5: Compute Overhead

- Benchmark: attention forward with and without EGA
- Single linear projection per key position + z-norm + sigmoid
- Expect < 5% overhead on attention compute

---

## Task Breakdown

### Phase 1: Core Implementation (katgpt-rs)
- [x] T1: Add `EgaGate` struct to `src/ega_attn.rs` behind `ega_attn` feature
- [x] T2: Add `gate_attention()` + energy computation to `src/ega_attn.rs`
- [x] T3: Wire `ega_attn` feature gate in `Cargo.toml` + `lib.rs`
- [x] T4: Add energy score computation + z-normalize + sigmoid gate utilities

### Phase 2: GOAT Proof
- [ ] T5: `ega_01_quality` example — val loss ablation on micro config
- [ ] T6: `ega_02_energy_profile` example — visualize energy distribution over sequence
- [ ] T7: `ega_03_eviction` example — KV eviction experiment using energy threshold
- [ ] T8: `ega_04_combined` example — EGA + DashAttn + SdpaOutputGate combined ablation

### Phase 3: riir-ai Integration (if GOAT passes)
- [ ] T9: Game-domain τ discovery — run energy profiling on Bomber/Go/FFT game states
- [ ] T10: Per-domain EGA LoRA adapter — `game_ega_lora.bin` as Secret A extension
- [ ] T11: EGA-aware KV cache eviction in inference pipeline

---

## Relationship to Other Plans

| Plan | Relationship |
|------|-------------|
| Plan 077 (SpectralQuant) | Same spectral lineage. EGA energy scores could feed into SQ compression decisions. |
| Plan 106 (DashAttention) | Orthogonal sparsity: α-entmax routing (DashAttn) + energy salience (EGA). |
| Plan 126 (RTPurbo) | RTPurbo finds retrieval heads; EGA finds energy-dominant positions. Different signals. |
| Plan 105 (GDN2) | GDN2 has recurrence gates; EGA has spectral gates. Complementary. |
| Plan 102 (TileRT) | EGA gate can be applied per-tile in the execution pipeline. |
| Plan 085 (Deep Manifold) | Both use fixed-point analysis. EGA's energy threshold is a fixed-point attractor. |

---

## Open/Closed Boundary

```
katgpt-rs (MIT)                      riir-ai (Private)
───────────────────────              ────────────────────────
EgaGate struct (generic)       →     Game-specific τ values
ega_attention_forward()        →     Per-domain w_proj LoRA
Energy score computation       →     Game energy profile data
Feature gate: ega_attn         →     game_ega_lora.bin (Secret A)
```

**Rule:** Generic EGA mechanism ships open. Game-specific tuning (τ, w_proj LoRA, energy profiles) stays private in riir-ai. Same pattern as WASM validators (generic trait in katgpt-rs, game implementations in riir-ai).

---

## Honest Assessment

**This is a secondary bet.** If LoRA converges, EGA improves attention quality and provides principled KV eviction. If LoRA doesn't converge, EGA's trained parameters (w_proj) don't help — you can't use EGA without training it.

The most valuable distillation might not be the attention gate itself, but the **energy score as a KV eviction signal** — even with random w_proj, the z-normalized energy might correlate with token importance. That's the modelless angle worth exploring first.

**Priority: Low.** Behind Issue 013 (riir-games), Issue 014 (Cold Tier), Issue 015 (MMO Backbone). Only pursue if bandwidth allows after pillar integration work.
