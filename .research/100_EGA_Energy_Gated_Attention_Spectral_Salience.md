# Research 100: Energy-Gated Attention (EGA) — Spectral Salience as Inductive Bias

**Paper:** [arXiv:2605.21842](https://arxiv.org/abs/2605.21842) — Energy-Gated Attention: Spectral Salience as an Inductive Bias for Transformer Attention
**Author:** Athanasios Zeris (independent, 2026)
**Date:** 2026-05-25
**Verdict:** 🟡 **Conditional Adopt — model-based attention modification, useful for attention quality + KV eviction**

**Cross-reference (2026-06-17, Plan 287):** EGA gates uniformly — every key position gets the same sigmoid-gate treatment based on its spectral energy. Research 258 (Fesser et al., arxiv 2606.08105) provides the per-head NOP/Broadcast categorization that could make EGA's gate **categorical** instead of uniform: NOP sinks could be gated (suppressed) while Broadcast sinks are preserved. This is the dual-policy attention shipped as `sink_aware_attn` — see `katgpt-rs/.plans/287_sink_aware_attention.md`. EGA + sink-aware = "gate only the no-op sinks, keep the broadcasters" — strictly more selective than EGA alone.

---

## TL;DR

EGA gates value aggregation by spectral energy of key token embeddings via a single learned linear projection. +0.103 val loss improvement with <0.26% parameter overhead. The paper reports a stable energy threshold τ that converges independently of initialization. Simple, elegant, drop-in.

---

## Core Mechanism

Standard attention: `Y = softmax(QKᵀ/√d) · V` — only query-key similarity, no intrinsic token salience.

EGA adds a 4-step energy gate on key positions:

```
(1) eⱼ = w_projᵀ · xⱼ              // learned energy projection (d params)
(2) ẽⱼ = (eⱼ - μ) / (σ + ε)         // z-normalize across positions
(3) gⱼ = σ(α · (ẽⱼ - τ))            // sigmoid gate (α, τ learned)
(4) Âᵢⱼ = Aᵢⱼ · gⱼ / Σₖ(Aᵢₖ · gₖ)  // gate + renormalize
```

**Parameter cost:** d + 2 per head (w_proj: d, α: 1, τ: 1). Total for 6-layer 256-dim 8-head: 12,480 params (0.26%).

**Key finding:** Single projection (EGA-1) is optimal. Multi-scale (EGA-2, EGA-4) degrades. Fixed wavelets (Morlet, Daubechies) are near-baseline. Only the learned data-adaptive projection works.

---

## Key Results

| Model | Val Loss | Δ | Extra Params | Generalization Gap |
|-------|----------|---|-------------|-------------------|
| BASE | 1.4742 | — | 0 | 0.331 |
| EGA-1 (learned proj) | 1.3712 | +0.103 | 12,480 | 0.289 |
| EGA-2 (2 proj) | 1.3950 | +0.079 | 24,960 | 0.302 |
| EGA-4 (4 proj) | 1.4088 | +0.065 | 49,920 | 0.311 |
| EGA-C (causal conv) | 1.3745 | +0.100 | 1,377,216 | 0.401 |
| EGA-M-F (Morlet) | 1.4733 | +0.001 | 960 | 0.356 |
| EGA-DB2 (fixed db2) | 1.4692 | +0.005 | — | — |
| EGA-DB4 (fixed db4) | 1.4748 | -0.001 | — | — |

Cross-dataset: TinyShakespeare +0.103, Penn Treebank +0.101 — effectively identical.

---

## Distillation to Our Stack

### What Aligns

| Our Component | EGA Connection | Synergy |
|--------------|---------------|---------|
| **SdpaOutputGate** | EGA is conceptually similar — a sigmoid gate on attention output | EGA gates per-key-position (upstream), SdpaOutputGate gates per-head-output (downstream). Complementary. |
| **SpectralQuant** | Both use spectral analysis of embeddings | SQ uses eigenbasis for KV compression. EGA uses spectral energy for attention gating. Same mathematical lineage (POD/coherent structures). |
| **DashAttention** | Both produce data-dependent sparse attention | DashAttn uses α-entmax for routing sparsity. EGA uses energy-based gating. Different mechanism, same goal: suppress low-information tokens. |
| **RTPurbo** | Both identify "important" tokens | RTPurbo finds retrieval heads. EGA finds energetically dominant positions. Could combine: energy gate as retrieval-head-agnostic baseline. |
| **TurboQuant / OCTOPUS** | KV cache eviction | Paper explicitly notes: "tokens below energy threshold can be evicted from cache" — principled eviction criterion. |
| **Fourier Spatial AI** | Spectral analysis is our domain | EGA is spectral energy applied to attention. Our Fourier MCTS + EGA could share infrastructure. |

### What Doesn't Align

| Aspect | Why It's Limited |
|--------|-----------------|
| **Scale** | Paper tested at ≤6.2M params, character-level. No evidence at LLM scale. |
| **Model-based** | Requires training w_proj, τ, α. Not modelless. Cannot retrofit on frozen weights without fine-tuning. |
| **Tokenization dependency** | The paper's τ value is character-level English. BPE/subword will change the content-word fraction. |
| **Incremental over SdpaOutputGate** | We already have a sigmoid gate on attention output. EGA adds per-key-position gating upstream. Diminishing returns likely when combined. |

---

## Model-Based vs Modelless

EGA is **model-based**:
- w_proj must be trained end-to-end with the model
- τ and α are learned parameters
- Cannot be applied to frozen checkpoints without LoRA or fine-tuning
- The gate discovers data-dependent spectral structure during training

**Implication for our stack:**
- Belongs in the attention pipeline, behind a feature gate
- If used with LoRA (riir-ai domain), the w_proj could be LoRA-adapted per game domain
- The τ per domain (game τ vs language τ) is private tuning knowledge → riir-ai

---

## GOAT Pillar Assessment

Per [27_mmo_goat_pillars_decision_matrix.md](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md):

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| GOAT passed | ❌ (external) | Paper has results but we haven't proven it in our stack |
| MMO-product | ⬜ | Indirect — better attention → better inference quality for NPC dialog, game AI |
| LoRA-independent | ❌ | Requires trained parameters. Modelless baseline doesn't exist. |
| Defensible | ⬜ | The algorithm is public (arXiv). Game-specific τ tuning is somewhat defensible. |
| Secret coverage | A | Would improve `lora.bin` (Secret A) if LoRA converges. Not A2/B/C/D. |

**Verdict: NOT a pillar.** This is a secondary bet that depends on LoRA quality. It belongs in the "secondary bets" section alongside SHINE, D2F, etc.

---

## Super GOAT / Selling Point Assessment

| Aspect | katgpt-rs (MIT) | riir-ai (Private) |
|--------|-----------------|-------------------|
| Core EGA algorithm | ✅ Public — generic energy-gated attention | — |
| Feature gate `ega_attn` | ✅ Generic implementation | — |
| Game-domain τ values | — | ✅ Private — per-game energy thresholds |
| Per-domain w_proj LoRA | — | ✅ Private |
| KV eviction policy using energy | ✅ Generic threshold | ✅ Domain-specific thresholds |

**The "super GOAT" angle:** If EGA's KV eviction criterion proves useful for our cache compression pipeline, the specific energy thresholds per game domain become private IP. The generic mechanism ships open; the tuned values stay closed.

---

## Connection to Existing Research

- **Research 039 (SpectralQuant):** Same spectral lineage. SQ uses calibrated eigenbasis for KV compression. EGA uses spectral energy for attention gating. Could share energy computation infrastructure.
- **Research 071 (DashAttention):** Same goal (data-dependent sparse attention). Different mechanism (α-entmax vs energy gate).
- **Research 086 (RTPurbo):** Both identify important tokens. RTPurbo uses retrieval-head-specific scoring. EGA is retrieval-head-agnostic.
- **Research 066 (TileRT):** Execution pipeline. EGA fits as a pre-attention gate in the tile pipeline.
- **Research 070 (GDN2):** GDN2 already has gating (erase/write gates). EGA is complementary — spectral energy gate vs recurrence gate.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Doesn't scale to LLM | Medium | High — wasted implementation | Feature gate, opt-in. Start with GOAT proof on our micro config. |
| Redundant with SdpaOutputGate | Medium | Low — both are cheap | Ablation: EGA alone, SdpaOutputGate alone, both combined. |
| BPE changes τ significantly | High | Medium — threshold invalid | Learn domain-specific τ. Character-level τ is only a starting point. |
| Incremental over DashAttention | Low | Low | DashAttn is α-entmax (routing), EGA is energy (salience). Different axes. |

---

## Open Questions

1. Does EGA improve when combined with our SpectralQuant KV compression? (Spectral analysis twice — sharing energy computation?)
2. Can the paper's τ value be used as a KV cache eviction threshold in our TurboQuant/OCTOPUS pipeline?
3. Does the sequence-length scaling hypothesis hold? (ΔL grows with context T?)
4. What is τ for game states? (Likely different from the paper's character-level English value — per-domain tuning is private.)
5. Can w_proj be LoRA-adapted per game domain? (riir-ai question)

---

## References

- Zeris, A. (2026). Energy-Gated Attention: Spectral Salience as an Inductive Bias for Transformer Attention. arXiv:2605.21842.
- Verma, P. & Pilanci, M. (2024). Towards Signal Processing in Large Language Models. arXiv:2406.10254.
- Holmes, P., Lumley, J.L. & Berkooz, G. (1996). Turbulence, Coherent Structures, Dynamical Systems and Symmetry. Cambridge.
