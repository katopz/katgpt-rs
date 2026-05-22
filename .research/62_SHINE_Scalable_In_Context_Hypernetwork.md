# Research 62: SHINE — Scalable In-Context Hypernetwork for Mapping Context to LoRA

> **Paper:** [SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass](https://arxiv.org/pdf/2602.06358) — Liu, Wang, Mao, Gelberg, Maron, Zhang (PKU/Oxford/Technion/NVIDIA), ICML 2026
> **Code:** https://github.com/MuLabPKU/SHINE
> **Date:** 2026-05, distilled 2026-05
> **Related Research:** 04 (LoRA Architecture), 19 (TTT), 37 (REAP Model-Based/Modelless), 38 (SDAR), 54 (ASFT), 55 (TriMode), 58 (GRAM), 59 (MoE Speculative), 60 (MeMo)
> **Related Plans:** 025 (Model vs Modelless Bandit), 050 (Feature Gate Audit), 092 (Freeze/Thaw), 094 (MeMo Reflections + TIES), 097 (Delta Routing)
> **Verdict: TECHNIQUE EXTRACTION — Two distillable components: (1) Meta LoRA context-to-adapter generation for instant expert creation, (2) Alternating sparse M2P attention pattern for parameter-space processing. Fits model-based path of our spectrum. Feature gate `shine_hypernet` on riir-ai side.**

---

## TL;DR

SHINE is a hypernetwork that generates LoRA adapters from arbitrary context in a **single forward pass** — no gradient-based optimization. It reuses the frozen LLM's own parameters via "Meta LoRA" + learnable memory embeddings, then a lightweight M2P (Memory-to-Parameter) Transformer with alternating row/column sparse attention produces full LoRA weights for all layers. Training: pretrain on reconstruction+completion → instruction fine-tune on QA.

Key results:
- F1=55.6 on MS MARCO MQA (vs In-Context 69.4, SFT 33.0) with 0.3s amortized time (vs SFT 29.3s)
- SQuAD F1=63.6 vs TTT methods ≤59.4 (SEAL, PaST) — **single forward pass** beats iterative optimization
- No capacity saturation in scaling experiments (8B backbone, 6B pretraining tokens)
- SHINE-R recurrent variant handles 18K+ token contexts with linear LoRA growth

---

## Paper Architecture

### Three-Stage Pipeline

```
Stage 1: Memory Extraction
  Context tokens X + Learnable memory embeddings M₀
    → [X; M₀] fed to frozen LLM with Meta LoRA
    → Extract memory states M_i from each layer's last M tokens
    → Stack into M ∈ R^{L×M×H}

Stage 2: LoRA Generation (M2P Transformer)
  M + Positional encoding (layer + token)
    → Alternating column/row bidirectional attention (sparse)
    → 2-layer MLP per token
    → Reshape memory states into LoRA A,B matrices per layer

Stage 3: Downstream Inference
  Question → LLM + Generated LoRA → Answer
  (Context NOT in prompt — knowledge is in LoRA weights)
```

### Key Equations

**Memory length vs LoRA size:**
```
M = ⌈rD/H⌉  where r=LoRA rank, D=sum of input+output dims per layer, H=hidden dim
```
Ensures memory capacity ≥ LoRA parameter count.

**Alternating sparse attention (M2P Transformer):**
```
Odd layers:  Y_{:,j} = SelfAttn(Z_{:,j})  for j=1..M    [column: mix across layers]
Even layers: Y_{j,:} = SelfAttn(Z_{j,:})  for j=1..L    [row: mix across tokens]
```
Complexity: O(LM² + ML²) vs O((LM)²) for full attention. Saves ~90% FLOPs.

**LoRA weight generation:**
```
v = flatten(M̂[i,:,:])  for layer i
A = Reshape(v[t : t + I·r])   ∈ R^{I×r}
B = Reshape(v[t + I·r : t + I·r + r·O]) ∈ R^{r×O}
```

**Training objectives:**
```
Pretrain:  J = λ·J_RECON + (1-λ)·J_COMP   (λ=0.5)
  RECON: context → LoRA → reconstruct original
  COMP:  truncated context → LoRA → complete missing 10-30%

IFT:      J = -log P(a|q; Θ_GLoRA, Θ_LLM)
  context → LoRA → answer question without seeing context
```

### SHINE-R: Recurrent Long Context

```
Chunk 1 → Meta LoRA₁ → Gen LoRA₁
Chunk 2 → Meta LoRA₁+Meta LoRA₂(M2P) → Gen LoRA₂
Chunk 3 → Meta LoRA₁+Meta LoRA₃(M2P) → Gen LoRA₃
...
Final LoRA = concat(Gen LoRA₁, Gen LoRA₂, ..., Gen LoRA_K)
```
Each chunk: M2P generates two LoRAs — one updates Meta LoRA (recurrent state), one is saved. Handles infinite contexts with linear LoRA growth.

### Efficiency Comparison

| Method | Amortized Time | Generation Time | Peak Memory |
|--------|---------------|----------------|-------------|
| Naive | 0.0s | 11.0s | Baseline |
| In-Context | 0.0s | 14.2s | +C tokens in KV |
| SFT (10 epochs) | 29.3s | 11.0s | Training overhead |
| **SHINE** | **0.3s** | **11.0s** | +Meta LoRA params |

SHINE generation is identical to Naive/SFT — no context tokens in KV cache during inference.

---

## Mapping to Our Stack

### Conceptual Alignment: SHINE on the Model-Based/Modelless Spectrum

SHINE sits firmly on the **model-based** side of our REAP spectrum (Research 37):

| Spectrum Position | Our Component | SHINE Analog |
|------------------|---------------|--------------|
| **Modelless** (zero inference) | `BanditPruner` Q-values, `ConstraintPruner` rules | N/A |
| **Light model-based** (one pass) | `ScreeningPruner::relevance()`, DDTree | Memory Extraction (one LLM forward) |
| **Full model-based** (training) | G-Zero Phase 2 (GRPO/DPO), riir-gpu LoRA training | **SHINE full pipeline** (Meta LoRA → M2P → Gen LoRA) |
| **Hyper-model-based** (meta-training) | — | **SHINE pretraining** (train the hypernetwork itself) |

SHINE is **two levels deeper** than our current model-based path:
1. We train LoRA adapters (riir-gpu) — SHINE trains the thing that generates LoRA adapters
2. Our model-based path: data → LoRA training → adapter deployment
3. SHINE's path: data → hypernetwork training → context → instant adapter

### What We Already Have (Partial Overlap)

| SHINE Concept | Our Equivalent | Gap |
|---------------|----------------|-----|
| Meta LoRA (frozen LLM + learnable LoRA for memory extraction) | LoRA training infra (riir-gpu, 26 WGSL shaders) | Our LoRA is for downstream tasks, not for context encoding |
| Memory embeddings (learnable tokens appended to input) | DomainLatent (Plan 038, `domain_latent` feature) | DomainLatent is fixed per-domain, SHINE's are per-context |
| M2P Transformer (memory → parameters) | — | **No equivalent** — we don't generate weights from activations |
| Alternating sparse attention | HLA (Plan 057, `hla_attention`), AttentionMode enum | HLA is for inference speedup, not parameter-space processing |
| Reconstruction pretraining | Freeze/Thaw knowledge pipeline (Plan 092) | Freeze/Thaw stores bandit state, not reconstructed knowledge |
| SHINE-R recurrent chunking | — | No equivalent — our long context is KV cache compression |
| Instruction fine-tuning | ASFT (Plan 090, `asft_loss`), SDAR (Plan 073) | Our IFT trains LoRA directly, SHINE trains the hypernetwork |

### What We DON'T Have (Gaps)

#### 1. Context-to-LoRA Generation (Core Gap)

Our Expert Registry (riir-ai Plan 023) routes queries to **pre-trained** LoRA adapters. SHINE generates adapters **on-the-fly** from context. This is a fundamentally different paradigm:

```
Our approach:  query → classify domain → select pre-trained LoRA → infer
SHINE:         query + context → hypernetwork → instant LoRA → infer
```

SHINE eliminates the need for a LoRA library. Context IS the adapter specification.

**Implication:** For domains where context is dynamic (game replays, evolving codebases, session-specific knowledge), SHINE-style generation is superior. For static domains (pre-trained skills like Go, Bomber), our pre-trained LoRA approach is more efficient (zero amortized cost).

#### 2. Bidirectional Layer Communication

Our `AttentionMode::Bidirectional` exists for D2F (Plan 066), but is token-level. SHINE's M2P Transformer does bidirectional attention across **layers** (deep → shallow information flow). This mimics backpropagation — shallow layer LoRA depends on deep layer signals.

**Implication:** The alternating row/column pattern is a reusable sparse attention primitive beyond SHINE.

#### 3. Post-LayerNorm for Parameter Generation

SHINE uses post-layernorm (not pre-layernorm) for the M2P Transformer because parameter generation has huge distribution gaps between layers. Post-layernorm stabilizes this. Our inference uses pre-layernorm (standard for LLMs).

**Implication:** If we ever generate parameters from activations, post-layernorm is the right choice.

---

## Distillations for Our Stack

### D1: Meta LoRA Context-to-Adapter — Model-Based (riir-ai)

**What:** A lightweight hypernetwork that takes context tokens + learnable memory embeddings through a frozen LLM with Meta LoRA, producing per-layer LoRA weights via the M2P pattern.

**Why it fits:** Our Expert Registry currently requires pre-trained LoRA adapters per domain. SHINE-style generation enables:
- **Session-adaptive experts:** Game replay context → instant game-specific LoRA
- **Document-adaptive inference:** Codebase context → instant code-domain LoRA
- **Eliminates LoRA library overhead:** No need to store/train/maintain multiple adapters

**Where it lives:** `riir-ai/crates/riir-gpu/src/hypernet/` (new module, behind `shine_hypernet` feature gate).

**Architecture sketch:**
```rust
/// SHINE-style context-to-LoRA hypernetwork.
#[cfg(feature = "shine_hypernet")]
pub struct ShineHypernet {
    /// Meta LoRA weights applied to frozen LLM during memory extraction.
    meta_lora: MetaLoRAWeights,
    /// Learnable memory embeddings [M × H].
    memory_embeddings: Tensor, // [M, H]
    /// M2P Transformer: alternating column/row attention.
    m2p_layers: Vec<M2PLayer>,
    /// Layer positional encoding [L × 1 × H].
    layer_pos_enc: Tensor,
    /// Token positional encoding [1 × M × H].
    token_pos_enc: Tensor,
}

#[cfg(feature = "shine_hypernet")]
pub enum M2PAttentionMode {
    /// Odd layers: column attention (mix across layers per token position).
    Column,
    /// Even layers: row attention (mix across tokens per layer).
    Row,
}

/// Generate LoRA from context in a single forward pass.
#[cfg(feature = "shine_hypernet")]
pub fn context_to_lora(
    hypernet: &ShineHypernet,
    context_tokens: &[u32],
    frozen_llm: &TransformerWeights,
    config: &Config,
) -> GeneratedLoRA {
    // Stage 1: Memory extraction
    //   [context_tokens; memory_embeddings] → frozen LLM + meta_lora
    //   → extract M_i from each layer → stack into M [L, M, H]
    //
    // Stage 2: M2P Transformer
    //   M + pos_enc → alternating column/row attention → M̂ [L, M, H]
    //
    // Stage 3: Reshape to LoRA
    //   Per layer: flatten M̂[i] → sequentially reshape into A, B matrices
}
```

**GOAT proof:** Run `bomber_14_shine_expert` — generate LoRA from bomber game replay context, compare win rate vs pre-trained bomber LoRA vs no-LoRA baseline. Expect: context-generated LoRA > no-LoRA, but < pre-trained LoRA (since pre-trained has more optimization steps). If context-generated wins > 50% vs baseline, feature is GOAT-proved.

**Scope:** ~500 lines new code. Requires wgpu kernels for M2P attention (column/row alternating). Reuses existing `attention_score.wgsl` with masking changes.

**Priority:** Medium — requires riir-gpu infrastructure. Useful for session-adaptive experts but not critical for current game domain benchmarks.

### D2: Alternating Sparse Attention Pattern — Modelless (microgpt-core)

**What:** The row/column alternating attention decomposition from SHINE's M2P Transformer. A general-purpose sparse attention primitive for processing 2D token grids.

**Why it fits:** We already have `AttentionMode` enum with `Causal`, `Bidirectional`, `BlockCausal`, `SpKv`. Adding `Alternating2D` captures the M2P pattern for any grid-structured processing (not just LoRA generation).

**Where it lives:** `microgpt-core/src/types.rs` (extend `AttentionMode` enum).

**Architecture sketch:**
```rust
// In microgpt-core/src/types.rs
pub enum AttentionMode {
    // ... existing variants ...
    
    /// Alternating row/column attention for 2D grids (SHINE M2P pattern).
    /// Odd layers: column attention (mix rows). Even layers: row attention (mix columns).
    /// Complexity: O(LM² + ML²) vs O((LM)²) full attention.
    /// Used for parameter-space processing, not token-sequence inference.
    Alternating2D {
        /// Grid rows (e.g., layers for SHINE).
        rows: usize,
        /// Grid columns (e.g., memory tokens for SHINE).
        cols: usize,
    },
}
```

**GOAT proof:** Run existing `hla_attention` benchmarks with `Alternating2D` mode vs full bidirectional on same grid dimensions. Verify <10% quality loss with >5× speedup.

**Scope:** ~100 lines in types.rs. Attention dispatch in riir-engine/riir-gpu handles masking.

**Priority:** Low — useful primitive but no immediate consumer beyond D1.

### D3: Context-Informed Expert Selection (Bridge Pattern)

**What:** Instead of full SHINE hypernetwork, use the **memory extraction** stage (Meta LoRA + memory embeddings) to produce a context embedding, then use that embedding to select from our existing Expert Registry.

**Why it fits:** This bridges our pre-trained LoRA library with SHINE's context-awareness:
- Context → frozen LLM + Meta LoRA → memory states → mean pool → context embedding
- Context embedding → cosine similarity with Expert Registry embeddings → select best expert
- No M2P Transformer needed — just the memory extraction stage

**Architecture:**
```rust
/// Lightweight context-to-expert routing using Meta LoRA memory extraction.
#[cfg(feature = "shine_routing")]
pub fn route_context_to_expert(
    context: &[u32],
    meta_lora: &MetaLoRAWeights,
    memory_emb: &[f32],  // [M × H]
    expert_registry: &ExpertRegistry,
    frozen_llm: &TransformerWeights,
) -> (ExpertId, f32) {
    // 1. Extract memory states from context
    let memory_states = extract_memory(context, meta_lora, memory_emb, frozen_llm);
    // 2. Mean pool across layers and tokens → context embedding [H]
    let context_emb = mean_pool(memory_states);
    // 3. Compare with expert embeddings in registry
    expert_registry.find_best_match(&context_emb)
}
```

**GOAT proof:** Run `go_10_move_accuracy` with context-informed routing vs static domain routing. If accuracy improves >2%, feature is GOAT-proved.

**Priority:** Medium-High — simpler than D1, leverages existing Expert Registry infrastructure.

---

## What NOT To Do

1. **Don't build the full SHINE pretraining pipeline (6B tokens).** Our training infrastructure is game-domain focused. We train on thousands of game episodes, not billions of text tokens. The pretraining scale is prohibitive for our use case.

2. **Don't replace our LoRA training with SHINE-style generation for static domains.** Pre-trained LoRA adapters for Go, Bomber, FFT are already GOAT-proved. SHINE generation is for **dynamic** contexts where pre-training isn't possible.

3. **Don't use SHINE-R for long-context inference.** Our KV cache compression (SpectralQuant, TurboQuant) is more mature and handles production inference. SHINE-R is a research prototype.

4. **Don't add SHINE to the default feature set.** This is a model-based research feature. It requires GPU infrastructure (riir-gpu) and significant training data. Keep it opt-in.

5. **Don't implement the coupled cross-attention variant.** The paper's ablation (E.1) shows it's **worse** than vanilla alternating attention — the "Bitter Lesson" applies. General methods beat hand-designed priors.

6. **Don't compete with TTT methods on their own terms.** SHINE beats SEAL/PaST on SQuAD, but our system doesn't do test-time training. Our Freeze/Thaw (Plan 092) and bandit learning are different mechanisms. SHINE validates that single-pass generation > iterative optimization, which supports our modelless-first philosophy.

---

## Experimental Results (From Paper)

### Multi-Turn QA (MS MARCO MQA)

| Method | F1 Score | Amortized Time | Generation Time |
|--------|----------|---------------|-----------------|
| Naive | 23.2 | 0.0s | 11.0s |
| In-Context | 69.4 | 0.0s | 14.2s |
| SFT (LoRA r=8, 10 epochs) | 33.0 | 29.3s | 11.0s |
| **SHINE** | **55.6** | **0.3s** | **11.0s** |

SHINE achieves 83% of In-Context quality with 0.3s overhead (vs 29.3s for SFT). Generation speed matches Naive/SFT.

### Single-Hop QA (SQuAD)

| Method | F1 Score |
|--------|----------|
| Naive | 22.0 |
| In-Context | 86.8 |
| Generative Adapter (prior work) | 70.3 |
| **SHINE** | **63.6** |
| SEAL (TTT, n=200) | 58.2 |
| PaST (TTT, n=200) | 58.9 |

SHINE beats all TTT methods with single forward pass (no iterative optimization).

### Multi-Hop QA

| Method | HotpotQA | MuSiQue | 2WikiMulti |
|--------|----------|---------|------------|
| Naive | 26.9 | 11.8 | 27.8 |
| In-Context | 68.7 | 36.3 | 48.7 |
| **SHINE** | **59.0** | **28.5** | **60.2** |

SHINE captures semantic structure (multi-hop reasoning) without explicit CoT — the LoRA encodes relational knowledge.

### Scaling (No Saturation)

| Backbone | Validation PPL (40% pretrain) |
|----------|-------------------------------|
| Qwen3-0.6B | ~5.5 |
| Qwen3-1.7B | ~4.0 |
| Qwen3-8B | ~2.5 |

Consistent improvement with scale — no capacity bottleneck observed.

### Ablation: Alternating Attention vs Alternatives

| M2P Architecture | Val PPL |
|-----------------|---------|
| Linear small (2-layer, 8K hidden) | ~18 |
| Linear (8-layer, 8K hidden) | ~12 |
| Linear + residual | ~10 |
| Only last memory (no cross-layer) | ~8 |
| Only horizontal (no column) | ~3.5 |
| **Origin (alternating column/row)** | **~2.5** |

Key findings:
- Transformer >> Linear (7× better PPL)
- All-layer memory >> Last-layer only (3× better)
- Alternating ≈ Horizontal but converges faster

---

## Relationship to Existing Research

| Research | Overlap | Delta |
|----------|---------|-------|
| 04 (LoRA Architecture) | LoRA generation target | SHINE generates LoRA without gradient steps |
| 19 (TTT) | Test-time adaptation | SHINE is non-gradient TTT alternative (single pass) |
| 37 (REAP) | Model-based/modelless spectrum | SHINE is fully model-based, adds hyper-model layer |
| 38 (SDAR) | LoRA + auxiliary signals | SDAR gates per token, SHINE generates per context |
| 54 (ASFT) | Anchored SFT training | SHINE's pretrain (reconstruction) ≈ anchor task |
| 55 (TriMode) | Multi-mode inference | SHINE = context-to-weights mode (4th mode?) |
| 58 (GRAM) | Recursive reasoning via LoRA | GRAM does recursive inference, SHINE does instant generation |
| 59 (MoE Speculative) | Expert-level adaptation | SHINE replaces MoE expert routing with context-generated adapter |
| 60 (MeMo) | Reflection QA → knowledge | MeMo trains memory model, SHINE generates instant adapter |
| 61 (SLIME) | Loss formulation | SHINE uses standard CE, SLIME uses implicit margin |

---

## Feature Gate Design

### microgpt-rs (modelless side)

No feature gate needed. The alternating attention pattern (D2) would extend `AttentionMode` in `microgpt-core/src/types.rs` — available to all consumers.

### riir-ai (model-based side)

```toml
# riir-ai/crates/riir-gpu/Cargo.toml
[features]
shine_hypernet = []  # SHINE context-to-LoRA hypernetwork (Research 62)
shine_routing = ["shine_hypernet"]  # Lightweight: context extraction → expert routing only
```

The `shine_hypernet` feature enables:
1. `MetaLoRAWeights` struct + memory extraction forward pass
2. `M2PTransformer` with alternating column/row attention WGSL kernels
3. `context_to_lora()` end-to-end function
4. `ShineHypernetConfig` (meta_lora_rank, gen_lora_rank, m2p_layers, memory_length)

The `shine_routing` feature enables only the memory extraction + embedding comparison (D3).

### GOAT Proof Examples

```toml
# microgpt-rs/Cargo.toml
[[example]]
name = "bomber_14_shine_expert"
required-features = ["bomber"]  # Uses riir-gpu via REST for LoRA generation

[[example]]
name = "go_11_shine_routing"
required-features = ["go", "shine_routing"]
```

---

## Key Takeaways

1. **SHINE is a hyper-model, not a model.** It doesn't train LoRA — it trains the thing that generates LoRA. This is one meta-level above our current training pipeline. The conceptual insight: **context can be compressed into weights, not just into KV cache.**

2. **Single-pass generation beats iterative optimization.** SHINE (0.3s, one forward pass) beats SEAL/PaST (iterative gradient-based TTT, hundreds of articles) on SQuAD. This validates our modelless-first philosophy — efficient computation often beats expensive optimization.

3. **The M2P alternating attention pattern is a reusable primitive.** Row/column alternating attention for 2D grid processing is not SHINE-specific. It's useful for any parameter-space processing task.

4. **Context-as-adapter is complementary to pre-trained-adapter.** For static domains (Go, Bomber), pre-trained LoRA is optimal (zero runtime cost). For dynamic contexts (session-specific knowledge, evolving codebases), SHINE-style generation fills a gap.

5. **Post-layernorm for parameter generation.** Standard LLMs use pre-layernorm for gradient flow. But when generating parameters from activations (huge distribution gaps), post-layernorm is more stable. This is a useful engineering note.

6. **The Bitter Lesson applies to hypernetwork design.** SHINE's ablation shows that coupled cross-attention (using LoRA structural priors) is worse than vanilla alternating attention. General methods that leverage computation win over hand-designed priors.

7. **SHINE-R's recurrent pattern is interesting but immature.** Linear LoRA growth with context length is a clean idea, but the paper shows it still trails In-Context significantly. Our KV cache compression (SpectralQuant) is more production-ready.

---

## References

- Paper: https://arxiv.org/pdf/2602.06358
- Code: https://github.com/MuLabPKU/SHINE
- Related: Generative Adapter (Chen et al., 2025), ICAE (Ge et al., 2024), Text-to-LoRA (Charakorn et al., 2025), Doc-to-LoRA (Charakorn et al., 2026)
- Comparisons: SEAL (Zweiger et al., 2025), PaST (Tang et al., 2026)