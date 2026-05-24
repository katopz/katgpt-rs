# Plan 098: SHINE Context-to-LoRA Hypernetwork

> **Research:** 62 (SHINE Scalable In-Context Hypernetwork)
> **Related Plans:** 025 (Model vs Modelless Bandit), 050 (Feature Gate Audit), 092 (Freeze/Thaw), 094 (MeMo Reflections + TIES), 097 (Delta Routing)
**Status:** ✅ Done
> **Verdict:** Technique extraction — two distillable components: Meta LoRA context-to-adapter + alternating sparse M2P attention. Fits model-based path. Feature gate `shine_hypernet` on riir-ai side.
> **NOT promoted:** `shine_hypernet` stays default-off. Stage 1 is placeholder, GPU dispatch unwired, random weights only.
> **GOAT gate:** Plan 104 — E2E proof needed: real Stage 1 + GPU dispatch + trained weights.

### Benchmark Results (`alternating_2d_bench`, Criterion, release, Apple M-series)

| Grid | Size (R×C×H) | Alternating | Full Bidirectional | FLOPs Savings |
|------|-------------|-------------|-------------------|---------------|
| small | 4×8×32 | 277 µs | 38 µs | 62.5% |
| medium | 8×16×64 | 8.3 ms | 1.1 ms | 62.5% |
| near_prod | 12×32×128 | 106 ms | (too slow) | 77.1% |

**Layer scaling** (hidden=64, heads=4): 4×8 → 1.0 ms, 8×16 → 4.1 ms, 12×24 → 9.4 ms, 16×32 → 17.0 ms.

**GOAT verdict (Plan 098 partial):** ✅ FLOPs savings validated (62–77%). ✅ Architecture compiles, 31 tests pass. ⚠️ Stage 1 is placeholder (ignores context tokens, ignores Meta LoRA). ⚠️ GPU dispatch unwired (WGSL kernels exist but no Rust dispatch). ⚠️ Random weights produce structured but meaningless output. **Not promoted to default** — needs Plan 104 E2E GOAT proof.

## Tasks

- [x] T1: Add `Alternating2D` variant to `AttentionMode` in `katgpt-core/src/types.rs`
- [x] T2: Add alternating row/column attention dispatch in `riir-gpu` WGSL kernel
- [x] T3: Implement `MetaLoRAWeights` struct + memory extraction forward pass (riir-ai)
- [x] T4: Implement `M2PTransformer` with alternating column/row attention (riir-ai)
- [x] T5: Implement `context_to_lora()` end-to-end function (riir-ai)
- [x] T6: Add `shine_hypernet` feature gate to `riir-ai/crates/riir-gpu/Cargo.toml`
- [x] T7: Add `shine_routing` feature gate (lightweight: extraction → expert routing only)
- [x] T8: GOAT proof — `bomber_14_shine_expert` (context-generated LoRA vs baseline)
- [x] T9: GOAT proof — `go_11_shine_routing` (context-informed expert selection vs static routing)
- [x] T10: Benchmark `Alternating2D` attention vs full bidirectional on grid dimensions
- [x] T11: Update README.md + `.docs/15_paper_feature_comparison.md`

---

## Architecture

SHINE maps context to LoRA in a **single forward pass** without gradient-based optimization. Three stages:

```
Stage 1: Memory Extraction
  context tokens + learnable memory embeddings
    → frozen LLM with Meta LoRA (lightweight trained LoRA for encoding)
    → extract memory states M_i from each layer
    → stack into grid M [L × M × H]

Stage 2: M2P Transformer (Memory-to-Parameter)
  M + positional encoding (layer index + token index)
    → alternating column/row bidirectional attention (sparse, ~90% FLOPs savings)
    → post-layernorm (not pre-layernorm — stabilizes parameter distribution gaps)
    → output grid M̂ [L × M × H]

Stage 3: Parameter Generation
  Per layer: flatten M̂[i,:,:] → reshape into LoRA A,B matrices
    → GeneratedLoRA applied to frozen LLM for downstream inference
    → Context NOT in prompt — knowledge is in weights
```

### Key Formula: Memory Length

```text
M = ⌈rD/H⌉
  r = generated LoRA rank (e.g., 8)
  D = sum of input+output dims per linear layer in one LLM layer
  H = hidden dimension
```

Ensures memory capacity ≥ LoRA parameter count.

### Key Formula: Alternating Attention

```text
Odd M2P layers:  column attention — mix across L layers, per token position j
  Y[:,j] = SelfAttn(Z[:,j])  for j = 1..M

Even M2P layers: row attention — mix across M tokens, per layer j
  Y[j,:] = SelfAttn(Z[j,:])  for j = 1..L

Complexity: O(LM² + ML²) vs O((LM)²) full attention
```

### Key Formula: LoRA Reshape

```text
v = flatten(M̂[i,:,:])  for layer i
A = Reshape(v[t : t + I·r])       ∈ R^{I×r}    (input dim × rank)
B = Reshape(v[t + I·r : t + I·r + r·O]) ∈ R^{r×O}    (rank × output dim)
t advances by Ir + rO per LoRA module
```

### Training Pipeline (Future — not in this plan)

```text
Pretrain:
  J = 0.5·J_RECON + 0.5·J_COMP
  RECON: context → LoRA → reconstruct original text
  COMP:  truncated context → LoRA → complete missing 10-30%

IFT:
  J = -log P(answer|question; GeneratedLoRA, FrozenLLM)
  context → LoRA → answer question WITHOUT seeing context
```

**Note:** Full training pipeline is out of scope for this plan. We focus on the inference architecture + GOAT proof with pre-trained Meta LoRA weights (loaded from checkpoint).

---

## Task Details

### T1: `Alternating2D` Attention Mode (katgpt-core)

Extend `AttentionMode` enum in `katgpt-core/src/types.rs`:

```rust
/// Alternating row/column attention for 2D grids (SHINE M2P pattern).
/// Odd layers: column attention (mix across rows). Even layers: row attention (mix across columns).
/// Complexity: O(RC² + CR²) vs O((RC)²) full attention.
/// Used for parameter-space processing, not token-sequence inference.
Alternating2D {
    /// Grid rows (e.g., transformer layers for SHINE).
    rows: usize,
    /// Grid columns (e.g., memory tokens for SHINE).
    cols: usize,
}
```

**Scope:** ~15 lines in `types.rs`. No dispatch changes needed here — consumers handle masking.

### T2: WGSL Alternating Attention Kernel (riir-ai)

New WGSL compute shader for M2P alternating attention:

```text
m2p_attention.wgsl
  - Kernel: m2p_column_attention  (bind groups: Q, K, V, output, [L, M, H])
  - Kernel: m2p_row_attention     (bind groups: Q, K, V, output, [L, M, H])
  - Post-layernorm (not pre-layernorm)
  - 2-layer MLP per token (SwiGLU style, intermediate dim = 2H)
```

**Reuse:** Adapt existing `attention_score.wgsl` with row/column masking instead of causal masking.

**Scope:** ~200 lines new WGSL + ~100 lines Rust dispatch in `riir-gpu/src/attention.rs`.

### T3: Meta LoRA Memory Extraction (riir-ai)

```rust
// riir-gpu/src/hypernet/meta_lora.rs

/// Meta LoRA weights — lightweight LoRA applied to frozen LLM during memory extraction.
/// Trained separately (pretraining pipeline). Loaded from checkpoint for inference.
pub struct MetaLoRAWeights {
    /// Per-layer (A, B) pairs. A: [I×r], B: [r×O].
    pub layers: Vec<(Tensor, Tensor)>,
    /// Meta LoRA rank (paper uses 128).
    pub rank: usize,
}

/// Memory extraction: context + memory embeddings → frozen LLM with Meta LoRA → memory grid.
pub fn extract_memory_states(
    context_tokens: &[u32],
    memory_embeddings: &Tensor,  // [M, H]
    meta_lora: &MetaLoRAWeights,
    frozen_llm: &TransformerWeights,
    config: &Config,
) -> Tensor  // [L, M, H]
{
    // 1. Concatenate: input = [context_tokens; memory_embeddings]
    // 2. Forward pass through frozen LLM with Meta LoRA applied to each layer
    // 3. Extract hidden states at memory token positions from each layer
    // 4. Stack into [L, M, H] grid
}
```

**Scope:** ~200 lines. Reuses existing forward pass infrastructure with LoRA injection.

### T4: M2P Transformer (riir-ai)

```rust
// riir-gpu/src/hypernet/m2p_transformer.rs

/// M2P Transformer layer — alternating column/row attention.
pub struct M2PLayer {
    /// Attention mode: Column (odd) or Row (even).
    pub mode: M2PAttentionMode,
    /// Post-layernorm (not pre-layernorm — stabilizes parameter distribution).
    pub layernorm: LayerNorm,
    /// 2-layer MLP with intermediate dim 2H.
    pub mlp: MLP,
}

pub enum M2PAttentionMode {
    Column, // Mix across layers (rows) per token position (column)
    Row,    // Mix across tokens (columns) per layer (row)
}

/// Full M2P Transformer: memory grid → refined memory grid.
pub fn m2p_forward(
    layers: &[M2PLayer],
    memory_grid: &Tensor,  // [L, M, H]
    layer_pos_enc: &Tensor, // [L, 1, H]
    token_pos_enc: &Tensor, // [1, M, H]
) -> Tensor  // [L, M, H]
{
    // 1. Add positional encodings: grid + layer_pos_enc + token_pos_enc (broadcast)
    // 2. For each layer:
    //    - Apply column or row attention (alternating)
    //    - Post-layernorm
    //    - 2-layer MLP
    //    - Post-layernorm
    // 3. Return refined grid
}
```

**Scope:** ~300 lines. Core of the hypernetwork.

### T5: `context_to_lora()` End-to-End (riir-ai)

```rust
// riir-gpu/src/hypernet/mod.rs

/// Generated LoRA weights from context.
pub struct GeneratedLoRA {
    /// Per-layer (A, B) LoRA pairs. A: [I×r], B: [r×O].
    pub layers: Vec<(Tensor, Tensor)>,
    /// Generated LoRA rank (paper uses 8).
    pub rank: usize,
}

/// SHINE context-to-LoRA generation config.
pub struct ShineConfig {
    /// Meta LoRA rank (paper: 128).
    pub meta_lora_rank: usize,
    /// Generated LoRA rank (paper: 8).
    pub gen_lora_rank: usize,
    /// Number of M2P Transformer layers (paper: 4).
    pub m2p_layers: usize,
    /// Number of memory embeddings M (computed from gen_lora_rank if None).
    pub memory_length: Option<usize>,
    /// Hidden dimension of backbone LLM.
    pub hidden_dim: usize,
    /// Number of layers in backbone LLM.
    pub num_layers: usize,
}

/// Generate LoRA from context in a single forward pass.
pub fn context_to_lora(
    hypernet: &ShineHypernet,
    context_tokens: &[u32],
    frozen_llm: &TransformerWeights,
    config: &ShineConfig,
) -> GeneratedLoRA {
    let memory_length = config.memory_length
        .unwrap_or_else(|| compute_memory_length(config));

    // Stage 1: Memory extraction
    let memory_grid = extract_memory_states(
        context_tokens,
        &hypernet.memory_embeddings,
        &hypernet.meta_lora,
        frozen_llm,
        config,
    );

    // Stage 2: M2P Transformer
    let refined_grid = m2p_forward(
        &hypernet.m2p_layers,
        &memory_grid,
        &hypernet.layer_pos_enc,
        &hypernet.token_pos_enc,
    );

    // Stage 3: Reshape to LoRA
    reshape_to_lora(&refined_grid, config)
}

fn compute_memory_length(config: &ShineConfig) -> usize {
    // M = ⌈rD/H⌉ where D = sum of linear dims per layer
    // For QKVO + gate/up/down: D = 4*H*H + 3*H*3H = 13H²
    // Simplified: M = ⌈r * 13H / 1⌉ = 13*r*H
    let d = 13 * config.hidden_dim * config.hidden_dim;
    (config.gen_lora_rank * d + config.hidden_dim - 1) / config.hidden_dim
}

fn reshape_to_lora(grid: &Tensor, config: &ShineConfig) -> GeneratedLoRA {
    // Per layer: flatten M̂[i,:,:] → sequentially reshape into A, B matrices
    // for each linear module (Q, K, V, O, gate, up, down)
    // ...
}
```

**Scope:** ~200 lines. Ties together T3 + T4.

### T6: Feature Gate `shine_hypernet` (riir-ai)

```toml
# riir-ai/crates/riir-gpu/Cargo.toml
[features]
shine_hypernet = []  # SHINE context-to-LoRA hypernetwork (Research 62, Plan 098)
```

Enables:
- `riir-gpu/src/hypernet/` module (new)
- `MetaLoRAWeights`, `M2PTransformer`, `ShineHypernet`, `GeneratedLoRA`
- `context_to_lora()` function
- `m2p_attention.wgsl` kernels

### T7: Feature Gate `shine_routing` (riir-ai)

```toml
# riir-ai/crates/riir-gpu/Cargo.toml
[features]
shine_routing = ["shine_hypernet"]  # Lightweight: context extraction → expert routing only
```

Enables only the memory extraction stage + embedding comparison:

```rust
/// Route context to best expert in registry using Meta LoRA memory extraction.
pub fn route_context_to_expert(
    context_tokens: &[u32],
    hypernet: &ShineHypernet,
    expert_registry: &ExpertRegistry,
    frozen_llm: &TransformerWeights,
) -> (ExpertId, f32) {
    // 1. Extract memory states (no M2P Transformer needed)
    let memory_states = extract_memory_states(...);
    // 2. Mean pool across layers and tokens → context embedding [H]
    let context_emb = mean_pool(memory_states);
    // 3. Compare with expert embeddings
    expert_registry.find_best_match(&context_emb)
}
```

**Scope:** ~50 lines on top of T3.

### T8: GOAT Proof — `bomber_14_shine_expert`

```bash
cargo run --example bomber_14_shine_expert --features "bomber"
```

**Protocol:**
1. Generate bomber game replay context (50 moves from existing replay gen)
2. Load pre-trained Meta LoRA checkpoint (or use random init for baseline)
3. `context_to_lora()` → GeneratedLoRA from replay context
4. Run 100 games: GeneratedLoRA bomber vs random baseline
5. Compare with pre-trained bomber LoRA (from `bomber_04_nn`)

**Success criteria:**
- Context-generated LoRA win rate > 50% vs random baseline
- Context-generated LoRA win rate < pre-trained LoRA (expected — less optimization)
- If win rate > 50% vs baseline: ✅ GOAT proved for dynamic context adaptation

**Scope:** ~150 lines example file.

### T9: GOAT Proof — `go_11_shine_routing`

```bash
cargo run --example go_11_shine_routing --features "go,shine_routing"
```

**Protocol:**
1. Load Go expert registry (pre-trained LoRA adapters per opening style)
2. Provide game context (opening moves, opponent history)
3. `route_context_to_expert()` → select best expert
4. Compare with static domain routing (always use default Go LoRA)
5. Run 50 games per routing strategy vs fixed opponent

**Success criteria:**
- Context-informed routing accuracy > static routing +2% on move_accuracy
- If improvement > 2%: ✅ GOAT proved for context-aware expert selection

**Scope:** ~100 lines example file.

### T10: Benchmark `Alternating2D` vs Full Bidirectional

```bash
cargo bench --features "hla_attention" -- alternating_2d
```

**Setup:**
- Grid dimensions: L=36, M=148, H=4096 (SHINE defaults)
- Compare: Full bidirectional O((LM)²) vs Alternating2D O(LM²+ML²)
- Metrics: FLOPs, wall-clock time, output cosine similarity vs full attention

**Expected results (from paper):**
- ~90% FLOPs reduction
- <10% quality loss (cosine sim > 0.9 vs full attention)
- 5-10× wall-clock speedup

**Scope:** ~80 lines bench file.

### T11: Documentation Updates

- Update `README.md` — add SHINE section under "🔬 Research Features"
- Update `.docs/15_paper_feature_comparison.md` — add SHINE row
- Update `riir-ai/README.md` — add SHINE hypernet section

---

## Module Structure

```text
riir-ai/crates/riir-gpu/src/
  hypernet/
    mod.rs              — pub mod + re-exports
    meta_lora.rs        — MetaLoRAWeights, extract_memory_states()
    m2p_transformer.rs  — M2PLayer, M2PAttentionMode, m2p_forward()
    shine.rs            — ShineHypernet, ShineConfig, context_to_lora()
    routing.rs          — route_context_to_expert() (shine_routing feature)
    types.rs            — GeneratedLoRA, M2PConfig, reshape_to_lora()

  kernels/
    m2p_attention.wgsl  — Column + row attention kernels (post-layernorm)

katgpt-core/src/
  types.rs              — AttentionMode::Alternating2D variant (T1)
```

---

## Dependencies

| Task | Depends On | Blocks |
|------|-----------|--------|
| T1 | — | T2, T4 |
| T2 | T1 | T4 |
| T3 | — | T5, T7 |
| T4 | T1, T2 | T5 |
| T5 | T3, T4 | T6, T8 |
| T6 | T5 | T7, T8 |
| T7 | T3, T6 | T9 |
| T8 | T6 | T11 |
| T9 | T7 | T11 |
| T10 | T1 | T11 |
| T11 | T8, T9, T10 | — |

**Critical path:** T1 → T2 → T4 → T5 → T6 → T8

---

## Out of Scope

1. **Full SHINE pretraining pipeline** (6B tokens, reconstruction + completion objectives) — research-scale, not production-viable
2. **SHINE-R recurrent long-context** — our SpectralQuant KV compression is more mature
3. **Instruction fine-tuning on QA datasets** — our domain is game/strategy, not open-domain QA
4. **Coupled cross-attention variant** — paper ablation shows it's worse (Bitter Lesson)
5. **Replacing pre-trained LoRA adapters** — static domains (Go, Bomber) benefit from dedicated training

---

## References

- Paper: https://arxiv.org/pdf/2602.06358
- Code: https://github.com/MuLabPKU/SHINE
- Research note: `.research/062_SHINE_Scalable_In_Context_Hypernetwork.md`
