# Research 178: Rosetta Neurons — Cross-Model Alignment Distillation

**Paper**: "Rosetta Neurons: Mining the Common Units in a Model Zoo" (arXiv:2306.09346)
**Date**: 2026-06-07
**Status**: Verdict + Distillation

---

## 0. Paper TL;DR

Different vision models (ResNet, DINO, CLIP, MAE, GANs) converge on ~50 universal visual concepts (edges, colors, textures) despite completely different architectures and training objectives. The paper discovers these via:

1. **Pearson correlation mining** across model activation maps for same inputs
2. **Best Buddies** (mutual KNN) — filter to bidirectional matches only
3. **Self-clustering** — handle within-model redundancy (multiple neurons encoding same concept)
4. **Cross-model dictionary** — translation table: (model_A, layer, channel) ↔ (model_B, layer, channel)
5. **Activation-guided inversion** — steer generators without retraining, just by matching activation patterns

Key property: **zero training, zero labels, inference-time only**.

---

## 1. FUNDAMENTAL DISTILLATION

The paper is about *vision* neurons, but the underlying principle is architecture-agnostic:

### 1.1 The Core Abstraction: Cross-System Correlation Alignment

```
given: system_A produces activations a_i for inputs x
given: system_B produces activations b_j for inputs x
claim: if corr(a_i, b_j) > threshold AND corr is mutual (best buddies)
        then a_i and b_j represent the SAME concept
```

This is a **training-free concept alignment** algorithm. The inputs are the bridge — same inputs, different systems, find which internal states correlate.

### 1.2 The Five Atomic Operations

| Operation | Paper Name | Abstract Form | What It Needs |
|-----------|-----------|---------------|---------------|
| 1. Correlation Mining | Pearson over activation maps | `corr(sys_a.unit_i, sys_b.unit_j)` | Same inputs, different systems |
| 2. Mutual NN Filter | "Best Buddies" | `argmax_j corr(i,j) == i AND argmax_i corr(j,i) == j` | Correlation matrix |
| 3. Self-Clustering | Synonym groups | Cluster within-model units that fire similarly | Single system activations |
| 4. Dictionary Build | Cross-model dictionary | Map `(sys, unit) → concept_id` | Best buddies + clusters |
| 5. Activation Steering | Inversion via matched units | `optimize latent to match target activation` | Dictionary + generator |

### 1.3 Why This Matters For Us

We have TWO systems that process the SAME inputs:
- **katgpt-rs**: Multiple pruners process the same token sequences. Multiple drafters produce logits for the same context.
- **riir-ai**: Multiple LoRA adapters process similar game states. Multiple training checkpoints see the same data.

The Rosetta insight is: **you don't need to train alignment. You can mine it from activation correlation.**

---

## 2. MODELLESS APPLICATIONS (katgpt-rs)

### 2.1 GOAT 🐐 — "Best Buddies Drafting" for Speculative Decoding

**Idea**: The draft model and target model both produce logit distributions for the same context. Find token positions where they *mutually agree* (best buddies). Only speculate on those positions; fall back to autoregressive on disagreed positions.

**Implementation Path**:

```rust
// katgpt-core/src/traits.rs — new trait

/// Cross-model activation alignment via mutual nearest neighbors.
///
/// Distilled from Rosetta Neurons (arXiv:2306.09346).
/// Finds positions where draft and target models mutually agree,
/// enabling focused speculative decoding on high-confidence regions.
pub trait BestBuddyAligner: Send + Sync {
    /// Compute mutual agreement between draft and target marginals.
    /// Returns indices where both models rank each other as top-K match.
    fn mutual_agreement(
        &self,
        draft_marginals: &[f32],    // [vocab_size] from draft model
        target_marginals: &[f32],   // [vocab_size] from target model
        k: usize,                    // top-K for mutual NN
    ) -> Vec<usize>;

    /// Batch variant: process multiple positions.
    /// Returns per-position acceptance confidence [0.0, 1.0].
    fn batch_alignment_confidence(
        &self,
        draft: &[&[f32]],   // [num_positions][vocab_size]
        target: &[&[f32]],  // [num_positions][vocab_size]
        k: usize,
    ) -> Vec<f32>;
}
```

**Why GOAT**: This directly increases speculative acceptance rate WITHOUT any model training. Pure inference-time signal. The current `SpeculativeContext` already has `marginals_flat` and `p_distributions_flat` — we just need correlation computation on top.

**Performance**: Pearson correlation is O(V) per position (V=vocab_size). For V=32K this is ~32K multiply-adds. SIMD-able. Can be done in <1μs per position on modern hardware.

**Existing Code Integration Points**:
- `SpeculativeContext::marginals_flat` → draft marginals (already computed)
- `DraftResult::marginals` → target marginals (already computed)
- `build_dd_tree_speculative` → add alignment gate before tree expansion
- `RejectionReason` → add `LowAlignment` variant

**Verdict**: 🐐 **GOAT**. Zero training, pure inference, measurable acceptance rate improvement, clean trait integration.

---

### 2.2 GOAT 🐐 — "Rosetta Pruners": Cross-Pruner Agreement Mining

**Idea**: Run the same token sequence through ALL pruners (go, bomber, fft, monopoly, etc.). Find which pruners *agree* on which tokens at which depths. Build a "pruner dictionary" mapping (pruner_A, depth, token) ↔ (pruner_B, depth, token) = same constraint concept.

**Why This Works**: The paper found ~50 universal visual concepts across vision models. We should find ~N universal *constraint concepts* across game pruners. For example: "boundary token", "syntax-critical token", "semantic pivot token" — these should emerge regardless of which game pruner is active.

**Implementation Path**:

```rust
// src/pruners/rosetta_pruner.rs

/// Cross-pruner alignment via mutual agreement mining.
///
/// Discovers universal constraint concepts that emerge across
/// all game-specific pruners, regardless of game rules.
pub struct RosettaPruner {
    /// Pre-computed concept dictionary: (depth, token_idx) → concept_id
    concept_map: Vec<Vec<u32>>,  // [max_depth][vocab_size]
    /// Universal concepts that ALL pruners agree on.
    universal_concepts: Vec<ConstraintConcept>,
    /// Number of pruners that participated in mining.
    n_pruners: usize,
}

pub struct ConstraintConcept {
    id: u32,
    depth: usize,
    tokens: Vec<usize>,           // synonym tokens (self-clustering)
    agreement_ratio: f32,         // fraction of pruners that agree
}

impl ConstraintPruner for RosettaPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Universal concepts = always valid/invalid regardless of game
        let concept_id = self.concept_map[depth][token_idx];
        // Look up universal verdict
        self.universal_concepts[concept_id as usize].agreement_ratio > 0.8
    }
}
```

**Offline Phase**: Run batch of representative inputs through all pruners. Compute correlation matrix. Find best buddies. Build dictionary. Store as lookup table.

**Online Phase**: O(1) lookup per (depth, token). Zero allocation.

**Verdict**: 🐐 **GOAT for meta-pruning**. This is the "Rosetta Dictionary" but for constraints. Once built, it gives you a universal pruner that works across ALL domains. This is the "engine" play — one meta-pruner to rule them all.

---

### 2.3 SOLID — "Self-Clustering Synonym Discovery" for Token Patterns

**Idea**: Within a single model's marginals, cluster tokens that have similar activation patterns across contexts. These are "synonym tokens" — the paper's self-clustering applied to LLM output space.

**Already Exists**: `IlcClusterer` + `SynonymMap` in `src/distill/ilc.rs` does exactly this. The Rosetta paper validates the approach.

**Verdict**: Already implemented. No action needed. The paper provides theoretical justification for Plan 082b.

---

### 2.4 MARGINAL — "Cross-Tokenizer Rosetta Tokens"

**Idea**: Map tokens across different BPE vocabularies (GPT-2 vs LLaMA vs Gemma tokenizer). Find "Rosetta Tokens" — tokens that represent the same concept despite different tokenization.

**Problem**: This requires running the same text through multiple tokenizers and finding which token activations correlate. But we don't have activations from different tokenizers — we have discrete token IDs that don't share a continuous space.

**Workaround**: Use embedding cosine similarity as a proxy for correlation. Token A in tokenizer X and token B in tokenizer Y are "Rosetta Tokens" if their embeddings are mutual nearest neighbors.

**Verdict**: ⚠️ **Marginal**. Requires embedding tables from multiple tokenizers. Not useful unless we support multi-model serving with different tokenizers. Current architecture is single-model. Revisit if/when multi-model serving is needed.

---

### 2.5 MARGINAL — "Activation Steering via Marginal Manipulation"

**Idea**: The paper steers generators by matching activation patterns. For LLM inference, this would mean manipulating the marginal distribution to match a target concept, without changing the model weights.

**Problem**: We already do this — it's called `ScreeningPruner`. The `relevance()` method IS activation steering: it manipulates the log-prob distribution to favor domain-relevant tokens.

**Verdict**: ⚠️ Already exists as `ScreeningPruner`. The paper's activation-guided inversion is a generalization, but our current implementation covers the LLM-specific case. No new code needed.

---

## 3. MODEL-BASED APPLICATIONS (riir-ai)

### 3.1 GOAT 🐐 — Cross-Game LoRA Adapter Alignment

**Idea**: The paper aligns neurons across vision models. We align LoRA adapters across game domains. Run the same game states through bomber LoRA, Go LoRA, FFT LoRA → find which LoRA neurons correlate → build a "Rosetta LoRA Dictionary".

**Implementation Path**:

```rust
// riir-engine/src/rosetta_lora.rs

/// Cross-domain LoRA adapter alignment (Rosetta Neurons for game AI).
///
/// Mines mutual nearest neighbors between LoRA adapter weight spaces
/// to discover universal game concepts that transfer across domains.
pub struct RosettaLoraDictionary {
    /// Map: (domain_a, neuron_idx) → (domain_b, neuron_idx) with correlation
    pairs: Vec<LoraNeuronPair>,
    /// Universal concepts discovered across all domains.
    universal_concepts: Vec<GameConcept>,
}

pub struct LoraNeuronPair {
    domain_a: String,
    neuron_idx_a: usize,
    domain_b: String,
    neuron_idx_b: usize,
    correlation: f32,  // Pearson correlation
}

pub struct GameConcept {
    id: u32,
    /// All neurons across all domains that represent this concept.
    neurons: Vec<(String, usize)>,  // (domain, neuron_idx)
    /// Activation pattern centroid.
    centroid: Vec<f32>,
}
```

**Why GOAT**: This enables **cross-game skill transfer WITHOUT retraining**. Once you know that neuron 42 in bomber LoRA corresponds to neuron 17 in Go LoRA (both = "danger avoidance"), you can:
1. Transfer learned behaviors from one game to another
2. Initialize new game LoRA from aligned neurons of existing games
3. Build a "universal game AI" that works across games via the dictionary

**Integration with Existing Code**:
- `EgaLoraAdapter` — the LoRA weights to mine correlations over
- `analogy.rs` — already probes structural alignment via Dirichlet Energy. Rosetta adds *neuron-level* alignment on top
- `frame_coreset.rs` — the coreset selection can be guided by universal concepts

**Verdict**: 🐐 **GOAT**. This is the paper's core contribution, directly applicable to our multi-game LoRA architecture.

---

### 3.2 GOAT 🐐 — Activation Correlation Mining Across Training Checkpoints

**Idea**: Instead of cross-model alignment, do cross-checkpoint alignment. Run the same inputs through LoRA weights at epoch 1, 10, 50, 100 → find which neurons remain stable (high self-correlation across checkpoints) vs which evolve.

**Why This Matters**: 
- Stable neurons = "core concepts" that the model always maintains
- Evolving neurons = "learning frontier" — where active learning happens
- This tells you WHERE to focus LoRA training budget

**Implementation Path**:

```rust
/// Track concept evolution across training checkpoints via self-correlation.
pub struct ConceptEvolutionTracker {
    /// Per-checkpoint activations for a fixed probe set.
    checkpoint_activations: Vec<Vec<Vec<f32>>>,  // [checkpoint][probe][neuron]
    /// Neurons that remain stable (corr > 0.9) across all checkpoints.
    stable_neurons: Vec<usize>,
    /// Neurons that changed significantly (learning frontier).
    frontier_neurons: Vec<usize>,
}
```

**Integration**: 
- `critical_position.rs` — KL-based critical position detection. Rosetta adds *neuron-level* stability tracking
- `wall.rs` / `wall_config.rs` — attention gate training. Stability tracking tells you which gates are stable vs evolving

**Verdict**: 🐐 **GOAT for training diagnostics**. Gives unprecedented visibility into what LoRA is actually learning. Low implementation cost — just Pearson correlation over saved checkpoints.

---

### 3.3 SOLID — Cross-Domain KV Cache Alignment

**Idea**: Find which attention patterns are shared across game domains. If bomber and Go both attend to "danger positions" in similar ways, we can share KV cache entries.

**Problem**: KV cache is per-sequence, not per-neuron. The alignment would need to be at the *attention head* level, not the *individual entry* level.

**Workaround**: Align attention head outputs across games. If head 3 in bomber and head 7 in Go produce similar attention patterns for analogous states, we can share those heads via the Rosetta dictionary.

**Verdict**: ⚠️ **Solid but complex**. Requires careful handling of sequence-level vs neuron-level alignment. Lower priority than 3.1 and 3.2.

---

### 3.4 MARGINAL — Generative Steering via Activation Matching

**Idea**: Use the Rosetta LoRA dictionary to steer NPC behavior at inference time. Instead of retraining the LoRA, match activation patterns to a target concept and adjust the output.

**Problem**: This is basically what `ScreeningPruner` already does in katgpt-rs. For game AI, the equivalent is the `GameState` trait — you'd be steering the policy via reward shaping, which we already do.

**Verdict**: ⚠️ Already covered by existing mechanisms (`GameState`, `RolloutPolicy`, `DualLeoMixer`). The Rosetta insight adds theoretical justification but no new implementation.

---

## 4. CREATIVE FUSION IDEAS

### 4.1 GOAT 🐐 — "Concept Dictionary" as a Service

**The Play**: Build a pre-computed Rosetta Dictionary that maps concepts across:
- **Pruners** (katgpt-rs): universal constraint concepts
- **LoRA adapters** (riir-ai): universal game concepts
- **Training checkpoints** (riir-ai): concept evolution timelines

This dictionary becomes a **shared artifact** between the two projects. katgpt-rs uses it for inference-time optimization; riir-ai uses it for training-time focusing.

**Commercial**: The dictionary is the "engine" play. Once computed, it's reusable across all deployments. Different games/domains just query the dictionary.

```rust
// Shared between katgpt-rs and riir-ai
pub struct ConceptDictionary {
    /// Universal concept IDs.
    concepts: Vec<Concept>,
    /// Map from (system, unit_id) → concept_id.
    unit_to_concept: HashMap<(SystemId, usize), u32>,
    /// Map from concept_id → [(system, unit_id)].
    concept_to_units: Vec<Vec<(SystemId, usize)>>,
}

pub enum SystemId {
    Pruner(String),          // katgpt-rs pruner name
    LoraAdapter(String),     // riir-ai LoRA domain
    Checkpoint(String, u32), // riir-ai (domain, epoch)
}
```

**Verdict**: 🐐 **GOAT**. The Rosetta Dictionary is the single most valuable artifact from this paper. It's the "translation table" that enables everything else.

---

### 4.2 GOAT 🐐 — "Activation Budget Allocation" via Correlation Mining

**The Play**: Run correlation mining on DDTree marginals across many requests. Find which (depth, token) combinations are always correlated (universal) vs rarely correlated (noise). Allocate tree expansion budget proportionally.

**Why GOAT**: This directly addresses the `PositionWeightedBudget` allocation problem. Instead of heuristic `gamma` decay, use empirical correlation data to decide where to spend tree budget.

**Implementation**:

```rust
/// Data-driven budget allocation via activation correlation mining.
pub struct CorrelationBudgetAllocator {
    /// Pre-computed: fraction of time each (depth) position is in
    /// the top-K mutual agreement set across requests.
    depth_agreement_rate: Vec<f32>,  // [max_depth]
    /// Smoothing factor for online updates.
    ema_alpha: f32,
}

impl CorrelationBudgetAllocator {
    /// Allocate budget proportional to agreement rate.
    /// Positions where draft/target agree often → more budget.
    /// Positions where they rarely agree → less budget (fall back to autoregressive).
    pub fn allocate(&self, total_budget: usize) -> Vec<usize> {
        let total_agreement: f32 = self.depth_agreement_rate.iter().sum();
        self.depth_agreement_rate.iter().map(|rate| {
            ((rate / total_agreement) * total_budget as f32).round() as usize
        }).collect()
    }
}
```

**Verdict**: 🐐 **GOAT**. Replaces heuristic budget allocation with data-driven allocation. Clean integration with existing `PositionWeightedBudget`.

---

### 4.3 SOLID — "Ensemble Pruner" via Best Buddies

**Idea**: Run multiple pruners. Take only the "best buddies" — tokens where ALL pruners agree. This gives a high-precision ensemble pruner.

**Problem**: Current pruners are domain-specific (Go pruner knows Go rules, Bomber pruner knows Bomber rules). They don't agree on domain-specific tokens — they agree only on *syntactic* tokens (which `ConstraintPruner` already handles).

**Revised Idea**: Use best buddies to detect *when pruners disagree* — this signals domain-critical decisions. When pruners agree, the token is safe (low cost). When they disagree, the token is domain-sensitive (high cost, needs careful evaluation).

**Verdict**: ⚠️ **Solid insight for prioritization** but marginal as standalone feature. Better as a diagnostic tool.

---

## 5. VERDICT SUMMARY

### GOAT 🐐 (Implement First)

| # | Idea | Project | Effort | Impact |
|---|------|---------|--------|--------|
| 2.1 | Best Buddies Drafting | katgpt-rs | Low | ↑ Spec acceptance rate |
| 2.2 | Rosetta Pruners (meta-pruner) | katgpt-rs | Medium | Universal constraint engine |
| 3.1 | Cross-Game LoRA Alignment | riir-ai | Medium | Cross-game skill transfer |
| 3.2 | Checkpoint Concept Evolution | riir-ai | Low | Training diagnostics |
| 4.1 | Concept Dictionary Service | Both | Medium | Shared infrastructure |
| 4.2 | Correlation Budget Allocation | katgpt-rs | Low | Data-driven budget |

### Solid (Implement When Relevant)

| # | Idea | Project | Notes |
|---|------|---------|-------|
| 2.3 | Synonym Discovery (ILC) | katgpt-rs | Already exists |
| 3.3 | Cross-Domain KV Cache Alignment | riir-ai | Complex, lower priority |
| 4.3 | Ensemble Pruner via Best Buddies | katgpt-rs | Better as diagnostic |

### Marginal (Skip / Defer)

| # | Idea | Why |
|---|------|-----|
| 2.4 | Cross-Tokenizer Rosetta Tokens | Single-model architecture, revisit for multi-model |
| 2.5 | Activation Steering via Marginals | Already exists as ScreeningPruner |
| 3.4 | Generative Steering via Activation | Covered by GameState/RolloutPolicy |

---

## 6. IMPLEMENTATION PRIORITY

### Phase 1: Quick Wins (1-2 days each)

1. **Best Buddies Drafting** (2.1) — Add `BestBuddyAligner` trait, implement Pearson correlation + mutual NN filter in `build_dd_tree_speculative`. This is the fastest path to measurable improvement.

2. **Correlation Budget Allocation** (4.2) — Extend `PositionWeightedBudget` to accept empirical agreement rates. Online EMA update from speculative decoding results.

### Phase 2: Infrastructure (1 week)

3. **Concept Dictionary** (4.1) — Shared `ConceptDictionary` type in `katgpt-core`. Offline mining tool that runs correlation across pruners/adapters. Serialization (blake3 hash verification).

### Phase 3: Training Integration (1-2 weeks)

4. **Cross-Game LoRA Alignment** (3.1) — Offline mining across `EgaLoraAdapter` instances. Build Rosetta LoRA Dictionary. Use for new-game initialization.

5. **Checkpoint Concept Evolution** (3.2) — Run probes through saved checkpoints. Track stable vs frontier neurons. Display in training dashboards.

### Phase 4: Engine Play (2-3 weeks)

6. **Rosetta Pruners** (2.2) — Full meta-pruner with pre-computed universal constraint concepts. Integration with DDTree as a first-class pruner.

---

## 7. RUST IMPLEMENTATION NOTES

### Pearson Correlation (SIMD)

```rust
/// SIMD-optimized Pearson correlation for two f32 slices.
/// O(n) with auto-vectorization for chunks of 4 or 8.
#[inline]
pub fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len() as f32;
    
    let (sum_a, sum_b, sum_ab, sum_aa, sum_bb) = a.iter().zip(b.iter())
        .fold((0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32), |(sa, sb, sab, saa, sbb), (&x, &y)| {
            (sa + x, sb + y, sab + x * y, saa + x * x, sbb + y * y)
        });
    
    let num = n * sum_ab - sum_a * sum_b;
    let den = ((n * sum_aa - sum_a * sum_a) * (n * sum_bb - sum_b * sum_bb)).sqrt();
    
    if den < 1e-10 { 0.0 } else { num / den }
}
```

### Mutual Nearest Neighbor (Best Buddies)

```rust
/// Find mutual nearest neighbors between two correlation matrices.
/// Returns pairs (i, j) where j = argmax_k corr(i, k) AND i = argmax_k corr(k, j).
pub fn best_buddies(corr_matrix: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let n_a = corr_matrix.len();
    let n_b = corr_matrix[0].len();
    
    // Forward: for each i, find best j
    let forward: Vec<usize> = corr_matrix.iter()
        .map(|row| row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0)
        .collect();
    
    // Backward: for each j, find best i
    let mut backward = vec![0; n_b];
    for j in 0..n_b {
        let mut best_i = 0;
        let mut best_val = f32::NEG_INFINITY;
        for i in 0..n_a {
            if corr_matrix[i][j] > best_val {
                best_val = corr_matrix[i][j];
                best_i = i;
            }
        }
        backward[j] = best_i;
    }
    
    // Mutual: i → j AND j → i
    forward.into_iter().enumerate()
        .filter(|&(i, j)| backward[j] == i)
        .map(|(i, j)| (i, j))
        .collect()
}
```

---

## 8. KEY INSIGHT: WHY ROSETTA WORKS FOR US

The paper's fundamental insight is: **convergent representations emerge across systems that process the same data, regardless of architecture**. 

We have:
- **katgpt-rs**: Multiple "systems" (pruners, drafters, verifiers) processing the same token sequences
- **riir-ai**: Multiple "systems" (LoRA adapters, training checkpoints) processing similar game states

The Rosetta method gives us a **training-free way to discover shared structure** across these systems. This is exactly the "modelless" philosophy: extract value from alignment rather than from training.

The commercial play is clear: the **Concept Dictionary** is the engine's IP. Once you've mined universal concepts across all your pruners and adapters, you have a meta-layer that no competitor can replicate without doing the same mining.

---

## TL;DR

**6 GOAT ideas, 3 Solid, 3 Marginal.** The Rosetta Neurons paper distills into one core capability: **training-free cross-system concept alignment via mutual nearest neighbors on activation correlations**. For katgpt-rs, this means better speculative drafting and universal meta-pruners. For riir-ai, this means cross-game LoRA transfer and training diagnostics. The shared artifact is a **Concept Dictionary** that bridges both projects.
