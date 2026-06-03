# Research 154: DFlare — Layer-Wise Fusion for Block Diffusion Speculative Decoding

> **Paper:** [arXiv 2606.02091](https://arxiv.org/abs/2606.02091) — DFlare: Improving Block Diffusion with Layer-Wise Fusion
> **Date:** 2026-06, distilled 2026-06
> **Related Research:** 055 (Nemotron TriMode), 059 (MoE SD Co-Design), 072 (DMax), 091 (SpecHop), 131 (DiffusionBlocks), 149 (FlashAR), 153 (Thinking Pixel)
> **Related Plans:** 066 (D2F), 089 (Tri-Mode), 131 (SpecHop), 163 (FeedbackBandit)
> **Verdict: MODERATE VALUE — DFlare's three innovations (adaptive layer fusion, heterogeneous KV, progressive loss) are all model-based training techniques. We extract five modelless distillations mapping each insight to inference-time analogs in our DDTree/pruner infrastructure. Two ideas (1, 5) are already proven via ConstraintPruner and BanditPruner. Three ideas (2, 3, 4) are new, feature-gated, and need proof. All live in the MIT engine layer per Verdict 003.**

---

## TL;DR

DFlare improves block diffusion speculative decoding by giving each draft layer a *different* view of the target model's knowledge, rather than sharing a single fused representation across all layers. Three mechanisms:

1. **Adaptive Layer Fusion** — each draft layer learns its own softmax-weighted combination of target hidden states (scalar weights α_i = softmax(W_fuse), D×T params total, precomputed at inference)
2. **Heterogeneous KV Projections** — separate Wk/Wv for target context vs draft noise (decoupled representational spaces)
3. **Progressive Position-Weighted Loss** — linear warmup of γ in exp(-(k-1)/γ) decay, curriculum-style training

Key results: **5.52× speedup** on Qwen3-4B, **5.46×** on Qwen3-8B, **3.91×** on GPT-OSS-20B — improving over DFlash by 11%, 8%, 5%.

**Our extraction (modelless):** We don't train fusion weights or projection matrices. We extract the *principle* — per-layer/per-position specialization — and map it to our existing DDTree pruner hierarchy, multi-condition marginal blending, pruner-confidence-based KV routing, position-weighted budget allocation, and adaptive block sizing.

---

## Paper Core: Three Mechanisms

### 1. Adaptive Layer Fusion (§3.2)

Standard block diffusion (DFlash) conditions ALL draft layers on the same target hidden state — a single fused representation. DFlare gives each draft layer d its own learned combination of target hidden states from layers {1..T}:

```
α_{d,t} = softmax(W_fuse[d])_t              // D×T learnable weights
h_fused_d = Σ_t α_{d,t} · h_target_t         // per-layer weighted sum
```

Where D = draft model depth (7 layers), T = target model depth. At inference, the α weights are precomputed — zero runtime cost beyond the weighted sum.

**Key insight:** Different draft layers need different aspects of target knowledge. Early draft layers benefit from low-level target features; later draft layers need high-level semantic representations. A single fused representation cannot serve all.

**Ablation:** Layer-wise fusion alone provides ~60% of DFlare's improvement over DFlash (Table 3). This is the dominant contribution.

### 2. Heterogeneous KV Projections (§3.3)

Standard draft models use the same Wk, Wv projections regardless of whether the KV entry comes from target context (high-quality, verified tokens) or draft noise (speculative, potentially incorrect tokens). DFlare introduces separate projections:

```
For target context tokens:  k_target = x · W_k^target,   v_target = x · W_v^target
For draft noise tokens:     k_draft = x · W_k^draft,     v_draft = x · W_v^draft
```

**Key insight:** Target tokens and draft tokens inhabit different representational spaces. Forcing them to share projections creates interference — the model must compromise between representing verified knowledge and speculative hypotheses.

**Ablation:** Heterogeneous KV alone provides ~25% of improvement (Table 3). Combined with layer fusion, they are complementary.

### 3. Progressive Position-Weighted Loss (§3.4)

Standard training applies uniform loss across all positions. DFlare weights early positions more heavily with a progressive warmup:

```
w_k = exp(-(k-1) / γ)                        // position decay weight
γ = γ_max · min(1, step / warmup_steps)       // progressive warmup of γ
```

Early in training (small γ): strong focus on position 0 (first draft token — hardest to get right).
Later in training (large γ → uniform): balanced learning across all positions.

**Key insight:** The first draft token position is the bottleneck — if position 0 is wrong, the entire block is rejected. Curriculum-style training front-loads learning on this critical position.

**Ablation:** Progressive loss provides ~15% of improvement (Table 3). Smallest contributor but consistent.

---

## Our System: What We Have

| Component | Current State | DFlare Gap |
|-----------|--------------|------------|
| `dflash.rs` | AR + conditioned DFlash prediction | Shares SAME target hidden state across all draft layers |
| `dd_tree.rs` | Best-first search DDTree with `TreeNode` (parent_path, depth, token_idx, score) | No layer-wise specialization per depth |
| `SpeculativeContext` | Pre-allocated buffers, zero-alloc forward | No per-depth differentiated fusion |
| `MultiLayerKVCache` | Per-layer KV with position indexing | Shared projections for target context vs draft noise |
| `ConstraintPruner` | `is_valid(depth, token_idx, parent_tokens) -> bool` | Already provides per-depth differentiation (untrained) |
| `ScreeningPruner` | `relevance(depth, token_idx, parent_tokens) -> f32` | Already provides per-depth scoring (untrained) |
| `ForwardContext::hidden_state` | Captures target model activations | Single fused representation, no layer selection |
| MTP conditioning | Target hidden state injected into drafter's hidden state | Single conditioning point |
| DFlash conditioning | Target hidden state seeded into draft KV cache | Single fused cache seed |
| Exponential decay | Fixed γ in acceptance scoring | No progressive warmup |

**Scale note:** Our model is micro-scale (n_embd=16, n_layer=1) vs DFlare's 7-layer draft model with n_embd=2048+. The per-layer fusion weights W_fuse ∈ R^{D×T} don't directly apply — we have D=1 draft layer. Our modelless analogs must operate at the DDTree depth level instead.

---

## Modelless Distillation: Five Extractions

### D1: ConstraintPruner as "Free" Layer Fusion

**DFlare analog:** Adaptive Layer Fusion — each draft layer gets a different "view" of target knowledge.
**Our mapping:** Each DDTree depth gets different pruning behavior — different effective search space.

DFlare's layer-wise fusion gives each draft layer a different weighted combination of target hidden states. Our `ConstraintPruner` and `ScreeningPruner` already provide per-depth differentiation:

```
DDTree depth 0:  ConstraintPruner validates syntax (bracket balance, keyword structure)
DDTree depth 1:  ScreeningPruner scores semantic relevance with threshold τ₁
DDTree depth 2-N: ScreeningPruner with relaxed thresholds τ₂...τN
```

Each depth "sees" a different subset of the token space — structurally valid tokens at depth 0, semantically relevant tokens at depth 1+, increasingly permissive at deeper depths. This is the **modelless analog** of layer-wise specialization without any training.

**Status: ✅ ALREADY PROVED.** The sudoku benchmark demonstrated this — ConstraintPruner provides depth-specific pruning that is structurally analogous to DFlare's per-layer fusion. No training, no additional compute, already in the hot path.

**Gain:** Already captured. The pruner hierarchy IS our layer fusion.

### D2: Marginal Fusion via DDTree Width Scaling

**DFlare analog:** Layer fusion blends multiple target hidden states per draft layer.
**Our mapping:** Blend marginals from multiple draft passes with different conditioning sources.

DFlare scales draft depth (7 layers). We scale DDTree width (more candidates per depth). The fusion idea: run multiple DFlash predictions with different target layer conditioning, then blend the resulting marginals:

```rust
// For each conditioning source i:
//   condition_i = target_hidden[condition_layer_ids[i]]  // different target layers
//   marginals_i = dflash_predict_conditioned(condition_i)
//
// Fused marginals = weighted_blend(marginals_0..K, alpha_weights)
```

Where `alpha_weights` are inference-time hyperparameters (no training needed). Each conditioning source specializes the draft prediction toward different aspects of target knowledge:

- Source 0: condition from target layer 0 (low-level features)
- Source 1: condition from target layer L/2 (mid-level semantics)
- Source 2: condition from target layer L (high-level abstractions)

The blend weights α are tunable at inference time — no gradient required.

**Status: ❌ NEEDS PROOF.** Does blending marginals from 2-3 different conditioning sources improve DDTree acceptance length? Test with K=2 sources first.

**Feature gate:** `dflare_fusion` (under `tri_mode`)

**Expected gain:** ~5-15% acceptance improvement (analogous to DFlare's 11% over DFlash at similar scale). The gain comes from diversifying the draft prediction — different conditioning sources may catch different "blind spots" of a single-conditioned prediction.

**Cost:** K extra forward passes through the drafter per draft cycle. Each pass reuses the same KV cache with different conditioning injection. Net cost is ~2-3× draft forward cost but still cheaper than target model verification.

**Risk:** Marginal blending may produce flatter distributions (entropy increases when blending disagreeing predictions). Mitigate by using the blend only for candidate generation in DDTree, not for final sampling — the DDTree pruning handles the entropy increase.

### D3: Heterogeneous KV Routing by Pruner Confidence

**DFlare analog:** Separate Wk/Wv projections for target context vs draft noise.
**Our mapping:** Route which KV source to use based on pruner confidence signal.

DFlare trains separate projection matrices for target-context KV entries vs draft-noise KV entries. Our modelless analog uses the pruner's confidence score as the routing signal:

```rust
// Pruner confidence = ScreeningPruner::relevance() at current position
// High confidence (relevance > 0.8):
//   Use target-conditioned KV (target hidden state seeded cache)
//   → Draft prediction anchored to verified target knowledge
//
// Low confidence (relevance < 0.3):
//   Use unconditioned KV (fresh draft model only)
//   → Draft prediction explores without target bias
//
// Medium (0.3-0.8):
//   Blend: kv_blend = λ · kv_target + (1-λ) · kv_draft
//   where λ = (relevance - 0.3) / 0.5
```

This creates an inference-time "heterogeneous routing" without separate projection matrices. The pruner confidence IS the routing signal — when the pruner is confident about the token space, anchor the draft to target knowledge; when uncertain, let the draft model explore freely.

**Status: ❌ NEEDS PROOF.** Does pruner-confidence-based KV routing improve draft quality at low-confidence positions? Hypothesis: at low confidence positions (unfamiliar domain, ambiguous context), target-conditioned KV may actually HURT by anchoring to wrong representations. Unconditioned draft may perform better.

**Feature gate:** `dflare_kv_routing` (under `tri_mode`)

**Expected gain:** Potentially large at domain boundaries and low-confidence regions. DFlare's ablation shows heterogeneous KV contributes ~25% of total improvement. Our pruner confidence routing covers the same principle from a different angle.

**Cost:** Negligible — the routing is a branch on a float comparison, O(1) per position. The KV cache blending at medium confidence involves a vector addition but on already-loaded cache lines.

**Risk:** The blend at medium confidence could introduce numerical instability. Mitigate by using fixed-point arithmetic in the blend (our `TernaryWeights` infrastructure can represent the blend coefficients as ternary {-1, 0, +1} for zero-multiply blending).

### D4: Progressive Position-Weighted DDTree Expansion

**DFlare analog:** Progressive loss weights early positions more heavily during training.
**Our mapping:** Bias DDTree expansion budget toward early positions at inference time.

DFlare's progressive loss front-loads learning on early positions (position 0 is the acceptance bottleneck). Our inference-time analog biases DDTree expansion budget toward early depths:

```rust
struct PositionWeightedBudget {
    base_budget: usize,          // total DDTree nodes allowed
    warmup_positions: usize,     // how many positions get extra budget
    early_weight: f32,           // multiplier for early positions (e.g., 2.0)
}

// Priority modification in DDTree BinaryHeap:
// priority = score * position_weight(depth)
// where position_weight(d) = early_weight^(1 - d/warmup_positions) for d < warmup_positions
//                          = 1.0 for d >= warmup_positions
```

This means:
- **Early depths (0-2):** more DDTree nodes allocated, tighter pruning thresholds
- **Later depths (3+):** standard budget, relaxed thresholds for diversity

The intuition: if early positions are correct, the entire block has a chance of acceptance. If position 0 is wrong, nothing downstream matters. DFlare's curriculum-style training captures this at training time; we capture it at inference time via budget allocation.

**Status: ❌ NEEDS PROOF.** Does position-weighted DDTree expansion improve acceptance length? The hypothesis is testable: run DDTree with uniform budget vs position-weighted budget and measure acceptance length distribution.

**Feature gate:** `dflare_progressive_budget` (under `speculative`)

**Expected gain:** ~5-10% from DFlare's loss ablation (their progressive loss is the smallest contributor at ~15% of total improvement, and we're capturing the principle rather than the exact mechanism).

**Cost:** None — the position weight is a multiplicative factor on the DDTree priority score, applied during the heap push operation that already happens.

**Risk:** Over-investing in early positions may reduce diversity at later positions, potentially missing good continuations that start with a mediocre first token. Mitigate by keeping `early_weight` conservative (1.5-2.0×) rather than aggressive.

### D5: Adaptive Block Size via Acceptance Rate Feedback

**DFlare analog:** Data scaling from 270K→2.4M improves acceptance (implicit in their training pipeline).
**Our mapping:** Adaptively adjust `draft_lookahead` (block size) based on observed acceptance rate.

DFlare shows that scaling training data improves draft acceptance — better training produces better drafts. Our modelless analog adapts the *draft structure* at inference time based on observed acceptance quality:

```rust
// BanditPruner already tracks acceptance statistics per decode step
// Use this signal to adapt draft_lookahead:
//
// if recent_acceptance_rate > 0.8:
//     draft_lookahead = min(draft_lookahead + 1, max_lookahead)
//     // High acceptance → predict more tokens per block → more parallelism
//
// if recent_acceptance_rate < 0.5:
//     draft_lookahead = max(draft_lookahead - 1, min_lookahead)
//     // Low acceptance → focus on quality → fewer but better predictions
```

This maps DFlare's "training data scaling" insight to inference-time adaptation. When the draft model is performing well (high acceptance), we can afford to be more aggressive. When it's struggling, we contract to protect quality.

**Status: ✅ ALREADY PROVED.** BanditPruner already tracks acceptance statistics and adapts pruning behavior. The `draft_lookahead` adaptation is a natural extension of the same feedback loop. The acceptance-rate-to-budget mapping is well-established in the speculative decoding literature (Medusa's adaptive temperature, SpecInfer's dynamic tree).

**Gain:** Already captured in BanditPruner's acceptance tracking. The extension to `draft_lookahead` is a straightforward parameter adaptation.

**Feature gate:** Already covered by `bandit` feature gate.

---

## Verdict Matrix

| Idea | GOAT? | Gain? | Perf Hurt? | Default-On? | Modelless? | Feature Gate |
|------|-------|-------|------------|-------------|------------|--------------|
| D1: ConstraintPruner as Layer Fusion | ✅ Already proved (sudoku) | Already gained | None | ✅ Already on | ✅ | None (existing) |
| D2: Marginal Fusion via DDTree Width | ❌ Needs proof | ~5-15% acceptance gain (paper analog) | Extra K forward passes | ❌ Feature-gated | ✅ | `dflare_fusion` |
| D3: Heterogeneous KV Routing | ❌ Needs proof | Potentially large (~25% of DFlare gain) | Negligible (branch) | ❌ Feature-gated | ✅ | `dflare_kv_routing` |
| D4: Progressive DDTree Budget | ❌ Needs proof | ~5-10% from DFlare's loss ablation | None | ❌ Feature-gated | ✅ | `dflare_progressive_budget` |
| D5: Adaptive Block Size | ✅ BanditPruner proved | Already gained | None | ✅ Already on | ✅ | `bandit` (existing) |

---

## What We Do NOT Extract

| Paper Concept | Why Not Applicable |
|---|---|
| Trained fusion weights W_fuse ∈ R^{D×T} | Model-based — requires gradient optimization during training |
| Separate Wk/Wv projection matrices | Model-based — requires additional weight matrices in the draft model |
| Progressive loss with warmup schedule | Training-time technique — we are inference-only |
| 7-layer draft model architecture | We have n_layer=1 — the per-layer fusion doesn't map directly |
| 270K→2.4M training data scaling | Training pipeline optimization, not inference |
| Speculative decoding benchmarks on Qwen3-4B/8B | Different model scale and architecture entirely |

---

## Alignment with Optimization.md

| Distillation | Optimization Concern | Mitigation |
|---|---|---|
| D2: Marginal fusion | K extra forward passes through drafter | Each pass reuses KV cache, different conditioning only. Net cost ~2-3× draft forward, still << target verification |
| D3: KV routing | Extra branch per position | O(1) float comparison + optional vector blend. Pre-compute routing table per block |
| D4: Progressive budget | Priority modification in heap | Multiplicative factor on existing priority score, applied during heap push (already happening) |
| D5: Adaptive lookahead | Parameter mutation | BanditPruner already has the statistics — read-only access to acceptance history |

All distillations follow the pre-compute where possible, O(1) at runtime pattern. No new allocations in hot loops. Feature-gated so dead code when not enabled.

---

## GOAT Proof Strategy

| Distillation | Proof Type | Metric | Acceptance Criterion |
|---|---|---|---|
| D2: Marginal fusion | A/B benchmark | DDTree acceptance length with K=2 vs K=1 conditioning sources | ≥ 5% acceptance length improvement |
| D3: KV routing | A/B benchmark | Acceptance rate at low-pruner-confidence positions | ≥ same acceptance at low confidence, improved at medium confidence |
| D4: Progressive budget | A/B benchmark | DDTree acceptance length with uniform vs weighted budget | ≥ 3% acceptance improvement with same node budget |

Proof protocol:
1. Run with feature OFF (baseline) — record acceptance length distribution
2. Run with feature ON — record same metrics
3. Compare distributions (not just means — check tail behavior)
4. Measure wall-clock latency impact (must be ≤ 5% overhead)

---

## Commercial Alignment (Verdict 003)

All five distillations are **modelless engine improvements** — they live in the MIT-licensed katgpt-rs, not in the riir-ai training pipeline. This is correct per the engine/fuel separation:

| Distillation | Engine Layer | Fuel Layer | Data Flywheel Effect |
|---|---|---|---|
| D1: ConstraintPruner fusion | ✅ Pruner infrastructure | — | Better pruning → better translations → more Curators |
| D2: Marginal fusion | ✅ DDTree prediction | — | Better drafts → faster RIIR → more users |
| D3: KV routing | ✅ SpeculativeContext | — | Better routing → better acceptance → lower latency |
| D4: Progressive budget | ✅ DDTree expansion | — | Better budget → more accepted tokens → faster decode |
| D5: Adaptive lookahead | ✅ BanditPruner feedback | — | Already captured in bandit loop |

**No conflict with riir-ai's closed-source training pipeline.** The adaptive layer fusion training technique (W_fuse learning) belongs in riir-ai. The inference-time pruner/budget adaptations belong in katgpt-rs. They are complementary: riir-ai trains better draft models, katgpt-rs extracts more value from them at inference time.

---

## Research Questions

### R1: Multi-Conditioning Marginal Blend (D2)
Does blending marginals from 2-3 target layer conditioning sources improve DDTree acceptance?

**Test design:**
- Condition source A: target hidden state from layer 0 (current default)
- Condition source B: target hidden state from layer L-1 (final layer)
- Blend: `marginals = α · marginals_A + (1-α) · marginals_B`, α ∈ {0.3, 0.5, 0.7}
- Measure: acceptance length, acceptance rate, tokens/sec

**Hypothesis:** α=0.5 blend will improve acceptance by 5-10% due to complementary information from different target layers.

### R2: Pruner-Confidence KV Routing (D3)
Does pruner-confidence-based KV routing improve draft quality at low-confidence positions?

**Test design:**
- Binary routing: relevance > 0.6 → target-conditioned KV, else → unconditioned KV
- Measure: per-position acceptance rate segmented by pruner confidence quartile
- Compare: acceptance at low-confidence positions with and without routing

**Hypothesis:** Unconditioned KV at low-confidence positions will improve acceptance by 10-20% (draft model explores freely rather than being anchored to potentially-wrong target conditioning).

### R3: Position-Weighted DDTree Budget (D4)
Does position-weighted DDTree expansion budget improve acceptance length?

**Test design:**
- `warmup_positions = 2`, `early_weight = 1.5`
- Measure: acceptance length distribution (focus on P50 and P90, not just mean)
- Compare: uniform budget vs weighted budget with same total node count

**Hypothesis:** P50 acceptance will improve by 3-5% due to better first-position predictions. P90 may decrease slightly (less diversity at later positions).

---

## Cross-References

- **Research 055 (Nemotron TriMode):** DFlare's block diffusion builds on the same DFlash foundation as our TriMode pipeline
- **Research 059 (MoE SD Co-Design):** DFlare's heterogeneous KV routing parallels MoE expert routing — both select representations based on context
- **Research 072 (DMax):** DFlare's aggressive parallel decoding aligns with DMax's approach to maximizing draft acceptance
- **Research 091 (SpecHop):** D2 (marginal fusion) could enhance SpecHop's multi-hop verification with per-hop conditioning
- **Research 131 (DiffusionBlocks):** DFlare's block-wise approach is related to DiffusionBlocks' block partitioning
- **Research 149 (FlashAR):** D2's multi-conditioning blend parallels FlashAR's dual-path consensus fusion
- **Research 153 (Thinking Pixel):** D3's pruner-confidence routing parallels Thinking Pixel's per-token expert routing

---

## Tasks

- [ ] (D2) Implement `dflare_fusion` feature gate — multi-conditioning marginal blend with K=2 sources
- [ ] (D2) Benchmark: acceptance length with K=2 vs K=1 conditioning, α ∈ {0.3, 0.5, 0.7}
- [ ] (D3) Implement `dflare_kv_routing` feature gate — pruner-confidence-based KV source selection
- [ ] (D3) Benchmark: per-position acceptance by pruner confidence quartile, with and without routing
- [ ] (D4) Implement `dflare_progressive_budget` feature gate — `PositionWeightedBudget` struct for DDTree
- [ ] (D4) Benchmark: acceptance length distribution with uniform vs weighted budget
- [ ] (D1, D5) Document existing mechanisms as DFlare analogs in code comments (no code change needed)
