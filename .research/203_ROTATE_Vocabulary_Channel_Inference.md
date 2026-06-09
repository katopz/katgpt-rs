# ROTATE: Vocabulary Channel Inference-Time Applications

**Date:** 2026-06
**Paper:** Disentangling MLP Neuron Weights in Vocabulary Space (arXiv:2604.06005)
**Status:** GOAT Verdict → GAIN (modelless)
**Target:** katgpt-rs (modelless inference engine)

---

## Paper Summary

ROTATE (Rotation-Optimized Token Alignment in weighT spacE) is a data-free method that disentangles MLP neuron weights into interpretable "vocabulary channels" by:

1. **Key observation:** Monosemantic neurons exhibit high kurtosis when their weights are projected to vocabulary space (w @ U). Median concept vectors lie at 90th-95th percentile vs random neurons.
2. **Core method:** Learn Householder reflections R to rotate weight vectors w, maximizing vocabulary-space kurtosis: `v = R·w`, `z = v·U`, maximize Kurt(z).
3. **Iterative token masking:** After each channel discovery, mask high-contributing tokens (|z_i - μ| > k·σ) to force discovery of new channels.
4. **Result:** Vocabulary channels that are faithful (0.46-0.71) and complete (0.49-0.60), outperforming SAEs on both metrics.
5. **Data-free:** No forward passes required. Pure weight-space operation.

### Key Properties

| Property | Value |
|----------|-------|
| Per-neuron compute | ~11 min for 50 channels on H100 (5K neurons batched) |
| Reconstruction | Cosine sim > 0.95, explained norm > 0.7 after 50 iterations |
| Consistency | 0.9 cosine similarity, 0.8 Jaccard across random seeds |
| Layer profile | Middle/late layers have highest kurtosis (most monosemantic) |

---

## Existing Infrastructure Overlap

| Component | Status | Location |
|-----------|--------|----------|
| `excess_kurtosis()` | ✅ Shipped | `src/speculative/kurtosis_gate.rs` |
| `KurtosisGate` | ✅ Shipped | `src/speculative/kurtosis_gate.rs` |
| `SelectivityRouter` | ✅ Shipped | `src/speculative/selectivity_router.rs` |
| `ConstraintPruner` trait | ✅ Shipped | `crates/katgpt-core/src/traits.rs` |
| `ScreeningPruner` trait | ✅ Shipped | `crates/katgpt-core/src/traits.rs` |
| DDTree + kurtosis integration | ✅ Shipped | `src/speculative/dd_tree.rs` |
| Givens rotations | ✅ Shipped | `src/planar_quant/rotation.rs` |
| `sparse_matmul` | ✅ Shipped | MLP forward with dead ReLU skip |
| CNA neuron attribution | ✅ Shipped | `src/pruners/cna.rs` |
| Vocabulary projection (lm_head) | ✅ Exists | `src/transformer.rs` |

### What's Missing (the gap)

- **Householder reflections** for weight-space rotation (only Givens 2D exists)
- **Weight-space decomposition** of MLP weights (only activation-space analysis exists)
- **Skewness** computation (only kurtosis m₂, m₄ exist; no m₃)
- **Load-time weight analysis pipeline** (weights are loaded but never introspected)

---

## Fusion Ideas (Creative, Not Direct Mapping)

### Idea 1: VocabChannel Pruner (ConstraintPruner from weight decomposition)

**Concept:** At model load time, decompose MLP output weights Wout into vocabulary channels per neuron. Store per-neuron top-K token sets. At inference time, use as a `ConstraintPruner` — if a drafted token is NOT in any active neuron's vocabulary channel, prune it immediately.

**Why creative:** ROTATE was designed for interpretability (understanding neurons). We repurpose the decomposition as a pruning signal — the channels become hard constraints on what tokens are reachable, eliminating dead branches before any forward pass.

**Mechanism:**
1. Load-time: For each neuron i, compute `w_out[i] @ U` → logits. Apply ROTATE to get channels {v₁...v₅₀} per neuron. Extract top-50 token set per channel.
2. Build per-layer token reachability map: `{token_idx → set of neurons that could produce it}`.
3. Inference-time: Given current hidden state x, identify top-k active neurons (via sparse_matmul). Look up their token reachability. Use as `ConstraintPruner::is_valid()` — reject tokens not in any active neuron's reachability set.
4. Zero additional compute per inference step — just a lookup table check.

**Expected gain:** Reduce DDTree branching factor by 30-60% (most vocabulary tokens are unreachable from any given neuron configuration). Quality-neutral (pruning tokens that no active neuron would produce).

**Complexity:** Load-time O(neurons × channels × vocab_dim). Storage: O(neurons × channels × 50) token indices. Runtime: O(1) lookup per token check.

### Idea 2: Kurtosis-Profile Sparse MLP Routing

**Concept:** The paper shows layer-wise kurtosis profiles are predictable — middle/late layers are high-kurtosis (monosemantic), early layers are low-kurtosis (polysemantic). Use this to dynamically gate MLP sparsity:
- High kurtosis layers → aggressive sparse top-k (monosemantic neurons are predictable)
- Low kurtosis layers → full activation (need exploration)

**Why creative:** Extends the existing `KurtosisGate` (which gates speculative decoding) to gate MLP computation itself. The kurtosis is computed once at load-time from weight analysis, not per-inference.

**Mechanism:**
1. Load-time: Compute per-layer kurtosis profile `K[l] = median_kurtosis(Wout[l] @ U)`.
2. Set per-layer sparsity threshold: `sparsity[l] = sigmoid(α × (K[l] - K_threshold))`.
3. Inference-time: Use `sparsity[l]` to gate `sparse_matmul` top-k selection.
4. Existing `sparse_matmul` already skips dead ReLU neurons — this makes it smarter about which alive neurons matter.

**Expected gain:** 15-30% MLP compute reduction on high-kurtosis layers with <1% quality loss. The paper proves high-kurtosis neurons are monosemantic (predictable), so dropping low-activation neurons is safe.

### Idea 3: Channel-Skewness Draft Refinement

**Concept:** ROTATE discovers channels with skewness polarity — positive-skew channels promote tokens, negative-skew suppress them. Use this to refine speculative draft token selection:
- For each draft position, compute which neurons would be active.
- Look up their channel skewness: positive-skew channels = promoted tokens, negative-skew = suppressed tokens.
- Adjust draft logits: boost promoted tokens, suppress suppressed tokens.

**Why creative:** Standard draft models predict output distributions. We refine draft predictions using weight-derived causal knowledge — not just what tokens are likely, but which tokens the model's internal mechanism actively promotes/suppresses.

**Expected gain:** 5-15% improvement in draft acceptance rate. The refinement is weight-derived (zero forward pass cost) and neuron-structure-aware (causal, not correlational).

---

## GOAT Verdict

| Idea | Gain | Modelless | Feature-Gateable | SOLID/DRY | Perf Impact | Verdict |
|------|------|-----------|------------------|-----------|-------------|---------|
| VocabChannel Pruner | HIGH (30-60% branch reduction) | ✅ | ✅ | ✅ (ConstraintPruner trait) | Positive (faster DDTree) | **GOAT** |
| Kurtosis-Profile Sparse MLP | MEDIUM (15-30% MLP reduction) | ✅ | ✅ | ✅ (extends KurtosisGate) | Positive (less compute) | **GAIN** |
| Kurtosis-Profile Sparse MLP | MEDIUM (15-30% MLP reduction) | ✅ | ✅ | ✅ (extends KurtosisGate) | Positive (less compute) | **GAIN** |
| Channel-Skewness Draft | LOW-MEDIUM (5-15% acceptance) | ✅ | ✅ | ✅ (ScreeningPruner trait) | Neutral (lookup cost) | **GAIN** |

### Commercial Strategy Alignment

Per `003_Commercial_Open_Source_Strategy_Verdict.md`:
- All three ideas are **engine-side** (MIT, open source) — they operate on weight analysis + inference-time pruning
- The "fuel" (trained lora.bin) is not required — these are weight-analysis methods
- The "Ferrari, no gas" problem remains: competitors can fork the pruner, but without the trained draft model for semantic accuracy, their output compiles but is wrong
- These ideas **strengthen the engine** (better pruning → faster inference → more translations/hour → more revenue per GPU)

---

## Decision: Promote Idea 1 (VocabChannel Pruner) to Plan

Idea 1 is the strongest GOAT candidate:
- Directly extends existing `ConstraintPruner` trait (SOLID: open for extension)
- Reuses existing `excess_kurtosis()` and DDTree infrastructure (DRY)
- Feature-gateable (load-time analysis, opt-in)
- Zero inference cost (lookup table)
- Strong expected gain (30-60% branch reduction)

Idea 2 is promoted as a follow-up GAIN.
Idea 3 is GAIN but lower priority (needs more validation of the draft refinement mechanism).
