# Research 177: Domino — Decoupled Causal Modeling for Speculative Decoding (Modelless Distillation)

> **Source:** Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding (Huang et al., 2026, arXiv:2605.29707)
> **Date:** 2026-06
> **Status:** GOAT Verdict — ✅ GAIN (Modelless Path)

---

## TL;DR

Domino decouples causal dependency modeling from autoregressive draft execution in speculative decoding. A parallel draft backbone produces preliminary distributions for the whole block, then a lightweight GRU + low-rank correction head refines them with prefix-dependent causal info. Result: 5.49× speedup (Transformers), 5.8× throughput (SGLang), beating DFlash by ~17% acceptance length improvement at only +2.8% latency cost.

**Our verdict: GOAT for modelless.** The decoupling pattern (parallel base + cheap sequential correction) maps directly to our DDTree + ConstraintPruner architecture. We don't need the GRU or LoRA training — we need the **pattern**.

---

## Paper Core

### The Problem
Speculative decoding faces a quality-cost tradeoff:
- **Autoregressive drafters** (EAGLE-3): Good causal modeling, but γ sequential forward passes × LM head = linear cost growth
- **Parallel drafters** (DFlash): Low cost (1 forward pass), but no intra-block causal dependencies → lower acceptance length

### The Solution: Domino
1. **Parallel draft backbone** (DFlash architecture): Produces base logits L_base for all B positions in one forward pass
2. **Domino head**: Lightweight causal correction
   - GRU causal encoder summarizes preceding draft tokens into causal state S_{i-1}
   - Low-rank (r=256) MLP produces logit-space residual: ΔL = W2·σ(W1·[H_i; S_{i-1}])
   - Final logits: L_i = L_base_i + ΔL_i
3. **Teacher-forced training**: Feed ground-truth (not self-generated) tokens to causal encoder
4. **Base-anchored curriculum**: Linearly anneal loss from base-anchored to final, preventing backbone collapse

### Key Numbers
| Metric | DFlash | Domino | Δ |
|--------|--------|--------|---|
| Accept length (Qwen3-8B avg) | 6.06 | 7.17 | +18.3% |
| Speedup (Transformers, T=0) | 4.66× | 5.49× | +17.8% |
| Throughput (SGLang, conc=2) | 3.7× | 5.1× | +37.8% |
| Added Parameters | 0 | +56M (+5.3%) | — |
| Added latency | 0 | +2.8% | Negligible |
| Low-rank dim (correction) | — | r=256 | — |
| GRU hidden dim | — | 1024 | — |

---

## Modelless Distillation: The Decoupling Pattern

### What We Extract (No Training Required)

The paper's core insight is not the GRU or the LoRA head — it's the **decoupling pattern**:

```
Base (parallel, cheap) + Correction (sequential, lightweight) = Near-autoregressive quality at parallel cost
```

This pattern applies to our existing architecture in three concrete ways:

### 1. DominoPruner: Prefix-Conditioned Pruning (DDTree Layer)

**Current:** `ConstraintPruner::is_valid(depth, token, parent_tokens)` checks each token independently. `ScreeningPruner::relevance()` scores each token independently.

**Domino insight:** At depth i, the pruner should consider the *joint validity* of the prefix. A token that's valid at depth 3 might be invalid when the depth-1 token is X but valid when it's Y. We already have `parent_tokens` but don't use it efficiently.

**Modelless DominoPruner pattern:**
```rust
/// Domino-style prefix-conditioned pruner.
///
/// Phase 1 (parallel/base): Run existing ConstraintPruner for all tokens at each depth → base scores
/// Phase 2 (sequential/correction): For each depth, compute prefix-conditioned correction
///   based on the specific parent path taken in DDTree's best-first search.
///
/// The correction is O(1) per node (hash lookup of prefix pattern) instead of O(depth).
pub trait DominoPruner: ConstraintPruner {
    /// Compute prefix-conditioned correction to base validity.
    /// Returns true if the correction flips the base decision (base was wrong).
    /// This is the modelless equivalent of Domino's low-rank logit correction.
    fn causal_correction(&self, depth: usize, token: usize, prefix: &[usize], base_valid: bool) -> bool;
}
```

This is modelless because the "correction" is a deterministic hash-based lookup of known prefix-token patterns (from Sudoku rules, Rust syntax rules, game constraints), not a learned parameter.

### 2. Acceptance-Aware Tree Pruning (Verification Layer)

**Domino insight:** Position i's correction only matters if positions 1..i-1 are accepted. The paper uses teacher forcing for this reason — only train on the accepted-prefix regime.

**Modelless application:** In DDTree's best-first search, when evaluating whether to expand a node at depth d, we should weight the expansion priority by the *cumulative acceptance probability* of the path, not just the marginal probability. This is equivalent to Domino's teacher-forced training logic but implemented as a tree search heuristic:

```rust
/// Domino-weighted tree score: penalize deep branches unless their prefix is strong.
/// Maps to Domino's "only correct on accepted prefix" insight.
fn domino_score(base_score: f32, depth: usize, prefix_scores: &[f32]) -> f32 {
    // Product of prefix acceptance probs (teacher-forced regime)
    let prefix_strength: f32 = prefix_scores.iter().product();
    base_score * prefix_strength.powi(depth as i32)  // deeper = more selective
}
```

### 3. Logit-Space Residual Correction (DFlash Layer)

**Domino insight:** Correct logits in logit space, not hidden space. Hidden-space correction requires re-running the full LM head per position. Logit-space correction is O(vocab × r) per position.

**Modelless application:** Our `dflash_predict` produces independent marginals. We can apply a cheap, deterministic, **prefix-conditioned logit correction** without any model forward pass:

```rust
/// Modelless Domino correction: adjust draft marginals based on prefix.
/// No GRU, no learned parameters — just deterministic pattern-based adjustment.
///
/// For code: if prefix is "fn ", boost '{' probability and suppress ';'
/// For math: if prefix is "x =", boost comparison operators
/// For Sudoku: if row has digit d, zero out d in same row positions (we already do this)
fn domino_logit_correction(
    marginals: &mut [Vec<f32>],  // per-depth distributions
    prefix: &[usize],            // accepted tokens so far
    correction_table: &PrefixCorrectionTable,  // pre-computed lookup
) {
    for depth in 0..marginals.len() {
        if depth == 0 { continue; }  // first position: no prefix to condition on
        let correction = correction_table.lookup(depth, &prefix[..depth.min(prefix.len())]);
        // Apply as additive logit correction (Domino's ΔL pattern)
        for (i, c) in correction.iter().enumerate() {
            marginals[depth][i] += c;
        }
        // Re-normalize
        let sum: f32 = marginals[depth].iter().sum();
        for v in marginals[depth].iter_mut() { *v /= sum; }
    }
}
```

---

## GOAT Verdict

### Per 003_Commercial_Open_Source_Strategy_Verdict.md

| Criterion | Verdict | Reasoning |
|-----------|---------|-----------|
| **Modelless first** | ✅ | All three distillations are inference-time only. No LLM training. |
| **Engine/Fuel split** | ✅ | Pattern goes into katgpt-rs (MIT engine). Learned version (if any) goes to riir-ai (fuel). |
| **No perf hurt** | ✅ | PrefixCorrectionTable is O(1) lookup. DominoPruner correction is one hash check. No hot-path allocation. |
| **SOLID/DRY** | ✅ | New trait extends existing ConstraintPruner. No duplication. |
| **Tests/examples** | ✅ | Before/after DDTree acceptance rates with/without Domino correction on Sudoku + code examples. |
| **Default ON if GOAT+gain** | ✅ | Feature-gated as `domino_correction`, default on after benchmark proves no regression. |

### Why This Is GOAT

1. **DFlash is our backbone** — Domino builds directly on DFlash, which is already our core drafting mechanism
2. **The pattern is free** — The decoupling insight (parallel base + cheap sequential correction) costs almost nothing at inference time
3. **Teacher-forced alignment** — Our ConstraintPruner already operates in the "accepted-prefix regime" (prunes before verification). Domino's training insight validates our architecture
4. **Base-anchored curriculum maps to bandit** — Our bandit infrastructure can implement the same "anchor to base, gradually shift" pattern

### What We DON'T Take

- **GRU causal encoder**: Requires training. Not modelless.
- **Learned low-rank correction head**: Requires training + LoRA. Goes to riir-ai (model-based path).
- **Fused Triton kernels**: GPU-specific. Our CPU path doesn't need this.
- **DFlash backbone retraining**: We already have our DFlash implementation.

---

## Existing Related Work in katgpt-rs

| Component | Relation to Domino |
|-----------|-------------------|
| `dflash_predict` | Domino's parallel backbone (identical) |
| `dflash_predict_ar_with` | Domino's autoregressive path (what Domino improves upon) |
| `build_dd_tree_pruned` | Tree where DominoPruner would slot in |
| `SpeculativeVerifier` trait | Verification layer where acceptance-aware scoring applies |
| `ConstraintPruner` trait | Base that DominoPruner extends |
| `ScreeningPruner` trait | Relevance scoring that prefix-conditioned correction enhances |
| `spechop/` | Continuous multi-hop speculation — complementary |
| `spec_reconciliation/` | Speculative reconciliation — Domino's correction is a form of reconciliation |
| `d2f.rs` | Discrete diffusion forcing — parallel drafting alternative to DFlash |
| `diffusion_sampler.rs` | Block diffusion — related parallel approach |

---

## Fusion Ideas (Creative)

### 1. Domino + ThoughtFold
ThoughtFold (Plan 195) folds reasoning chains. Domino's prefix-conditioned correction could fold *within* a thought chain — correcting draft predictions based on the *already-folded* prefix rather than the full unfolded chain. This gives O(1) correction cost on folded chains.

### 2. Domino + VortexFlow
VortexFlow (Plan 196) routes KV cache through sparse channels. Domino's correction could be *channel-routed*: different correction tables for different semantic channels (code vs math vs natural language). The router decides which correction to apply.

### 3. Domino + TriggerGate
TriggerGate (Plan 176) gates between ANE/GPU/CPU. Domino's correction cost is so low (O(vocab × r) per position) that it could run on **ANE** while the parallel backbone runs on GPU. The "decoupling" becomes a **hardware decoupling** — not just conceptual.

---

## TL;DR

Domino's core insight — **parallel base + cheap sequential correction = near-AR quality at parallel cost** — is a GOAT pattern for katgpt-rs. We extract the decoupling pattern as three modelless mechanisms: DominoPruner (prefix-conditioned constraint), acceptance-aware tree scoring, and logit-space residual correction. No training needed. Feature-gate as `domino_correction`, default ON after benchmark validation.
