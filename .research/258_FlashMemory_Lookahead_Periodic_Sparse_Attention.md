# Research 258: FlashMemory-DeepSeek-V4 — Lookahead Periodic Sparse Attention

> **Source:** [FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention](https://arxiv.org/abs/2606.09079) — Yan Wang, Qifan Zhang, Jiachen Yu, Tian Liang, et al. (Tencent / HKUST-GZ / Tsinghua), 2026-06-07
> **Date:** 2026-06-17
> **Status:** Active — GOAT verdict
> **Classification:** Public (katgpt-rs modelless inference primitive)
> **Related Research:** 176 (VortexFlow), 225 (MSA), 086 (RTPurbo), 145 (Wall Attention), 213 (Still Perceiver), 233 (Attention Matching), 109 (Shard), 063 (OCTOPUS), 100 (EGA spectral salience)
> **Related Plans:** TBD (pending GOAT gate)
> **Training redirect:** The paper's dual-encoder indexer training (BCE/Focal loss on pre-computed hidden states) → riir-train. This note distills only the modelless inference paradigm.

---

## TL;DR

FlashMemory introduces **Lookahead Sparse Attention (LSA)**: instead of scoring every KV block at every decode step, a Memory Indexer triggers **every τ=64 decode steps** to batch-evaluate which compressed KV chunks will be needed in the upcoming window, fetching only those from CPU (cold pool) into GPU (hot pool). The paper uses a **sigmoid threshold** (≥0.5) for selection — not rigid Top-k — and a **3-layer union routing** (layers 10, 12, 20 with OR-mode consensus). Results: 13.5% KV cache footprint, +0.6% accuracy, 90% memory reduction at 500K context.

**Distilled for katgpt-rs (modelless):** The periodic batch-scoring architecture + sigmoid threshold + multi-layer union routing are all inference-time patterns that work with ANY block scorer (VortexFlow centroid, MSA max-pool, EGA spectral salience). The "lookahead" framing reframes sparse attention from *reactive per-step scoring* to *amortized periodic batch-scoring with cached decisions*.

---

## 1. Paper Core Findings

### 1.1 The Core Insight — Periodic Predictive Fetching

Standard sparse attention scores ALL KV blocks at EVERY decode step. FlashMemory observes that >90% of decode steps are "context-independent" — the current token doesn't need historical KV. So: score KV importance **every τ steps** (not every step), and cache the selection decision for the intervening τ−1 steps.

### 1.2 Sigmoid Threshold Selection (not Top-k)

The paper replaces rigid Top-k with a **sigmoid-activated threshold**:
```
I_{t,s} = σ(Σ_h w_{l,h} · ReLU(q_{l,h} · K^IComp_s))    // sigmoid, not softmax
C^{MemComp}_t = { C^Comp_s | I_{t,s} ≥ 0.5 }              // threshold, not top-k
```
This selects a **dynamic number** of blocks per query — context-independent queries retrieve ~0 blocks, context-dense queries retrieve many. **Aligns with our "sigmoid never softmax" rule (AGENTS.md).**

### 1.3 3-Layer Union Routing (OR-mode)

Indexers on layers 10, 12, 20 independently score. A block is fetched if **ANY** layer predicts score ≥ 0.5:
```
C^{MemComp}_t = ∪_{l ∈ {10,12,20}} { C^Comp_s | I^{(l)}_{t,s} ≥ 0.5 }
```
OR-mode is deliberately conservative — it's a "safety-net" union that avoids false-negative drops.

### 1.4 Memory Hierarchy — CPU Cold Pool / GPU Hot Pool

- **CPU cold pool**: all compressed KV entries (pre-computed, frozen)
- **GPU hot pool**: only the fetched subset (updated every τ steps)
- The native Lightning Indexer then operates on the hot pool for fine-grained Top-k

### 1.5 Results

| Benchmark | DS-V4-Flash | FM-DS-V4 | Memory |
|-----------|-------------|----------|--------|
| LongBench-v2-L (493K) | 68.1% (1.80 GB) | **70.0%** (0.18 GB) | 90% reduction |
| RULER (512K) | 88.3% (1.87 GB) | **89.6%** (0.18 GB) | 90% reduction |
| Average | 76.9% (0.93 GB) | **77.5%** (0.10 GB) | 86.5% reduction |

### 1.6 Failure Mode — Dense Global Memory (MRCR)

On MRCR (Multi-Range Context Retrieval), accuracy drops from 76.0% to 48.0%. Even with oracle golden chunks at 50%, accuracy still drops ~2%. **Some tasks require dense global memory that sparse fetching fundamentally cannot serve.** This is an important cautionary tale.

### 1.7 Length Generalization Ceiling

Generalizes safely up to **2× training context length**. Beyond that, accuracy collapses (OOD positional embeddings).

---

## 2. Distillation — Modelless Path

### 2.1 What Maps Directly (the modelless inference paradigm)

| FlashMemory Concept | katgpt-rs Equivalent | Status |
|---------------------|---------------------|--------|
| Periodic refresh every τ steps | NOT shipped — all our scorers run per-step | ⚠️ Gap |
| Sigmoid threshold selection | EGA sigmoid gate (R100), sigmoid margin (R061) | ✅ Pattern exists |
| 3-layer union routing | Multi-head attention (implicit), no explicit OR-mode | ⚠️ Gap |
| Block max-pool scoring | MSA (R225), VortexFlow centroid (R176) | ✅ Shipped |
| CPU cold / GPU hot tier | Memory tier concept exists (Plasma/Hot/Warm/Cold) | ✅ Conceptual |
| Compressed KV entries | OCTOPUS (R063), SpectralQuant (R039), Shard (R109) | ✅ Shipped |

### 2.2 The Modelless Primitive — Periodic Batched Sparse Scoring

The distilled primitive is a **control-flow change**, not a new scorer:

```rust
// Current: score every step
for step in decode_loop {
    let scores = score_all_blocks(query, kv_cache);  // O(seq_len) per step
    let selected = top_k_or_threshold(scores);
    attend(query, &kv_cache[selected]);
}

// FlashMemory-distilled: score every τ steps, cache decision
let mut cached_selection: Option<Vec<usize>> = None;
for step in decode_loop {
    if step % tau == 0 || cached_selection.is_none() {
        let scores = score_all_blocks(query, kv_cache);  // O(seq_len) per τ steps
        let selected = sigmoid_threshold_select(scores, 0.5);  // sigmoid, not top-k
        cached_selection = Some(selected);
    }
    attend(query, &kv_cache[cached_selection.unwrap()]);
}
```

**Amortization gain:** scoring cost reduced by factor τ (e.g., τ=64 → 64× less scoring compute).

### 2.3 Sigmoid Threshold vs Top-k

Our current sparse attention uses Top-k (fixed budget). FlashMemory proves sigmoid threshold is better for variable-density contexts:
- Context-independent queries → ~0 blocks selected (free)
- Context-dense queries → many blocks selected (accurate)

The threshold `I_{t,s} ≥ 0.5` is equivalent to `sigmoid(score) ≥ 0.5` which is equivalent to `score ≥ 0` — a natural decision boundary.

### 2.4 Multi-Layer Union Routing

Instead of scoring at one layer, score at K strategic layers and union the selections:
```rust
fn union_select(layers: &[LayerScore], threshold: f32) -> Vec<usize> {
    let mut selected = HashSet::new();
    for layer in layers {
        for (block_idx, &score) in layer.scores.iter().enumerate() {
            if sigmoid(score) >= threshold {
                selected.insert(block_idx);
            }
        }
    }
    selected.into_iter().collect()
}
```
This is a "safety-net" — any layer that thinks a block is important gets it fetched.

### 2.5 What's NOT Modellessly Distillable

- **The trained dual-encoder indexer** — requires supervised training on pre-computed labels → riir-train
- **Cross-Layer Majority Voting for golden labels** — training data pipeline → riir-train
- **True lookahead prediction** — requires a learned mapping from current state to future KV needs. Modellessly, we can only do *amortized current-state scoring*, not future-state prediction.

---

## 3. Fusion Ideas

### F1: VortexFlow × FlashMemory — Periodic Vortex Scoring

Replace VortexFlow's per-step block scoring with periodic batch scoring (every τ=64 steps). Use VortexFlow's centroid dot-product as the scorer, but cache the selection. **Gain:** 64× less scoring overhead, same selection quality (KV importance doesn't change much across 64 decode steps in practice).

### F2: EGA × FlashMemory — Sigmoid Energy Gate + Periodic Refresh

EGA's energy-gated sigmoid (`g = σ(α · (ẽ − τ))`) IS the FlashMemory sigmoid threshold. Combine: use EGA's z-normalized energy score as the periodic batch scorer, refresh every τ steps. **Gain:** spectral salience + amortized cost.

### F3: OCTOPUS × FlashMemory — Compressed KV Tier Management

OCTOPUS compresses KV to octahedral encoding. FlashMemory's CPU cold pool / GPU hot pool is the natural tier boundary: compressed OCTOPUS entries live in CPU cold, fetched subset lives in GPU hot. **Gain:** OCTOPUS compression × tiered access = ultra-low memory for long context.

### F4: Wall Attention × FlashMemory — Gate-Derived Periodic Scoring

Wall Attention's diagonal forget gates produce per-channel retention scores. Use these as the periodic batch scorer: every τ steps, compute Wall gate prefix sums, threshold the blocks whose gates have decayed below τ_decay. **Gain:** zero-overhead scoring (gates already computed) + periodic refresh.

---

## 4. Verdict: GOAT

**One-line reasoning:** FlashMemory's modelless distillation (periodic batch-scoring + sigmoid threshold + multi-layer union routing) provides a provable gain (τ× scoring cost reduction, memory tiering) over our per-step Top-k sparse attention — but it's an optimization of existing sparse attention, not a new capability class.

**GOAT gate criteria (before promoting to default):**
- G1: Periodic scoring with τ=64 must show <1% quality degradation vs per-step scoring on needle-in-haystack
- G2: Sigmoid threshold must match or beat Top-k at equivalent average budget
- G3: Multi-layer union routing must not inflate budget >2× vs single-layer
- G4: Scoring cost reduction must be measurable (≥10× fewer scorer calls per 1K tokens)
- G5: MRCR-style dense-memory tasks must degrade gracefully (not collapse)

---

## 5. What Stays Where (3-Repo Discipline)

| Component | Repo | Why |
|-----------|------|-----|
| Periodic batch-scoring framework | katgpt-rs (MIT) | Generic sparse attention control flow |
| Sigmoid threshold selector | katgpt-rs (MIT) | Generic selection primitive |
| Multi-layer union router | katgpt-rs (MIT) | Generic multi-head consensus |
| Game-side τ tuning (per-NPC context density) | riir-ai (private) | Game-specific parameterization |
| Trained dual-encoder indexer | riir-train (private) | Training know-how |

---

## 6. Limitations and Failure Modes (from paper §3.3)

1. **Context-independent overhead leak** — sigmoid gater leaks marginal background probability, accumulating false positives at extreme lengths. Fix: tighter threshold or entropy-based collapse detection.
2. **MRCR dense-memory breakdown** — some tasks need dense global attention. Sparse fetching fundamentally cannot serve them. Must detect and fall back to full attention.
3. **Length generalization ceiling** — safe up to 2× training length. Beyond that, OOD positional embeddings cause collapse.

---

## TL;DR

**Verdict: GOAT.** FlashMemory's modelless distillation is "switch from per-step Top-k sparse attention to periodic (every τ=64 steps) batch-scoring with sigmoid threshold selection and multi-layer union routing." The periodic refresh amortizes scoring cost by τ×, the sigmoid threshold enables variable-budget selection (0 blocks for context-independent queries), and the union routing provides safety-net redundancy. Fusion targets: VortexFlow (periodic centroid scoring), EGA (sigmoid energy gate as scorer), OCTOPUS (compressed KV tier management), Wall Attention (gate-derived scoring). The trained indexer → riir-train. Failure mode cautionary tale: dense-memory tasks (MRCR) collapse — must detect and fall back to full attention. No files beyond this note per GOAT protocol; plan creation deferred pending GOAT gate validation.
