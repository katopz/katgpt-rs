# Research 436: FlashMemory-DeepSeek-V4 — Lookahead Periodic Sparse Attention

> **Source:** [FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention](https://arxiv.org/abs/2606.09079) — Yan Wang, Qifan Zhang, Jiachen Yu, Tian Liang, et al. (Tencent / HKUST-GZ / Tsinghua), 2026-06-07
> **Date:** 2026-06-17
> **Status:** Gain — actionable (validated 2026-08-13). Originally PASS (micro-transformer scale, no long-context serving). Revised to Gain after verifying: 4090 (24GB) + Kimi-K3-0.40B (395M, hybrid MLA/KDA, stack already has `kimi_k3/decoder_layer.rs`) + Bonsai dspark-Q4_1 (1.95GB, Qwen3.5, 256K native context) form a concrete validation path. FlashMemory's 90% KV reduction directly solves the "256K context KV cache (≈67GB) doesn't fit in 24GB VRAM" problem. **M3 validation COMPLETE (2026-08-13):** G1 PASS (Bench 021, real weights, cos ≥ 0.96 at 128-8192 tokens), G3 PASS (169 tests), G4 PASS (Bench 022, alloc-free steady state), G5 M3 scaling curve DONE (Bench 024, 74% reduction at ≤8K — the 90% requires 256K on 4090), NIAH diagnostic DONE (Bench 023, pattern preservation r=0.965). Plan 337 filed (riir-train indexer training recipe). Phase 2 scale test + G2 COMPLETE (2026-08-14, Bench 671 on 4090: 1.8x decode at 64K, 256K measured; G5 ~50% synthetic - data-dependent, real 74% <=4K). Issue 584 resolved + removed (2026-08-15) - Benches 021-026 + 671 are the record.
> **Classification:** Public (katgpt-rs modelless inference primitive)
> **Related Research:** 176 (VortexFlow), 225 (MSA), 086 (RTPurbo), 145 (Wall Attention), 213 (Still Perceiver), 233 (Attention Matching), 109 (Shard), 063 (OCTOPUS), 100 (EGA spectral salience)
> **Related Plans:** riir-train Plan 337 (indexer training - in flight; Issue 584 GOAT gate evaluated 2026-08-14)
> **Related Issues:** 584 (FlashMemory sparse attention validation on 4090 + Kimi-K3-0.40B → Bonsai dspark-Q4_1)
> **Training redirect:** The paper's dual-encoder indexer training (BCE/Focal loss on pre-computed hidden states) → riir-train. This note distills only the modelless inference paradigm. Issue 584 Phase 2 will file a riir-train Plan for the indexer training recipe scaled to a single 4090.
>
> **PASS-Redirects (synthesis):** Fu et al. [arXiv:2606.04511 "SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference"] — concurrent lookahead sparse attention (NVIDIA/ByteDance/MIT, same month as FlashMemory). Adds a trained 4th "Forecast" projection (KL-divergence, 0.41% params, frozen backbone) for one-layer-ahead block selection + async UVA CPU→GPU prefetch overlap. The lookahead CONCEPT is already distilled here (R436); SparDA's genuinely-novel piece (trained Forecast) is a training recipe → riir-train, not modelless; the UVA persistent kernel is H100/A100 PCIe Gen5 serving infrastructure, out of scope for the CPU/modelless stack. Per-layer lookahead (SparDA) vs periodic τ=64 refresh (FlashMemory) are two solutions to the same amortize-selection-cost problem — neither applies at micro-transformer scale (n_layer=1). Closest substrate: `VortexFlow::forward_indexer` takes the attention Query (not a separate Forecast); `PerGroupTopKRouter` already does per-GQA-group routing.
> **Redirect update (2026-09-06, post-FlashMemory-shipping):** both dismissal grounds above are superseded — FlashMemory SHIPPED (Issue 584 closed; Bench 671: 1.8× decode at 64K on 4090), so "micro-transformer scale" no longer describes the corridor, and the 4090 lane IS a PCIe machine with the KV wall this corridor exists for. Full per-track distillation: katgpt-rs **Research 539**. Filed outputs: katgpt-rs **Issue 730** (deterministic prefill KV-offload double-buffer — the prefill wall needs no learner), riir-train **Issue 524** (Plan 337 recipe deltas: soft-target BCE, max-pool labels, FMID mass storage), riir-ai **Issue 880** (belief-Forecast Warm-tier prefetch fusion idea).
> **PASS-Redirects (synthesis):** Lintai Hou [arXiv:2609.02881 "Graph Machine: Towards Better Pretraining via Edges"] — GM-2 replaces 75% of dense layers with sparse layers that retrieve 2–4 of 4,096 tokens per KV head via a TRAINED pointer/edge state (persistent cross-layer edge indices+weights, differentiable 2-hop referral). Corroborates this corridor's headroom datum (marginal loss delta at 0.6B, 19% less ref+att compute) but the load-bearing component — the referral projections/temperatures/mixing matrices — is gradient-learned during from-scratch pretraining, so it cannot retrofit onto served dense checkpoints (our serving is GGUF-checkpoint-bound; RAT+/FlashMemory stay the consumable shapes). The modelless-extractable pieces already ship: top-s sparsify ≈ top-k (PKM/radix-select), the edge-factor prior softmax(qk + log w) ≈ additive attention-logit bias (SP-KV `GateBias`, katgpt-kv Issue 727).

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

## 5. What Stays Where (4-Repo Discipline)

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

> **PASS-Redirects (synthesis):** Bertsch et al. [arXiv:2608.10296 "Cracks in the Foundation: Seemingly Minor Architectural Choices Impact Long Context Extension"] — expectation calibration for the Bonsai dspark (Qwen3.5) target: QK-norm architectures (Qwen 3 family = headwise QK norm + GQA) place less attention on needle tokens at prefill and carry less sink mass, so Plan 337 indexer training data will show weaker/diffuser needle signatures than a Llama-class model would — the same reduction ratio may need more lenient sigmoid thresholds. (Bonsai 256K context is native, so the paper extension findings do not apply.) Compound-degradation recipe → riir-train Research 420; no action here.

> **PASS-Redirects (synthesis):** Kim, Jin & Kim [arXiv:2608.14333 "Beyond Capacity: Scalable MoE LLM Inference via High-Bandwidth Flash with Direct GPU and HBM Paths"] — the expert-weight twin of SparDA's one-layer-ahead KV-block prefetch: lookahead selection + async weight fetch overlapped with preceding compute, but with EXACT early top-k (reformulated router logits α·(P_base + P_attn), α rank-invariant under positive scaling) instead of prediction. Confirms the periodic-lookahead/async-prefetch pattern generalizes beyond attention blocks to expert weights. Hardware (HBF stacks, UCIe dual-path) out of scope; our Kimi router's additive noaux_tc bias falls in the paper's own excluded class, so no action here.
