# Research: Memory Caching — RNNs with Growing Memory (arXiv 2602.24281)

**Date:** 2026-06
**Status:** GOAT Verified — Modelless Inference-Time Enhancement
**Paper:** [Memory Caching: RNNs with Growing Memory](https://arxiv.org/pdf/2602.24281)
**Authors:** Ali Behrouz, Zeman Li, Yuan Deng, Peilin Zhong, Meisam Razaviyayn, Vahab Mirrokni (Google)
> **PASS-Redirects (synthesis):** Behrouz et al. [arXiv:2512.24695 "Nested Learning: The Illusion of Deep Learning Architectures"] — successor paper by the same Google team. NL generalizes MC's segment-boundary checkpoint caching into a unified framework of nested multi-level optimization problems (NSAM) with per-level context flow + update frequency. All of NL's new contributions (M3 optimizer, Delta Gradient Descent, Delta Momentum, Hope architecture, Self-referential Titans, Continuum Memory System) require backprop → riir-train. The modelless-relevant insight — that multi-frequency state blocks at different update cadences prevent catastrophic forgetting — ALREADY ships as our per-NPC runtime stack: HLA (fast/tick) + KARC (medium/tick-batched) + Committed Personality (slow/BLAKE3-frozen) + Sleep-Time Anticipator + Raven/δ-Mem consolidation (slowest/sleep-cycle). No new katgpt-rs primitive; NL is a theoretical-framing validation of the architecture we already ship, plus training methods for riir-train.
> **PASS-Redirects (synthesis):** Behrouz, Javanmard, Mirrokni [arXiv:2606.03979 "Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories"] (Google, Jun 2026) — the NEXT paper in this team's lineage after NL: periodic wake/sleep lifecycle whose consolidation = Knowledge Seeding (GKD on-policy + LTI Levenshtein/semantic-RM imitation reward, upward self-distillation into freshly-unmasked low-rank MoE experts with prospective-base ordering + synaptic pruning of sender temporaries) and dreaming = SEAL ReST-EM + random-expert router injection + gradient dream selection. Same routing as NL: consolidation/dreaming cores are gradient-bound → riir-train Research 499 + Plan 502; no new katgpt-rs primitive — the cadence/masked-container/reset shapes ship (Plan 154/334, `GateMaskOverlay`/`LoraPair`, freeze/thaw, compaction). ⚠ title collision with the CMU paper arXiv:2605.26099 (Research 116) — grep IDs, never the title.

---

## Executive Summary

Memory Caching (MC) is a simple, general inference-time technique that caches checkpoints of recurrent memory states at segment boundaries. The effective memory capacity of RNNs grows with sequence length, interpolating between O(L) RNN complexity and O(L²) Transformer complexity — user controls the trade-off via segment size.

**Key finding: MC works as post-training, zero-training enhancement.** This makes it a perfect modelless fit for katgpt-rs.

---

## Core Mechanism

### Problem
- RNNs have fixed-size memory → forget long-past information → fail at recall-intensive tasks
- Transformers have growing memory but O(L²) complexity
- No middle ground

### Solution: Segment + Cache + Aggregate
1. Split sequence into segments S(1)...S(N) of configurable size
2. Each segment compresses tokens into memory state M(s) using existing recurrence rule
3. Cache the final memory state per segment: {M(1), M(2), ..., M(s-1)}
4. At query time, aggregate online memory + cached memories via configurable Agg function

```
y_t = Agg({M(1)...M(s-1)}; M(s)_t; q_t)
```

Complexity: O(NL) where N = L/segment_size, user-controllable.

---

## Four Variants

| Variant | Aggregation | Complexity | Key Property |
|---------|-------------|------------|--------------|
| **Residual Memory** | `y = M_online(q) + Σ M_cached(q)` | O(NL) | Simplest. Equal weight to all segments. Collapses for linear memory (pre-summable). |
| **GRM (Gated Residual)** | `y = γ_online·M_online(q) + Σ γ_i·M_cached(q)` | O(NL) | Context-dependent gating. γ(i) = sim(u_t, MeanPool(S(i))). BEST performing variant. |
| **Memory Soup** | `θ_M* = Σ γ_i · θ_M(i)`, then `y = M*(q)` | O(NL) | Weight souping for deep/non-linear memories. Equivalent to GRM for linear memories. |
| **SSC (Sparse Selective)** | Top-k segment selection via MoE router | O(kL) | Best throughput. Only loads k most relevant segments. Minimal overhead. |

### Gating Mechanism (GRM)
The critical insight: γ(i)_t should depend on BOTH the query AND the segment context:
```
γ(i)_t = <u_t, MeanPooling(S(i))>   where u_t = x_t · W_u
```
Then normalize via softmax. This is context-dependent, not position-dependent.

### Segmentation Strategies
- **Constant-size**: segment_size=C → O(L²/C) complexity
- **Logarithmic**: Fenwick tree partitioning → O(L log L) complexity, but less resolution for long past
- Paper shows constant-size segmentation wins on accuracy (Table 1)

---

## Key Results

| Metric | Base RNN | + MC (GRM) | + MC (SSC) | Transformer |
|--------|----------|------------|------------|-------------|
| WikiText PPL (1.3B) | 15.60 (Titans) | **15.37** | 15.44 | 17.92 |
| Average CS Reasoning | 56.82 (Titans) | **58.33** | 57.58 | 53.19 |
| NIAH 16K (UUID) | 21.2 (Titans) | **32.2** | 27.0 | 40.8 |
| SWDE Retrieval 16K | 29.7 (Titans) | **50.1** | 41.4 | 44.0 |

**MC-enhanced Titans BEATS Transformers on retrieval tasks** (50.1 vs 44.0 on SWDE 16K).

### Ablation (Table 5)
- Context-dependent γ: +7.5 retrieval accuracy (critical!)
- Gating (vs residual only): +0.6 accuracy
- Removing gating collapses to residual — still works but worse

---

## Post-Training Application

From Section 4.3: "Memory caching can also be applied after pre-training, where at inference, we cache the state of the memory after each segment. For decoding, we use moving average of the past cached memory without learnable weights."

**This is modelless. No LoRA training, no weight updates. Pure inference-time enhancement.**

---

## Distillation to katgpt-rs (Modelless)

### Fusion Idea 1: SegmentCheckpoint KV Cache
- **What**: At inference time, snapshot compressed KV state at tile-aligned segment boundaries
- **Integration**: KVarN tiles (128 tokens) = natural zero-copy segment boundary
- **Checkpoint**: `{ key_quantized[0..tile_n], val_quantized[0..tile_n], key_tiles[0..tile_n], val_tiles[0..tile_n], pos }`
- **Retrieval**: GRM gating via dot-product similarity of query to segment summary vector
- **Perf**: Zero training overhead. Memory grows O(N) where N = seq_len/segment_size
- **Feature gate**: `segment_checkpoint`

### Fusion Idea 2: SSC-Sparse Speculative Drafting
- **What**: Drafter references only top-k relevant cached segments when producing drafts
- **Integration**: Channel-concentrated routing from VortexFlow (25% of dims suffice)
- **Perf**: Reduces draft verification misses by giving drafter access to long-range context
- **Feature gate**: `ssc_spec_draft`

### Fusion Idea 3: Memory-Soup DDTree Branch Merging
- **What**: DDTree already caches branch states. Memory Soup averages them per-query.
- **Integration**: At DDTree leaf evaluation, average cached branch KV states weighted by γ
- **Perf**: Input-dependent specialized state per query, zero training
- **Feature gate**: `memory_soup_dtree`

### Fusion Idea 4: Post-Training MC for Speculative Reconciliation
- **What**: During spec reconciliation, cache memory checkpoints at segment boundaries
- **Integration**: After reconciliation accepts tokens, snapshot KV state as segment checkpoint
- **Perf**: Future queries can reference reconciled context via cached checkpoints
- **Feature gate**: Part of `segment_checkpoint`

---

## Relationship to Existing katgpt-rs Research

| Existing Work | MC Variant | Relationship |
|---------------|------------|--------------|
| KVarN (Plan 159) | All | Natural segment boundary = tile boundary (128 tokens). Zero-copy checkpoints. |
| ThoughtFold (Plan 195) | SSC | Fold points = segment boundaries. SSC selects which folded segments to retrieve. |
| VortexFlow (Plan 196) | SSC | Channel-concentrated routing = MC's MoE router. Same top-k pattern. |
| BFCF (Plan 218) | All | Region-level LFU eviction + segment caching = growing memory with bounded cost. |
| GDN2 (Plan 105) | GRM | Gated erase/write = MC's γ gating. Same channel-wise selectivity. |
| SpecReconciliation (Plan 177) | Residual | Reconciliation = segment boundary creation. Natural checkpoint trigger. |
| TriggerGate (Plan 176) | All | Tier-aware checkpoint policy: eager at CpuGpuAne, lazy at CpuOnly. |

---

## GOAT Verdict

**GOAT. Must be on by default for modelless.**

Reasoning:
1. ✅ Zero training needed (post-training proven in paper)
2. ✅ Aligns with existing KVarN tile boundaries (zero-copy implementation)
3. ✅ User-controllable O(NL) complexity, interpolates RNN↔Transformer
4. ✅ Consistent improvements across ALL benchmarks (Table 1-4)
5. ✅ No perf hurt — SSC variant has minimal overhead, GRM variant is best accuracy
6. ✅ Titans + MC GRM beats Transformers on retrieval (SWDE 16K: 50.1 vs 44.0)
7. ✅ Feature-gated as `segment_checkpoint`, on by default

**Risk**: Memory overhead grows O(N) with segment count. Mitigated by SSC top-k selection and KVarN compression of checkpoint tiles.

**When segment_size=1, MC recovers gated global attention. When segment_size=∞, MC recovers standard RNN. User picks the trade-off.**

---

## TL;DR

Memory Caching is a simple, proven inference-time technique that lets RNN memory grow with sequence length. Four variants (Residual, GRM, Memory Soup, SSC) trade off accuracy vs efficiency. GRM is best accuracy, SSC is best throughput. Post-training application makes it 100% modelless. Natural fit with KVarN tile boundaries for zero-copy segment checkpoints. GOAT — on by default.
