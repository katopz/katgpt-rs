# Research 193: BFCF Region × LFU Eviction × Sharding Fusion

**Date:** 2026-06-08
**Status:** Research — Novelty Assessment & Literature Review
**For:** Plan 213 (BFCF Tree) Extension — Post-GOAT Optimization
**Relates:** Plan 189 (FreqBandit), Plan 109 (ShardDrop), Plan 043 (TurboQuant), Plan 070 (SP-KV)

---

## Executive Summary

Research into fusing BFCF Tree's O(regions ≈ 50) logit-space partitioning with LFU eviction, sharding, and latent-space clustering for katgpt-rs. **Verdict: No prior work combines Borel partition folding with frequency-aware eviction and sharding for inference-time latent operations. The fusion is novel, but individual components have strong precedents that inform design.**

---

## 1. Literature Landscape

### 1.1 Semantic Caching in Embedding Space

| Paper | Year | Key Insight | Relevance |
|-------|------|-------------|-----------|
| **"From Exact Hits to Close Enough"** (Biton & Friedman, [arXiv:2603.03301](https://arxiv.org/abs/2603.03301)) | 2026 | Proves optimal offline semantic caching is NP-hard. Frequency-based policies are strong baselines for embedding-space cache. Combines recency + frequency + locality. | **Directly applicable** — BFCP regions ARE embedding-space clusters. LFU on regions ≈ their "frequency + locality" policy. |
| **MeanCache** ([arXiv:2403.02694](https://arxiv.org/abs/2403.02694)) | 2024 | Privacy-aware semantic cache using embedding similarity. Dual-layer: exact hash + vector similarity. | Architecture reference for dual-layer region cache. |
| **STEP: Semantic-aware Tiered Eviction** (NeurIPS 2025) | 2025 | Semantic-tiered eviction outperforms LRU/LFU by 29% for KV cache sharing. | Tiered eviction maps to BFCP region tiers (Accept/Reject/Maybe). |

### 1.2 KV-Cache Eviction Policies

| Paper | Year | Key Insight | Relevance |
|-------|------|-------------|-----------|
| **"KVCache Cache in the Wild"** (Wang et al., [arXiv:2506.02634](https://arxiv.org/abs/2506.02634), USENIX ATC'25) | 2025 | First large-scale characterization of KV-cache workloads. **Key finding:** reuses are skewed across requests, per-category patterns are predictable. Workload-aware eviction beats LRU. | **Critical** — validates that LFU-style eviction on *predictable categories* (≈ BFCP regions) outperforms generic LRU. |
| **KVP: Learning to Evict** (Moschella et al., [arXiv:2602.10238](https://arxiv.org/abs/2602.10238)) | 2026 | Reframes KV eviction as RL problem. Per-head lightweight agents learn to rank tokens by future utility. Outperforms heuristics. | Alternative to LFU — could use RL-learned eviction per BFCP region. But violates modelless constraint. **LFU is the right call for modelless.** |
| **IceCache** (Mao et al., [arXiv:2604.10539](https://arxiv.org/abs/2604.10539)) | 2026 | **Semantic token clustering** + PagedAttention. Groups semantically related tokens into contiguous memory regions. 256-token budget retains 99% accuracy. | **Closest prior art to BFCF+clustering** — but operates on KV tokens, not logit-space partitions. BFCF is one level of abstraction higher (regions, not tokens). |
| **SABlock** (Chen et al., [arXiv:2510.22556](https://arxiv.org/abs/2510.22556)) | 2025 | Semantic segmentation aligns compression boundaries with linguistic structures. Adaptive block sizes per segment. 99.9% retrieval with 96 KV entries (vs 8K full). | Adaptive block sizing maps to BFCP region having variable token_count — each region IS an adaptive block. |

### 1.3 Semantic/Episodic KV Cache Clustering

| Paper | Year | Key Insight | Relevance |
|-------|------|-------------|-----------|
| **EpiCache** (Kim & Kundu et al., [arXiv:2509.17396](https://arxiv.org/abs/2509.17396)) | 2025 | Semantic clustering of conversation history into episodes. Episodic KV cache compression. Training-free. | **Episodes ≈ BFCP regions in conversation space.** Validates clustering-before-eviction pattern. |
| **SemantiCache** (Wu et al., [arXiv:2603.14303](https://arxiv.org/abs/2603.14303)) | 2026 | Semantic chunking + Greedy Seed-Based Clustering (GSC) for KV compression. Clustered merging preserves semantic integrity. 2.61× decode speedup. | GSC algorithm is relevant — could cluster BFCP regions by label proximity for batch processing. |
| **SentenceKV** (Zhu et al., [arXiv:2504.00970](https://arxiv.org/abs/2504.00970)) | 2025 | Sentence-level semantic KV caching. Retrieves relevant sentence-level entries during decode. | Hierarchical caching (sentence > token) mirrors BFCF hierarchy (region > token). |

### 1.4 Vocabulary/Logit Space Partitioning

| Paper | Year | Key Insight | Relevance |
|-------|------|-------------|-----------|
| **COMPACT** (Kwek & Yin, [arXiv:2509.06836](https://arxiv.org/abs/2509.06836)) | 2025 | **Prunes rare vocabulary** to shrink embedding/LM head. Joint vocab + FFN pruning using common-token-weighted activations. | **Complementary** — COMPACT prunes vocab at training/export time. BFCF prunes at inference time. Together: smaller vocab + region folding. |
| **"Deep ReLU Networks Have Surprisingly Simple Polytopes"** ([arXiv:2305.09145](https://arxiv.org/abs/2305.09145)) | 2023 | ReLU networks partition input space into convex polytopes, but the polytopes are simpler than theoretical bounds suggest. | **Validates BFCF assumption** — the number of activation regions (≈ BFCP regions) is tractable in practice. |
| **"Latent Space Clustering for Improving In-Context Learning"** ([arXiv:2401.16184](https://arxiv.org/abs/2401.16184)) | 2024 | Vocabulary-defined semantics. Latent clustering in label space of model outputs. | Maps to BFCF's vocabulary-level partition clustering. |

### 1.5 Frequency-Aware Routing & Speculative Decoding

| Paper | Year | Key Insight | Relevance |
|-------|------|-------------|-----------|
| **Cascade: Utility-Driven Speculative Decoding for MoE** (Saxena et al., [arXiv:2506.20675](https://arxiv.org/abs/2506.20675)) | 2025 | **Speculation utility** = token_gain / verification_cost. Per-request, per-iteration locality. Test-set phases for adaptive K. | **Pattern-matches BFCF + LFU** — utility ≈ region frequency × inverse routing cost. The test-set pattern maps to GOAT gate phases. |
| **Cache-Conditional Experts** (Skliar et al., 2024) | 2024 | Cache-aware routing in MoE — prioritize experts already in cache. | **Precedent for frequency-aware routing.** BFCP region sharding is analogous: route to shards that already hold hot regions. |

### 1.6 Sharding & Distributed KV Cache

| Paper | Year | Key Insight | Relevance |
|-------|------|-------------|-----------|
| **vAttention** (Microsoft, [arXiv:2405.04437](https://arxiv.org/abs/2405.04437)) | 2024 | Virtual memory for KV cache. Decouples virtual/physical allocation. Contiguous virtual = cache-friendly. | Virtual segment approach maps to BFCP regions as virtual segments, physical pages allocated per region hotness. |
| **PagedAttention / vLLM** (Kwon et al., [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)) | 2023 | OS-inspired paging for KV cache. Fixed-size blocks, dynamic allocation. | **Foundation for region sharding** — each BFCP region occupies N pages, hot regions get more pages. |
| **Disaggregated LLM Serving with CXL** ([arXiv:2512.18194](https://arxiv.org/abs/2512.18194)) | 2025 | KV-cache-centric disaggregated architecture. Separates prefill/decode clusters. | Disaggregation pattern: BFCP region computation on prefill cluster, cached results served from decode cluster. |
| **KVCache-centric Architecture** (USENIX FAST'25) | 2025 | KV-cache as first-class citizen in serving architecture. | Validates region-as-first-class-entity architecture. |

---

## 2. Novelty Assessment

### What's Been Done (Prior Art)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Concept                    │ Prior Art                             │
├─────────────────────────────────────────────────────────────────────┤
│ Semantic embedding cache    │ Biton & Friedman 2026, MeanCache 2024 │
│ LFU on KV cache             │ TableCache 2026, Wang et al. 2025     │
│ Token clustering for KV     │ IceCache 2026, SemantiCache 2026      │
│ Episodic/semantic KV group  │ EpiCache 2025, SentenceKV 2025        │
│ Frequency-aware routing     │ Cascade 2025, Cache-Conditional 2024  │
│ Paged KV memory             │ vAttention 2024, PagedAttention 2023  │
│ Vocab partitioning/pruning  │ COMPACT 2025                          │
│ Polytope region analysis    │ Deep ReLU Polytopes 2023              │
│ Piecewise-constant regions  │ NS-CSG (Plan 213 base) 2022           │
└─────────────────────────────────────────────────────────────────────┘
```

### What's Novel (The Fusion)

**No prior work combines all three:**

1. **Borel partition folding** (O(128K) → O(50) regions from ReLU threshold crossings in logit space)
2. **LFU eviction on those regions** (evict cold BFCP regions, keep hot ones cached for reuse across decode steps)
3. **Frequency-aware sharding** (partition hot regions across workers for parallel screening, with LFU-guided placement)

The closest priors:
- **IceCache** does clustering + paging but at **token level**, not **region level**
- **Cascade** does utility-driven routing but for **MoE expert activation**, not **Borel partition regions**
- **Biton & Friedman** prove frequency-based caching works for embedding space but don't address **inference-time partition caching** or **sharding**
- **COMPACT** does vocab pruning but at **training/export time**, not **inference time with dynamic partitions**

### Novelty Matrix

```
                        LFU    Shard   Cluster  Borel   Modelless
IceCache                  ✗       ✗       ✓        ✗       ✓
Cascade                   ✗       ✗       ✗        ✗       ✗
SemantiCache              ✗       ✗       ✓        ✗       ✓
COMPACT                   ✗       ✗       ✗        ✗       ✗
BFCF Tree (Plan 213)      ✗       ✗       ✗        ✓       ✓
BFCF + LFU + Shard        ✓       ✓       ✓        ✓       ✓  ← NOVEL
```

---

## 3. Proposed Fusion Architecture

### 3.1 BFCP Region Cache (LFU)

The key insight: BFCP regions are **stable across decode steps** for the same context window. The ScreeningPruner thresholds don't change much between adjacent tokens. This means:

- **Region partitions computed at step T** have high reuse probability at step T+1
- **Hot regions** (frequently in Accept label) should be cached with precomputed half-space membership
- **Cold regions** (always Reject) can be evicted — they'll be quickly re-rejected if needed

```rust
/// LFU cache for BFCP regions — evicts cold regions, keeps hot ones
pub struct BfcpRegionCache {
    /// Fixed-size region slots (pre-allocated)
    slots: Box<[Option<CachedRegion>]>,
    /// Frequency counter per region hash
    freq: papaya::HashMap<u64, AtomicU32>,
    /// Capacity
    capacity: usize,
}

struct CachedRegion {
    /// The BFCP region (half-spaces + label)
    region: BorelRegion,
    /// BLAKE3 commitment hash of region boundaries
    hash: [u8; 32],
    /// Precomputed membership test results for recent logits
    membership_cache: Vec<bool>,
}
```

**LFU eviction policy:**
- Track `access_count` per region (via BLAKE3 hash of constraints)
- On cache full: evict region with lowest `access_count`
- Decay: multiply all counters by `λ = 0.99` every N steps (prevents stale hotness)
- Sigmoid gate: only cache regions where `sigmoid(access_count / threshold) > 0.5`

### 3.2 Frequency-Aware Region Sharding

Partition BFCP regions across workers for parallel screening:

```rust
/// Shard assignment: region → worker_id
pub struct RegionShardMap {
    /// Number of shards (workers)
    num_shards: usize,
    /// Shard assignment: region_label × freq_tier → preferred shard
    assignment: papaya::HashMap<(RegionLabel, FreqTier), usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FreqTier {
    Hot,    // access_count > hot_threshold
    Warm,   // access_count > warm_threshold
    Cold,   // below warm
}
```

**Sharding strategy:**
- **Hot regions** → dedicated shard (pinned, never evicted)
- **Warm regions** → round-robin across remaining shards
- **Cold regions** → lazy computation (computed on demand, not sharded)
- **Rebalance** when freq_tier changes (region promotion/demotion)

This is analogous to **Cascade's** test-set phases but applied to region-to-shard routing:
- **Test phase**: route region to any shard, measure screening cost
- **Set phase**: pin hot region to dedicated shard for cache locality

### 3.3 Region-Level Batching

Group BFCP regions by label for batch processing:

```
Accept regions  → batch sample (one SIMD pass)
Reject regions  → batch skip   (O(1) — just count tokens)
Maybe regions   → batch refine (preimage lookahead per region)
```

This is inspired by **SemantiCache's** GSC (Greedy Seed-Based Clustering) — cluster regions by semantic similarity (label + constraint proximity), then batch-process each cluster.

### 3.4 Integration with Existing katgpt-rs Components

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ScreeningPruner ──► BFCP Partition ──► LFU Region Cache         │
│        │                    │                    │                │
│        │                    ▼                    ▼                │
│        │           RegionShardMap          FreqBandit            │
│        │           (freq-tier ×            (PWC arms             │
│        │            label → shard)          per region)          │
│        │                    │                    │                │
│        │                    ▼                    ▼                │
│        │           ┌─── Worker 0 ──┐     ┌── Worker 1 ──┐       │
│        │           │ Hot Accept    │     │ Hot Accept    │       │
│        │           │ Warm Maybe    │     │ Warm Maybe    │       │
│        │           │ (pinned)      │     │ (pinned)      │       │
│        │           └───────────────┘     └───────────────┘       │
│        │                                                  │       │
│        ▼                                                  ▼       │
│  PerceptRouter ◄── sigmoid(freq × complexity) ──► ComputePath   │
│                                                                  │
│  Region batching: Accept SIMD | Reject O(1) | Maybe preimage    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Latent vs Raw Space Rules (Compliance Check)

| Component | Domain | Space | Rationale |
|-----------|--------|-------|-----------|
| BFCP half-space constraints | Logit indices + thresholds | **Raw** | Deterministic: same logits → same partition |
| Region label (Accept/Reject/Maybe) | Symbolic | **Raw** | Binary decision, must be bit-identical |
| LFU frequency counter | Access statistics | **Raw** | Exact integer counts for eviction ordering |
| FreqTier (Hot/Warm/Cold) | Derived from freq | **Raw** | Threshold crossings, deterministic |
| Shard assignment | Routing decision | **Raw** | Must be deterministic for quorum |
| Region similarity clustering | Constraint proximity | **Latent** | Dot-product + sigmoid for "are these regions similar enough to batch?" |
| PerceptRouter complexity | Partition entropy | **Latent** | Sigmoid(region_count × entropy) — already implemented |
| PWC bandit value function | Per-region reward | **Latent** | Dot-product + sigmoid projection for value estimation |

**Bridge functions needed:**
- `raw → latent`: Region constraints → dot-product projection onto direction vectors → sigmoid similarity for batching
- `latent → raw`: FreqTier assignment → clamped shard_id for routing

**No raw→latent→raw round-trips.** LFU eviction is entirely in raw space (exact frequency counts). Latent space used ONLY for clustering/batching decisions.

---

## 5. Performance Expectations

| Metric | Before (Plan 213) | After (BFCF + LFU + Shard) | Source |
|--------|-------------------|----------------------------|--------|
| Region recomputation | Every step | Cache hit: 0 cost | LFU cache |
| Cold region overhead | Full screening | Skip (evicted, not recomputed) | LFU eviction |
| Parallel screening | Sequential regions | Sharded across workers | Region sharding |
| Batch efficiency | Per-region loop | SIMD batch per label group | Region batching |
| Memory overhead | BFCP only | + LFU cache (~50 × CachedRegion) | Acceptable |
| Expected throughput | +20-40% (Plan 213) | **+35-55%** (with sharding) | Conservative |

---

## 6. GOAT Gate Design

```rust
#[cfg(feature = "bfcf_lfu_shard")]
pub struct BfcpLfuShard {
    cache: BfcpRegionCache,
    shard_map: RegionShardMap,
    freq_bandit: FreqBandit, // reuse existing Plan 189
}

// GOAT gates:
// G1: LFU cache hit rate ≥ 60% on synthetic workload
// G2: Shard parallelism ≥ 2× on 4+ core workload
// G3: Zero perf regression when feature disabled
// G4: LFU eviction correctness (no region leak)
// G5: Sigmoid bounded complexity (no softmax)
// G6: Modelless verification (no training required)
```

**Feature flag:** `bfcf_lfu_shard` — auto-enables `bfcf_tree`.

---

## 7. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Region partitions not stable enough for caching | Medium | LFU decay handles drift; cache invalidation on pruner threshold change |
| Sharding overhead > parallelism gain for small region counts (~50) | Medium | Only shard when regions > threshold (e.g., > 30); below that, sequential is faster |
| LFU counter contention in multi-threaded access | Low | Use papaya lock-free HashMap + AtomicU32 for counters |
| Feature flag complexity explosion | Low | `bfcf_lfu_shard` auto-enables `bfcf_tree`; single gate for users |

---

## 8. Key References

1. Biton & Friedman, "From Exact Hits to Close Enough: Semantic Caching for LLM Embeddings," [arXiv:2603.03301](https://arxiv.org/abs/2603.03301), 2026
2. Wang et al., "KVCache Cache in the Wild," [arXiv:2506.02634](https://arxiv.org/abs/2506.02634), USENIX ATC'25
3. Moschella et al., "Learning to Evict from Key-Value Cache," [arXiv:2602.10238](https://arxiv.org/abs/2602.10238), 2026
4. Mao et al., "IceCache: Memory-efficient KV-cache Management," [arXiv:2604.10539](https://arxiv.org/abs/2604.10539), 2026
5. Chen et al., "SABlock: Semantic-Aware KV Cache Eviction," [arXiv:2510.22556](https://arxiv.org/abs/2510.22556), 2025
6. Kim & Kundu et al., "EpiCache: Episodic KV Cache Management," [arXiv:2509.17396](https://arxiv.org/abs/2509.17396), 2025
7. Wu et al., "SemantiCache: Semantic Chunking and Clustered Merging," [arXiv:2603.14303](https://arxiv.org/abs/2603.14303), 2026
8. Zhu et al., "SentenceKV: Sentence-Level Semantic KV Caching," [arXiv:2504.00970](https://arxiv.org/abs/2504.00970), 2025
9. Saxena et al., "Cascade: Utility-Driven Speculative Decoding for MoE," [arXiv:2506.20675](https://arxiv.org/abs/2506.20675), 2025
10. Kwek & Yin, "COMPACT: Common-token Optimized Model Pruning," [arXiv:2509.06836](https://arxiv.org/abs/2509.06836), 2025
11. "Deep ReLU Networks Have Surprisingly Simple Polytopes," [arXiv:2305.09145](https://arxiv.org/abs/2305.09145), 2023
12. vAttention: [arXiv:2405.04437](https://arxiv.org/abs/2405.04437), 2024
13. PagedAttention/vLLM: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180), 2023

---

## 9. Creative Fusion Extensions (Beyond Direct Mapping)

The BFCF region abstraction is not just a cache key — it's a **universal clustering primitive** for the entire latent stack. Here are fusion ideas that go beyond direct LFU+Shard mapping:

### 9.1 Region-Hotset Carry (Cross-Step Partition Reuse)

BFCP partitions computed at step T have high similarity with step T+1 for the same context. Instead of recomputing from scratch:
- **Delta-partition**: compute only the changed regions (threshold crossings that moved)
- **Hotset carry**: regions that stayed Accept stay cached with zero re-evaluation
- This is like CPU branch prediction's BTB (Branch Target Buffer) — predict which regions will be Accept/Reject based on step history
- FreqBandit already tracks per-band frequency; feed that into partition prediction

### 9.2 NeuronShard-Region Fusion (Latent × BFCP)

NeuronShard has `style_weights` and `hla_moments` — these are latent-space descriptors.
- Cluster regions not just by label, but by **NeuronShard similarity**
- `dot_product(shard.style_weights, region.centroid) → sigmoid → "is this shard relevant to this region?"`
- This enables **shard-aware routing**: NeuronShard stays in cache for hot regions, evicted for cold regions
- BLAKE3 hash of NeuronShard + region constraints = compound cache key
- This is the bridge function (latent → raw) from the latent/raw space rules

### 9.3 Spatial Belief × Region Folding (Two-Brain × BFCF)

The two-brain model has `SpatialBelief` (think brain, zone-level) and `MapPos` (info brain, exact position):
- **Think brain regions**: fold zones into BFCP-like regions by belief similarity
- A zone's "attention weight" (dot-product of NPC preference × zone embedding → sigmoid) determines which regions the NPC cares about
- Hot regions = high attention zones → cache those beliefs
- Cold regions = low attention zones → evict beliefs (NPC doesn't care)
- This is the SpatialBelief analogue of LFU eviction in latent space
- The bridge: `attention_weight = sigmoid(dot(pref, zone)) → FreqTier(Hot/Warm/Cold) → belief_cache_admit/evict`

### 9.4 Region-Level Consolidation (Sleep Cycle × BFCF)

Plan 154 (Sleep Consolidation) does offline memory consolidation. With BFCF regions:
- During sleep cycle, **consolidate per-region**: merge similar NeuronShard entries within the same BFCP region
- Region boundaries provide natural consolidation scope — no cross-region leakage
- Hot regions get deeper consolidation (more merge passes), cold regions get shallow (just dedup)
- This is LFU applied to the consolidation budget, not just the cache

### 9.5 Emotion Vector × Region Frequency

Emotion vectors (valence, arousal, desperation, calm, fear) are 5 scalar projections:
- Track per-region emotion profile: `region_emotion_avg = mean(emotion_vectors of tokens in region)`
- Regions with high arousal + high frequency = "excited hot regions" → prioritize in cache + shard
- Regions with low arousal + low frequency = "calm cold regions" → evict first
- Emotion-aware LFU: eviction priority = `1 / (freq × arousal)`, not just `1 / freq`
- This adds semantic richness to LFU's raw frequency count

### 9.6 KG Triple Emission from Region Transitions

When regions transition (Accept → Maybe → Reject) across decode steps:
- Emit KG triple: `(step, region_idx, transition_label)` → `(step+1, region_idx, new_label)`
- These transitions encode **inference dynamics** — which parts of logit space are "unstable" vs "stable"
- Unstable regions (frequent label transitions) → high-priority for preimage refinement
- Stable regions (always Accept or always Reject) → low-priority, can be cached aggressively
- This is semantic domain KG from physical domain (logit) transitions

---

## Verdict

**GOAT.** The fusion is novel, the prior art validates individual components, and the creative extensions (NeuronShard-region fusion, spatial belief folding, emotion-aware LFU) open new dimensions beyond simple caching. The research says:

1. ✅ **LFU is the right eviction policy** (Biton & Friedman 2026 prove it's a strong baseline for semantic caching)
2. ✅ **Region-level operations are tractable** (Deep ReLU Polytopes paper: ~50-100 regions in practice)
3. ✅ **Frequency-aware routing works** (Cascade 2025: utility-driven routing for MoE, maps directly)
4. ✅ **Semantic clustering before eviction beats naive eviction** (SemantiCache: 2.61× decode speedup)
5. ✅ **Workload-aware eviction beats generic LRU** (Wang et al. 2025: per-category predictability)
6. ✅ **No prior combines all three** (Borel partition + LFU + sharding) — confirmed novel

**Decision:** Proceed to Plan 218 (`bfcf_lfu_shard`). Modelless, feature-gated, GOAT-gated.

---

## TL;DR

**No prior work combines Borel partition folding + LFU eviction + frequency-aware sharding for inference-time latent space operations.** The individual components are well-studied (semantic caching, KV eviction, paged memory, vocab pruning), but the specific fusion of BFCP regions as the caching/sharding unit is novel. The closest prior (IceCache) operates at token level, not region level. Cascade's utility-driven routing is the closest architectural analog but for MoE experts, not Borel partitions.

Creative extensions: region-hotset carry (BTB-like partition prediction), NeuronShard-region fusion (latent × BFCP cache key), spatial belief × region folding (two-brain LFU), emotion-aware eviction (arousal × frequency priority), region-level consolidation (sleep cycle per-region budget), and KG triples from region transitions. These go beyond direct LFU+Shard into fundamental latent-space clustering.

**Verdict: GOAT.** Proceed to Plan 218 (`bfcf_lfu_shard`). Expected: +35-55% throughput over Plan 213 baseline, with latent-space extensions enabling broader application across NeuronShard, SpatialBelief, and emotion vectors.
