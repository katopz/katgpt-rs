# Plan 140: CachePrune — Summed-Area Table + Rolling Hash for KV Cache Analysis

> **Research:** [101 — CachePrune Privacy-Aware KV Cache Sharing](../.research/101_CachePrune_Privacy_Aware_Fine_Grained_KV_Cache_Sharing.md)
> **Paper:** [arXiv:2605.23640](https://arxiv.org/abs/2605.23640) — Token-granularity KV cache sharing with SAT attention analysis
> **Feature Gate:** `cache_prune` (opt-in, NOT default-on)
> **Status:** ✅ Phase 1–3 complete (SAT + rolling hash + sensitivity trait + examples)
> **GOAT Pillar:** ❌ Not a pillar — infrastructure optimization supporting future MMO serving. See [MMO GOAT Pillars Decision Matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md).
> **Domain:** `katgpt-rs` — generic SAT + rolling hash + SensitivityDetector trait. Game-specific sensitivity patterns and per-domain ρ thresholds stay in `riir-ai`.
> **Blocks:** None. Enables future MMO backbone (Issue 015) optimization.

---

## Summary

Extract the two most reusable algorithmic primitives from CachePrune: (1) **Summed-Area Table (SAT)** for O(1) rectangular region sum queries on attention matrices, and (2) **Rolling Hash** for O(n) variable-length segment matching. Add a generic `SensitivityDetector` trait for selective KV sharing. All modelless, no training, no model changes.

---

## Why

1. **SAT is broadly useful:** O(1) intra/inter attention computation benefits DashAttention, RTPurbo, SpectralQuant, EGA, and any attention analysis pipeline
2. **Rolling hash improves SpecHop:** Sub-prompt matching across hops instead of prefix-only matching
3. **Future MMO readiness:** Token-granularity KV sharing is the correct primitive for multi-tenant game serving (Issue 015)
4. **Modelless:** No training, no model changes. Pure algorithmic infrastructure
5. **Small footprint:** SAT ~100 lines, rolling hash ~200 lines. Minimal code surface

---

## Architecture

### Module Structure

```
src/
├── cache_prune/                    ← new module, feature-gated
│   ├── mod.rs                      ← public API + re-exports
│   ├── sat.rs                      ← SummedAreaTable for attention analysis
│   ├── rolling_hash.rs             ← Rolling hash retrieval
│   └── sensitivity.rs              ← SensitivityDetector trait + MaskedSegment
```

### A. SummedAreaTable (Generic Attention Analysis)

```rust
/// Summed-area table (integral image) for O(1) rectangular region sum queries.
///
/// Preprocesses an n×n attention matrix in-place in O(n²) time.
/// After preprocessing, any rectangular region sum is O(1).
///
/// Use cases:
/// - Intra/inter attention ratio (self-contextualization)
/// - Per-segment importance scoring
/// - Retrieval head identification
pub struct SummedAreaTable<'a> {
    data: &'a mut [Vec<f32>],  // n×n, modified in-place
    n: usize,
}

impl<'a> SummedAreaTable<'a> {
    /// Build SAT in-place from attention matrix. O(n²).
    pub fn build(attention: &'a mut [Vec<f32>]) -> Self;

    /// Query sum of rectangular region [x1..=x2] × [y1..=y2]. O(1).
    pub fn region_sum(&self, x1: usize, x2: usize, y1: usize, y2: usize) -> f32;

    /// Compute intra-attention of substring [l..=r]. O(1).
    /// Sum of attention from positions l..r to positions l..r.
    pub fn intra_attention(&self, l: usize, r: usize) -> f32;

    /// Compute inter-attention of substring [l..=r] to prefix [0..l). O(1).
    pub fn inter_attention(&self, l: usize, r: usize) -> f32;

    /// Self-contextualization score: intra - inter. O(1).
    /// Positive = segment is self-contained (reusable).
    pub fn contextualization_score(&self, l: usize, r: usize) -> f32;

    /// Find optimal reusable substring within [start..=end]. O((end-start)²).
    /// Returns (l, r) maximizing contextualization_score, with min_length constraint.
    pub fn best_reusable_segment(
        &self,
        start: usize,
        end: usize,
        min_length: usize,
    ) -> Option<(usize, usize)>;
}
```

### B. RollingHash (Variable-Length Segment Retrieval)

```rust
/// Polynomial rolling hash for O(n) variable-length segment matching.
/// Uses Mersenne prime 2^61-1 with random base.
///
/// Two-phase retrieval:
/// 1. Prefix filtering: slide window, O(1) per shift
/// 2. Full verification: SHA-256 (blake3 in our case) to eliminate collisions
pub struct RollingHash {
    base: u64,
    modulus: u64,    // 2^61 - 1
    powers: Vec<u64>, // precomputed base^i mod modulus
}

impl RollingHash {
    pub fn new(max_length: usize) -> Self;

    /// Compute prefix hash array for a token sequence. O(n).
    pub fn prefix_hashes(&self, tokens: &[u32]) -> Vec<u64>;

    /// Hash of substring [l..r) from prefix hash array. O(1).
    pub fn substring_hash(&self, prefixes: &[u64], l: usize, r: usize) -> u64;

    /// Slide window hash: update from old to new in O(1).
    pub fn slide(&self, old_hash: u64, old_token: u32, new_token: u32, window_size: usize) -> u64;
}

/// KV segment pool for variable-length segment matching.
pub struct KvSegmentPool {
    segments: Vec<CachedSegment>,
    prefix_index: HashMap<u64, Vec<usize>>, // prefix hash → segment indices
}

pub struct CachedSegment {
    pub token_hashes: Vec<u32>,   // blake3 hashes of tokens
    pub prefix_hash: u64,          // rolling hash of first 128 tokens
    pub full_hash: [u8; 32],       // blake3 of full segment
    pub start: usize,
    pub end: usize,
}

impl KvSegmentPool {
    /// Find matching segments in an incoming request. O(n + candidates).
    pub fn find_matches(&self, request_tokens: &[u32], roller: &RollingHash) -> Vec<MatchResult>;
}
```

### C. SensitivityDetector Trait (Generic Interface)

```rust
/// Trait for identifying sensitive tokens in a prompt.
/// Implementations are domain-specific and live in riir-ai.
pub trait SensitivityDetector: Send + Sync {
    /// Name of the detector (for logging).
    fn name(&self) -> &str;

    /// Produce a binary sensitivity mask for the token sequence.
    /// M[i] = true means token i is sensitive (excluded from cross-user sharing).
    fn detect(&self, tokens: &[u32], text: &str) -> Vec<bool>;
}

/// Default: strict masking (everything is sensitive).
pub struct StrictDetector;

/// Default: nothing is sensitive (open sharing).
pub struct OpenDetector;

/// A segment of tokens bounded by sensitive tokens.
pub struct MaskedSegment {
    pub start: usize,
    pub end: usize,
    pub is_reusable: bool,
    pub contextualization_score: f32,
    pub recompute_indices: Vec<usize>, // tokens to recompute when reused
}
```

---

## Tasks

### Phase 1: SAT Primitive (Core Value)

- [x] **T1:** Implement `SummedAreaTable::build` — in-place O(n²) preprocessing
- [x] **T2:** Implement `SummedAreaTable::region_sum` — O(1) rectangular query
- [x] **T3:** Implement `SummedAreaTable::intra_attention` / `inter_attention` / `contextualization_score`
- [x] **T4:** Implement `SummedAreaTable::best_reusable_segment` — optimal substring search with min_length
- [x] **T5:** GOAT proof: verify SAT correctness on 4×4–64×64 attention matrices with known ground truth
- [x] **T6:** GOAT proof: benchmark SAT build + query vs naive O(n²) scan for n=64, n=256, n=512

### Phase 2: Rolling Hash Retrieval

- [x] **T7:** Implement `RollingHash::new` with Mersenne prime 2^61-1 and precomputed powers
- [x] **T8:** Implement `RollingHash::prefix_hashes` and `substring_hash`
- [x] **T9:** Implement `RollingHash::slide` for O(1) window update
- [x] **T10:** Implement `KvSegmentPool::find_matches` with two-phase retrieval
- [x] **T11:** GOAT proof: verify rolling hash matches direct for all substrings, 0 false negatives

### Phase 3: Sensitivity Trait + Integration

- [x] **T12:** Implement `SensitivityDetector` trait + `StrictDetector` + `OpenDetector`
- [x] **T13:** Implement `MaskedSegment` derivation from sensitivity mask
- [x] **T14:** Wire `cache_prune` feature gate in `Cargo.toml`
- [x] **T15:** Example: `cache_prune_01_sat_bench` — benchmark SAT vs naive on synthetic attention matrices
- [x] **T16:** Example: `cache_prune_02_segment_match` — demonstrate rolling hash segment matching

### Phase 4: Cross-Feature Integration

- [x] **T17:** Wire SAT into DashAttention for per-head sparsity analysis (requires `dash_attn`)
- [x] **T18:** Wire SAT into RTPurbo for retrieval head identification (requires `rt_turbo`)
- [x] **T19:** Wire rolling hash into SpecHop for sub-prompt matching (requires `spechop`)

---

## GOAT Proof Plan

| # | Proof | Threshold | Method |
|---|-------|-----------|--------|
| G1 | SAT region_sum correctness | ≤1e-6 max error vs naive sum | Compare on 100 random 8×8–64×64 matrices |
| G2 | SAT build + query throughput | ≥10M queries/sec on 256×256 matrix | Benchmark vs naive O(n) scan |
| G3 | Rolling hash collision rate | 0 false negatives on 10K random segments | Verify with blake3 ground truth |
| G4 | Segment matching latency | ≤10ms for 10K-token request vs 1000 cached segments | Benchmark prefix filter + verification |

---

## Feature Gate

```toml
[features]
cache_prune = []  # CachePrune SAT + rolling hash + sensitivity masking (Plan 140, Research 101)
```

NOT default-on. This is opt-in infrastructure. Will become default-on when MMO backbone (Issue 015) ships.

---

## What Stays in riir-ai (Private)

| Component | Why Private | Plan |
|-----------|-------------|------|
| `BomberSensitivityDetector` | Bomber strategy tokens are game IP | riir-ai Plan TBD |
| `GoSensitivityDetector` | Go move sequences reveal strategy | riir-ai Plan TBD |
| `FftSensitivityDetector` | FFT party composition is competitive | riir-ai Plan TBD |
| Per-domain ρ recompute thresholds | Tuned game parameters | riir-ai config |
| MMO cross-player KV reuse policy | Per-game cache sharing rules | riir-ai Issue 015 |
| `GameKvSegmentPool` with game-specific eviction | Game-tuned cache policy | riir-ai Issue 015 |

---

## Dependencies

- **No new external dependencies.** SAT is pure math on `Vec<f32>`. Rolling hash uses arithmetic on `u64`. Verification uses existing `blake3`.
- **Internal:** Uses existing `PagedKVCache` (Plan 011), `DashAttention` (Plan 106), `SpecHop` (Plan 131) — but only as integration targets, not hard dependencies.

---

## Honest Assessment

**What this plan actually delivers:** Two reusable algorithmic primitives (SAT + rolling hash) and a trait for game-specific sensitivity detection. The SAT is genuinely useful for our attention analysis pipeline. The rolling hash improves SpecHop matching. The full CachePrune system (multi-tenant KV pool, sensitivity detector pipeline, GPU→CPU attention transfer) is **not in scope** — that requires MMO backbone (Issue 015) to be meaningful.

**Risk:** This could be over-engineering for single-user inference. The SAT primitive is the clear value extraction; the rolling hash and sensitivity trait are "nice to have" that justify the feature gate. If SAT alone proves useful, the rest can wait.

**Alignment with GOAT pillars:** This is NOT a pillar. It's infrastructure that supports the MMO backbone (Gap 2 in the decision matrix). It becomes valuable when Issue 015 ships and multiple players share KV cache in the same game world.
