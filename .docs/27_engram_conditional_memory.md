# Engram — Conditional Memory (Plan 299)

Hash-addressed, sigmoid-fused static pattern memory — the **first conditional-memory
axis** in the katgpt stack. Where Raven (RSM/dMoE) routes **computation** per token,
Engram routes **memory lookups** per token. The paper's U-shape scaling law proves
the hybrid is strictly better than either axis alone (§3).

**Research:** [`katgpt-rs/.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md`](../.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md)
**Plan:** [`katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md`](../.plans/299_Engram_Hash_Addressed_Pattern_Memory.md)
**Benchmark:** [`katgpt-rs/.benchmarks/299_engram_goat.md`](../.benchmarks/299_engram_goat.md)
**Feature flag:** `engram` (opt-in — G6 deferred to riir-ai integration)

---

## TL;DR

The open half of the Engram Super-GOAT. A generic, hash-addressed, sigmoid-fused
static pattern memory primitive in `katgpt-core`. The mechanism reduces to:

```text
hash_keys = multi_head_hash(n_gram_suffix(input_ids))   # K=16 deterministic hashes, O(1)
e_t       = concat(table[k] for k in hash_keys)          # multi-head retrieval
α_t       = σ(RMSNorm(q_t) · RMSNorm(W_K e_t) / √d)     # sigmoid gate (NEVER softmax)
output_t  = α_t · (W_V e_t)                              # gated residual contribution
h_t      += output_t                                     # residual fuse
```

No training, no backprop. The table is populated offline and frozen; updates are
atomic Arc swaps via `EngramHotSwap`. The whole pipeline is zero-allocation on the
hot path (caller provides scratch buffers).

**GOAT status:** G1/G2/G4 PASS (48 ns/retrieval, ρ=1.0, bit-deterministic commitment).
G6 (effective depth) is the load-bearing gate but requires a live inference
pipeline — **deferred to riir-ai integration**, feature stays opt-in.

---

## API Surface

```rust
use katgpt_core::engram::{
    // Core types
    EngramHash, CanonicalId, TokenId, K_MAX,                 // hashing
    EngramTable, InMemoryEngramTable, EngramTableBuilder,    // table
    HashHead, multi_head_hash,                               // multi-head hash
    SigmoidFusionConfig, sigmoid_fuse_into,
    sigmoid_fuse_multi_branch_into, rmsnorm_into,            // sigmoid kernel
    IDENTITY_KERNEL, conv_causal_into,                       // depthwise causal conv (§2.3)
    SurjectiveMap, TokenizerSpec, compress_token,
    build_surjective_map,                                    // tokenizer compression (§2.2)
    EngramHotSwap,                                           // atomic table replacement
    ZipfianCacheHierarchy, CacheTier, ColdFetcher,
    ZipfianStats, CacheResult,                               // tiered cache (§2.5)
    EngramTableId, build_merkle_root,                        // BLAKE3 commitment
    EngramConfig, fuse_into_hidden_state,                    // end-to-end fuse hook
};
```

### Trait

```rust
pub trait EngramTable: Send + Sync {
    fn lookup_into(&self, hash_keys: &[EngramHash; K_MAX], out: &mut [f32]) -> usize;
    fn commitment(&self) -> [u8; 32];
    fn num_slots(&self) -> usize;
    fn dim(&self) -> usize;
}
```

The `out` buffer is `K_MAX * D` long; the implementation writes K slot vectors
into it row-major (`out[k*D..(k+1)*D]`). Empty / collision-missed slots are
written as zeros so the caller can treat the output uniformly.

---

## When to enable

**Enable when:**
- Your inference stack spends depth on **static knowledge reconstruction**
  (named entities, formulaic phrases, idioms) that could be served by O(1)
  lookup instead. The paper's §6.1 LogitLens shows Engram's layer 5 ≈ MoE
  baseline's layer 12.
- You want **retrieval-augmented generation without per-query vector DB
  lookups** — Engram's deterministic hash addressing means indices are known
  before forward, enabling async prefetch overlapping preceding-layer compute
  (paper §2.5).
- You want a **conditional-memory axis** complementary to your conditional-
  computation axis (MoE/Raven). The U-shape scaling law (§3) says the hybrid
  is strictly better.

**Do NOT enable when:**
- You don't have a learned embedding table to populate the slots from. The
  open primitive is **table plumbing**, not a model — the table population
  is the caller's responsibility (typically via offline distillation of
  common N-grams into direction vectors).
- Your workload doesn't have a Zipfian N-gram distribution. The cache
  hierarchy in `cache.rs` helps a lot, but Engram's wins compound with
  real-world token distribution skew.
- You need end-to-end proven quality (G6). G6 is deferred to riir-ai
  integration — until then, you're betting on the paper's claims without
  reproducing them locally.

---

## Performance characteristics

| Operation | Complexity | Measured (M-series, release) | Notes |
|---|---|---|---|
| `multi_head_hash` | O(suffix.len()) | ~10 ns for 3-token suffix | Stack-only, no alloc |
| `InMemoryEngramTable::lookup_into` | O(K_MAX × D) | **48 ns/retrieval** (G1) | Direct slice-index + memcpy |
| `sigmoid_fuse_into` | O(D) | sub-µs for D ≤ 128 | SIMD (NEON/AVX2) RMSNorm+dot fused |
| `compress_token` | O(1) | ~1 ns | Direct index into Box<[CanonicalId]> |
| `EngramHotSwap::commitment_fast` | O(1) | <1 ns | Atomic u64 load |
| `ZipfianCacheHierarchy::lookup_cached` | O(1) hot, O(N) on evict | sub-µs plasma hit | Eviction scan only on full cap |
| `EngramTableId::from_table` | O(N × D) | build-time only | BLAKE3 Merkle root, cached |

All hot-path operations are **zero-allocation** (caller provides scratch buffers).
The build path (`build_surjective_map`, `build_merkle_root`, `EngramTableBuilder::build`)
allocates freely — these are infrequent control-plane operations.

---

## Comparison vs Raven (Research 006)

| Axis | Raven (conditional **computation**) | Engram (conditional **memory**) |
|---|---|---|
| What's routed | Active parameters per token | Lookup slots per token |
| Mechanism | Top-K hidden-state routing to experts | N-gram hash → table → sigmoid gate |
| Latency | O(routed experts) | **O(1)** (constant retrieval count) |
| Best for | Compositional reasoning (dynamic compute) | Knowledge retrieval (static patterns) |
| Update mode | Training-time expert assignment | Atomic Arc swap of frozen table |
| Paper's hybrid claim | Both are suboptimal alone; U-shape optimum is ~80% Raven + ~20% Engram (§3, Fig 3) | |

**Rule of thumb:** if your bottleneck is "the model wastes depth reconstructing
known facts", add Engram. If your bottleneck is "the model can't reason deeply
enough about novel combinations", add Raven. Most real workloads want both.

---

## Latent vs raw boundary (AGENTS.md)

- **Latent** (local, never synced): slot contents (`[f32; D]` direction vectors),
  the sigmoid gate scalar, the cached plasma entries.
- **Raw** (syncable audit artifact): `EngramTableId` (32-byte BLAKE3 Merkle root),
  `EngramHotSwap::commitment_fast` (low 8 bytes of the root, as u64).
- **Bridge**: `EngramTableId::from_table(&table)` computes the raw commitment
  from the latent slot contents. Two tables with identical slots share an ID
  regardless of head configuration — this is the contract for content-addressed
  sync.

---

## References

- **Plan 299:** [`katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md`](../.plans/299_Engram_Hash_Addressed_Pattern_Memory.md)
- **Research 278:** [`katgpt-rs/.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md`](../.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md)
- **Source paper:** Cheng, Zeng, Dai, Chen et al. (Peking U. + DeepSeek-AI), "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models", [arXiv:2601.07372](https://arxiv.org/pdf/2601.07372), 13 Jan 2026.
- **Sibling research:** 006 (Raven RSM — the complementary *computation* axis), 196 (KG Latent Octree — spatial lookup, different substrate), 262 (Lore ContentStore — Merkle-blob lookup), 268 (Forensic Asset Fingerprinting — BLAKE3-seeded addressing, same primitive different domain), 276 (Personality-Weighted Composition — same sigmoid×direction kernel, different source).
- **Private guide (riir-ai):** `riir-ai/.research/147_Engram_Conditional_Memory_NPC_Guide.md` — the selling-point guide for NPC domain use.
- **Chain commitment half (TODO):** `riir-chain/.research/001_Engram_LatCal_Commitment_Bridge.md` — file when the LatCal bridge starts.
