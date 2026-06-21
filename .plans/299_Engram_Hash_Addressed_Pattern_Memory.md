# Plan 299: Engram — Hash-Addressed Pattern Memory (Open Primitive)

**Date:** 2026-06-21
**Research:** [katgpt-rs/.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md](../.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md)
**Private guide:** [riir-ai/.research/147_Engram_Conditional_Memory_NPC_Guide.md](../../../riir-ai/.research/147_Engram_Conditional_Memory_NPC_Guide.md)
**Source paper:** [arXiv:2601.07372](https://arxiv.org/pdf/2601.07372) — Engram, Cheng et al. 2026 (DeepSeek-AI / Peking U.)
**Target:** `katgpt-rs/crates/katgpt-core/src/engram/` (new module)
**Cargo feature:** `engram` (opt-in, default OFF — promote to default-on after G1–G7 GOAT gate passes; per AGENTS.md GOAT gate rule)
**Status:** Active — Phases 1-3 (core), 5 partial (commitment), 7 partial (forward) done; Phases 4 (tokenizer), 5 hotswap, 6 (cache), 7 GOAT gates, 8 docs pending

---

## Goal

Ship the **open half** of the Engram Super-GOAT (Research 278 / Guide 147): a generic, hash-addressed, sigmoid-fused static pattern memory primitive in `katgpt-core`. The mechanism: N-gram-suffix → multi-head hash → O(1) embedding-table lookup → context-aware sigmoid gate → residual-fuse into hidden state. **No training, no backprop.** The table is a frozen snapshot; updates are atomic Arc swaps.

This is the **first conditional-memory axis** in our stack — complementary to Raven (conditional computation). The U-shape scaling law (paper §3) proves hybrid is strictly better than either alone. The open primitive is the adoption hook; the private selling-point guide (riir-ai R147) is the moat; the chain commitment half (riir-chain R001, TODO) is what makes it chain-committable.

**No game semantics, no chain IP, no NPC types.** This is pure inference-time math: a hash table + a sigmoid kernel + an atomic swap. The host (game runtime, recommender, code completion engine) supplies the table population and the query.

**GOAT gate** (per AGENTS.md): feature flag `engram`, default OFF. Promote to default-on only after G1–G7 pass. Demote to experimental if any gate fails. Benchmarks live in `katgpt-rs/.benchmarks/299_engram_goat.md`.

---

## Architecture

```
katgpt-rs/crates/katgpt-core/src/engram/
├── mod.rs              ← public API: EngramTable trait, EngramHash, K_MAX
├── hash.rs             ← MultiHeadHash, HashHead, multi_head_hash() — prime-table mult-XOR
├── table.rs            ← InMemoryEngramTable — papaya-backed frozen table
├── tokenizer.rs        ← SurjectiveMap, compress_token() — NFKC + lowercase collapse
├── kernel.rs           ← sigmoid_fuse_into() — RMSNorm + dot + sigmoid + scale
├── conv.rs             ← depthwise causal conv (paper §2.3) — kernel 4, dilation = max N
├── hotswap.rs          ← EngramHotSwap — AtomicPtr<Box<EngramTable>>, lock-free reads
├── cache.rs            ← ZipfianCacheHierarchy — plasma/hot/warm/cold tiered
├── commitment.rs       ← EngramTableId ([u8; 32]) + Merkle root of slot hashes
└── forward.rs          ← fuse_into_hidden_state() — end-to-end residual fuse hook
```

Plus tests in `crates/katgpt-core/src/engram/` (unit) and `tests/bench_299_engram_goat.rs` (GOAT gates).

---

## Phase 1 — Core Types & Hashing Skeleton (CORE)

### Tasks

- [x] **T1.1** Create module skeleton `crates/katgpt-core/src/engram/mod.rs` with feature gate `#[cfg(feature = "engram")]`. Add `engram` feature to `crates/katgpt-core/Cargo.toml` (deps: `blake3` already there; `papaya` already there per AGENTS.md; no new deps). Export from `crates/katgpt-core/src/lib.rs` behind feature gate.
- [x] **T1.2** Define `EngramHash(pub u64)` — `#[repr(transparent)]`, `Copy + Eq + Hash`. Zero-cost newtype.
- [x] **T1.3** Define `HashHead { n: u8, k: u8, modulus: u64, seed: u64 }` — one prime-table hash configuration. Pre-computed at table build time, immutable.
- [x] **T1.4** Define `K_MAX = 16` const (paper uses 8 heads × 2 N-gram orders = 16). Fixed-size array `[EngramHash; K_MAX]` per retrieval — zero alloc.
- [x] **T1.5** Implement `multi_head_hash(suffix: &[CanonicalId], heads: &[HashHead; K_MAX]) -> [EngramHash; K_MAX]` in `hash.rs`. Multiplicative-XOR per head: `hash = (seed XOR suffix_fold) % modulus` where `suffix_fold = Σᵢ suffix[i] · MULTIPLIERS[i]`. SIMD-friendly (4 or 8 heads at once when suffix is fixed-size `[u64; 3]`).
- [x] **T1.6** Unit tests: empty suffix → all-zero hashes; same suffix → same hash (determinism); different suffix → different hash (no trivial collisions); K heads independent (changing one head's seed changes only its hash).
- [ ] **T1.7** Property test: `proptest` over random `[CanonicalId; 3]` suffixes — verify determinism + uniform distribution modulo prime (chi-square test on 10K samples).

---

## Phase 2 — Frozen Table + Lookup (CORE)

### Tasks

- [x] **T2.1** Define `EngramTable` trait in `mod.rs` (per R278 §2.1 — `lookup_into`, `commitment`, `num_slots`, `const D`).
- [x] **T2.2** Implement `InMemoryEngramTable` in `table.rs`:
  - `slots: Box<[[f32; D]>` — flat `[N × D]` row-major, 64-byte aligned (per ContiguousWeights pattern, R102)
  - `heads: [HashHead; K_MAX]` — pre-computed at build
  - `commitment_cache: OnceCell<[u8; 32]>` — lazy BLAKE3
  - Lookup: `slots[hash as usize % N]` — direct index, O(1), cache-friendly
- [x] **T2.3** Implement `EngramTableBuilder`:
  - `from_iter(items: impl Iterator<Item = ([f32; D], frequency: u64)>)` — populate table, write to slots indexed by hash mod N
  - Hash collision handling: last-write-wins (paper uses prime moduli + multi-head to dilute; collisions are a quality issue, not a correctness issue — the sigmoid gate filters noise)
  - `build_with_commitment()` — compute Merkle root of slot hashes (binary Merkle per R262), cache in `commitment_cache`
- [x] **T2.4** Implement `lookup_into(&self, hash_keys: &[EngramHash; K_MAX], out: &mut [f32]) -> usize`:
  - For each `k`, copy `slots[hash_keys[k].0 as usize % N]` into `out[k*D..(k+1)*D]`
  - Return hit count (slots that contain non-zero data)
  - Zero-allocation: caller provides `out` of size `K_MAX × D`
- [x] **T2.5** Unit tests: empty table → all zeros out; single-slot populated → lookup hits; K-head retrieval fills all K slots; commitment deterministic (same content → same BLAKE3).
- [ ] **T2.6** Performance: micro-bench `lookup_into` on 1M-slot table — target < 50 ns per K=16 retrieval (cache-resident, SIMD-friendly).

---

## Phase 3 — Sigmoid Fusion Kernel (CORE)

### Tasks

- [x] **T3.1** Define `SigmoidFusionConfig { tau: f32, rmsnorm_eps: f32 }` in `kernel.rs`. Defaults: `tau = √D` (matches paper), `rmsnorm_eps = 1e-6`.
- [x] **T3.2** Implement `rmsnorm_into(x: &[f32], eps: f32, out: &mut [f32])` — reuse existing `types::rmsnorm` pattern from `transformer.rs` if compatible; else inline.
- [x] **T3.3** Implement `sigmoid_fuse_into(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32], config: &SigmoidFusionConfig)`:
  ```text
  q_norm = RMSNorm(q); k_norm = RMSNorm(k)
  gate = sigmoid(dot(q_norm, k_norm) / config.tau)
  for j in 0..D: out[j] = gate * v[j]
  ```
  SIMD-accelerated when `D % 8 == 0` (NEON/AVX2 dispatch via `simd::simd_dot_f32`).
- [x] **T3.4** **CRITICAL — never softmax.** Add a static assertion / doc comment that this kernel uses sigmoid per AGENTS.md. No `softmax` symbol in this module.
- [x] **T3.5** Unit tests:
  - `q == k` → gate ≈ 1.0 (after RMSNorm, dot ≈ D, sigmoid(D/√D) high)
  - `q == -k` → gate ≈ 0.0
  - `q ⊥ k` → gate ≈ 0.5
  - Ranking preservation: for fixed `q`, varying `k`, the gate ranking matches cosine ranking (rank-correlation > 0.95)
- [ ] **T3.6** Multi-branch variant `sigmoid_fuse_multi_branch_into(q_per_branch: &[&[f32]; M], k_per_branch: &[&[f32]; M], v: &[f32], out_per_branch: &mut [&mut [f32]; M], config)` — paper §2.4. Single shared `v`, M distinct gates. Default `M = 1` (single-branch); mHC users opt in to `M = 4`.
- [ ] **T3.7** Depthwise causal conv `conv_causal_into(v_tilde: &[f32], out: &mut [f32], kernel: [f32; 4], dilation: usize)` — paper §2.3 eq 5. Init kernel to identity (zero conv → pure residual) per paper's "Conv Zero Init" hyperparameter.

---

## Phase 4 — Tokenizer Compression (CORE)

### Tasks

- [ ] **T4.1** Define `SurjectiveMap { raw_to_canonical: Box<[CanonicalId]> }` in `tokenizer.rs`. Pre-computed at build time from a tokenizer spec; immutable.
- [ ] **T4.2** Implement `compress_token(raw_id: TokenId, projection: &SurjectiveMap) -> CanonicalId` — direct index lookup, O(1).
- [ ] **T4.3** Implement `build_surjective_map(tokenizer: &dyn TokenizerSpec) -> SurjectiveMap` — for each raw token ID, compute its canonical form via:
  - Decode raw token to bytes
  - NFKC normalize (use `unicode-normalization` crate — verify it's already a dep; if not, add behind feature)
  - Lowercase
  - Re-encode → canonical bytes → hash to `CanonicalId` (BLAKE3 → first 8 bytes as u64)
  - Build equivalence classes (canonical → list of raws)
- [ ] **T4.4** Unit tests:
  - `"Apple"` and `"␣apple"` → same canonical ID
  - `"A"` and `"a"` → same canonical
  - Distinct semantic tokens → distinct canonicals
  - Surjectivity: every raw ID maps to exactly one canonical ID
  - Compression ratio: 23% reduction target on a 128k tokenizer (per paper Appendix C)
- [ ] **T4.5** Serialization: `SurjectiveMap::save_to_bytes` / `load_from_bytes` — postcard format, BLAKE3-committed. Reuse `serialize.rs` patterns.

---

## Phase 5 — Atomic HotSwap + Commitment (CORE)

### Tasks

- [ ] **T5.1** Define `EngramHotSwap` in `hotswap.rs` — mirror `SenseHotSwap` pattern (`sense/hotswap.rs`):
  - `table: AtomicPtr<Box<dyn EngramTable>>`
  - `lock: AtomicBool` — set during swap, cleared after
  - `current_commitment: AtomicU64` — low 8 bytes of BLAKE3, for fast identity check
- [ ] **T5.2** Implement `swap(new_table: Box<dyn EngramTable>)`:
  - Spin-wait on `lock.compare_exchange(false, true)`
  - Compute new commitment if not cached
  - Atomic pointer swap
  - Update `current_commitment`
  - Drop old table (ref-counted via `Arc` if needed for in-flight readers)
  - Clear `lock`
- [ ] **T5.3** Implement `with_table<R>(&self, f: impl FnOnce(&dyn EngramTable) -> R) -> R`:
  - Spin-wait if `lock` is set
  - Load pointer, call `f(table)`
  - (Reader holds a borrowed reference for the duration of `f` — swap waits for all readers if needed, or uses epoch-based reclamation)
- [ ] **T5.4** Decide on memory reclamation strategy:
  - **Option A (simple):** `lock` blocks readers during swap. Swap is rare (table updates are infrequent), so this is OK if swap latency < 1ms.
  - **Option B (lock-free):** `crossbeam-epoch` for safe reclamation. Adds a dep.
  - **Default: Option A.** Promote to Option B only if G5 (reader atomicity under high swap rate) demands it.
- [x] **T5.5** Implement `EngramTableId(pub [u8; 32])` in `commitment.rs` — content-addressed identity. Methods: `from_table(table: &dyn EngramTable) -> Self`, `verify(table: &dyn EngramTable) -> bool`.
- [x] **T5.6** Implement `build_merkle_root(slots: &[[f32; D]]) -> [u8; 32]` — binary Merkle tree (R262 infrastructure). Leaves = `BLAKE3(slot_bytes)`; internal = `BLAKE3(left || right)`; root = table identity.
- [ ] **T5.7** Unit tests:
  - Same content → same `EngramTableId`
  - Different content → different `EngramTableId`
  - `EngramTableId::verify` returns true for the table that produced it
  - HotSwap: 1000 swaps in a row, no leak (valgrind / `loom` clean)
  - HotSwap reader atomicity: `loom` model checker — reader sees either old or new table, never a mix
- [ ] **T5.8** **G5 gate** — concurrent reader/writer test: reader does 1M lookups, writer does 100 swaps. Verify 0 torn reads.

---

## Phase 6 — Zipfian Cache Hierarchy (CORE)

### Tasks

- [ ] **T6.1** Define `CacheTier` enum: `Plasma` (in-process L1 / shared mem), `Hot` (HBM / DRAM), `Warm` (host DRAM), `Cold` (NVMe / network).
- [ ] **T6.2** Define `ZipfianCacheHierarchy { hot_cache: LruCache<EngramHash, [f32; D]>, warm_source: Box<dyn EngramTable>, cold_fetcher: Option<Box<dyn ColdFetcher>> }` in `cache.rs`.
- [ ] **T6.3** Implement `lookup_cached(&self, hash: EngramHash) -> CacheResult`:
  - Check `hot_cache` (plasma + hot tier — in-process)
  - On miss, fall through to `warm_source.lookup_into()`
  - On warm miss, fall through to `cold_fetcher` if present (Lore ContentStore `ChunkFetcher` pattern, R262)
  - Promote to `hot_cache` on hit
- [ ] **T6.4** Implement `ZipfianStats { hits_plasma, hits_hot, hits_warm, hits_cold, misses }` — per-tier counters for diagnostics.
- [ ] **T6.5** Implement adaptive hot-cache sizing: monitor hit rate over a sliding window; grow/shrink `hot_cache` capacity to maintain ≥90% plasma+hot hit rate.
- [ ] **T6.6** **G3 gate** — simulate 10K retrievals from 1M-slot table with Zipf(s=1.1) distribution. Verify 90%+ plasma+hot, <1% cold. Bench: < 200 ns amortized per retrieval including tiering.
- [ ] **T6.7** Unit tests: all-in-hot → 100% hot hits; all-in-cold → 100% cold hits; promotion works (cold lookup → hot lookup next time).

---

## Phase 7 — End-to-End Fuse + GOAT Gate

### Tasks

- [x] **T7.1** Implement `fuse_into_hidden_state(hidden_state: &mut [f32], query: &[f32], table: &dyn EngramTable, hash_keys: &[EngramHash; K_MAX], config: &EngramConfig)` in `forward.rs`:
  - Allocate K retrievals + K gates on caller-provided scratch buffers
  - Lookup K patterns
  - For each pattern: compute `k = W_K · e`, `v = W_V · e`, sigmoid-fuse into hidden_state
  - Sum the K contributions into hidden_state (residual add)
- [x] **T7.2** Define `EngramConfig { fusion: SigmoidFusionConfig, k_heads: usize, conv_kernel: Option<[f32; 4]>, multi_branch: Option<usize> }` — host-configurable.
- [ ] **T7.3** **G1 gate** — `tests/bench_299_engram_goat.rs::g1_lookup_latency`:
  - 1M-slot table, D=128
  - Retrieve K=16 patterns in single call
  - Target: < 200 ns per retrieval (amortized over K=16 = ~3.2 µs total), zero allocation
  - Apple Silicon NEON SIMD path
- [ ] **T7.4** **G2 gate** — `g2_sigmoid_ranking_preserved`:
  - Generate 100 synthetic pattern vectors + 100 hidden-state queries
  - For each query, compute cosine similarity to all 100 patterns (ground truth ranking)
  - Compute sigmoid gate (with RMSNorm) → ranking
  - Spearman rank-correlation > 0.95
- [ ] **T7.5** **G4 gate** — `g4_table_identity_deterministic`:
  - Generate random table contents, compute `EngramTableId`
  - Re-build table from same contents, compute `EngramTableId` again
  - Verify bit-identical (1000 random tables)
  - **G4 chain-half stub**: convert 10K `EngramHash` values to `LatCalFixed` (mock — actual bridge in riir-chain), serialize, deserialize, round-trip bit-identical
- [ ] **T7.6** **G6 gate** — `g6_effective_depth_smoke` (smoke version, full validation in riir-ai integration):
  - On the existing Bomber or Go inference pipeline, log per-layer LogitLens divergence
  - With vs without Engram fused at layer 2
  - Target: divergence at layer 5 with Engram ≤ divergence at layer 12 without
  - (This is a smoke test — full G6 runs in riir-ai integration per P TBD)
- [ ] **T7.7** **G7 gate** — `cargo test --workspace --all-features` with `engram` on: 0 regressions in 7400+ tests.
- [ ] **T7.8** **GOAT verdict**:
  - G1–G7 all PASS → promote `engram` from opt-in to default-on in `crates/katgpt-core/Cargo.toml` default features
  - Any gate FAIL → demote to experimental, file `.issues/` with diagnosis, do not promote
  - Per AGENTS.md: demote the loser (e.g., if Engram-Enabled causes regression in Raven path, demote Engram; if Engram path adds zero value, demote Engram)
- [ ] **T7.9** Add `katgpt-rs/README.md` Feature Showcase entry for Engram (after G1–G7 pass). Cross-link to Research 278 + Plan 299.
- [ ] **T7.10** Add example `examples/engram_demo.rs` — populate a small table from a few sentences, retrieve by N-gram suffix, sigmoid-fuse into a hidden state. ~150 lines, runs without GPU.

---

## Phase 8 — Documentation

### Tasks

- [ ] **T8.1** Module-level rustdoc in `engram/mod.rs`: what it does, when to use, the sparsity-axis framing (conditional memory vs conditional computation), reference to Research 278.
- [ ] **T8.2** Add `katgpt-rs/.docs/` entry — likely `27_engram_conditional_memory.md` covering: trait surface, when to enable, performance characteristics, comparison vs Raven (the other axis).
- [ ] **T8.3** Add `katgpt-rs/.benchmarks/299_engram_goat.md` with G1–G7 results table.
- [ ] **T8.4** Update `katgpt-rs/README.md` Feature Showcase + Paper Feature Comparison Matrix (`.docs/15_paper_feature_comparison.md`).

---

## File Change Summary

| File | Change |
|------|--------|
| `crates/katgpt-core/Cargo.toml` | Add `engram` feature (deps: blake3, papaya already present; `unicode-normalization` optional for tokenizer compression) |
| `crates/katgpt-core/src/lib.rs` | Export `engram` module behind feature gate |
| `crates/katgpt-core/src/engram/mod.rs` | Public API: EngramTable trait, EngramHash, K_MAX, EngramConfig |
| `crates/katgpt-core/src/engram/hash.rs` | MultiHeadHash, HashHead, multi_head_hash() |
| `crates/katgpt-core/src/engram/table.rs` | InMemoryEngramTable, EngramTableBuilder |
| `crates/katgpt-core/src/engram/tokenizer.rs` | SurjectiveMap, compress_token(), build_surjective_map() |
| `crates/katgpt-core/src/engram/kernel.rs` | sigmoid_fuse_into(), rmsnorm_into(), SigmoidFusionConfig |
| `crates/katgpt-core/src/engram/conv.rs` | Depthwise causal conv (paper §2.3) |
| `crates/katgpt-core/src/engram/hotswap.rs` | EngramHotSwap — AtomicPtr<Box<EngramTable>> |
| `crates/katgpt-core/src/engram/cache.rs` | ZipfianCacheHierarchy — tiered cache |
| `crates/katgpt-core/src/engram/commitment.rs` | EngramTableId, build_merkle_root() |
| `crates/katgpt-core/src/engram/forward.rs` | fuse_into_hidden_state() end-to-end hook |
| `tests/bench_299_engram_goat.rs` | G1–G7 GOAT gate tests |
| `examples/engram_demo.rs` | End-to-end demo |
| `benches/engram_micro.rs` | Criterion micro-benchmarks (lookup, sigmoid_fuse, hotswap) |

**Estimated total:** ~2500–3000 LOC across engine + tests + benches + example.

---

## Dependencies & Cross-References

- **Research note (open):** `katgpt-rs/.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md`
- **Private selling-point guide:** `riir-ai/.research/147_Engram_Conditional_Memory_NPC_Guide.md`
- **Chain commitment half (TODO):** `riir-chain/.research/001_Engram_LatCal_Commitment_Bridge.md` (the chain commitment half — file when work on the LatCal bridge starts)
- **Existing primitives reused:**
  - `SenseHotSwap` (`katgpt-rs/crates/katgpt-core/src/sense/hotswap.rs`) — AtomicPtr pattern
  - `MerkleOctree` / `MerkleProof` (R221, P253) — binary Merkle root
  - `simd::simd_dot_f32`, `simd::simd_outer_product_acc` — SIMD kernels
  - `types::rmsnorm` — RMSNorm helper (if signature fits)
  - `ChunkFetcher` trait (R262) — cold-tier fetcher pattern
  - `papaya::HashMap` (per AGENTS.md) — lock-free hash map for slot index
  - `blake3` (per AGENTS.md) — commitments
  - `Uuid::now_v7()` (per AGENTS.md) — snapshot IDs (for the hotswap version tag)

---

## TL;DR

Plan 299 = **Engram open primitive** — hash-addressed, sigmoid-fused static pattern memory in `katgpt-core`. Phase 1: hashing. Phase 2: frozen table + lookup. Phase 3: sigmoid fusion kernel (NEVER softmax per AGENTS.md). Phase 4: tokenizer compression (surjective V→V'). Phase 5: AtomicPtr hotswap + BLAKE3 commitment. Phase 6: Zipfian cache hierarchy (plasma/hot/warm/cold). Phase 7: end-to-end fuse + G1–G7 GOAT gate. Phase 8: docs. Feature flag `engram`, default OFF until G1–G7 pass. The open half of the Super-GOAT (Research 278) — private half is riir-ai Guide 147, chain half is riir-chain R001 (TODO).
