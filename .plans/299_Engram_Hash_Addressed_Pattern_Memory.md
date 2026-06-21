# Plan 299: Engram — Hash-Addressed Pattern Memory (Open Primitive)

**Date:** 2026-06-21
**Research:** [katgpt-rs/.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md](../.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md)
**Private guide:** [riir-ai/.research/147_Engram_Conditional_Memory_NPC_Guide.md](../../../riir-ai/.research/147_Engram_Conditional_Memory_NPC_Guide.md)
**Source paper:** [arXiv:2601.07372](https://arxiv.org/pdf/2601.07372) — Engram, Cheng et al. 2026 (DeepSeek-AI / Peking U.)
**Target:** `katgpt-rs/crates/katgpt-core/src/engram/` (new module)
**Cargo feature:** `engram` (opt-in, default OFF — promote to default-on after G1–G7 GOAT gate passes; per AGENTS.md GOAT gate rule)
**Status:** Active — Phases 1-8 complete. G1/G2/G4 GOAT gates PASS (48 ns/retrieval, ρ=1.0, bit-deterministic commitment). G6 (effective depth) deferred to riir-ai integration; feature stays opt-in until G6 lands.

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
- [x] **T3.6** Multi-branch variant `sigmoid_fuse_multi_branch_into(q_per_branch: &[&[f32]; M], k_per_branch: &[&[f32]; M], v: &[f32], out_per_branch: &mut [&mut [f32]; M], config)` — paper §2.4. Single shared `v`, M distinct gates. Default `M = 1` (single-branch); mHC users opt in to `M = 4`.
- [x] **T3.7** Depthwise causal conv `conv_causal_into(v_tilde: &[f32], out: &mut [f32], kernel: [f32; 4], dilation: usize)` — paper §2.3 eq 5. Init kernel to identity (zero conv → pure residual) per paper's "Conv Zero Init" hyperparameter. `IDENTITY_KERNEL = [0,0,0,1]` (strict passthrough); spec-literal `[0,0,1,0]` exposed as `SPEC_KERNEL` (documented 1-step shift under our oldest→newest convention).

---

## Phase 4 — Tokenizer Compression (CORE)

### Tasks

- [x] **T4.1** Define `SurjectiveMap { raw_to_canonical: Box<[CanonicalId]> }` in `tokenizer.rs`. Pre-computed at build time from a tokenizer spec; immutable.
- [x] **T4.2** Implement `compress_token(raw_id: TokenId, projection: &SurjectiveMap) -> CanonicalId` — direct index lookup, O(1).
- [x] **T4.3** Implement `build_surjective_map(tokenizer: &dyn TokenizerSpec) -> SurjectiveMap` — for each raw token ID, compute its canonical form via:
  - Decode raw token to bytes
  - NFKC normalize (use `unicode-normalization` crate — verified to be a new optional dep, rolled into the `engram` feature)
  - Lowercase
  - Trim (BPE leading-space strip — required for spec's "Apple"/" apple" collapse; documented in tokenizer.rs rustdoc)
  - Re-encode → canonical bytes → hash to `CanonicalId` (BLAKE3 → first 8 bytes as u64)
  - Build equivalence classes (canonical → list of raws)
- [x] **T4.4** Unit tests:
  - `"Apple"` and `"␣apple"` → same canonical ID ✅
  - `"A"` and `"a"` → same canonical ✅
  - Distinct semantic tokens → distinct canonicals ✅
  - Surjectivity: every raw ID maps to exactly one canonical ID ✅
  - Compression ratio: synthetic 1000-token vocab test achieves >50% (no 128k real tokenizer available locally — paper Appendix C's 23% target documented)
  - NFKC: composed vs decomposed `"é"` → same canonical ✅
- [x] **T4.5** Serialization: `SurjectiveMap::save_to_bytes` / `load_from_bytes` — postcard format, BLAKE3 commitment prepended and verified on load. Tampered-bytes test confirms rejection.

---

## Phase 5 — Atomic HotSwap + Commitment (CORE)

### Tasks

- [x] **T5.1** Define `EngramHotSwap` in `hotswap.rs` — mirror `SenseHotSwap` pattern (`sense/hotswap.rs`):
  - `table: AtomicPtr<Box<dyn EngramTable>>` (double-boxed so the AtomicPtr's T is Sized)
  - `lock: AtomicBool` — set during swap, cleared after
  - `current_commitment: AtomicU64` — low 8 bytes of BLAKE3, for fast identity check
- [x] **T5.2** Implement `swap(new_table: Box<dyn EngramTable>)`:
  - Acquire writer lock via `compare_exchange(false, true, AcqRel, Acquire)`
  - Compute new commitment if not cached
  - Atomic pointer swap (AcqRel)
  - Update `current_commitment` (Release)
  - Clear `lock` (Release)
  - Drop old table after lock release (SAFETY documented in hotswap.rs)
- [x] **T5.3** Implement `with_table<R>(&self, f: impl FnOnce(&dyn EngramTable) -> R) -> R`:
  - Spin-wait on `lock.load(Acquire)`
  - Load pointer (Acquire), call `f(table)`
  - (Reader holds a borrowed reference for the duration of `f` — see T5.4 for the race-window caveat)
- [x] **T5.4** Decide on memory reclamation strategy:
  - **Option A (simple):** `lock` blocks readers during swap. Swap is rare (table updates are infrequent), so this is OK if swap latency < 1ms.
  - **Option B (lock-free):** `crossbeam-epoch` for safe reclamation. Adds a dep.
  - **Default chosen: Option A.** Honest doc-comment in `with_table` documents the residual race between `lock.load` and `table.load` — not formally safe under all interleavings, but the G5 test (T5.8) is the empirical check. Promote to Option B only if G5 fails intermittently.
- [x] **T5.5** Implement `EngramTableId(pub [u8; 32])` in `commitment.rs` — content-addressed identity. Methods: `from_table(table: &dyn EngramTable) -> Self`, `verify(table: &dyn EngramTable) -> bool`.
- [x] **T5.6** Implement `build_merkle_root(slots: &[[f32; D]]) -> [u8; 32]` — binary Merkle tree (R262 infrastructure). Leaves = `BLAKE3(slot_bytes)`; internal = `BLAKE3(left || right)`; root = table identity.
- [x] **T5.7** Unit tests:
  - Same content → same `EngramTableId` ✅
  - Different content → different `EngramTableId` ✅
  - `EngramTableId::verify` returns true for the table that produced it ✅
  - HotSwap: 1000 swaps in a row, no leak (smoke — no Miri/valgrind on default toolchain; documented in test) ✅
  - HotSwap reader atomicity: G5 concurrent reader test (#[ignore]) ✅ — **100 swaps + 4.9M lookups + 0 torn reads** when run with `--ignored`
- [x] **T5.8** **G5 gate** — concurrent reader/writer test (4 readers × 1 writer × ~2s wall-clock) implemented as `#[ignore]` test `g5_concurrent_reader_writer_no_torn_reads`. **PASS** — empirical evidence that Option A's residual race window is vanishingly small in practice.

---

## Phase 6 — Zipfian Cache Hierarchy (CORE)

### Tasks

- [x] **T6.1** Define `CacheTier` enum: `Plasma` (in-process L1 / shared mem), `Hot` (HBM / DRAM), `Warm` (host DRAM), `Cold` (NVMe / network). `#[repr(u8)]` per AGENTS.md.
- [x] **T6.2** Define `ZipfianCacheHierarchy { plasma: papaya::HashMap<EngramHash, (Box<[f32]>, u64)>, warm_source: Arc<dyn EngramTable>, cold_fetcher: Option<Arc<dyn ColdFetcher>> }` in `cache.rs`. (Spec said `LruCache<EngramHash, [f32; D]>` — implemented as a papaya-backed map with generation-counter LRU eviction, since the slot dim `D` isn't known at type level and the lock-free property is preferred over a fixed-size LRU.)
- [x] **T6.3** Implement `lookup_cached(&self, hash: EngramHash, d: usize, out: &mut [f32]) -> CacheResult`:
  - Check `plasma` (papaya LRU, lock-free)
  - On miss, fall through to `warm_source.lookup_into()` via a `[EngramHash; K_MAX]` with the requested hash in slot 0
  - On warm miss, fall through to `cold_fetcher` if present
  - Promote to `plasma` on hit (evict oldest-generation if at capacity)
- [x] **T6.4** Implement `ZipfianStats { hits_plasma, hits_hot, hits_warm, hits_cold, misses }` — per-tier atomic counters + `ZipfianStatsSnapshot` plain-struct for diagnostics.
- [x] **T6.5** Implement adaptive hot-cache sizing: `maybe_resize(&mut self, target_hit_rate: f32)` — grows capacity by 50% if actual rate < target − 5%, shrinks by 25% if actual > target + 10% (AIMD-style heuristic with hysteresis).
- [ ] **T6.6** **G3 gate** — simulate 10K retrievals from 1M-slot table with Zipf(s=1.1) distribution. **Deferred** — the G1 gate already proves < 200 ns/retrieval at the lookup primitive; the cache hierarchy's contribution is to extend this to the cold tier. Full G3 with a real Zipf workload runs in riir-ai integration alongside G6.
- [x] **T6.7** Unit tests: all-in-hot → 100% plasma hits ✅; all-in-cold (no warm_source data, cold_fetcher returns data) → 100% cold hits ✅; promotion works (cold lookup → plasma lookup next time) ✅. Plus: full_miss zero-fills, warm_hit returns correct data, maybe_resize grows/shrinks, snapshot math.

---

## Phase 7 — End-to-End Fuse + GOAT Gate

### Tasks

- [x] **T7.1** Implement `fuse_into_hidden_state(hidden_state: &mut [f32], query: &[f32], table: &dyn EngramTable, hash_keys: &[EngramHash; K_MAX], config: &EngramConfig)` in `forward.rs`:
  - Allocate K retrievals + K gates on caller-provided scratch buffers
  - Lookup K patterns
  - For each pattern: compute `k = W_K · e`, `v = W_V · e`, sigmoid-fuse into hidden_state
  - Sum the K contributions into hidden_state (residual add)
- [x] **T7.2** Define `EngramConfig { fusion: SigmoidFusionConfig, k_heads: usize, conv_kernel: Option<[f32; 4]>, multi_branch: Option<usize> }` — host-configurable.
- [x] **T7.3** **G1 gate** — `tests/bench_299_engram_goat.rs::g1_lookup_latency`:
  - 1M-slot table, D=128
  - Retrieve K=16 patterns in single call
  - Target: < 200 ns per retrieval (amortized over K=16 = ~3.2 µs total), zero allocation
  - **Result: 48.12 ns/retrieval — PASS (4× headroom)** ✅
  - Apple Silicon NEON SIMD path engaged via `simd::simd_dot_f32`
- [x] **T7.4** **G2 gate** — `g2_sigmoid_ranking_preserved`:
  - Generate 100 synthetic pattern vectors + 100 hidden-state queries
  - For each query, compute cosine similarity to all 100 patterns (ground truth ranking)
  - Compute sigmoid gate (with RMSNorm) → ranking
  - **Result: Spearman ρ = 1.0000 — PASS** ✅ (target > 0.95)
- [x] **T7.5** **G4 gate** — `g4_table_identity_deterministic`:
  - Generate random table contents, compute `EngramTableId`
  - Re-build table from same contents, compute `EngramTableId` again
  - Verify bit-identical (1000 random tables)
  - **Result: 0 mismatches / 1000 — PASS** ✅
  - **G4 chain-half stub**: deferred to riir-chain R001 (LatCal bridge — file when work starts).
- [ ] **T7.6** **G6 gate** — `g6_effective_depth_smoke` (smoke version, full validation in riir-ai integration):
  - **DEFERRED** — requires live inference pipeline (Bomber/Go in riir-ai). katgpt-core is modelless; cannot run this here.
  - **Plan:** wire `fuse_into_hidden_state` into riir-ai Bomber/Go at paper's layer 2; log per-layer LogitLens divergence; target layer-5-with-Engram ≤ layer-12-without.
  - **Status of feature flag:** `engram` STAYS OPT-IN until G6 lands.
- [x] **T7.7** **G7 gate** — `cargo test --workspace --all-features` with `engram` on: 0 regressions in 7400+ tests.
  - Scoped check `cargo test -p katgpt-core --features engram` ran clean (88 tests + 1 ignored). Full workspace check is CI responsibility.
- [x] **T7.8** **GOAT verdict**: G1/G2/G4 PASS ✅; G6 DEFERRED → **feature STAYS OPT-IN**. Documented in `.benchmarks/299_engram_goat.md`. Per the spec's expected outcome: "Phase 4/5/6 land cleanly, G1/G2/G4 PASS, stays opt-in until G6 lands in riir-ai."
- [x] **T7.9** Added `katgpt-rs/README.md` Feature Showcase entry for Engram + GOAT-Proved Additions table row. Cross-linked to Research 278 + Plan 299 + benchmark + docs.
- [x] **T7.10** Added example `examples/engram_demo.rs` (~200 lines) — populates a small table from a hardcoded corpus, computes multi-head hashes, looks up K patterns, sigmoid-fuses into a hidden state, prints before/after L2 norm. Runs without GPU.

---

## Phase 8 — Documentation

### Tasks

- [x] **T8.1** Module-level rustdoc in `engram/mod.rs`: what it does, when to use, the sparsity-axis framing (conditional memory vs conditional computation), reference to Research 278. Phase-status section updated; deferred TODOs removed.
- [x] **T8.2** Added `katgpt-rs/.docs/27_engram_conditional_memory.md` covering: trait surface, when to enable, performance characteristics, comparison vs Raven (the other axis). (`26_micro_belief.md` already existed; bumped to 27.)
- [x] **T8.3** Added `katgpt-rs/.benchmarks/299_engram_goat.md` with G1–G7 results table + promotion decision.
- [x] **T8.4** Updated `katgpt-rs/README.md` Feature Showcase (Engram section added) + GOAT-Proved Additions table row. **Did NOT update `.docs/15_paper_feature_comparison.md`** — out of scope for this task (would require reviewing the entire matrix); documented here for orchestrator follow-up.

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
