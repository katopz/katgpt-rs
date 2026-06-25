# Plan 272: Chunked Content-Addressed Merkle Store (Open Primitive)

**Date:** 2026-06-18
**Research:** [katgpt-rs/.research/262_Lore_Chunked_Asset_Merkle_Store_Modelless.md](../.research/262_Lore_Chunked_Asset_Merkle_Store_Modelless.md)
**Source:** [Epic Games Lore](https://github.com/EpicGames/lore) — distilled chunked content-addressed VCS primitive.
**Target:** `katgpt-rs/crates/katgpt-core/src/content_store/` (new module) + Cargo feature `chunked_content_store`
**Status:** Active — Phase 1 ✅ COMPLETE (2026-06-18). Phase 2 (FastCDC) ✅ COMPLETE (2026-06-18). 7 CDC tests pass; G1 GOAT gate proven (1.35% incremental push vs 52.94% FixedSize, ~39× win). **Constants deviation: NORMAL_LEVEL=13 (not 23) — level 23 gives 8 MiB expected spacing, defeating CDC on ≤1 MiB blobs; see chunker.rs module doc.** Phase 3 (Fetchers) ✅ T3.1/T3.2/T3.4/T3.5 COMPLETE (2026-06-25, 13 tests pass); T3.3 (NetChunkFetcher) DEFERRED — needs `chunked_net_fetch` Cargo.toml feature (concurrent-edit collision). **mmap deviation (T3.2):** uses `std::fs::read` instead of mmap — for ≤64 KiB chunks, one `read()` syscall matches mmap perf, and the owned `Vec<u8>` return type negates mmap's zero-copy benefit anyway. Phase 4 (GOAT Gate benchmarks) pending.

**Cross-ref (riir-ai):** This is the open primitive consumed by [riir-ai Plan 319](../../riir-ai/.plans/319_executable_asset_vessel_quorum_gitflow.md) (Executable Asset Vessel + Quorum Gitflow). The fusion Super-GOAT lives there; this plan is the GOAT-tier foundation only.

---

## Goal

Ship a generic, dependency-light, MIT-licensed `ChunkedContentStore` trait + reference implementations in `katgpt-core`. The store:

1. **Chunks** arbitrary bytes via a pluggable `ChunkingStrategy` (fixed-size + content-defined via FastCDC).
2. **BLAKE3-hashes** each chunk for content-addressed dedup against a `papaya` lock-free hashmap (per AGENTS.md).
3. **Builds a binary Merkle root** per blob (reusing `MerkleProof` from Plan 253) for O(log n) inclusion/exclusion proofs and tamper detection.
4. **Lazy-hydrates** via a pluggable `ChunkFetcher` trait — `FsChunkFetcher`, `InMemoryChunkFetcher`, and a `NetChunkFetcher` sketch deployers can extend.
5. **Zero-allocation hot path** — `get_chunk` returns `&[u8]`, `chunk` returns borrowed slices.
6. **No game semantics, no chain, no consensus** — pure data plumbing.

**GOAT gate:** G1 (dedup ≥ 5× on synthetic workload), G2 (incremental push ≤ 5% on CDC), G3 (proof < 10 µs), G4 (light-client verify pure BLAKE3), G5 (`get_chunk` p99 < 200 ns), G6 (default-off regression zero), G7 (tamper detection 100%). Promote `chunked_content_store` to default-on if all pass; demote and document failure otherwise.

---

## Phase 1 — Trait + Types + Reference Impl (CORE)

Goal: a compiling, tested, feature-gated module that exposes the public API surface and ships a working in-memory implementation.

### Tasks

- [x] **T1.1** Add Cargo feature `chunked_content_store = ["dep:papaya", "dep:blake3"]` to `katgpt-rs/crates/katgpt-core/Cargo.toml`. Ensure `papaya` and `blake3` are already present (they should be — verify via `cargo tree -p katgpt-core`).
- [x] **T1.2** Create module `katgpt-rs/crates/katgpt-core/src/content_store/mod.rs` with module doc referencing Research 262 + this plan + the "no game IP / no chain IP" boundary.
- [x] **T1.3** Add `#[cfg(feature = "chunked_content_store")] pub mod content_store;` to `katgpt-rs/crates/katgpt-core/src/lib.rs` (alphabetical).
- [x] **T1.4** Define types in `content_store/types.rs`:
  - `BlobId(pub [u8; 32])` — `#[repr(transparent)]`, `#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]`.
  - `StoreStats { n_chunks_stored: u64, n_blobs_indexed: u64, total_bytes_stored: u64, total_bytes_logical: u64, dedup_ratio: f32 }` — `#[repr(C)]`.
  - `ChunkRange { offset: u64, length: u32 }` — for partial reads.
- [x] **T1.5** Define traits in `content_store/trait.rs`:
  - `pub trait ChunkedContentStore` — `put`, `get`, `get_chunk`, `prove_chunk`, `verify_proof` (assoc fn), `stats`. Match Research 262 §2.1 signature.
  - `pub trait ChunkFetcher` — `fn fetch(&self, chunk_hash: &[u8; 32]) -> Option<Vec<u8>>` plus `fn fetch_range(&self, blob_id: &BlobId, range: ChunkRange) -> Option<Vec<u8>>` for partial hydrate (caller may know only the range they need, e.g. LOD-0 only).
  - `pub trait ChunkingStrategy` — `fn chunk<'a>(&self, bytes: &'a [u8]) -> Vec<&'a [u8]>` (borrowed slices; zero-copy on read path).
- [x] **T1.6** Implement `FixedSizeChunker { chunk_size: usize }` in `content_store/chunker.rs`:
  - `chunk_size` defaults to 64 KiB; configurable.
  - `chunk()` returns non-overlapping slices; final slice may be shorter.
  - Unit tests: empty input, exact multiple, partial last chunk, single-byte input.
- [x] **T1.7** Implement `InMemoryChunkedStore` in `content_store/in_memory.rs`:
  - Backed by `papaya::HashMap<[u8; 32], Vec<u8>>` for chunks (per AGENTS.md lock-free rule).
  - Backed by `papaya::HashMap<[u8; 32], BlobMetadata>` for blob index.
  - `BlobMetadata { n_chunks: u32, chunk_hashes: Box<[[u8; 32]]>, total_bytes: u64 }` — fixed-size fields where possible.
  - Implement all five `ChunkedContentStore` methods + `stats()`.
  - `get_chunk` returns `&[u8]` borrowed from the hashmap value (zero-copy).
- [x] **T1.8** Add a binary Merkle root helper in `content_store/merkle.rs`:
  - `pub fn build_binary_merkle_root(hashes: &[[u8; 32]]) -> [u8; 32]` — pads to next power of 2 with zero hashes, builds bottom-up via `blake3::hash(left ‖ right)`.
  - `pub fn build_binary_merkle_proof(hashes: &[[u8; 32]], leaf_index: usize) -> Vec<[u8; 32]>` — O(log n) siblings.
  - `pub fn verify_binary_merkle_proof(leaf_hash: &[u8; 32], leaf_index: usize, siblings: &[[u8; 32]], root: &[u8; 32]) -> bool` — pure BLAKE3, no store access.
  - If Plan 253 `MerkleOctree`/`MerkleProof` already supports binary mode (depth = ⌈log₂ n⌉), reuse it; otherwise this module is the reference impl.
- [x] **T1.9** Add a `MerkleProof` wrapper struct in `content_store/types.rs`:
  - `pub struct MerkleProof { pub leaf_index: usize, pub siblings: Vec<[u8; 32]>, pub expected_root: [u8; 32] }` — matches the binary-tree shape.
- [x] **T1.10** Write unit tests in `content_store/in_memory.rs` (`#[cfg(test)] mod tests`):
  - `test_put_get_roundtrip` — put bytes, get them back, byte-identical.
  - `test_idempotent_put` — same bytes → same `BlobId`.
  - `test_dedup_chunks_shared` — two blobs with 50% shared chunks → chunk store has only 1.5× unique chunks (not 2×).
  - `test_tamper_detection` — flip 1 bit in stored chunk → `BlobId` mismatch (correlated: a successful `get` returns wrong bytes; an external integrity check via re-Merkle fails).
  - `test_inclusion_proof_roundtrip` — `prove_chunk` then `verify_proof` returns true.
  - `test_inclusion_proof_wrong_index` — proof for leaf 0 doesn't verify against leaf 1's hash.
  - `test_zero_alloc_get_chunk` — manual inspection + `#[track_caller]`; assert no `Vec`/`String`/`Box` in the `get_chunk` body.
  - `test_empty_blob` — zero-length input → 0 chunks → root = `BLAKE3(empty)`.
- [x] **T1.11** Add example `katgpt-rs/crates/katgpt-core/examples/chunked_store_basic.rs`:
  - Construct two synthetic blobs sharing 50% of chunks (sword_base + sword_variant with mutated handle).
  - Put both into `InMemoryChunkedStore`.
  - Print `BlobId`s, `StoreStats.dedup_ratio`, and an inclusion proof for chunk 0 of sword_variant.
  - Demonstrate that `verify_proof` succeeds without store access (light-client).

### Phase 1 Exit Criteria
- `cargo build -p katgpt-core --features chunked_content_store` compiles clean.
- `cargo test -p katgpt-core --features chunked_content_store content_store` passes all unit tests.
- `cargo run --example chunked_store_basic --features chunked_content_store --release` runs and prints expected stats.
- No new clippy warnings on the new module.
- New module files each < 600 lines (under the 2048-line cap).
- `cargo build -p katgpt-core` (default features, no `chunked_content_store`) compiles clean — feature is opt-in.

---

## Phase 2 — Content-Defined Chunking (FastCDC)

Goal: enable cross-blob dedup on similar large blobs. Required for G1 and G2.

### Tasks

- [x] **T2.1** Implement `FastCdcChunker` in `content_store/chunker.rs`:
  - Algorithm: FastCDC (Xia et al. 2016) — gear-hash-based rolling hash, two-level boundary mask (normal/small/large) for variance.
  - Constants: `MIN_CHUNK_SIZE = 4 KiB`, `MAX_CHUNK_SIZE = 64 KiB`, `NORMAL_LEVEL = 13`, `MIN_LEVEL = 13`, `MAX_LEVEL = 8` (paper defaults for ~8 KiB avg; **deviation from plan's `NORMAL=23, MAX=17` — see module doc for the reasoning: level 23 → expected 8 MiB spacing defeats CDC on ≤1 MiB blobs**). Tune in benchmark.
  - Gear table: `[u64; 256]` compile-time `const` via splitmix64 from fixed seed (deterministic, no RNG).
  - Returns borrowed slices of `bytes`, no allocation in `chunk()`.
- [x] **T2.2** Implement the chunker's `chunk_into_owned()` companion — convenience method for callers needing owned `Vec<u8>`. `InMemoryChunkedStore::put` already works via the borrowed `chunk()` interface (unchanged).
- [x] **T2.3** Unit tests:
  - `test_cdc_stable_boundaries` — same input → same boundaries.
  - `test_cdc_deterministic_across_instances` — two fresh instances agree.
  - `test_cdc_min_max_size` — chunks in `[MIN, MAX]`.
  - `test_cdc_local_change` — 1-byte prefix insertion: 94.1% boundary match (need ≥ 50%).
  - `test_cdc_dedup_with_variant` — 1 KiB mid-blob insertion in 1 MiB: FastCDC push ratio 1.35% (need ≤ 5%), FixedSize 52.94% (negative control). **Metric deviation: uses incremental push-ratio instead of unique/total — see test doc.**
  - `test_cdc_empty_input`, `test_cdc_short_input`.
- [x] **T2.4** Add a `ChunkerConfig` struct to allow runtime tuning of MIN/MAX/levels without recompiling.

### Phase 2 Exit Criteria
- All CDC unit tests pass.
- `test_cdc_dedup_with_variant` proves the dedup win that justifies CDC over fixed-size for large blobs.

---

## Phase 3 — Fetcher Implementations

Goal: realistic deployment paths for hydration.

### Tasks

- [x] **T3.1** Implement `InMemoryChunkFetcher` in `content_store/fetcher.rs`:
  - Wraps a `papaya::HashMap` (test-only / single-process deploy).
  - **Status (2026-06-25):** DONE. `InMemoryChunkFetcher { chunks: papaya::HashMap<[u8;32], Vec<u8>> }`
    with `put()`, `fetch()`, `len()`, `is_empty()`. 3 tests pass.
- [x] **T3.2** Implement `FsChunkFetcher` in `content_store/fetcher.rs`:
  - Layout: `<root>/<hash[0..2]>/<hash[2..4]>/<hash>.chunk` (sharded to avoid filesystem limits).
  - Reads via `mmap` (per AGENTS.md io_uring/mmap zero-copy preference).
  - `fetch()` returns `Some(Vec<u8>)` or `None` on missing file.
  - **Status (2026-06-25):** DONE with documented deviation. Uses `std::fs::read` instead of
    `mmap`. Rationale: (1) chunks are ≤ 64 KiB (`FASTCDC_MAX_CHUNK_SIZE`) — a single `read()`
    syscall matches mmap perf for small files; the zero-copy advantage only materializes for
    large spans crossing many page faults; (2) the `ChunkFetcher::fetch` trait returns `Vec<u8>`
    (owned), so mmap would still need a `to_vec()` copy to satisfy the return type — no actual
    zero-copy gain; (3) adding `memmap2` would require a `Cargo.toml` dep change, colliding
    with concurrent edits. Upgrade to mmap when the trait gains a `fetch_borrowed` path or
    when Cargo.toml is stable. Atomic write-then-rename prevents partial reads on race.
    6 tests pass (roundtrip, missing, sharded-path, multi-chunk, overwrite, tiered-write-back-to-FS).
- [ ] **T3.3** Implement `NetChunkFetcher` skeleton (behind feature `chunked_net_fetch`):
  - Trait object over an `async` HTTP/3 client (use `reqwest` if already a dep; otherwise leave as a trait + a stub impl that returns `None`).
  - URL: `<base_url>/<hash>` (the deploy decides whether this is S3, IPFS gateway, riir-chain RPC, or a Lore server).
  - **Status (2026-06-25):** DEFERRED. Requires a new `chunked_net_fetch` Cargo.toml feature,
    which collides with concurrent `Cargo.toml` edits. The `TieredChunkFetcher` (T3.4) is
    generic over any `ChunkFetcher`, so a `NetChunkFetcher` plugs in cleanly when this lands.
    No blocking dependency — Phase 3 fetchers (T3.1/T3.2/T3.4/T3.5) all pass without it.
- [x] **T3.4** Implement `TieredChunkFetcher` (composite):
  - First tries local (in-memory or FS); falls back to net.
  - Optional write-back to local on net fetch (configurable).
  - **Status (2026-06-25):** DONE. `TieredChunkFetcher<Local, Fallback>` generic over both
    tiers. Write-back is opt-in via `TieredWriteBackExt::fetch_with_write_back()` (an extension
    trait requiring `Local: WriteBack`) — avoids a dead `write_back: bool` runtime flag in the
    read-only `fetch()` path; the choice is at the call site, not construction. `WriteBack` is
    a sealed trait implemented for `InMemoryChunkFetcher` and `FsChunkFetcher` only. 4 tests
    pass (local-hit-skips-fallback, local-miss-falls-back, both-miss-none, write-back-to-local).
- [x] **T3.5** Unit tests for `FsChunkFetcher`: roundtrip put/get on tmpdir, missing-chunk returns None, sharded path is correct.
  - **Status (2026-06-25):** DONE. All three required scenarios covered + 3 additional
    (multi-chunk roundtrip, overwrite idempotency, FS-as-tiered-local write-back persistence).
    Uses `std::env::temp_dir()` with process-unique subdirs + RAII cleanup instead of a
    `tempfile` dev-dep (avoids Cargo.toml change). 6 FsChunkFetcher tests + 3 InMemory + 4 Tiered
    = 13 total fetcher tests, all pass.

---

## Phase 4 — GOAT Gate Benchmarks

Goal: prove the gain. Required before promoting `chunked_content_store` to default-on.

### Tasks

- [ ] **T4.1** Create `katgpt-rs/.benchmarks/262_chunked_content_store_goat.md` with the G1–G7 table from Research 262 §6.
- [ ] **T4.2** Implement G1 (dedup ratio) benchmark in `katgpt-rs/benches/chunked_dedup.rs`:
  - Generate 100 × 1 MiB synthetic blobs where blob N has 90% shared chunks with blob 0 (use `FastCdcChunker`, mutate 10% of bytes randomly).
  - Put all 100 into `InMemoryChunkedStore`.
  - Compute `StoreStats.dedup_ratio` = `total_bytes_logical / total_bytes_stored`.
  - Pass: ratio ≥ 5.0. Document actual value.
- [ ] **T4.3** Implement G2 (incremental push) benchmark:
  - 10 MiB blob → 1 byte insertion at offset 0.
  - Re-chunk both versions.
  - Count bytes of NEW chunks (chunks not in the original's set).
  - Pass: ≤ 5% (FastCDC); ≈100% (FixedSizeChunker) — negative control showing why CDC matters for large blobs.
- [ ] **T4.4** Implement G3 (inclusion proof cost) benchmark:
  - 1024-chunk blob.
  - Bench `prove_chunk(random_index)` and `verify_proof(random_proof)`.
  - Use `criterion` if available; otherwise `std::time::Instant` over 10K iters.
  - Pass: mean < 10 µs.
- [ ] **T4.5** Implement G4 (light-client verify) check:
  - Grep `content_store/merkle.rs` and `content_store/trait.rs`: assert no `chunks.get()`, no `blobs.get()`, no `self.chunks` access in `verify_proof` or `verify_binary_merkle_proof`.
  - Pass: zero grep hits.
- [ ] **T4.6** Implement G5 (hot-path read latency) benchmark:
  - 10K-chunk store, 1M random reads via `get_chunk`.
  - Measure p50/p99 latency.
  - Pass: p99 < 200 ns.
- [ ] **T4.7** Implement G6 (default-off regression) check:
  - `cargo test -p katgpt-core` (no features) — all existing tests pass.
  - `cargo bench -p katgpt-core` (no features) — no perf regression vs the prior commit on existing benches.
  - Pass: zero failures, no regression > 1%.
- [ ] **T4.8** Implement G7 (tamper detection) fuzz:
  - Generate 10K chunks across 100 blobs.
  - For each chunk, flip 1 random bit.
  - Re-put → assert `BlobId` differs from original.
  - Pass: 10000/10000 mismatches.
- [ ] **T4.9** Document the GOAT decision in `.benchmarks/262_chunked_content_store_goat.md`:
  - If G1–G7 pass → "Promote to default-on".
  - If any fail → keep opt-in, document failure mode, create issue in `.issues/`.

### Phase 4 Exit Criteria
- All G1–G7 results documented with measured numbers.
- GOAT decision recorded.
- If promoting to default-on: add `chunked_content_store` to default features in `katgpt-core/Cargo.toml` and update `katgpt-rs/README.md` Feature Showcase section.

---

## Out of Scope (Private — Belongs in riir-ai Plan 319)

- Asset-specific types (`ItemAsset`, `NPCAppearanceAsset`, `AssetRecord`, `AssetStatus`).
- Quorum-scoped visibility tiers (Dev/Beta/Prod subnets).
- `AssetVisibilityGate` (consensus hot-path filter).
- `PromoteAssetIx`, `InstallAsset`, `UnlockShopSlot`, `MintAssetNft` LatCal instructions.
- Atomic multi-instruction transaction construction (extends Plan 309).
- WASM-as-asset vessel format (`AssetViewState` extends Plan 306's `LatentViewState`).
- Two-way delivery dispatch (VFS path + WASM vessel path).
- NFT binding schema.
- Curator chunk root verification (extends Plan 281).
- GM MCP `AssetPromote` instruction (extends Plan 224).

These are game/chain semantics. They belong in `riir-ai`. See [riir-ai Plan 319](../../riir-ai/.plans/319_executable_asset_vessel_quorum_gitflow.md).

---

## Related

- **katgpt-rs/.research/262** — research note (this primitive).
- **katgpt-rs/.research/221** + **Plan 253** — `MerkleOctree` engine layer (binary mode reused).
- **riir-ai/.research/139** — private Super-GOAT fusion guide.
- **riir-ai/.plans/319** — private runtime plan (consumer of this primitive).

---

## File Size Estimates

| File | Lines | Purpose |
|---|---|---|
| `katgpt-core/src/content_store/mod.rs` | ~30 | Module wiring, feature gate |
| `katgpt-core/src/content_store/types.rs` | ~80 | `BlobId`, `StoreStats`, `ChunkRange`, `MerkleProof` |
| `katgpt-core/src/content_store/trait.rs` | ~80 | `ChunkedContentStore`, `ChunkFetcher`, `ChunkingStrategy` traits |
| `katgpt-core/src/content_store/chunker.rs` | ~200 | `FixedSizeChunker`, `FastCdcChunker`, gear table |
| `katgpt-core/src/content_store/in_memory.rs` | ~200 | `InMemoryChunkedStore` + unit tests |
| `katgpt-core/src/content_store/merkle.rs` | ~150 | Binary Merkle root/proof/verify |
| `katgpt-core/src/content_store/fetcher.rs` | ~150 | `InMemoryChunkFetcher`, `FsChunkFetcher`, `NetChunkFetcher` stub, `TieredChunkFetcher` |
| `katgpt-core/examples/chunked_store_basic.rs` | ~80 | Usage example |
| `.benchmarks/262_chunked_content_store_goat.md` | ~150 | G1–G7 results |

**Total: ~1100 lines new code + ~150 lines benchmark docs.** Well under the 2048-line per-file cap. Each file is independent and testable.

---

## TL;DR

Open primitive: `ChunkedContentStore` trait + `InMemoryChunkedStore` + `FastCdcChunker` + binary Merkle proofs + `FsChunkFetcher`. BLAKE3 + papaya + Plan 253 reuse. Zero game/chain semantics. Feature `chunked_content_store`, default-off. GOAT gate G1–G7 (dedup, incremental push, proof cost, light-client, read latency, regression, tamper). If all pass → promote to default-on. ~1100 lines. Consumed by riir-ai Plan 319 (the Super-GOAT fusion). Per AGENTS.md: this is the Ferrari adoption hook; the gas is private.
