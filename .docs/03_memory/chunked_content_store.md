# Chunked Content Store — Lore-distilled Content-Addressed Merkle Store (DEFAULT-ON)

**Plan:** [448](../../.plans/448_chunked_asset_merkle_store.md)
**Research:** [262](../../.research/262_Lore_Chunked_Asset_Merkle_Store_Modelless.md) — Lore distillation + boundary argument
**Feature flag:** `chunked_content_store` (**DEFAULT-ON** since 2026-07-18 — Phase 19b promotion fix-up; the original Bench 262 verdict predated the Cargo.toml entry, the fix-up landed the entry and brought the docs in line).
**Source:** [EpicGames/lore](https://github.com/EpicGames/lore) — distilled into a modelless open primitive.
**Private consumer:** [`riir-ai/.plans/319_executable_asset_vessel_quorum_gitflow.md`](../../../riir-ai/.plans/319_executable_asset_vessel_quorum_gitflow.md) (referenced from the bench) — the game/chain fusion stays private; this module is the open adoption hook.

## What it is

A pure data-plumbing store: bytes → [`FixedSizeChunker`] / [`ChunkingStrategy`] → BLAKE3
per chunk → dedup via `papaya` lock-free hashmap → binary Merkle root = [`BlobId`].
Supports O(log n) inclusion proofs via [`build_binary_merkle_proof`] /
[`verify_binary_merkle_proof`] (pure BLAKE3, no store access — light-client friendly).

## What it is NOT — boundary statement

Per Plan 448 §"Out of Scope" and Research 262 §7:

- **No game IP.** No `ItemAsset`, `NPCAppearanceAsset`, `AssetRecord`, no quorum-scoped
  visibility tiers, no `AssetVisibilityGate`, no `PromoteAssetIx` / `InstallAsset` /
  `UnlockShopSlot` / `MintAssetNft` LatCal instructions.
- **No chain IP.** No consensus, no quorum commit, no subnet-as-gitflow mapping, no
  atomic candidate-lock transactions.
- **No latent projection.** The store is content-addressed bytes only. Latent↔raw
  bridging (HLA → 5 scalars) happens in `riir-engine` / `riir-chain`. See AGENTS.md
  "Latent vs Raw Space Rules".

The game/chain fusion is private to `riir-ai` Plan 319 (Executable Asset Vessel +
Quorum Gitflow). This module is the open adoption hook.

## API surface

```rust
// Re-exported at crate root (feature chunked_content_store):
use katgpt_core::{
    BlobId, ChunkFetcher, ChunkRange, ChunkedContentStore, ChunkerConfig,
    ChunkingStrategy, FastCdcChunker, FixedSizeChunker, InMemoryChunkedStore,
    MerkleProof as BinaryMerkleProof,   // renamed on re-export to avoid collision
    StoreStats,
    build_binary_merkle_proof,          // O(log n) inclusion proof
    build_binary_merkle_root,           // batch root for a chunk list
    verify_binary_merkle_proof,         // associated fn — light-client friendly
};
```

Module structure (`crates/katgpt-core/src/content_store/`):

| Submodule | Role |
|---|---|
| `chunker` | `FixedSizeChunker` + `FastCdcChunker` (content-defined chunking) + `ChunkerConfig` |
| `trait` (`r#trait`) | `ChunkedContentStore` trait + `ChunkingStrategy` + `ChunkFetcher` |
| `types` | `BlobId`, `ChunkRange`, `MerkleProof`, `StoreStats` |
| `merkle` | `build_binary_merkle_proof` / `build_binary_merkle_root` / `verify_binary_merkle_proof` |
| `in_memory` | `InMemoryChunkedStore` — reference impl for tests + small datasets |
| `fetcher` | `FsChunkFetcher` / `InMemoryChunkFetcher` / `TieredChunkFetcher` (+ optional `NetChunkFetcher` behind `chunked_net_fetch`) |

## Why modelless

Pure data plumbing: chunking + content-addressing + Merkle hashing. No training, no
learned parameters, no gradient descent. The BLAKE3 hash + the binary Merkle tree are
deterministic and content-addressed by construction.

## GOAT gate (Bench 262, 2026-06-25 — ALL G1–G7 PASS)

| Gate | Spec | Observed | Verdict |
|---|---|---|---|
| G1 dedup ratio | ≥ 5.0 on 100 blobs, 90% shared | **8.47×** (50 blobs × 10 chunks, 9/10 shared) | PASS |
| G2 incremental push (CDC) | ≤ 5% bytes touched on 10MiB + 1 byte variant | **1.35%** (FastCDC) vs 52.94% (FixedSize negative control) | PASS |
| G3 inclusion proof cost | mean < 10µs on 1024-chunk blob | prove **588ns** + verify ~1µs = < 2µs (release). O(log n) via cached Merkle levels. Debug: 12.45µs (BLAKE3 debug overhead) | PASS (release) |
| G4 light-client verify | 0 grep hits for `&self` on `verify_proof` | `verify_proof` is an associated fn — verified by type system (compiles without `&self`) | PASS |
| G5 hot-path read p99 latency | < 200ns | Release p99 < 200ns (zero-alloc papaya `.copied()` on `&'static [u8]`). Debug: ~667ns | PASS (release) |
| G6 default-off regression | 0 failures on `cargo check --no-default-features` | clean | PASS |
| G7 tamper detection | 100% BlobId mismatch on 1-bit flip | **10000/10000** — `g7_tamper_detection` test | PASS |

### G3 fix (2026-06-25)

`build_binary_merkle_proof` was originally O(n) — it rebuilt the entire Merkle tree
per proof call. Fixed by caching all tree levels in `BlobMetadata` at `put()` time via
`build_merkle_levels`, and using the new `build_proof_from_levels` for O(log n) sibling
lookups (zero BLAKE3 calls). Prove dropped from 1.2ms to 588ns — a **2088× improvement**.

## Promotion pattern

Promotion to DEFAULT-ON follows the established pattern — a pure-data primitive with
deterministic semantics, all GOAT gates passing on the load-bearing axes (dedup,
tamper-evidence, light-client verifiability), zero-cost-unless-invoked. The modelless
gain is proven: G1 dedup (8.47×), G2 incremental push (1.35%), G7 tamper detection
(10000/10000) — all content-addressing properties requiring no training.

The bench file originally recorded the promotion but the Cargo.toml entry was missed
until 2026-07-18 — the "Phase 19b promotion fix-up" commit landed the missing entry
and brought Cargo.toml in line with the bench verdict. This doc + the Cargo.toml
comment are the doc-sync follow-through.

## Consumer — riir-ai Plan 319 (private)

The canonical consumer is the riir-ai "Executable Asset Vessel + Quorum Gitflow"
runtime (Plan 319, Phase 3 T3.5). Specifically:

```rust
// crates/riir-wasm/src/asset_vessel_sidecar.rs (moved from riir-ffi 2026-09-06):
impl<S: ChunkedContentStore> AssetStoreAdapter<S> { ... }
```

`AssetStoreAdapter<S>` wraps any `katgpt_core::content_store::ChunkedContentStore`
as an `AssetStore` for the vessel sidecar. Tested in
[`riir-chain/tests/e2e_nft_execute_permission.rs`](../../../riir-chain/tests/e2e_nft_execute_permission.rs)
(re-homed from `riir-ffi/tests/` when the ffi dev-edge flipped, then again at
the 2026-09-06 `riir-ffi` dissolution — the chain root package now owns it).
The private side (game/chain IP) stays in riir-ai; this open primitive is the
adoption hook.

## Usage

```rust
use katgpt_core::{
    InMemoryChunkedStore, FixedSizeChunker, ChunkerConfig, ChunkedContentStore,
};

let mut store = InMemoryChunkedStore::new();
let chunker = FixedSizeChunker::new(ChunkerConfig::default());
let blob_id = store.put(&chunker, b"hello world").unwrap();
let bytes = store.get(blob_id).unwrap();
assert_eq!(bytes, b"hello world");

// Light-client inclusion proof (no &self on verify):
let proof = store.prove(blob_id, 0).unwrap();
// verify_binary_merkle_proof(&leaf_hash, &proof, &root) — pure BLAKE3
```

## References

- Plan: [`katgpt-rs/.plans/448_chunked_asset_merkle_store.md`](../../.plans/448_chunked_asset_merkle_store.md)
- Research: [`katgpt-rs/.research/262_Lore_Chunked_Asset_Merkle_Store_Modelless.md`](../../.research/262_Lore_Chunked_Asset_Merkle_Store_Modelless.md)
- Benchmark: [`katgpt-rs/.benchmarks/262_chunked_content_store_goat.md`](../../.benchmarks/262_chunked_content_store_goat.md)
- Source code: [`katgpt-rs/crates/katgpt-core/src/content_store/`](../../crates/katgpt-core/src/content_store/)
- Private consumer: [`riir-ai/.plans/319_executable_asset_vessel_quorum_gitflow.md`](../../../riir-ai/.plans/319_executable_asset_vessel_quorum_gitflow.md)
- Upstream reference: [EpicGames/lore](https://github.com/EpicGames/lore)
