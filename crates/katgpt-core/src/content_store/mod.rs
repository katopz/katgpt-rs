//! ChunkedContentStore — Lore-distilled chunked content-addressed Merkle store.
//!
//! Distilled from [Epic Games Lore](https://github.com/EpicGames/lore) into a
//! modelless open primitive. See:
//! - **Plan 448** (`katgpt-rs/.plans/448_chunked_asset_merkle_store.md`)
//! - **Research 262** (`katgpt-rs/.research/262_Lore_Chunked_Asset_Merkle_Store_Modelless.md`)
//!
//! ## What this is
//!
//! A pure data-plumbing store: bytes → [`FixedSizeChunker`] / `ChunkingStrategy`
//! → BLAKE3 per chunk → dedup via `papaya` lock-free hashmap → binary Merkle
//! root = [`BlobId`]. Supports O(log n) inclusion proofs via
//! [`build_binary_merkle_proof`] / [`verify_binary_merkle_proof`] (pure BLAKE3,
//! no store access — light-client friendly).
//!
//! ## What this is NOT — boundary statement
//!
//! Per Plan 448 §"Out of Scope" and Research 262 §7:
//! - **No game IP.** No `ItemAsset`, `NPCAppearanceAsset`, `AssetRecord`, no
//!   quorum-scoped visibility tiers, no `AssetVisibilityGate`, no
//!   `PromoteAssetIx` / `InstallAsset` / `UnlockShopSlot` / `MintAssetNft`
//!   LatCal instructions.
//! - **No chain IP.** No consensus, no quorum commit, no subnet-as-gitflow
//!   mapping, no atomic candidate-lock transactions.
//! - **No latent projection.** The store is content-addressed bytes only.
//!   Latent↔raw bridging (HLA → 5 scalars) happens in `riir-engine` /
//!   `riir-chain`. See AGENTS.md "Latent vs Raw Space Rules".
//!
//! The game/chain fusion is private to `riir-ai` Plan 319 (Executable Asset
//! Vessel + Quorum Gitflow). This module is the open adoption hook.
//!
//! ## GOAT gate
//!
//! DEFAULT-ON since 2026-07-18 (Phase 19b promotion fix-up — the bench file
//! `.benchmarks/262_chunked_content_store_goat.md` had recorded promotion but
//! `Cargo.toml` was never updated; this comment + the Cargo.toml `default`
//! entry are the fix-up). G1–G7 ALL PASS (dedup 8.47×, CDC 1.35%, inclusion
//! proof <2µs, tamper 10000/10000). G4 (light-client verify) is enforced
//! structurally: [`ChunkedContentStore::verify_proof`] is an associated fn
//! that takes only the proof + leaf hash — no `&self`. Pure modelless +
//! zero-cost-unless-invoked.
//!
//! NOTE: `apply_operator_into` is NOT re-exported at crate root — it collides with
//! `analytic_lattice::apply_operator_into` when both features are on. Access it
//! via `katgpt_core::mag::apply_operator_into`.
//!
//! ChunkedContentStore — Lore-distilled chunked content-addressed Merkle store (Plan 448, Research 262).
//! Open primitive: chunks → BLAKE3 → dedup via papaya → binary Merkle root. No game/chain IP.
//! Consumed by riir-ai Plan 319 (Executable Asset Vessel + Quorum Gitflow).
//!
//! NOTE: the binary-Merkle `MerkleProof` here is renamed on re-export to
//! `BinaryMerkleProof` to avoid colliding with `merkle_octree::MerkleProof`
//! when both features are active simultaneously (caught by `cargo check
//! --all-features`). Internal callers still reach the type via
//! `crate::content_store::MerkleProof`.

// `trait` is a reserved keyword; the source file is `trait.rs` but the module
// is referenced via the raw identifier `r#trait`. Re-exports below hide this.
pub mod chunker;
pub mod fetcher;
pub mod in_memory;
pub mod merkle;
#[allow(non_snake_case)]
pub mod r#trait;
pub mod types;

#[cfg(test)]
mod goat;

pub use chunker::{
    ChunkerConfig, DEFAULT_CHUNK_SIZE, FASTCDC_MAX_CHUNK_SIZE, FASTCDC_MAX_LEVEL,
    FASTCDC_MIN_CHUNK_SIZE, FASTCDC_MIN_LEVEL, FASTCDC_NORMAL_LEVEL, FastCdcChunker,
    FixedSizeChunker,
};
#[cfg(feature = "chunked_net_fetch")]
pub use fetcher::NetChunkFetcher;
pub use fetcher::{
    FsChunkFetcher, InMemoryChunkFetcher, TieredChunkFetcher, TieredWriteBackExt, WriteBack,
};
pub use in_memory::InMemoryChunkedStore;
pub use merkle::{build_binary_merkle_proof, build_binary_merkle_root, verify_binary_merkle_proof};
pub use r#trait::{ChunkFetcher, ChunkedContentStore, ChunkingStrategy};
pub use types::{BlobId, ChunkRange, MerkleProof, StoreStats};
