//! `engram` — Hash-Addressed Pattern Memory (open primitive).
//!
//! Plan 299 / Research 278. Implements the **open half** of the Engram
//! Super-GOAT: a generic, hash-addressed, sigmoid-fused static pattern memory
//! primitive for inference-time context injection. No training, no backprop —
//! the table is a frozen snapshot and updates go through atomic Arc swaps.
//!
//! # The sparsity-axis framing
//!
//! Raven = conditional **computation** (which experts to fire).
//! Engram = conditional **memory** (which patterns to recall).
//! The U-shape scaling law (paper §3) proves the hybrid is strictly better
//! than either alone. This is the **first conditional-memory axis** in the
//! katgpt stack.
//!
//! # Pipeline
//!
//! ```text
//! N-gram suffix → multi-head hash → O(1) slot lookup →
//! sigmoid gate (RMSNorm · dot · σ) → residual-fuse into hidden state
//! ```
//!
//! Every step is **zero-allocation** in the hot path: callers provide scratch
//! + out buffers, lookups index a flat `Box<[f32]>` row-major array directly
//! by `hash mod N`.
//!
//! # Phase status (this file)
//!
//! - ✅ Phase 1 (hashing) — [`hash`].
//! - ✅ Phase 2 (frozen table + lookup) — [`table`].
//! - ✅ Phase 3 (sigmoid fusion kernel, T3.1–T3.7) — [`kernel`] + [`conv`].
//! - ✅ Phase 4 (tokenizer compression) — [`tokenizer`].
//! - ✅ Phase 5 (commitment + hotswap) — [`commitment`] + [`hotswap`].
//! - ✅ Phase 6 (Zipfian cache hierarchy) — [`cache`].
//! - ✅ Phase 7 partial (T7.1–T7.2 forward fuse) — [`forward`].
//! - ✅ Phase 7 GOAT gates (T7.3–T7.10) — `tests/bench_299_engram_goat.rs`.
//!   G6 (effective depth) is deferred to riir-ai integration; feature stays
//!   opt-in until G6 lands there.
//! - ✅ Phase 8 (docs) — `.docs/27_engram_conditional_memory.md`,
//!   `.benchmarks/299_engram_goat.md`, README entry.
//!
//! # CRITICAL — never softmax
//!
//! Per AGENTS.md, this module **uses sigmoid, not softmax**, everywhere. The
//! fusion kernel's gate is a single scalar `sigmoid(dot(q_norm, k_norm) /
//! tau)`. There is no `softmax` symbol anywhere in this module.
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! - Slot contents (latent patterns) → latent, frozen, BLAKE3-committed.
//! - `EngramTableId` (commitment) → raw, syncable audit artifact.
//! - Sigmoid gate output → latent scalar, residual-added in caller's hidden
//!   state (latent stays latent; bridge at the sync boundary).
//!
//! # References
//!
//! - Plan: [`katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md`]
//! - Research: [`katgpt-rs/.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md`]
//! - Source paper: [arXiv:2601.07372](https://arxiv.org/pdf/2601.07372) —
//!   Engram, Cheng et al. 2026.

// All engram submodules are now landed (Phases 1–8). Phase 7 GOAT gate
// results live in katgpt-rs/.benchmarks/299_engram_goat.md.

mod cache;
mod commitment;
mod conv;
mod forward;
mod hash;
mod hotswap;
mod kernel;
mod table;
mod tokenizer;

pub use cache::{
    CacheResult, CacheTier, ColdFetcher, ZipfianCacheHierarchy, ZipfianStats, ZipfianStatsSnapshot,
};
pub use commitment::{EngramTableId, build_merkle_root};
pub use conv::{IDENTITY_KERNEL, SPEC_KERNEL, ZERO_KERNEL, conv_causal_into};
pub use forward::{EngramConfig, fuse_into_hidden_state};
pub use hash::{HashHead, multi_head_hash};
pub use hotswap::EngramHotSwap;
pub use kernel::{
    SigmoidFusionConfig, rmsnorm_into, sigmoid_fuse_into, sigmoid_fuse_multi_branch_into,
};
pub use table::{EngramTableBuilder, InMemoryEngramTable};
pub use tokenizer::{
    SurjectiveMap, SurjectiveMapLoadError, TokenizerSpec, build_surjective_map, compress_token,
    try_compress_token,
};

#[cfg(test)]
mod tests;

/// Maximum number of heads retrieved per query.
///
/// Per the Engram paper, K = 8 heads × 2 N-gram orders = 16. Fixed at compile
/// time so the lookup output and scratch buffers are stack-sized arrays with
/// zero allocation.
pub const K_MAX: usize = 16;

/// A 64-bit hash slot key produced by [`multi_head_hash`].
///
/// `#[repr(transparent)]` over `u64` — zero-cost newtype. Lookup is
/// `slots[hash.0 as usize % N]`, so equality + hash + copy must all be
/// trivial (they are).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct EngramHash(pub u64);

/// A canonical (tokenizer-compressed) token id.
///
/// Phase 4's `SurjectiveMap` collapses raw tokenizer IDs (e.g. `"Apple"` vs
/// `"apple"`) to a shared `CanonicalId`. Until Phase 4 lands, callers may
/// pass the raw token id cast directly: `CanonicalId(raw_id as u64)`.
///
/// `#[repr(transparent)]` over `u64` so a `&[CanonicalId]` slice has the same
/// layout as `&[u64]` — required for SIMD-friendly hashing in
/// [`multi_head_hash`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct CanonicalId(pub u64);

/// A raw (uncompressed) tokenizer token id.
///
/// Carried through the public API so the caller can opt out of Phase 4's
/// surjective compression and address patterns by raw id directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct TokenId(pub u32);

/// Frozen, content-addressed engram pattern table.
///
/// The two known implementations are:
/// - [`InMemoryEngramTable`] — flat `Box<[f32]>` row-major array, direct
///   `hash mod N` lookup. This is the only implementation in Phase 1–3.
/// - (Phase 6) `ZipfianCacheHierarchy` — tiered cache wrapping a warm
///   source. Deferred.
///
/// # Hot-path contract
///
/// [`EngramTable::lookup_into`] is **zero-allocation**: the caller provides
/// an `out` slice of size `K_MAX * dim()` and the implementation writes the
/// K retrieved slot vectors into it row-major (`out[k*D..(k+1)*D]`). The
/// return value is the number of non-empty slots hit (for diagnostics;
/// retrieval fills all K slots regardless, with zero-filled rows on miss).
///
/// # Commitment contract
///
/// [`EngramTable::commitment`] returns a 32-byte BLAKE3 root over the slot
/// contents. Two tables with the same slot contents produce the same root —
/// this is the table identity that crosses the sync boundary as an audit
/// artifact. The latent slot contents themselves never sync.
pub trait EngramTable: Send + Sync {
    /// Look up K_MAX slot vectors, writing them row-major into `out`.
    ///
    /// `out` MUST be at least `K_MAX * dim()` long (debug_asserted). Returns
    /// the number of slots that contained non-zero data (i.e. were
    /// populated). Empty / collision-missed slots are written as zeros so
    /// the caller can treat the output uniformly.
    fn lookup_into(&self, hash_keys: &[EngramHash; K_MAX], out: &mut [f32]) -> usize;

    /// BLAKE3 commitment over the slot contents (Merkle root of per-slot
    /// hashes). Cached after first call.
    fn commitment(&self) -> [u8; 32];

    /// Number of slots in the table (the modulus used for
    /// `hash.0 as usize % num_slots`).
    fn num_slots(&self) -> usize;

    /// Dimensionality of each slot vector (the hidden-state dimension D).
    fn dim(&self) -> usize;
}
