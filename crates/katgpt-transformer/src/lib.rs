//! katgpt-transformer: Transformer substrate types shared between katgpt-rs root
//! and riir-engine.
//!
//! Pure data + loaders — no forward logic. The forward kernels stay in the
//! katgpt-rs root crate because they compose cognitive primitives
//! (`crate::hla`, `crate::sleep`, `crate::tf_loop`, `crate::gdn2`, etc.)
//! that do not exist in this substrate crate.
//!
//! See `README.md` for the architectural rationale (why this is a separate
//! crate from `katgpt-core`).

mod contiguous;
mod context;
mod kv_cache;
mod mtp;
mod weights;

pub use contiguous::ContiguousWeights;
pub use context::PrefillContext;
#[cfg(feature = "wall_attention")]
pub use context::{GateStatistics, WallPrefixState};
pub use kv_cache::{
    KVCache, KVLayerSnapshot, KVSnapshot, MultiLayerKVCache, PagedKVCache, RavenKVCache,
    preload_kv_cache,
};
/// Page size in tokens for [`PagedKVCache`] (tuneable, must be power of 2).
pub use kv_cache::PAGE_SIZE;
pub use mtp::{MtpProjection, load_mtp_projection, project_target_activation};
pub use weights::{LayerWeights, TransformerWeights};

// Contiguous ternary loader (Plan 148, gated `plasma_path`).
#[cfg(feature = "plasma_path")]
pub use contiguous::load_ternary_bits;

// Decode stage for specialized forward paths (Plan 102: TileRT pipeline).
/// Different stages have different optimization opportunities:
/// - Draft: can skip screening, reduced KV writes, approximate attention
/// - Verify: exact attention, full KV write, enable screening
/// - Sample: SIMD-only, no attention needed
/// - BeliefDraft: MLP-only forward via BeliefDrafter, no attention needed (Plan 217)
#[cfg(feature = "decode_specialize")]
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecodeStage {
    /// Batch-friendly, attention-heavy, needs full KV write.
    Prefill,
    /// Small batch, can skip screening, matmul-heavy.
    Draft,
    /// Single batch, needs exact attention, KV read-heavy.
    Verify,
    /// SIMD-only, no attention needed.
    Sample,
    /// MLP-only forward via BeliefDrafter — no attention, no KV (Plan 217).
    /// The belief drafter uses a lightweight MLP to predict next hidden states
    /// instead of running the full transformer forward pass.
    BeliefDraft,
}
