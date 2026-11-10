//! MUX-Latent Context Compression — inference-time context compression via
//! vocabulary superposition. Distilled from LCLM (arXiv:2606.09659).
//!
//! Architecture: Input tokens → MUX superposition encoder → latent slots
//!                                                    → domain_latent mid-layer injection
//!
//! No training required. Uses existing MUX infrastructure for zero-allocation
//! position-weighted superposition with lossless separation guarantee.

mod buffer;
mod config;
mod context;
mod encoder;
mod expand;
mod inject;
mod prefill;
mod spectral_lod;

#[cfg(feature = "mux_latent_context")]
mod octree_bridge;
#[cfg(feature = "mux_latent_context")]
mod patcher;
#[cfg(feature = "mux_latent_context")]
mod wire;

pub use buffer::{BufferStats, EvictionPolicy, LatentContextBuffer};
pub use config::{CompressionRatio, MuxLatentConfig};
pub use context::{CompressedContext, LatentSegment};
pub use encoder::MuxLatentEncoder;
pub use expand::{expand_all, expand_segment, select_segments_to_expand};
pub use inject::{CompressionSummary, LatentPrefillAdapter, MixedPrefillSequence, PrefillEntry};
pub use prefill::{CompressedPrefillPlan, CompressionMetadata, forward_prefill_with_compression};
pub use spectral_lod::SpectralLOD;

#[cfg(feature = "mux_latent_context")]
pub use patcher::{DirtyTracker, LatentPatcher};
#[cfg(feature = "mux_latent_context")]
pub use wire::{LatentPatch, LatentPatchBatch, PatchReceipt, PatchRejection};

#[cfg(feature = "mux_latent_context")]
pub use octree_bridge::{
    MortonCode, OctreeLod, TernaryDir, TernaryValue, octree_leaf_to_patch_weights,
    patch_weights_to_octree_leaf,
};
