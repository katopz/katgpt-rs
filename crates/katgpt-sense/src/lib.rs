//! katgpt-sense — KG Latent Octree Sense substrate (Plan 221 / 235 / 236 / 237 / 240 / 262 / 331).
//!
//! Generic sense substrate: octree construction, reconstruction, BAKE
//! precision-gated embedding update, LOD routing, schema-centroid init,
//! sector projection, and SenseModule serialization. NPCs compose modules at
//! spawn time and query at ~45ns/tick via bitwise dot-product.
//!
//! Spun out of `katgpt-core::sense` (Issue 007 Phase E Tier 2 #7, Plan 338).
//! The NPC-runtime half (`brain`, `backend`, `batch`, `gm`, `hotswap`,
//! `bandit`) had already moved to `riir-engine::sense::*` in Phase C. The
//! tightly-coupled `spectral_threat` (depends on `linoss`) stayed in
//! katgpt-core as composition code (`katgpt_core::sense_threat`), re-exported
//! back through the `katgpt_core::sense::spectral_threat` shim path.
//!
//! Depends only on `katgpt-types` (the leaf): SIMD kernels, ScaleBoundary,
//! TemporalDerivativeKernel, MerkleOctree/MerkleProof, depth-invariance
//! classifier, leaky_step. Zero katgpt-core dependency — the cycle is broken.
//!
//! Note: `belief` (per-NPC affect state, 8-dim) is unrelated to `katgpt-hla`
//! (Higher-order Linear Attention, a Transformer attention layer). The
//! shared "HLA" acronym was a naming collision resolved in Issue 195.

pub mod bake;
#[cfg(feature = "sense_lod")]
pub mod lod;
pub mod octree;
pub mod reconstruction;

/// Re-export the SyncBank relation-memory module (Issue 930) so consumers of
/// the reconstruction accessor resolve `katgpt_sense::sync_bank::SyncBank`
/// without a direct katgpt-types dep. The module itself is unconditional in
/// katgpt-types (the TemporalDerivativeKernel precedent); the
/// `sync_bank` FEATURE gates the ReconstructionState wiring.
pub use katgpt_types::sync_bank;
#[cfg(feature = "depth_invariance")]
pub mod reconstruction_depth_invariance;
#[cfg(feature = "schema_centroid")]
pub mod schema_centroid;
#[cfg(feature = "sector_projection")]
pub mod sector;
pub mod serialize;

#[cfg(feature = "bake_precision")]
pub use bake::{BakePrecisionStore, BakeSession, PrecisionEntry};
pub use bake::{
    DEFAULT_OBS_PRECISION, UNINFORMATIVE_PRECISION, bake_regularize, bake_update, bake_update_mean,
    bake_update_precision, exploration_priority, informed_prior_precision, precision_to_confidence,
};
pub use octree::{KgEmbedding, SenseOctreeBuilder};
pub use reconstruction::{
    OctreeNodeId, ReconstructionConfig, ReconstructionResult, ReconstructionState, TraversalAction,
    TripleEvidence, compare_reconstruction,
};
#[cfg(all(feature = "schema_centroid", feature = "bake_precision"))]
pub use schema_centroid::schema_init_with_precision;
#[cfg(feature = "schema_centroid")]
pub use schema_centroid::{
    CentroidStats, SchemaCentroidCache, compute_centroid, schema_init_entity,
};
#[cfg(feature = "sector_projection")]
pub use sector::SectorProjection;
