//! Sense Composition — KG Latent Octree (Plan 221).
//!
//! Compresses game domain knowledge into fixed-type ternary bit-plane sense modules.
//! NPCs compose modules at spawn time and query at ~45ns/tick via bitwise dot-product.
//!
//! Issue 007 Phase C (2026-06-28): the NPC-runtime half of this module
//! (`brain`, `backend`, `batch`, `gm`, `hotswap`, `bandit`) moved to
//! `riir-engine::sense::*`. Only the generic sense substrate (octree,
//! reconstruction, bake, lod, etc.) remains here.

pub mod bake;
#[cfg(feature = "sense_lod")]
pub mod lod;
pub mod octree;
pub mod reconstruction;
#[cfg(feature = "depth_invariance")]
pub mod reconstruction_depth_invariance;
#[cfg(feature = "schema_centroid")]
pub mod schema_centroid;
#[cfg(feature = "sector_projection")]
pub mod sector;
pub mod serialize;
#[cfg(feature = "spectral_threat")]
pub mod spectral_threat;

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
#[cfg(feature = "spectral_threat")]
pub use spectral_threat::{CombatRhythmTracker, SpectralThreatFeatures};
