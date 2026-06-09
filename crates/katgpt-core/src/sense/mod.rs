//! Sense Composition — KG Latent Octree (Plan 221).
//!
//! Compresses game domain knowledge into fixed-type ternary bit-plane sense modules.
//! NPCs compose modules at spawn time and query at ~45ns/tick via bitwise dot-product.

pub mod bake;
pub mod bandit;
pub mod batch;
pub mod brain;
pub mod gm;
pub mod hotswap;
pub mod octree;
#[cfg(feature = "schema_centroid")]
pub mod schema_centroid;
pub mod serialize;

pub use bake::{
    DEFAULT_OBS_PRECISION, UNINFORMATIVE_PRECISION, bake_regularize, bake_update, bake_update_mean,
    bake_update_precision, exploration_priority, informed_prior_precision, precision_to_confidence,
};
pub use bandit::{SenseTrial, SenseTrialLog};
pub use brain::{NpcBrain, SenseOverride};
pub(crate) use gm::dispatch_gm_action;
pub use gm::{NpcBrainSnapshot, SenseError};
pub use hotswap::SenseHotSwap;
pub use octree::{KgEmbedding, SenseOctreeBuilder};
#[cfg(all(feature = "schema_centroid", feature = "bake_precision"))]
pub use schema_centroid::schema_init_with_precision;
#[cfg(feature = "schema_centroid")]
pub use schema_centroid::{
    CentroidStats, SchemaCentroidCache, compute_centroid, schema_init_entity,
};
