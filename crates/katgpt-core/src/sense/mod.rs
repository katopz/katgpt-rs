//! Sense Composition — KG Latent Octree (Plan 221).
//!
//! Compresses game domain knowledge into fixed-type ternary bit-plane sense modules.
//! NPCs compose modules at spawn time and query at ~45ns/tick via bitwise dot-product.

pub mod bandit;
pub mod batch;
pub mod brain;
pub mod gm;
pub mod hotswap;
pub mod octree;
pub mod serialize;

pub use bandit::{SenseTrial, SenseTrialLog};
pub use brain::{NpcBrain, SenseOverride};
pub(crate) use gm::dispatch_gm_action;
pub use gm::{NpcBrainSnapshot, SenseError};
pub use hotswap::SenseHotSwap;
pub use octree::{KgEmbedding, SenseOctreeBuilder};
