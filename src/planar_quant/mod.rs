//! PlanarQuant: 2D Givens rotation KV cache compression.
//!
//! Replaces TurboQuant's O(d²) WHT with O(d) block-diagonal 2D rotations.
//! Per adjacent pair: (cos θ_i · v0 - sin θ_i · v1, sin θ_i · v0 + cos θ_i · v1)
//! 256 FMAs for d=128 vs TurboQuant's 16,384.

pub mod kv_cache;
pub mod rotation;
pub mod types;

pub use kv_cache::PlanarQuantKVCache;
pub use rotation::{generate_givens_rotations, rot2_apply, rot2_inverse};
pub use types::{PlanarQuantConfig, PlanarQuantLayer};
