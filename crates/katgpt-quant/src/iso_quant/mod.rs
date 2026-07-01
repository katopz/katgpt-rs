//! IsoQuant: 4D quaternion rotation KV cache compression.
//!
//! Replaces TurboQuant's O(d²) WHT with O(d) block-diagonal 4D rotations.
//! Per group of 4: T(v) = q_L * v * conj(q_R)  [full mode]
//!                 T(v) = q_L * v                 [fast mode]
//! 512 FMAs for d=128 vs TurboQuant's 16,384.

pub mod kv_cache;
pub mod rotation;
pub mod types;

pub use kv_cache::IsoQuantKVCache;
pub use rotation::{generate_unit_quaternions, quat_conjugate, quat_multiply};
pub use types::{IsoQuantConfig, IsoQuantLayer, IsoQuantMode};
