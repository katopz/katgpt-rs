//! OCTOPUS: Octahedral Triplet KV Cache Compression.
//!
//! Based on "OCTOPUS: Octahedral Quantization for KV Cache Compression"
//! (arXiv:2605.21226). Data-oblivious triplet codec that dominates at
//! 2-3 bit extreme compression via:
//!
//! 1. **Triplet decomposition** — groups rotated coordinates into contiguous
//!    3-blocks instead of per-coordinate quantization
//! 2. **Octahedral map** — S² → [-1,1]² equal-area parameterization
//! 3. **Non-uniform bit split** — (b+1, b-1) for direction/norm is MSE-optimal
//! 4. **Joint 3×3 rounding** — encoder-only optimization (6-14% MSE reduction)
//!
//! Production stack position: between SpectralQuant (calibrated, higher quality)
//! and TurboQuant (legacy baseline). Best data-oblivious codec at extreme compression.

pub mod codebook;
pub mod octahedral;
pub mod triplet;

pub use codebook::{ScalarCodebook, build_norm_codebook, build_oct_codebook};
pub use octahedral::{oct_decode, oct_encode};
pub use triplet::{Triplet, decompose, recompose};
pub mod types;

pub mod encode;

pub use types::{OctopusCodebook, OctopusConfig, OctopusLayer, TripletIndices};

pub mod forward;
pub mod kv_cache;

pub use kv_cache::OctopusKVCache;
