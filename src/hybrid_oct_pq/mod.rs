//! Hybrid OCTOPUS-encoding + PlanarQuant-rotation KV cache compression.
//!
//! Combines PlanarQuant's O(d) 2D Givens block-diagonal rotation with
//! OCTOPUS's octahedral triplet encoding. Achieves near-OCTOPUS MSE quality
//! (within 5%) at PlanarQuant's rotation speed (256 FMAs for d=128 vs
//! OCTOPUS's 16,384).
//!
//! Pipeline:
//!   Encode:  normalize → PQ 2D rotate → decompose triplets → OCT encode → bit-pack
//!   Decode:  unpack → OCT decode triplets → recompose → PQ inverse rotate → rescale

pub mod kv_cache;
pub mod types;

pub use kv_cache::HybridOctPqKVCache;
pub use types::{HybridOctPqConfig, HybridOctPqLayer};
