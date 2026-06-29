//! ShardKV: Asymmetric KV cache compression (Plan 147, Research 109).
//!
//! Inspired by the Shard paper: K and V have different structural properties
//! requiring different compression methods.
//!
//! - **K path** (prefill): undo RoPE → PCA rotation → water-fill bit allocation → Lloyd-Max quantize
//! - **K path** (decode): Hadamard rotation → 8-bit Lloyd-Max streaming (guaranteed lossless)
//! - **V path** (prefill): Hadamard rotation → K-means VQ (groups of 4, 256 codebook) → 2 bits/elem
//! - **V path** (decode): Hadamard rotation → 8-bit Lloyd-Max streaming (guaranteed lossless)
//! - **Sink + window**: attention sinks and recency window stored losslessly
//!
//! Reuses spectralquant's `SpectralRotation`, `LloydMaxQuantizer`, `BitAllocator`,
//! and `waterfill_bits` for the K path.

pub mod kv_cache;
pub mod rope;
pub mod types;

pub use kv_cache::ShardKVCache;
pub use rope::{RopeFreqs, reapply_rope, undo_rope};
pub use types::{ShardCalibration, ShardConfig, ShardLayer, VqCodebook};
