//! TurboQuant: Near-optimal vector quantization for KV cache compression.
//!
//! Based on "TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate"
//! (arXiv:2504.19874). Compresses KV cache from f32 to 2-4 bits per coordinate.
//!
//! Architecture:
//! - Random rotation → Beta-distributed coordinates
//! - Lloyd-Max codebook → optimal scalar quantizer
//! - Bit-packed storage → 8-16× compression

pub mod codebook;
pub mod forward;
pub mod kv_cache;
pub mod rotation;
pub mod types;

pub use codebook::compute_codebook;
pub use kv_cache::TurboQuantKVCache;
pub use rotation::{generate_qjl_matrix, generate_rotation_matrix};
pub use types::{TurboQuantCodebook, TurboQuantKVCacheConfig, TurboQuantLayer};
