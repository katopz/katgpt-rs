//! Types for ShardKV asymmetric KV cache compression.
//!
//! Shard (Research 109) uses different compression paths for K and V:
//! - K: undo RoPE → PCA rotation → water-fill bit allocation → Lloyd-Max quantize
//! - V: Hadamard rotation → vector quantization (VQ) for prefill, 8-bit Lloyd-Max for decode

use crate::spectralquant::types::LloydMaxCodebook;

/// VQ codebook for vector quantization (groups of channels).
///
/// Stores centroids as a flat (codebook_size × group_size) f32 array.
/// Each centroid is a `group_size`-dimensional vector.
#[derive(Debug, Clone)]
pub struct VqCodebook {
    /// Centroids: (codebook_size × group_size) flattened, row-major.
    pub centroids: Vec<f32>,
    /// Number of entries in the codebook (e.g., 256).
    pub codebook_size: usize,
    /// Number of channels per VQ group (e.g., 4).
    pub group_size: usize,
}

/// Configuration for ShardKV cache.
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Number of transformer layers.
    pub n_layers: usize,
    /// KV dimension (head_dim × n_kv_heads).
    pub kv_dim: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Number of attention-sink tokens stored at FP16.
    pub sink_tokens: usize,
    /// Number of recent-window tokens stored at FP16.
    pub window_tokens: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// VQ group size for V path (default: 4).
    pub v_vq_group_size: usize,
    /// VQ codebook size for V path (default: 256).
    pub v_vq_codebook_size: usize,
    /// Average bits per coordinate for K path.
    pub avg_bits_k: f32,
    /// Average bits per coordinate for V path (prefill, used for VQ sizing).
    pub avg_bits_v: f32,
    /// Minimum bits for tail dimensions (K path).
    pub min_tail_bits: u8,
    /// Maximum bits per dimension (K path).
    pub max_bits: u8,
    /// Bits per coordinate for decode streaming (default: 8 = lossless).
    pub decode_stream_bits: u8,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            n_layers: 1,
            kv_dim: 128,
            head_dim: 128,
            max_seq_len: 512,
            sink_tokens: 4,
            window_tokens: 64,
            seed: 42,
            v_vq_group_size: 4,
            v_vq_codebook_size: 256,
            avg_bits_k: 4.0,
            avg_bits_v: 2.0,
            min_tail_bits: 1,
            max_bits: 8,
            decode_stream_bits: 8,
        }
    }
}

/// Per-layer calibration for the K path (spectral decomposition).
#[derive(Debug, Clone)]
pub struct ShardCalibration {
    /// Eigenvector matrix V (head_dim × head_dim), row-major.
    /// Columns sorted by eigenvalue descending.
    pub k_eigenvectors: Vec<f32>,
    /// Eigenvalues sorted descending.
    pub k_eigenvalues: Vec<f32>,
    /// Head dimension.
    pub head_dim: usize,
    /// Effective dimensionality: (Σλ_i)² / Σ(λ_i²).
    pub k_d_eff: f32,
}

/// Per-layer state for ShardKV: calibration + fitted codebooks.
#[derive(Debug, Clone)]
pub struct ShardLayer {
    /// K-path calibration.
    pub calibration: ShardCalibration,
    /// K-path per-dim bit widths from water-fill (None for uniform).
    pub k_bits_per_dim: Option<Vec<u8>>,
    /// K-path per-dim semantic codebooks (water-fill path).
    pub k_per_dim_codebooks: Option<Vec<LloydMaxCodebook>>,
    /// K-path shared semantic codebook (uniform path).
    pub k_semantic_codebook: Option<LloydMaxCodebook>,
    /// K-path tail codebook.
    pub k_tail_codebook: LloydMaxCodebook,
    /// V-path VQ codebook for prefill (Hadamard + K-means on groups of channels).
    pub v_vq_codebook: VqCodebook,
    /// V-path decode streaming codebook (8-bit Lloyd-Max for Hadamard-rotated data).
    pub decode_v_codebook: LloydMaxCodebook,
    /// K-path effective dimensionality (integer ceiling of k_d_eff).
    pub d_eff: usize,
    /// V-path effective bits per element from VQ (codebook_bits / group_size).
    pub v_bits_per_elem: f32,
    /// K-path semantic (high-energy) bits per coordinate.
    pub k_b_high: u8,
    /// K-path tail bits per coordinate.
    pub k_b_low: u8,
    /// V-path decode streaming bits per coordinate (default 8 = lossless).
    pub decode_stream_bits: u8,
}
