//! Core types for TurboQuant KV cache compression.

/// Lloyd-Max codebook for scalar quantization of Beta-distributed values.
///
/// Field order: Vec (24B, 8-aligned) → usize (8B) → f32 (4B) → u8 (1B)
/// eliminates inter-field padding.
#[derive(Debug, Clone)]
pub struct TurboQuantCodebook {
    /// Centroid values (2^bits entries).
    pub centroids: Vec<f32>,
    /// Decision boundaries (2^bits - 1 entries).
    pub boundaries: Vec<f32>,
    /// Dimension the codebook was computed for.
    pub dim: usize,
    /// MSE per coordinate at this (dim, bits) setting.
    pub mse_per_coord: f32,
    /// Bits per coordinate.
    pub bits: u8,
}

/// Per-layer quantization state (rotation matrix + codebooks).
#[derive(Debug, Clone)]
pub struct TurboQuantLayer {
    /// Random rotation matrix (dim × dim), stored row-major.
    pub rotation: Vec<f32>,
    /// Transposed rotation matrix (dim × dim), stored row-major.
    /// Precomputed so that inverse rotation uses SIMD row-major
    /// `simd_matmul_rows` instead of scalar column-stride multiply.
    pub rotation_t: Vec<f32>,
    /// QJL projection matrix for residual estimation.
    pub qjl_matrix: Vec<f32>,
    /// Codebook for key cache.
    pub key_codebook: TurboQuantCodebook,
    /// Codebook for value cache.
    pub val_codebook: TurboQuantCodebook,
}

/// Configuration for TurboQuant KV cache.
#[derive(Debug, Clone)]
pub struct TurboQuantKVCacheConfig {
    /// Number of layers.
    pub n_layers: usize,
    /// KV dimension (head_dim × n_kv_head).
    pub kv_dim: usize,
    /// Maximum sequence length (block_size).
    pub max_seq_len: usize,
    /// Random seed for rotation matrix (deterministic).
    pub seed: u64,
    /// Bits per key coordinate (default: 3).
    pub key_bits: u8,
    /// Bits per value coordinate (default: 3).
    pub val_bits: u8,
}
