//! Core types for PlanarQuant KV cache compression.

/// Per-layer PlanarQuant state.
#[derive(Debug, Clone)]
pub struct PlanarQuantLayer {
    /// Key rotations: (cos θ, sin θ) per pair — ceil(kv_dim/2) pairs.
    pub key_rotations: Vec<[f32; 2]>,
    /// Value rotations: (cos θ, sin θ) per pair.
    pub val_rotations: Vec<[f32; 2]>,
    /// Lloyd-Max codebook centroids for keys (2^bits entries).
    pub key_centroids: Vec<f32>,
    /// Lloyd-Max codebook boundaries for keys (2^bits - 1 entries).
    pub key_boundaries: Vec<f32>,
    /// Lloyd-Max codebook centroids for values.
    pub val_centroids: Vec<f32>,
    /// Lloyd-Max codebook boundaries for values.
    pub val_boundaries: Vec<f32>,
}

/// Configuration for PlanarQuant KV cache.
#[derive(Debug, Clone)]
pub struct PlanarQuantConfig {
    /// Number of transformer layers.
    pub n_layers: usize,
    /// KV dimension (head_dim × n_kv_heads). Padded to even.
    pub kv_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Random seed for rotation generation (deterministic).
    pub seed: u64,
    /// Bits per key coordinate (2-4).
    pub key_bits: u8,
    /// Bits per value coordinate (2-4).
    pub val_bits: u8,
}
