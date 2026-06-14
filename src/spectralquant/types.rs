//! Types for SpectralQuant KV cache compression.

/// Lloyd-Max codebook for per-dimension quantization.
///
/// Field order: Vec (24B, 8-aligned) then u8 (1B). 7 bytes trailing padding,
/// but no inter-field waste — this is already optimal for Vec→u8 layout.
#[derive(Debug, Clone)]
pub struct LloydMaxCodebook {
    /// Centroid values (2^n_bits entries).
    pub centroids: Vec<f32>,
    /// Bits per coordinate.
    pub n_bits: u8,
}

/// Result of offline calibration per (layer, head, kv_type).
/// Computed once, serialized with model weights.
///
/// Field order: Vec (24B, 8-aligned) → usize (8B) → Option<f32> (8B) → f32 (4B)
/// eliminates inter-field padding between usize and f32.
#[derive(Debug, Clone)]
pub struct SpectralQuantCalibration {
    /// Eigenbasis matrix V (d_h × d_h), row-major.
    /// Columns are eigenvectors sorted by eigenvalue descending.
    pub eigenvectors: Vec<f32>,
    /// Eigenvalues from covariance eigendecomposition, sorted descending.
    pub eigenvalues: Vec<f32>,
    /// Number of calibration samples collected.
    pub n_samples: usize,
    /// Min components for 95% cumulative variance.
    pub var_95: usize,
    /// Min components for 99% cumulative variance.
    pub var_99: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Spectral gap: λ_{d_eff} / λ_{d_eff+1}.
    pub spectral_gap: Option<f32>,
    /// Effective dimensionality: (Σλ_i)² / Σ(λ_i²).
    /// Typically 4–6 at d_h=128.
    pub d_eff: f32,
}

/// Water-fill bit allocation result.
///
/// Field order: Vec (24B, 8-aligned) → usize (8B) → Option<Vec<u8>> (24B) → u8/bool (packed).
#[derive(Debug, Clone)]
pub struct WaterfillAllocation {
    /// First d_eff eigenvalues (used for marginal gain computation).
    pub eigenvalues: Vec<f32>,
    /// Per-semantic-dim bit widths from water-fill.
    pub bits_per_dim: Vec<u8>,
    /// Effective dimensionality.
    pub d_eff: usize,
    /// Total bits allocated to semantic subspace.
    pub total_semantic_bits: usize,
    /// Per-dim maximum bits (None = uncapped).
    pub max_bits: Option<u8>,
    /// Per-dim minimum bits.
    pub min_bits: u8,
    /// Formula version tag for serialization.
    pub formula_version: u8,
    /// Whether water-fill is enabled (v2 path).
    pub use_water_fill: bool,
}

/// Per-layer SpectralQuant state: calibration + fitted codebooks.
///
/// Field order: Vec/struct (24B+, 8-aligned) → usize (8B) → u8 (1B, packed together).
/// Eliminates 7 bytes of padding between usize and u8 fields.
#[derive(Debug, Clone)]
pub struct SpectralQuantLayer {
    /// Calibration data (eigenvectors, eigenvalues, d_eff).
    pub calibration: SpectralQuantCalibration,
    /// QJL sign matrix: (qjl_dim × d_eff) Rademacher ±1.
    pub qjl_signs: Vec<f32>,
    /// Tail regime codebook (shared across all tail dims).
    pub tail_codebook: LloydMaxCodebook,
    /// Shared semantic codebook (v1 uniform path, None for v2).
    pub semantic_codebook: Option<LloydMaxCodebook>,
    /// Per-dim semantic codebooks (v2 water-fill path, None for v1).
    pub per_dim_semantic_codebooks: Option<Vec<LloydMaxCodebook>>,
    /// Per-dim semantic bits (v2 water-fill path, None for v1).
    pub semantic_bits_per_dim: Option<Vec<u8>>,
    /// Precomputed full [kv_dim] bits-per-dim array (semantic + tail).
    /// Built once at construction so per-token store/dequantize paths avoid rebuilding it.
    pub packed_bits: Vec<u8>,
    /// Effective dimensionality (integer ceiling of d_eff).
    pub d_eff: usize,
    /// Semantic regime bits per coordinate.
    pub b_high: u8,
    /// Tail regime bits per coordinate.
    pub b_low: u8,
}

/// Configuration for SpectralQuant KV cache.
///
/// Field order: u64 (8B) → usize (8B) → f32 (4B) → u8/bool (1B, packed).
/// Eliminates inter-field padding on 64-bit targets.
#[derive(Debug, Clone)]
pub struct SpectralQuantKVCacheConfig {
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Number of layers.
    pub n_layers: usize,
    /// KV dimension (head_dim × n_kv_head).
    pub kv_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Max Lloyd-Max iterations.
    pub lloyd_max_iter: usize,
    /// Number of calibration samples to collect.
    pub calibration_samples: usize,
    /// QJL projection dimension.
    pub qjl_dim: usize,
    /// Average bits per coordinate across all dimensions.
    pub avg_bits: f32,
    /// Minimum bits for tail dimensions.
    pub min_tail_bits: u8,
    /// Maximum bits per dimension.
    pub max_bits: u8,
    /// Water-fill minimum bits per dim.
    pub wf_min_bits: u8,
    /// Water-fill maximum bits per dim.
    pub wf_max_bits: u8,
    /// Whether to use water-fill allocation (v2).
    pub use_water_fill: bool,
}
