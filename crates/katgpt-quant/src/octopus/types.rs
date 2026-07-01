//! Core types for OCTOPUS octahedral triplet KV cache compression.
//!
//! Key types:
//! - [`OctopusConfig`] — compression hyperparameters (bit split, dims, seeds)
//! - [`OctopusLayer`] — per-layer rotation + dual codebooks
//! - [`OctopusCodebook`] — paired norm + oct-direction codebooks with (b+1, b-1) split
//! - [`TripletIndices`] — packed quantized indices for one triplet

use super::codebook::ScalarCodebook;

/// Configuration for OCTOPUS KV cache compression.
///
/// The nominal `key_bits` / `val_bits` are split non-uniformly:
/// - Direction (ξ, η): `bits + 1` bits each → oct codebook
/// - Norm (ρ): `bits - 1` bits → norm codebook
///
/// Total bits per triplet: `2(bits+1) + (bits-1) = 3·bits + 1`.
/// This non-uniform split gives 31-41% MSE reduction over uniform at d=128.
#[derive(Debug, Clone)]
pub struct OctopusConfig {
    /// Deterministic rotation seed.
    pub seed: u64,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// KV dimension (head_dim × n_kv_heads). Should be power of 2 for WHT.
    pub kv_dim: usize,
    /// Maximum sequence length (block size).
    pub max_seq_len: usize,
    /// Nominal bits per value coordinate.
    pub val_bits: u8,
    /// Nominal bits per coordinate (actual split: dir=b+1, nrm=b-1).
    pub key_bits: u8,
    /// Enable QJL 1-bit residual for score-attention (adds ~0.5 bpc).
    pub use_qjl_residual: bool,
    /// Enable joint 3×3 rounding in encoder (6-14% MSE gain, encoder-only).
    pub use_joint_rounding: bool,
}

impl OctopusConfig {
    /// Direction bits for given nominal bits: `b + 1`.
    #[must_use]
    #[inline]
    pub fn dir_bits(nominal_bits: u8) -> u8 {
        nominal_bits + 1
    }

    /// Norm bits for given nominal bits: `b - 1`.
    ///
    /// Returns 1 if nominal_bits is 2 (minimum 1 bit for norm).
    #[must_use]
    #[inline]
    pub fn nrm_bits(nominal_bits: u8) -> u8 {
        (nominal_bits - 1).max(1)
    }

    /// Total bits per triplet for given nominal bits.
    ///
    /// `2·(b+1) + (b-1) = 3b + 1`
    #[must_use]
    #[inline]
    pub fn bits_per_triplet(nominal_bits: u8) -> usize {
        let dir = Self::dir_bits(nominal_bits) as usize;
        let nrm = Self::nrm_bits(nominal_bits) as usize;
        2 * dir + nrm
    }

    /// Effective bits per scalar coordinate.
    ///
    /// Each triplet covers 3 coordinates, so: `(3b + 1) / 3`.
    #[must_use]
    #[inline]
    pub fn effective_bits_per_scalar(nominal_bits: u8) -> f64 {
        Self::bits_per_triplet(nominal_bits) as f64 / 3.0
    }

    /// Create a default config for testing with common transformer dimensions.
    #[must_use]
    pub fn for_testing() -> Self {
        Self {
            seed: 42,
            n_layers: 4,
            kv_dim: 64,
            max_seq_len: 256,
            key_bits: 2,
            val_bits: 2,
            use_qjl_residual: false,
            use_joint_rounding: true,
        }
    }
}

/// Paired codebook for one side (key or value) of OCTOPUS compression.
///
/// Contains separate scalar codebooks for:
/// - **Norm** ρ ∈ [0,1] — Beta(3/2, (d-3)/2) marginal, `b-1` bits
/// - **Oct-direction** (ξ, η) ∈ [-1,1] — triangular marginal, `b+1` bits
///
/// The non-uniform bit split (b+1 for direction, b-1 for norm) is MSE-optimal
/// because direction errors dominate: E[ρ²] = 3/d → 0 while direction variance is O(1).
#[derive(Debug, Clone)]
pub struct OctopusCodebook {
    /// Codebook for triplet norm ρ ∈ [0,1].
    pub norm: ScalarCodebook,
    /// Codebook for oct-coordinate (ξ or η) ∈ [-1,1].
    pub oct: ScalarCodebook,
    /// Direction bits per oct-coordinate (b+1).
    pub dir_bits: u8,
    /// Norm bits per triplet norm (b-1).
    pub nrm_bits: u8,
}

impl OctopusCodebook {
    /// Build a paired codebook for given dimension and nominal bits.
    pub fn build(dim: usize, nominal_bits: u8) -> Self {
        let dir_bits = OctopusConfig::dir_bits(nominal_bits);
        let nrm_bits = OctopusConfig::nrm_bits(nominal_bits);
        Self {
            norm: super::codebook::build_norm_codebook(dim, nrm_bits),
            oct: super::codebook::build_oct_codebook(dir_bits),
            dir_bits,
            nrm_bits,
        }
    }
}

/// Per-layer OCTOPUS quantization state.
///
/// Each layer has its own rotation matrix (deterministic from seed + layer offset)
/// and separate key/value codebooks. The rotation reuses TurboQuant's
/// random orthogonal matrix (same WHT-equivalent approach).
#[derive(Debug, Clone)]
pub struct OctopusLayer {
    /// Random orthogonal rotation matrix (kv_dim × kv_dim), stored column-major.
    pub rotation: Vec<f32>,
    /// Codebook pair for key compression.
    pub key_codebook: OctopusCodebook,
    /// Codebook pair for value compression.
    pub val_codebook: OctopusCodebook,
    /// QJL second rotation signs (optional, for score-attention bias reduction).
    pub qjl_matrix: Option<Vec<f32>>,
}

/// Packed quantized indices for one triplet.
///
/// Each triplet stores 3 indices:
/// - `i_xi` — oct-coordinate ξ centroid index (dir_bits wide)
/// - `i_eta` — oct-coordinate η centroid index (dir_bits wide)
/// - `i_rho` — norm centroid index (nrm_bits wide)
///
/// Total index bits per triplet: `2·dir_bits + nrm_bits`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TripletIndices {
    /// Oct-coordinate ξ index → oct codebook.
    pub i_xi: u16,
    /// Oct-coordinate η index → oct codebook.
    pub i_eta: u16,
    /// Norm ρ index → norm codebook.
    pub i_rho: u16,
}

impl TripletIndices {
    /// Create zero indices (all pointing to first centroid).
    #[must_use]
    pub fn zero() -> Self {
        Self {
            i_xi: 0,
            i_eta: 0,
            i_rho: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_split() {
        // nominal=2 → dir=3, nrm=1
        assert_eq!(OctopusConfig::dir_bits(2), 3);
        assert_eq!(OctopusConfig::nrm_bits(2), 1);

        // nominal=3 → dir=4, nrm=2
        assert_eq!(OctopusConfig::dir_bits(3), 4);
        assert_eq!(OctopusConfig::nrm_bits(3), 2);

        // nominal=4 → dir=5, nrm=3
        assert_eq!(OctopusConfig::dir_bits(4), 5);
        assert_eq!(OctopusConfig::nrm_bits(4), 3);
    }

    #[test]
    fn test_bits_per_triplet() {
        // nominal=2: 2*3 + 1 = 7
        assert_eq!(OctopusConfig::bits_per_triplet(2), 7);
        // nominal=3: 2*4 + 2 = 10
        assert_eq!(OctopusConfig::bits_per_triplet(3), 10);
        // nominal=4: 2*5 + 3 = 13
        assert_eq!(OctopusConfig::bits_per_triplet(4), 13);
    }

    #[test]
    fn test_effective_bits_per_scalar() {
        // nominal=2: 7/3 ≈ 2.33
        let eff = OctopusConfig::effective_bits_per_scalar(2);
        assert!((eff - 7.0 / 3.0).abs() < 1e-10, "effective bits = {eff}");

        // nominal=3: 10/3 ≈ 3.33
        let eff = OctopusConfig::effective_bits_per_scalar(3);
        assert!((eff - 10.0 / 3.0).abs() < 1e-10, "effective bits = {eff}");
    }

    #[test]
    fn test_octopus_codebook_build() {
        let cb = OctopusCodebook::build(128, 2);
        assert_eq!(cb.dir_bits, 3);
        assert_eq!(cb.nrm_bits, 1);
        // dir=3 bits → 8 oct centroids, nrm=1 bit → 2 norm centroids
        assert_eq!(cb.oct.centroids.len(), 8);
        assert_eq!(cb.norm.centroids.len(), 2);
    }

    #[test]
    fn test_codebook_quantize_dequantize() {
        let cb = OctopusCodebook::build(128, 3);
        // Norm quantize
        let rho = 0.15f32; // near √(3/128)
        let rho_idx = cb.norm.quantize(rho);
        let rho_recon = cb.norm.dequantize(rho_idx);
        assert!(
            (rho_recon - rho).abs() < 0.2,
            "norm roundtrip: {rho} → idx {rho_idx} → {rho_recon}"
        );

        // Oct quantize
        let xi = 0.3f32;
        let xi_idx = cb.oct.quantize(xi);
        let xi_recon = cb.oct.dequantize(xi_idx);
        assert!(
            (xi_recon - xi).abs() < 0.3,
            "oct roundtrip: {xi} → idx {xi_idx} → {xi_recon}"
        );
    }

    #[test]
    fn test_config_for_testing() {
        let cfg = OctopusConfig::for_testing();
        assert_eq!(cfg.key_bits, 2);
        assert_eq!(cfg.kv_dim, 64);
        assert!(cfg.use_joint_rounding);
        assert!(!cfg.use_qjl_residual);
    }

    #[test]
    fn test_triplet_indices_zero() {
        let idx = TripletIndices::zero();
        assert_eq!(idx.i_xi, 0);
        assert_eq!(idx.i_eta, 0);
        assert_eq!(idx.i_rho, 0);
    }

    #[test]
    fn test_triplet_indices_default() {
        let idx = TripletIndices::default();
        assert_eq!(idx.i_xi, 0);
        assert_eq!(idx.i_eta, 0);
        assert_eq!(idx.i_rho, 0);
    }
}
