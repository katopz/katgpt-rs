//! Core types for Hybrid OCTOPUS-encoding + PlanarQuant-rotation codec.
//!
//! Combines PlanarQuant's O(d) 2D Givens rotation with OCTOPUS's octahedral
//! triplet encoding ((b+1, b-1) bit split). This gives OCTOPUS-quality MSE
//! at PlanarQuant rotation speed (256 FMAs for d=128 vs OCTOPUS's 16,384).

use crate::octopus::types::OctopusCodebook;

/// Configuration for hybrid OCTOPUS-encoding + PlanarQuant-rotation codec.
///
/// Uses PlanarQuant's 2D Givens rotation (O(d) FMAs) with OCTOPUS's
/// octahedral triplet encoding ((b+1, b-1) bit split).
#[derive(Debug, Clone)]
pub struct HybridOctPqConfig {
    /// Random seed for 2D Givens rotation generation (deterministic).
    pub seed: u64,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// KV dimension (head_dim × n_kv_heads). Padded to even for PQ rotation.
    pub kv_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Nominal bits per key coordinate. OCTOPUS splits: dir=b+1, nrm=b-1.
    pub key_bits: u8,
    /// Nominal bits per value coordinate.
    pub val_bits: u8,
    /// Enable joint 3×3 rounding in OCT encoder (6-14% MSE gain).
    pub use_joint_rounding: bool,
}

impl HybridOctPqConfig {
    /// Create a default config for testing.
    #[must_use]
    pub fn for_testing() -> Self {
        Self {
            seed: 42,
            n_layers: 2,
            kv_dim: 64,
            max_seq_len: 256,
            key_bits: 2,
            val_bits: 2,
            use_joint_rounding: true,
        }
    }
}

/// Per-layer hybrid state: PQ 2D rotations + OCT dual codebooks.
///
/// Combines:
/// - PlanarQuant's per-pair (cos θ, sin θ) rotations — ceil(kv_dim/2) pairs
/// - OCTOPUS's paired norm + oct-direction codebooks per side
#[derive(Debug, Clone)]
pub struct HybridOctPqLayer {
    /// Key 2D Givens rotations: (cos θ, sin θ) per pair.
    pub key_rotations: Vec<[f32; 2]>,
    /// Value 2D Givens rotations.
    pub val_rotations: Vec<[f32; 2]>,
    /// OCTOPUS key codebook pair (norm + oct-direction).
    pub key_codebook: OctopusCodebook,
    /// OCTOPUS value codebook pair.
    pub val_codebook: OctopusCodebook,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_for_testing() {
        let cfg = HybridOctPqConfig::for_testing();
        assert_eq!(cfg.key_bits, 2);
        assert_eq!(cfg.kv_dim, 64);
        assert!(cfg.use_joint_rounding);
    }
}
