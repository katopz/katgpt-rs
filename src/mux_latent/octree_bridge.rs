//! Bridge between KG Latent Octree and MUX Latent Patches.
//!
//! Maps octree leaf nodes (TernaryDir) to MUX latent weight vectors and back.
//! The octree uses 2-bit ternary nodes {-1, 0, +1}. A depth-7 octree has 128 nodes
//! = 256 bits = 4 × u64 = 32 bytes, which matches a single X8 latent slot.
//!
//! Segment_id maps 1:1 to octree morton code.

use crate::mux_latent::config::CompressionRatio;

/// Ternary direction from KG octree: {-1, 0, +1}.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum TernaryValue {
    Negative = -1,
    Zero = 0,
    Positive = 1,
}

impl TernaryValue {
    /// Convert to f32 weight for MUX latent slot.
    pub fn to_weight(self) -> f32 {
        match self {
            Self::Negative => -1.0,
            Self::Zero => 0.0,
            Self::Positive => 1.0,
        }
    }

    /// Convert from f32 weight, thresholding at ±0.5.
    pub fn from_weight(w: f32) -> Self {
        if w > 0.5 {
            Self::Positive
        } else if w < -0.5 {
            Self::Negative
        } else {
            Self::Zero
        }
    }
}

/// A KG octree leaf node carrying a ternary direction.
/// 32 bytes: 4 × u64 bitmask representation of 128 ternary nodes.
#[derive(Debug, Clone)]
pub struct TernaryDir {
    /// Bitmask representation: 2 bits per ternary node.
    /// 00 = Zero, 01 = Positive, 10 = Negative.
    /// 128 nodes × 2 bits = 256 bits = 4 × u64.
    pub bitmask: [u64; 4],
}

impl TernaryDir {
    /// Create a zero-valued direction.
    pub fn zero() -> Self {
        Self { bitmask: [0u64; 4] }
    }

    /// Get the ternary value at the given node index (0..127).
    pub fn get(&self, index: usize) -> TernaryValue {
        assert!(index < 128, "Octree node index must be < 128, got {index}");
        let u64_idx = index / 32;
        let bit_offset = (index % 32) * 2;
        let bits = ((self.bitmask[u64_idx] >> bit_offset) & 0b11) as u8;
        match bits {
            0 => TernaryValue::Zero,
            1 => TernaryValue::Positive,
            2 => TernaryValue::Negative,
            _ => TernaryValue::Zero, // 3 is unused, default to Zero
        }
    }

    /// Set the ternary value at the given node index.
    pub fn set(&mut self, index: usize, value: TernaryValue) {
        assert!(index < 128, "Octree node index must be < 128, got {index}");
        let u64_idx = index / 32;
        let bit_offset = (index % 32) * 2;
        let bits = match value {
            TernaryValue::Zero => 0u64,
            TernaryValue::Positive => 1u64,
            TernaryValue::Negative => 2u64,
        };
        // Clear existing 2 bits
        self.bitmask[u64_idx] &= !(0b11 << bit_offset);
        // Set new value
        self.bitmask[u64_idx] |= bits << bit_offset;
    }

    /// Convert to MUX latent weights (8 × f32).
    /// Maps 128 ternary nodes to 8 weights by averaging groups of 16.
    pub fn to_weights(&self) -> [f32; 8] {
        let mut weights = [0.0f32; 8];
        for (group, weight_slot) in weights.iter_mut().enumerate() {
            let base = group * 16;
            let mut sum = 0.0f32;
            for i in 0..16 {
                sum += self.get(base + i).to_weight();
            }
            *weight_slot = sum / 16.0;
        }
        weights
    }

    /// Convert from MUX latent weights to ternary direction.
    /// Each weight maps to 16 ternary nodes (broadcast).
    pub fn from_weights(weights: &[f32; 8]) -> Self {
        let mut dir = Self::zero();
        for (group, &w) in weights.iter().enumerate() {
            let tv = TernaryValue::from_weight(w);
            let base = group * 16;
            for i in 0..16 {
                dir.set(base + i, tv);
            }
        }
        dir
    }

    /// Size in bytes (always 32).
    pub const fn size_bytes(&self) -> usize {
        32
    }
}

/// Morton code utilities for segment_id ↔ octree position mapping.
pub struct MortonCode;

impl MortonCode {
    /// Encode (x, y) coordinates into a morton code (segment_id).
    /// Supports up to 16 bits per axis (max depth 16, 65536 positions).
    pub fn encode(x: u32, y: u32) -> u32 {
        let mut result = 0u32;
        let mut x = x;
        let mut y = y;
        let mut bit = 0u32;
        while x > 0 || y > 0 {
            result |= (x & 1) << bit;
            result |= (y & 1) << (bit + 1);
            x >>= 1;
            y >>= 1;
            bit += 2;
        }
        result
    }

    /// Decode a morton code back into (x, y) coordinates.
    pub fn decode(morton: u32) -> (u32, u32) {
        let mut x = 0u32;
        let mut y = 0u32;
        let mut m = morton;
        let mut bit = 0u32;
        while m > 0 {
            x |= (m & 1) << bit;
            m >>= 1;
            y |= (m & 1) << bit;
            m >>= 1;
            bit += 1;
        }
        (x, y)
    }
}

/// LOD (Level of Detail) mapping from octree depth to compression ratio.
pub struct OctreeLod;

impl OctreeLod {
    /// Map octree depth to compression ratio.
    /// Depth 3 = X16
    /// Depth 5 = X8
    /// Depth 7 = X4 (fine-grained, less compression)
    pub fn depth_to_ratio(depth: usize) -> CompressionRatio {
        match depth {
            0..=3 => CompressionRatio::X16,
            4..=5 => CompressionRatio::X8,
            _ => CompressionRatio::X4,
        }
    }

    /// Map compression ratio to recommended octree depth.
    pub fn ratio_to_depth(ratio: CompressionRatio) -> usize {
        match ratio {
            CompressionRatio::X4 => 7,
            CompressionRatio::X8 => 5,
            CompressionRatio::X16 => 3,
        }
    }

    /// Number of latent slots at a given octree depth.
    pub fn slot_count(depth: usize) -> usize {
        1 << depth.min(7)
    }
}

/// Bridge function: convert TernaryDir at octree leaf to a LatentPatch-compatible weight vector.
///
/// This is the core bridge: octree leaf → MUX weights → wire patch.
/// Returns `[f32; 8]` suitable for `LatentPatch::new()`.
pub fn octree_leaf_to_patch_weights(dir: &TernaryDir) -> [f32; 8] {
    dir.to_weights()
}

/// Bridge function: convert wire patch weights back to TernaryDir.
///
/// Inverse of `octree_leaf_to_patch_weights`.
pub fn patch_weights_to_octree_leaf(weights: &[f32; 8]) -> TernaryDir {
    TernaryDir::from_weights(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_value_roundtrip() {
        for tv in [
            TernaryValue::Negative,
            TernaryValue::Zero,
            TernaryValue::Positive,
        ] {
            let w = tv.to_weight();
            let back = TernaryValue::from_weight(w);
            assert_eq!(tv, back, "Roundtrip failed for {tv:?}");
        }
    }

    #[test]
    fn test_ternary_dir_get_set() {
        let mut dir = TernaryDir::zero();
        dir.set(0, TernaryValue::Positive);
        dir.set(1, TernaryValue::Negative);
        dir.set(127, TernaryValue::Positive);

        assert_eq!(dir.get(0), TernaryValue::Positive);
        assert_eq!(dir.get(1), TernaryValue::Negative);
        assert_eq!(dir.get(127), TernaryValue::Positive);
        assert_eq!(dir.get(2), TernaryValue::Zero); // Default
    }

    #[test]
    fn test_ternary_dir_to_weights_roundtrip() {
        // Fill entire groups to ensure the average exceeds ±0.5 threshold
        let mut dir = TernaryDir::zero();
        for i in 0..16 {
            dir.set(i, TernaryValue::Positive);
        }
        for i in 16..32 {
            dir.set(i, TernaryValue::Negative);
        }

        let weights = dir.to_weights();
        assert!(
            (weights[0] - 1.0).abs() < f32::EPSILON,
            "Group 0 should be +1.0"
        );
        assert!(
            (weights[1] - (-1.0)).abs() < f32::EPSILON,
            "Group 1 should be -1.0"
        );

        // Roundtrip should be exact for uniform groups
        let dir2 = TernaryDir::from_weights(&weights);
        assert_eq!(dir2.get(0), TernaryValue::Positive);
        assert_eq!(dir2.get(16), TernaryValue::Negative);
    }

    #[test]
    fn test_morton_code_roundtrip() {
        for &(x, y) in &[(0, 0), (1, 0), (0, 1), (15, 15), (100, 200)] {
            let morton = MortonCode::encode(x, y);
            let (x2, y2) = MortonCode::decode(morton);
            assert_eq!((x, y), (x2, y2), "Morton roundtrip failed for ({x}, {y})");
        }
    }

    #[test]
    fn test_morton_code_ordering() {
        let m00 = MortonCode::encode(0, 0);
        let m10 = MortonCode::encode(1, 0);
        let m01 = MortonCode::encode(0, 1);
        let m11 = MortonCode::encode(1, 1);

        assert!(m00 < m10);
        assert!(m00 < m01);
        assert!(m10 < m11);
    }

    #[test]
    fn test_octree_lod_mapping() {
        assert_eq!(OctreeLod::depth_to_ratio(3), CompressionRatio::X16);
        assert_eq!(OctreeLod::depth_to_ratio(5), CompressionRatio::X8);
        assert_eq!(OctreeLod::depth_to_ratio(7), CompressionRatio::X4);

        assert_eq!(OctreeLod::ratio_to_depth(CompressionRatio::X16), 3);
        assert_eq!(OctreeLod::ratio_to_depth(CompressionRatio::X8), 5);
        assert_eq!(OctreeLod::ratio_to_depth(CompressionRatio::X4), 7);
    }

    #[test]
    fn test_bridge_functions() {
        // Use uniform groups so averaging is exact
        let mut dir = TernaryDir::zero();
        for i in 0..128 {
            let group = i / 16;
            let value = match group % 3 {
                0 => TernaryValue::Positive,
                1 => TernaryValue::Negative,
                _ => TernaryValue::Zero,
            };
            dir.set(i, value);
        }

        let weights = octree_leaf_to_patch_weights(&dir);
        for w in &weights {
            assert!(*w >= -1.0 && *w <= 1.0, "Weight {w} out of range");
        }

        // Verify uniform groups roundtrip correctly
        assert_eq!(weights[0], 1.0); // Group 0: all Positive
        assert_eq!(weights[1], -1.0); // Group 1: all Negative
        assert_eq!(weights[2], 0.0); // Group 2: all Zero

        let dir2 = patch_weights_to_octree_leaf(&weights);
        assert_eq!(dir2.get(0), TernaryValue::Positive);
        assert_eq!(dir2.get(16), TernaryValue::Negative);
        assert_eq!(dir2.get(32), TernaryValue::Zero);
    }

    #[test]
    fn test_ternary_dir_size() {
        let dir = TernaryDir::zero();
        assert_eq!(dir.size_bytes(), 32);
    }

    #[test]
    fn test_slot_count() {
        assert_eq!(OctreeLod::slot_count(3), 8);
        assert_eq!(OctreeLod::slot_count(5), 32);
        assert_eq!(OctreeLod::slot_count(7), 128);
    }
}
