//! Position-free KV cache compaction.
//!
//! Un-rotates RoPE from keys before compaction, compacts in position-free latent space,
//! then re-rotates on retrieval. This decouples compaction from positional encoding.
//!
//! RoPE convention: LLaMA/Gemma style "rotate half".
//! Pairs: `vec[i]` with `vec[i + half]` where `half = head_dim / 2`.
//! Forward rotation:  x' = x*cos(θ) - y*sin(θ),  y' = x*sin(θ) + y*cos(θ)
//! Inverse rotation:  x  = x'*cos(θ) + y'*sin(θ), y  = -x'*sin(θ) + y'*cos(θ)

use half::f16;

/// Compactor that operates in position-free (RoPE-removed) space.
///
/// The key insight: RoPE encodes absolute position via rotation. If we un-rotate
/// keys before compaction, the compactor sees "pure semantic" keys without position
/// interference. After compaction, we re-rotate with adjusted positions.
#[derive(Debug, Clone)]
pub struct PositionFreeCompactor {
    /// RoPE base frequency (theta).
    pub rope_theta: f32,
    /// Head dimension.
    pub head_dim: usize,
    /// Pre-computed RoPE frequency table: `freq[i] = 1.0 / (theta^(2i / head_dim))`.
    /// Cached at construction so per-call un-rotate/re-rotate avoid recomputing it.
    freqs: Vec<f32>,
}

impl PositionFreeCompactor {
    /// Create a new position-free compactor with the given RoPE parameters.
    pub fn new(rope_theta: f32, head_dim: usize) -> Self {
        let freqs = Self::compute_freq_table(rope_theta, head_dim);
        Self {
            rope_theta,
            head_dim,
            freqs,
        }
    }

    /// Build the RoPE frequency table.
    ///
    /// `freq[i] = 1.0 / (theta^(2i / head_dim))` for `i` in `0..half`.
    /// Pure function of `(rope_theta, head_dim)` — safe to cache once.
    fn compute_freq_table(rope_theta: f32, head_dim: usize) -> Vec<f32> {
        let half = head_dim / 2;
        let mut freqs = Vec::with_capacity(half);
        for i in 0..half {
            let exponent = 2.0 * i as f32 / head_dim as f32;
            freqs.push(1.0 / rope_theta.powf(exponent));
        }
        freqs
    }

    /// Un-rotate RoPE from keys, returning position-free key buffer.
    ///
    /// # Arguments
    /// * `keys` - Flat f16 key buffer, shape `[seq_len * head_dim]`
    /// * `start_pos` - Starting position index of this key sequence
    ///
    /// # Returns
    /// Position-free keys in f32 for compaction processing.
    ///
    /// Inverse rotation (un-rotate):
    ///   x_new = x*cos(θ) + y*sin(θ)
    ///   y_new = -x*sin(θ) + y*cos(θ)
    pub fn un_rotate_keys(&self, keys: &[f16], start_pos: usize) -> Vec<f32> {
        if self.head_dim == 0 || keys.is_empty() {
            return Vec::new();
        }

        let head_dim = self.head_dim;
        let half = head_dim / 2;
        let seq_len = keys.len() / head_dim;

        // Convert f16 → f32
        let mut out = Vec::with_capacity(keys.len());
        for &v in keys {
            out.push(v.to_f32());
        }

        // Use cached frequency table (built once at construction).
        let freqs = &self.freqs;

        // For each token at position p, un-rotate pairs (i, i+half)
        for t in 0..seq_len {
            let pos = (start_pos + t) as f32;
            let base = t * head_dim;

            // Process in chunks of 4 for auto-vectorization
            let chunks = half / 4;
            let remainder = half % 4;

            for c in 0..chunks {
                let i = c * 4;

                // Unroll 4 pairs
                for j in 0..4 {
                    let idx = i + j;
                    let angle = pos * freqs[idx];
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    let x = out[base + idx];
                    let y = out[base + idx + half];

                    // Inverse rotation
                    out[base + idx] = x * cos_a + y * sin_a;
                    out[base + idx + half] = -x * sin_a + y * cos_a;
                }
            }

            // Handle remainder
            for j in 0..remainder {
                let idx = chunks * 4 + j;
                let angle = pos * freqs[idx];
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                let x = out[base + idx];
                let y = out[base + idx + half];

                out[base + idx] = x * cos_a + y * sin_a;
                out[base + idx + half] = -x * sin_a + y * cos_a;
            }
        }

        out
    }

    /// Re-rotate keys with new positions after compaction.
    ///
    /// # Arguments
    /// * `keys` - Position-free keys in f32, shape `[seq_len * head_dim]`
    /// * `new_start_pos` - New starting position for the compacted sequence
    ///
    /// # Returns
    /// Re-rotated keys in f16.
    ///
    /// Forward rotation:
    ///   x_new = x*cos(θ) - y*sin(θ)
    ///   y_new = x*sin(θ) + y*cos(θ)
    pub fn re_rotate_keys(&self, keys: &[f32], new_start_pos: usize) -> Vec<f16> {
        if self.head_dim == 0 || keys.is_empty() {
            return Vec::new();
        }

        let head_dim = self.head_dim;
        let half = head_dim / 2;
        let seq_len = keys.len() / head_dim;

        // Work in f32, convert to f16 at the end
        let mut buf = keys.to_vec();

        // Use cached frequency table (built once at construction).
        let freqs = &self.freqs;

        // For each token at position p, apply forward rotation
        for t in 0..seq_len {
            let pos = (new_start_pos + t) as f32;
            let base = t * head_dim;

            let chunks = half / 4;
            let remainder = half % 4;

            for c in 0..chunks {
                let i = c * 4;

                for j in 0..4 {
                    let idx = i + j;
                    let angle = pos * freqs[idx];
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    let x = buf[base + idx];
                    let y = buf[base + idx + half];

                    // Forward rotation
                    buf[base + idx] = x * cos_a - y * sin_a;
                    buf[base + idx + half] = x * sin_a + y * cos_a;
                }
            }

            for j in 0..remainder {
                let idx = chunks * 4 + j;
                let angle = pos * freqs[idx];
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                let x = buf[base + idx];
                let y = buf[base + idx + half];

                buf[base + idx] = x * cos_a - y * sin_a;
                buf[base + idx + half] = x * sin_a + y * cos_a;
            }
        }

        // Convert f32 → f16
        buf.iter().map(|&v| f16::from_f32(v)).collect()
    }

    /// Compute the position offset for the compacted cache.
    ///
    /// After compaction, compacted tokens occupy positions `[new_start_pos..new_start_pos+compact_len]`.
    /// For RoPE continuation, new tokens should start at `original_start + original_len`.
    /// Therefore: `position_offset = original_start + original_len - compact_len`.
    ///
    /// This is the `position_offset` field in `CompactKVCache`.
    pub fn compute_position_offset(
        &self,
        original_start: usize,
        original_len: usize,
        compact_len: usize,
    ) -> usize {
        if compact_len == 0 {
            return original_start;
        }
        original_start
            .saturating_add(original_len)
            .saturating_sub(compact_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_free_compactor_new() {
        let compactor = PositionFreeCompactor::new(10000.0, 64);
        assert_eq!(compactor.rope_theta, 10000.0);
        assert_eq!(compactor.head_dim, 64);
    }

    #[test]
    fn test_compute_position_offset() {
        let compactor = PositionFreeCompactor::new(10000.0, 64);
        // original_start=10, original_len=100, compact_len=25
        // offset = 10 + 100 - 25 = 85
        let offset = compactor.compute_position_offset(10, 100, 25);
        assert_eq!(offset, 85);

        // Edge case: zero compact_len returns original_start
        let offset_zero = compactor.compute_position_offset(10, 100, 0);
        assert_eq!(offset_zero, 10);

        // Compact equals original length: offset = original_start
        let offset_same = compactor.compute_position_offset(5, 50, 50);
        assert_eq!(offset_same, 5);
    }

    #[test]
    fn test_un_rotate_re_rotate_roundtrip() {
        // Un-rotate then re-rotate should recover original within f16 precision
        let compactor = PositionFreeCompactor::new(10000.0, 8);
        let start_pos = 10;

        // Create keys with non-trivial values so rotation has an effect
        let original_f16: Vec<f16> = (0..8)
            .map(|i| f16::from_f32((i as f32 + 1.0) * 0.5))
            .collect();

        // Un-rotate at start_pos=10
        let unrotated = compactor.un_rotate_keys(&original_f16, start_pos);

        // Re-rotate back at same start_pos=10
        let rerotated_f16 = compactor.re_rotate_keys(&unrotated, start_pos);

        // Convert back to f32 for comparison
        let rerotated_f32: Vec<f32> = rerotated_f16.iter().map(|v| v.to_f32()).collect();
        let original_f32: Vec<f32> = original_f16.iter().map(|v| v.to_f32()).collect();

        // Should recover original within f16 precision (~1e-3 relative)
        for (orig, round) in original_f32.iter().zip(rerotated_f32.iter()) {
            let diff = (orig - round).abs();
            let tol = orig.abs().max(1e-3) * 1e-3;
            assert!(
                diff < tol.max(5e-4),
                "Roundtrip mismatch: original={orig}, round={round}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_un_rotate_identity_at_pos_zero() {
        // At position 0, all angles are 0 → cos=1, sin=0 → identity
        let compactor = PositionFreeCompactor::new(10000.0, 8);

        let keys: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32 + 1.0)).collect();

        let unrotated = compactor.un_rotate_keys(&keys, 0);

        for (orig, unr) in keys.iter().zip(unrotated.iter()) {
            let diff = (orig.to_f32() - unr).abs();
            assert!(
                diff < 1e-6,
                "Identity at pos=0 failed: orig={}, unr={}, diff={}",
                orig.to_f32(),
                unr,
                diff
            );
        }
    }

    #[test]
    fn test_re_rotate_produces_f16() {
        let compactor = PositionFreeCompactor::new(10000.0, 8);

        let keys_f32: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let result = compactor.re_rotate_keys(&keys_f32, 5);

        assert_eq!(result.len(), 8, "Output size should match input size");
        // Verify it's f16 (compile-time, but also check values are finite f16 range)
        for (i, v) in result.iter().enumerate() {
            let back = v.to_f32();
            assert!(
                back.is_finite(),
                "Result at index {i} is not finite: {back}"
            );
        }
    }

    #[test]
    fn test_un_rotate_nontrivial_rotation() {
        // Verify that un-rotate at non-zero position actually changes values
        let compactor = PositionFreeCompactor::new(10000.0, 8);

        let keys: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32 + 1.0)).collect();

        let unrotated = compactor.un_rotate_keys(&keys, 100);

        // At pos=100, angles are non-zero → values should differ
        let mut any_different = false;
        for (orig, unr) in keys.iter().zip(unrotated.iter()) {
            if (orig.to_f32() - unr).abs() > 1e-6 {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Un-rotate at pos=100 should produce different values"
        );
    }

    #[test]
    fn test_freq_table_correctness() {
        let compactor = PositionFreeCompactor::new(10000.0, 8);
        let freqs = &compactor.freqs;
        let half = 4; // head_dim / 2

        assert_eq!(freqs.len(), half);

        // freq[0] = 1.0 / (10000^(0/8)) = 1.0 / 1.0 = 1.0
        assert!((freqs[0] - 1.0).abs() < 1e-6, "freq[0] should be 1.0");

        // freq[1] = 1.0 / (10000^(2/8)) = 1.0 / (10000^0.25) ≈ 0.1
        let expected_1 = 1.0 / 10000.0_f32.powf(0.25);
        assert!(
            (freqs[1] - expected_1).abs() < 1e-6,
            "freq[1] should be {expected_1}, got {}",
            freqs[1]
        );
    }

    #[test]
    fn test_multi_token_un_rotate_re_rotate_roundtrip() {
        // Multi-token sequence roundtrip
        let compactor = PositionFreeCompactor::new(10000.0, 16);
        let seq_len = 5;
        let start_pos = 20;

        // seq_len tokens × head_dim = 80 f16 values
        let original_f16: Vec<f16> = (0..seq_len * 16)
            .map(|i| f16::from_f32(((i % 16) as f32 + 1.0) * 0.3))
            .collect();

        let unrotated = compactor.un_rotate_keys(&original_f16, start_pos);
        assert_eq!(unrotated.len(), seq_len * 16);

        let rerotated_f16 = compactor.re_rotate_keys(&unrotated, start_pos);
        assert_eq!(rerotated_f16.len(), seq_len * 16);

        let original_f32: Vec<f32> = original_f16.iter().map(|v| v.to_f32()).collect();
        let rerotated_f32: Vec<f32> = rerotated_f16.iter().map(|v| v.to_f32()).collect();

        for (i, (orig, round)) in original_f32.iter().zip(rerotated_f32.iter()).enumerate() {
            let diff = (orig - round).abs();
            let tol = orig.abs().max(1e-3) * 1e-3;
            assert!(
                diff < tol.max(5e-4),
                "Token {i}: roundtrip mismatch, orig={orig}, round={round}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_empty_keys() {
        let compactor = PositionFreeCompactor::new(10000.0, 8);

        let empty: Vec<f16> = vec![];
        let unrotated = compactor.un_rotate_keys(&empty, 0);
        assert!(unrotated.is_empty());

        let empty_f32: Vec<f32> = vec![];
        let rerotated = compactor.re_rotate_keys(&empty_f32, 0);
        assert!(rerotated.is_empty());
    }
}
