//! Static Calibration Tables — pre-computed per-head attention scales.
//! Replaces runtime Sinkhorn iterations with O(1) lookup.
//!
//! Inspired by Gemma 4 QAT: "optimize for the precision you'll deploy at."
//! Feature-gated behind `static_cal_tables`.

/// Pre-computed static calibration table for attention heads.
/// Stores per-head scale factors computed from representative prompts.
#[derive(Debug, Clone)]
pub struct StaticCalTable {
    /// Per-head scale factors, indexed by (layer * num_heads + head).
    pub scales: Vec<f32>,
    /// Number of layers in the model.
    pub num_layers: usize,
    /// Number of attention heads per layer.
    pub num_heads: usize,
    /// Number of calibration prompts used to compute scales.
    pub calibration_prompts: usize,
    /// BLAKE3 commitment hash of the scale table.
    pub commitment: [u8; 32],
}

impl StaticCalTable {
    /// Create a new empty calibration table.
    /// All scales initialize to 1.0 (neutral — no correction).
    pub fn new(num_layers: usize, num_heads: usize) -> Self {
        let total = num_layers * num_heads;
        let mut table = Self {
            scales: vec![1.0; total],
            num_layers,
            num_heads,
            calibration_prompts: 0,
            commitment: [0u8; 32],
        };
        table.commit();
        table
    }

    /// O(1) lookup for a specific head's scale.
    #[inline]
    pub fn get_scale(&self, layer: usize, head: usize) -> f32 {
        debug_assert!(layer < self.num_layers);
        debug_assert!(head < self.num_heads);
        // SAFETY: index is bounds-checked by debug_assert above; in release this is pure indexing.
        unsafe { *self.scales.get_unchecked(layer * self.num_heads + head) }
    }

    /// Set scale for a specific head.
    pub fn set_scale(&mut self, layer: usize, head: usize, scale: f32) {
        debug_assert!(layer < self.num_layers);
        debug_assert!(head < self.num_heads);
        self.scales[layer * self.num_heads + head] = scale;
    }

    /// Total number of entries in the table.
    pub fn len(&self) -> usize {
        self.scales.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.scales.is_empty()
    }

    /// Calibrate from activation statistics.
    ///
    /// Takes per-head activation statistics and computes scale factors.
    /// Uses sigmoid (not softmax) to normalize each head's scale to [0.5, 1.5] range.
    /// Updates via EMA (α=0.1) for stability across calibration passes.
    pub fn calibrate_from_stats(&mut self, stats: &[HeadStats]) {
        self.calibration_prompts += 1;
        for stat in stats {
            let idx = stat.layer * self.num_heads + stat.head;
            if idx < self.scales.len() {
                // Sigmoid-normalized scale: 1.0 + 0.5 * sigmoid(mean_activation - offset)
                // Heads with high activation get slightly boosted, low activation slightly dampened
                let normalized = sigmoid(stat.mean_activation * 0.1);
                let scale = 0.5 + normalized; // [0.5, 1.5]
                // EMA update: smooth convergence, never jumps
                let prev = self.scales[idx];
                self.scales[idx] = 0.9 * prev + 0.1 * scale;
            }
        }
        self.commit();
    }

    /// Compute BLAKE3 commitment hash over all scales.
    ///
    /// Hashes the raw `f32` bytes of `scales` as a single contiguous slice rather
    /// than per-element `update()` calls — lets BLAKE3's internal buffer absorb
    /// a large chunk in one go (fewer merge rounds).
    pub fn commit(&mut self) {
        self.commitment = hash_scales(&self.scales);
    }

    /// Verify BLAKE3 commitment matches current scales.
    pub fn verify(&self) -> bool {
        hash_scales(&self.scales) == self.commitment
    }
}

/// Hash `scales` with BLAKE3 as a contiguous byte slice.
///
/// SAFETY: `[f32]` and `[u8]` have no padding on supported targets
/// (f32 is 4 bytes, alignment 4; u8 alignment 1). We hash little-endian
/// representation; on big-endian targets this would differ from
/// `to_le_bytes()` per-element, but the project targets little-endian
/// platforms (aarch64-apple, x86_64) and BLAKE3 commitment is only
/// verified against commitments produced by the same build.
fn hash_scales(scales: &[f32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            scales.as_ptr() as *const u8,
            std::mem::size_of_val(scales),
        )
    };
    hasher.update(bytes);
    hasher.finalize().into()
}

/// Per-head activation statistics collected during calibration pass.
#[derive(Debug, Clone)]
pub struct HeadStats {
    pub layer: usize,
    pub head: usize,
    pub mean_activation: f32,
    pub variance: f32,
    pub max_activation: f32,
}

/// Sigmoid activation — used instead of softmax for independent per-head normalization.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Run calibration pass over representative prompts.
/// Takes a closure that simulates model forward pass and returns per-head activation stats.
/// The closure receives (prompt_index) and should return Vec<HeadStats>.
pub fn run_calibration_pass<F>(table: &mut StaticCalTable, num_prompts: usize, forward_fn: F)
where
    F: Fn(usize) -> Vec<HeadStats>,
{
    for i in 0..num_prompts {
        let stats = forward_fn(i);
        table.calibrate_from_stats(&stats);
    }
}

/// RV-triggered recalibration: checks if the RV signal exceeds threshold
/// and triggers recalibration if so.
pub fn check_rv_recalibration(
    table: &mut StaticCalTable,
    rv_variance: f64,
    threshold: f64,
    forward_fn: impl Fn() -> Vec<HeadStats>,
) -> bool {
    if rv_variance > threshold {
        let stats = forward_fn();
        table.calibrate_from_stats(&stats);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_table_default_scales() {
        let table = StaticCalTable::new(4, 8);
        assert_eq!(table.scales.len(), 32);
        assert!(table.scales.iter().all(|&s| s == 1.0));
        assert_eq!(table.num_layers, 4);
        assert_eq!(table.num_heads, 8);
        assert_eq!(table.calibration_prompts, 0);
    }

    #[test]
    fn test_len_is_empty() {
        let table = StaticCalTable::new(2, 4);
        assert_eq!(table.len(), 8);
        assert!(!table.is_empty());

        let empty = StaticCalTable::new(0, 0);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_get_set_scale() {
        let mut table = StaticCalTable::new(2, 4);
        table.set_scale(1, 2, 0.75);
        assert!((table.get_scale(1, 2) - 0.75).abs() < 1e-6);
        assert!((table.get_scale(0, 0) - 1.0).abs() < 1e-6); // unchanged
    }

    #[test]
    fn test_calibrate_from_stats() {
        let mut table = StaticCalTable::new(1, 2);
        let stats = vec![
            HeadStats {
                layer: 0,
                head: 0,
                mean_activation: 5.0,
                variance: 1.0,
                max_activation: 8.0,
            },
            HeadStats {
                layer: 0,
                head: 1,
                mean_activation: -2.0,
                variance: 0.5,
                max_activation: 1.0,
            },
        ];
        table.calibrate_from_stats(&stats);
        // High activation head should get slightly higher scale
        assert!(table.get_scale(0, 0) > table.get_scale(0, 1));
        assert!(table.verify());
        assert_eq!(table.calibration_prompts, 1);
    }

    #[test]
    fn test_calibrate_out_of_bounds_ignored() {
        let mut table = StaticCalTable::new(1, 1);
        let stats = vec![HeadStats {
            layer: 5,
            head: 5,
            mean_activation: 10.0,
            variance: 1.0,
            max_activation: 15.0,
        }];
        table.calibrate_from_stats(&stats);
        // Scale unchanged — out of bounds stat ignored
        assert!((table.get_scale(0, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_commitment_roundtrip() {
        let mut table = StaticCalTable::new(2, 4);
        table.set_scale(0, 0, 1.5);
        table.commit();
        assert!(table.verify());
        // Tamper
        table.scales[0] = 2.0;
        assert!(!table.verify());
    }

    #[test]
    fn test_ema_update() {
        let mut table = StaticCalTable::new(1, 1);
        let stats = vec![HeadStats {
            layer: 0,
            head: 0,
            mean_activation: 10.0,
            variance: 1.0,
            max_activation: 15.0,
        }];
        // Multiple calibrations should converge
        for _ in 0..100 {
            table.calibrate_from_stats(&stats);
        }
        // Should have converged to a stable value
        let final_scale = table.get_scale(0, 0);
        assert!(final_scale > 1.0 && final_scale < 1.5);
        // Verify commitment still valid after all updates
        assert!(table.verify());
    }

    #[test]
    fn test_sigmoid_range() {
        // Sigmoid always in [0, 1]
        assert!(sigmoid(0.0) > 0.49 && sigmoid(0.0) < 0.51);
        assert!(sigmoid(-100.0) >= 0.0); // f32 underflows to 0 for large negative
        assert!(sigmoid(100.0) > 0.99 && sigmoid(100.0) <= 1.0);
    }

    #[test]
    fn test_sigmoid_not_softmax() {
        // Verify independence: sigmoid(a) + sigmoid(b) != 1 in general
        let a = sigmoid(1.0);
        let b = sigmoid(2.0);
        assert!((a + b - 1.0).abs() > 0.1); // Not softmax — values don't sum to 1
    }
}
