//! Static Calibration Tables — pre-computed per-head attention scales.
//! O(1) per-head scale lookup. NOT a KVarN Sinkhorn replacement: the KVarN
//! branch that consumed this was dead since the Issue 015 extraction and was
//! deleted by Issue 897 — a per-row scale ahead of KVarN's per-row RTN is
//! absorbed by the RTN min/max, so it reduces to plain RTN (HISTORY.md).
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
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, std::mem::size_of_val(scales))
    };
    hasher.update(bytes);
    hasher.finalize().into()
}

/// Per-head activation statistics collected during calibration pass.
#[derive(Debug, Clone, Copy)]
pub struct HeadStats {
    pub layer: usize,
    pub head: usize,
    pub mean_activation: f32,
    pub variance: f32,
    pub max_activation: f32,
}

/// Sigmoid activation — used instead of softmax for independent per-head normalization.
/// Delegates to [`katgpt_core::simd::fast_sigmoid`] (Cephes polynomial).
#[inline]
fn sigmoid(x: f32) -> f32 {
    katgpt_core::simd::fast_sigmoid(x)
}

/// Run calibration pass over representative prompts.
/// Takes a closure that simulates model forward pass and returns per-head activation stats.
/// The closure receives (prompt_index) and should return `Vec<HeadStats>`.
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

/// Bind a calibration table to the WEIGHTS it was fitted against
/// (Issue 841 §B-3, seam 2 of 4).
///
/// ⛔ **`StaticCalTable::commitment` is the identity of the TABLE, not of the
/// weights** — it is a BLAKE3 over `scales`, so it moves when the table is
/// recalibrated and is *constant across a weight swap*, which is precisely the
/// event that makes the table wrong. The two are independent and neither
/// substitutes for the other: `verify()` answers *were these scales
/// corrupted*, this answers *are these scales still describing the live
/// weights*.
///
/// A wrapper, deliberately, rather than a field: [`StaticCalTable::get_scale`]
/// is a shipped signature returning a bare `f32` and it is untouched, so a
/// caller that has not opted in compiles and behaves byte-identically. The
/// guarded read is an additive sibling.
///
/// ```ignore
/// use katgpt_core::calibration_staleness::SnapshotIdentity;
/// let bound = table.bound_to(store.snapshot_id()?);
/// // … after somebody swaps the snapshot …
/// match bound.get_scale_checked(store.snapshot_id()?, layer, head) {
///     Some(s) => apply(s),
///     None => refit_or_abstain(),   // never a default — see the module docs
/// }
/// ```
#[cfg(feature = "calibration_staleness")]
pub use bound::BoundStaticCal;

#[cfg(feature = "calibration_staleness")]
mod bound {
    use super::StaticCalTable;
    use katgpt_core::calibration_staleness::{SnapshotBound, SnapshotId};

    impl StaticCalTable {
        /// Bind this table to the snapshot identity it was calibrated under.
        ///
        /// ⚠ This ASSERTS a fact the type cannot check: that the scales in
        /// `self` were fitted against those weights. Calling it to silence a
        /// refusal re-attaches a stale table under a fresh-looking identity,
        /// which is worse than the original defect — the guard then certifies
        /// it. Same warning as `SnapshotBound::rebind`, at the seam where the
        /// mistake is easiest to make.
        #[inline]
        #[must_use]
        pub fn bound_to(self, fitted: SnapshotId) -> SnapshotBound<Self> {
            SnapshotBound::new(self, fitted)
        }
    }

    /// The guarded read. An extension trait because `SnapshotBound` is
    /// katgpt-core's and `StaticCalTable` is ours — an inherent impl on the
    /// combination is not ours to write.
    pub trait BoundStaticCal {
        /// `Some(scale)` while the binding still describes the live weights,
        /// `None` once the snapshot identity has moved.
        ///
        /// ⛔ `None` is the whole point and must not be turned into `1.0` by
        /// the caller. A neutral scale is right for a table that was never
        /// fitted and wrong for one that was: the cold start has no claim to
        /// be wrong about, a stale head does, and silently substituting the
        /// identity hides a refit that somebody owes.
        fn get_scale_checked(&self, current: SnapshotId, layer: usize, head: usize)
            -> Option<f32>;
    }

    impl BoundStaticCal for SnapshotBound<StaticCalTable> {
        #[inline]
        fn get_scale_checked(
            &self,
            current: SnapshotId,
            layer: usize,
            head: usize,
        ) -> Option<f32> {
            self.get(current).map(|t| t.get_scale(layer, head))
        }
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

    /// Issue 841 §B-3 seam 2 — the guarded read over a shipped `f32` return.
    /// Every arm asserts the INDEPENDENCE of two identities that are easy to
    /// confuse: `commitment` says whether these SCALES were corrupted, the
    /// binding says whether they still describe the LIVE WEIGHTS.
    #[cfg(feature = "calibration_staleness")]
    mod bound_static_cal {
        use super::*;
        use crate::static_cal::BoundStaticCal;
        use katgpt_core::calibration_staleness::{SnapshotId, Staleness};

        fn table() -> StaticCalTable {
            let mut t = StaticCalTable::new(2, 4);
            t.set_scale(1, 2, 0.75);
            t.commit();
            t
        }

        fn id(version: u64, byte: u8) -> SnapshotId {
            SnapshotId::new(version, [byte; 32])
        }

        /// Fresh: the guarded read is BIT-IDENTICAL to the shipped one, so
        /// opting in changes no number.
        #[test]
        fn fresh_reads_are_bit_identical_to_the_shipped_accessor() {
            let raw = table().get_scale(1, 2);
            let fitted = id(3, 0xAB);
            let bound = table().bound_to(fitted);
            assert_eq!(bound.get_scale_checked(fitted, 1, 2), Some(raw));
            assert_eq!(bound.staleness(fitted), Staleness::Fresh);
        }

        /// ⛔ THE POINT, and the distinction the whole seam exists for: the
        /// table is INTACT — `verify()` passes and `commitment` has not moved
        /// — and it is nonetheless describing weights that are no longer
        /// there. Integrity and freshness are different questions and the
        /// shipped `get_scale` answers neither.
        #[test]
        fn an_intact_table_still_refuses_once_the_weights_moved() {
            let fitted = id(3, 0xAB);
            let t = table();
            let commitment_before = t.commitment;
            let bound = t.bound_to(fitted);

            let moved = id(4, 0xAB); // same weights bytes, next generation
            assert_eq!(bound.staleness(moved), Staleness::VersionMoved);
            assert_eq!(bound.get_scale_checked(moved, 1, 2), None);

            let still = bound.peek_unchecked();
            assert!(still.verify(), "the table itself was never corrupted");
            assert_eq!(still.commitment, commitment_before, "its own identity is unmoved");
        }

        /// The mirror: RECALIBRATING moves the table's own commitment and
        /// leaves the binding alone. Two axes, neither derivable from the
        /// other — which is why a `commitment`-only check could never have
        /// implemented this rule.
        #[test]
        fn recalibration_moves_the_tables_commitment_and_not_the_binding() {
            let fitted = id(3, 0xAB);
            let mut t = table();
            let before = t.commitment;
            t.set_scale(0, 0, 1.25);
            t.commit();
            assert_ne!(t.commitment, before, "recalibration must move the table hash");

            let bound = t.bound_to(fitted);
            assert_eq!(bound.staleness(fitted), Staleness::Fresh);
            assert_eq!(bound.get_scale_checked(fitted, 0, 0), Some(1.25));
        }

        /// A swap that changed the weights without bumping the generation is
        /// its own verdict — the repair is a broken bump site, not a refit.
        #[test]
        fn a_weight_change_under_a_held_generation_is_commitment_moved() {
            let fitted = id(3, 0xAB);
            let bound = table().bound_to(fitted);
            let sneaky = id(3, 0xCD);
            assert_eq!(bound.staleness(sneaky), Staleness::CommitmentMoved);
            assert!(bound.get_scale_checked(sneaky, 1, 2).is_none());
        }

        /// The unwired caller — no generation counter anywhere — gets refusal,
        /// not a plausible scale. `StaticCalTable::new` fills every entry with
        /// the neutral 1.0, which is exactly the number that would look fine.
        #[test]
        fn an_unversioned_binding_refuses_rather_than_returning_the_neutral_scale() {
            let bound = StaticCalTable::new(2, 4).bound_to(SnapshotId::UNVERSIONED);
            assert_eq!(
                bound.peek_unchecked().get_scale(0, 0),
                1.0,
                "the neutral scale is what a refusal would otherwise look like"
            );
            assert_eq!(bound.staleness(SnapshotId::UNVERSIONED), Staleness::Unversioned);
            assert_eq!(bound.get_scale_checked(SnapshotId::UNVERSIONED, 0, 0), None);
        }
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
