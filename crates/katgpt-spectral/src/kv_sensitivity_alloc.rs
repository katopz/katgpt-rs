//! Behavioral-damage-weighted bit allocation — Issue 960 T2 (Research 633:
//! arXiv:2606.09864). A weight INPUT to the shipped water-fill allocator
//! ([`crate::spectral::waterfill_bits`], [`crate::spectral::BitAllocator`]):
//! where SpectralQuant weights the water-fill by structural rank
//! (eigenvalues — top-`d_eff` semantics-first), this module weights it by
//! the T0a probe's BEHAVIORAL damage ranking
//! ([`crate::kv_sensitivity::probe_key_channels`]), so the bit budget lands
//! on the channels whose quantization actually moves the attention pattern.
//!
//! The error model that makes damage a water-fill weight: a uniform min–max
//! scalar quantizer at `b` bits has error scale ∝ `4^−b` (each extra bit
//! quarters the grid step), so a channel's damage reads
//! `damage_c(b) ≈ D_c / 4^b`. `D_c = damage_c(b₀) · 4^{b₀}` — the probe
//! damage extrapolated to the bit-0 scale — predicts the damage at ANY bit
//! width, and the greedy water-fill minimizing `Σ_c D_c / 4^{b_c}` under a
//! total-bit budget is exactly the shipped allocator with `D_c` in the
//! eigenvalue slot. No new search, no new quantizer: one weight vector.
//! The absolute scale of the weights is irrelevant (the greedy compares
//! marginal gains, invariant under a common positive factor) — only the
//! RANKING carries information.
//!
//! T1's caveat binds here (Bench 954): on random weights the behavioral
//! ranking co-locates with channel range (9/12 behavior-in-outliers — the
//! negative control), so a T2 win validates the PROCEDURE (allocator +
//! instrument), not the model-level law; the trained-model read is T3. The
//! T0b dissociation (K behavioral damage vs structural weighting ρ = −0.038,
//! straddling zero) is the precondition that makes this weight input
//! informative: structural and behavioral are different orderings on this
//! model, so the two arms are a real comparison.
//!
//! GOAT protocol (issue 960): three arms at EQUAL average bits — (1)
//! behavioral damage weighting (this module), (2) the shipped structural
//! weighting projected per raw channel (`Σ_i bits_i·R[i][c]²`, the T0b
//! projection), (3) uniform. Retention read = the joint key read
//! ([`crate::kv_sensitivity::joint_key_read`]) at the allocated widths.
//! Calibration is one-shot (~20 prompts) like the probe — exempt from the
//! steady-state standard; the decode-path surface is the
//! [`BitTable`] lookup, which is O(1) and allocation-free.

use crate::spectral::waterfill_bits;

/// The behavioral water-fill weight vector: `D_c = damage_c · 4^{probe_bits}`
/// (f64 for the allocator's gain arithmetic). Order: channel-ascending —
/// i.e. pass the probe's `damage` field REORDERED by `channel` (the probe
/// returns damage-desc).
///
/// # Panics
/// Panics when `damage` is empty or `probe_bits > 31` (the `4^bits` scale
/// must stay exact in f64; the probe clamps to `[1, 24]` anyway).
pub fn behavioral_waterfill_weights(damage: &[f32], probe_bits: u32) -> Vec<f64> {
    assert!(
        !damage.is_empty(),
        "behavioral_waterfill_weights: damage must be non-empty"
    );
    assert!(
        probe_bits <= 31,
        "behavioral_waterfill_weights: probe_bits {probe_bits} exceeds the exact-f64 4^k scale"
    );
    let scale = 4f64.powi(probe_bits as i32);
    damage.iter().map(|&d| d as f64 * scale).collect()
}

/// A per-channel bit-width table — the allocation's decode-path surface.
///
/// One `u8` per channel, produced once per allocation decision
/// (calibration-class) and read per token afterwards: [`BitTable::bits`]
/// is an O(1) index read with no allocation (G4). `channel` is the column
/// index in the `[seq × d]` row-major layout the probe and the joint read
/// use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitTable {
    bits: Vec<u8>,
}

impl BitTable {
    /// The allocated width for `channel`.
    ///
    /// # Panics
    /// Panics when `channel >= d` — a caller bug (the layout's channel
    /// count is fixed at allocation time).
    #[inline]
    pub fn bits(&self, channel: usize) -> u8 {
        self.bits[channel]
    }

    /// The whole table (channel-ascending), for callers that sweep channels.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.bits
    }

    /// Number of channels.
    #[inline]
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// True when there are no channels (degenerate probe shape).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Total bits `Σ_c b_c` — the equal-bits law the GOAT comparison
    /// stands on: at the same `avg_bits`, every arm's total is identical.
    pub fn total_bits(&self) -> usize {
        self.bits.iter().map(|&b| b as usize).sum()
    }
}

/// Allocate per-channel bit widths from a weight vector at a fixed average
/// budget, via the SHIPPED water-fill ([`waterfill_bits`]): every channel
/// starts at `min_bits`, then the budget is spent one bit at a time on the
/// highest marginal gain `w_c / 4^{b_c}` (ties: lowest channel index —
/// deterministic; uniform weights therefore round-robin channel-ascending).
///
/// Equal-bits law: `total = round(avg_bits · d)`, and the returned table's
/// [`BitTable::total_bits`] equals `total` exactly whenever
/// `min_bits ≤ avg_bits ≤ max_bits` (asserted) — the GOAT premise that the
/// three arms spend the SAME budget.
///
/// # Panics
/// Panics when `weights` is empty, when any weight is negative or NaN
/// (a ranking cannot be signed), when ALL weights are zero (nothing to
/// rank — an under-read probe; fail loud rather than silently under-spend),
/// when `avg_bits < min_bits` (budget below the floor) or
/// `avg_bits > max_bits` (budget above the cap — the water-fill would
/// terminate short and break the equal-bits law), or when
/// `max_bits > 24` (the quantizer's exact-grid ceiling).
pub fn allocate_bits(weights: &[f64], avg_bits: f32, min_bits: u8, max_bits: u8) -> BitTable {
    assert!(
        !weights.is_empty(),
        "allocate_bits: weights must be non-empty"
    );
    assert!(
        weights.iter().all(|w| w.is_finite() && *w >= 0.0),
        "allocate_bits: weights must be finite and non-negative"
    );
    assert!(
        weights.iter().any(|&w| w > 0.0),
        "allocate_bits: all-zero weights under-read the probe — nothing to rank"
    );
    assert!(
        max_bits <= 24,
        "allocate_bits: max_bits {max_bits} exceeds the quantizer's 24-bit exact-grid ceiling"
    );
    assert!(
        avg_bits >= min_bits as f32,
        "allocate_bits: avg_bits {avg_bits} below the min_bits floor {min_bits}"
    );
    assert!(
        avg_bits <= max_bits as f32,
        "allocate_bits: avg_bits {avg_bits} above the max_bits cap {max_bits}"
    );
    let d = weights.len();
    let total_bits = (avg_bits * d as f32).round() as usize;
    let bits = waterfill_bits(weights, total_bits, min_bits, Some(max_bits));
    assert_eq!(
        bits.iter().map(|&b| b as usize).sum::<usize>(),
        total_bits,
        "allocate_bits: water-fill under-spent the budget — equal-bits law violated"
    );
    BitTable { bits }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_are_damage_scaled_by_bit0_extrapolation() {
        let w = behavioral_waterfill_weights(&[0.5, 0.25, 0.0], 2);
        // D_c = damage · 4^2 = damage · 16
        assert_eq!(w, vec![8.0, 4.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "nothing to rank")]
    fn all_zero_weights_refuse() {
        let _ = allocate_bits(&[0.0, 0.0, 0.0, 0.0], 2.0, 1, 8);
    }

    #[test]
    #[should_panic(expected = "below the min_bits floor")]
    fn avg_below_floor_refuses() {
        let _ = allocate_bits(&[1.0, 2.0], 0.5, 1, 8);
    }

    #[test]
    #[should_panic(expected = "above the max_bits cap")]
    fn avg_above_cap_refuses() {
        let _ = allocate_bits(&[1.0, 2.0], 9.0, 1, 8);
    }

    #[test]
    fn uniform_weights_round_robin_lowest_index_first() {
        // d=4, avg 1.5 → total 6, base 4 → +2 extras, round-robin ascending.
        let t = allocate_bits(&[1.0, 1.0, 1.0, 1.0], 1.5, 1, 8);
        assert_eq!(t.as_slice(), &[2, 2, 1, 1]);
        assert_eq!(t.total_bits(), 6);
    }

    #[test]
    fn uniform_integer_average_is_flat() {
        let t = allocate_bits(&[1.0, 1.0, 1.0, 1.0], 2.0, 1, 8);
        assert_eq!(t.as_slice(), &[2, 2, 2, 2]);
    }

    #[test]
    fn heavier_weight_never_receives_fewer_bits() {
        // No per-dim cap binds here (max 8), so the greedy is monotone in
        // the weight ranking.
        let t = allocate_bits(&[16.0, 4.0, 1.0, 0.0], 2.0, 1, 8);
        let b = t.as_slice();
        assert!(b[0] >= b[1] && b[1] >= b[2] && b[2] >= b[3]);
        assert_eq!(t.total_bits(), 8);
        // The zero-weight channel stays at the floor.
        assert_eq!(b[3], 1);
    }

    #[test]
    fn equal_bits_law_holds_across_weightings() {
        // The GOAT premise: total is set by the budget, never the weights.
        let w_behavioral = behavioral_waterfill_weights(&[0.9, 0.1, 0.5, 0.3], 2);
        let w_structural = vec![0.31, 0.29, 0.30, 0.28];
        let w_uniform = vec![1.0, 1.0, 1.0, 1.0];
        for avg in [1.25f32, 1.5, 2.0, 3.0] {
            let tb = allocate_bits(&w_behavioral, avg, 1, 8).total_bits();
            let ts = allocate_bits(&w_structural, avg, 1, 8).total_bits();
            let tu = allocate_bits(&w_uniform, avg, 1, 8).total_bits();
            assert_eq!(tb, ts, "avg {avg}");
            assert_eq!(ts, tu, "avg {avg}");
        }
    }

    #[test]
    fn bit_table_accessor_reads_back() {
        let t = allocate_bits(&[4.0, 1.0], 2.0, 1, 8);
        assert_eq!(t.len(), 2);
        assert!(!t.is_empty());
        assert!(t.bits(0) >= t.bits(1));
    }

    #[test]
    #[should_panic(expected = "weights must be finite")]
    fn negative_weight_refuses() {
        let _ = allocate_bits(&[1.0, -2.0], 2.0, 1, 8);
    }

    #[test]
    fn weights_scale_invariance() {
        // Only the ranking carries information: a common positive factor
        // changes no width.
        let a = allocate_bits(&[8.0, 2.0, 1.0], 2.0, 1, 8);
        let b = allocate_bits(&[800.0, 200.0, 100.0], 2.0, 1, 8);
        assert_eq!(a.as_slice(), b.as_slice());
    }
}
