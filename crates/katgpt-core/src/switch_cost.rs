//! Issue 663 — `SwitchCostTable`: directed pairwise switch-difficulty table.
//!
//! Distillation of Skill Entropy (Research 484 / arXiv:2608.05139, He et al.
//! "Toward Skill-Native LLMs"). The paper's measure is pure counter math —
//! no training — so it ships here as a modelless primitive:
//!
//! ```text
//! SkE(a, b) = (½·Acc(a) + ½·Acc(b) + α) / (Acc(a, b) + α)    α = 0.1
//! ```
//!
//! where `Acc(a)` is solo success rate of mode `a`, `Acc(a, b)` the success
//! rate when `b` is attempted immediately after `a`. SkE ≈ 1 → chaining adds
//! no difficulty; SkE ≫ 1 → hard switch. The measure is **directed**
//! (`SkE(a,b) ≠ SkE(b,a)` — the paper's Planning→Info-Extraction is ~4.4
//! while the reverse is cheap; our border-piling bug is flee→reposition hard,
//! reposition→flee easy).
//!
//! # Cold-start semantics
//!
//! At zero trials an accuracy is *undefined*, not zero. This module uses the
//! neutral prior `0.5` for any accuracy with zero trials, which makes a cold
//! table evaluate to exactly `1.0` everywhere (numerator `0.6` == denominator
//! `0.6` at α = 0.1): no measured evidence ⇒ no assumed difficulty. Once
//! counters fill, α keeps every ratio finite. Consumers that need a warm-up
//! floor before arming triggers (Research 484 §6.1) use [`SwitchCostTable::ske_if_armed`].
//!
//! # Domain classification (latent, local, never synced)
//!
//! The table is per-NPC / per-archetype derived state: updated from runtime
//! telemetry, freeze/thaw-able as a [`SwitchCostSnapshot`] (a `#[repr(transparent)]`
//! POD newtype — bitwise-committable, BLAKE3-able). Only raw scalars/events
//! cross any boundary. No sync dependency, no replay coupling.
//!
//! # Factorization (Eq. 7)
//!
//! For large mode sets the O(N²) pair grid is impractical to measure. The
//! paper factorizes through mode *families*: `SkE(a,b) ≈ SkE(a, fam_b) ·
//! SkE(fam_a, b)` — a leave-cost × land-cost product. [`FactorizedSwitchCost`]
//! stores O(N·F) counters and routes real switch telemetry into family cells.
//!
//! Feature: `switch_cost` (opt-in). Promotion requires a riir-ai consumer A/B
//! (F1: SkE-gated preemptive re-estimation vs the coherence-only arm).
//!
//! SwitchCostTable — directed pairwise switch-difficulty table (skill-entropy
//! distillation, Research 484 / arXiv:2608.05139, Issue 663). Opt-in per the
//! issue's GOAT-gate discipline: promotion to default requires a riir-ai
//! consumer A/B (F1: SkE-gated preemptive re-estimation vs the coherence-only
//! arm on the Issue-054 stuck-rate scenario).

use bytemuck::{Pod, Zeroable};

/// Paper-default Laplace smoothing (§3.1; α = 0.1).
pub const DEFAULT_ALPHA: f32 = 0.1;

/// Neutral accuracy returned for a zero-trial measurement.
///
/// With this prior a cold table is exactly 1.0 at every pair (0.6/0.6 at
/// α = 0.1) — evidence-free switches are cost-free, not free wins/losses.
pub const NEUTRAL_ACC: f32 = 0.5;

/// Directed pairwise switch-difficulty table over a bounded mode set.
///
/// Modes are indices into a consumer's bounded enum (behavior-FSM states,
/// quest objective kinds, LEO goal kinds, cognition runtime selections).
/// All state is fixed-size `[u32]` counters plus the α constant —
/// `#[repr(C)]`, `Copy`, and [`Pod`], so snapshots are bitwise-stable and
/// committable.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SwitchCostTable<const N: usize> {
    solo_success: [u32; N],
    solo_trials: [u32; N],
    /// `[a][b]`: `b` attempted immediately after `a`.
    pair_success: [[u32; N]; N],
    pair_trials: [[u32; N]; N],
    alpha: f32,
}

// Safety: `#[repr(C)]` over `[u32; N]` / `[[u32; N]; N]` / `f32` only — every
// field is 4-byte sized and aligned, so the struct has no interior padding
// for any `N` and any bit pattern is a valid instance.
unsafe impl<const N: usize> Pod for SwitchCostTable<N> {}
unsafe impl<const N: usize> Zeroable for SwitchCostTable<N> {}

impl<const N: usize> Default for SwitchCostTable<N> {
    fn default() -> Self {
        Self::new(DEFAULT_ALPHA)
    }
}

impl<const N: usize> SwitchCostTable<N> {
    /// Empty table with the given Laplace smoothing (`alpha > 0`).
    #[inline]
    pub fn new(alpha: f32) -> Self {
        debug_assert!(alpha > 0.0, "alpha=0 makes zero-trial pairs 0/0 NaN");
        Self {
            solo_success: [0; N],
            solo_trials: [0; N],
            pair_success: [[0; N]; N],
            pair_trials: [[0; N]; N],
            alpha,
        }
    }

    /// Record one solo trial of `mode`.
    #[inline]
    pub fn record_solo(&mut self, mode: usize, success: bool) {
        debug_assert!(mode < N);
        self.solo_trials[mode] = self.solo_trials[mode].saturating_add(1);
        if success {
            self.solo_success[mode] = self.solo_success[mode].saturating_add(1);
        }
    }

    /// Record one pair trial: `b` attempted immediately after `a`.
    #[inline]
    pub fn record_switch(&mut self, a: usize, b: usize, success: bool) {
        debug_assert!(a < N && b < N);
        self.pair_trials[a][b] = self.pair_trials[a][b].saturating_add(1);
        if success {
            self.pair_success[a][b] = self.pair_success[a][b].saturating_add(1);
        }
    }

    /// Bitwise read-only snapshot (freeze/thaw artifact).
    ///
    /// The snapshot type carries no `record_*` methods — a thawed view cannot
    /// mutate telemetry. It is `#[repr(transparent)]` over the table, so the
    /// two share layout and any POD commitment covers both.
    #[inline]
    pub fn snapshot(&self) -> SwitchCostSnapshot<N> {
        SwitchCostSnapshot(*self)
    }

    /// Laplace smoothing in use.
    #[inline]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Solo trials recorded for `mode`.
    #[inline]
    pub fn solo_trials(&self, mode: usize) -> u32 {
        debug_assert!(mode < N);
        self.solo_trials[mode]
    }

    /// Pair trials recorded for the ordered switch `a → b`.
    #[inline]
    pub fn pair_trials(&self, a: usize, b: usize) -> u32 {
        debug_assert!(a < N && b < N);
        self.pair_trials[a][b]
    }

    /// Solo success rate of `mode` ([`NEUTRAL_ACC`] at zero trials).
    #[inline]
    pub fn solo_accuracy(&self, mode: usize) -> f32 {
        debug_assert!(mode < N);
        accuracy(self.solo_success[mode], self.solo_trials[mode])
    }

    /// Pair success rate of the ordered switch `a → b` ([`NEUTRAL_ACC`] at
    /// zero trials).
    #[inline]
    pub fn pair_accuracy(&self, a: usize, b: usize) -> f32 {
        debug_assert!(a < N && b < N);
        accuracy(self.pair_success[a][b], self.pair_trials[a][b])
    }

    /// Directed switch difficulty `SkE(a, b)` — hot lookup, zero-alloc.
    ///
    /// ≈1: chaining adds no difficulty; ≫1: hard switch. Directed by
    /// construction (the pair counters are ordered).
    #[inline]
    pub fn ske(&self, a: usize, b: usize) -> f32 {
        debug_assert!(a < N && b < N);
        let num = 0.5 * self.solo_accuracy(a) + 0.5 * self.solo_accuracy(b) + self.alpha;
        let den = self.pair_accuracy(a, b) + self.alpha;
        num / den
    }

    /// [`Self::ske`] gated on a warm-up floor (Research 484 §6.1).
    ///
    /// Returns `None` until the ordered pair has at least `min_pair_trials`
    /// observations — proactive triggers (F1/F2) must not fire on noise.
    #[inline]
    pub fn ske_if_armed(&self, a: usize, b: usize, min_pair_trials: u32) -> Option<f32> {
        if self.pair_trials(a, b) >= min_pair_trials {
            Some(self.ske(a, b))
        } else {
            None
        }
    }

    /// Mean directed SkE along a mode sequence (paper Eq. 4).
    ///
    /// Averages SkE over **consecutive** pairs `seq[i] → seq[i+1]` — the pair
    /// counters measure immediate adjacency. Sequences shorter than 2 modes
    /// contain no switches and return the neutral cost `1.0`.
    #[inline]
    pub fn sequence_entropy(&self, seq: &[usize]) -> f32 {
        mean_pairwise(|a, b| self.ske(a, b), seq)
    }
}

/// Read-only, freeze/thaw-friendly view of a [`SwitchCostTable`].
///
/// `#[repr(transparent)]` over the table — same layout, same POD
/// commitment — but carries only the read API. A snapshot cannot record.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct SwitchCostSnapshot<const N: usize>(SwitchCostTable<N>);

// Safety: `#[repr(transparent)]` delegates layout to the inner table, which
// is `Pod` for every `N` (manual impl above).
unsafe impl<const N: usize> Pod for SwitchCostSnapshot<N> {}
unsafe impl<const N: usize> Zeroable for SwitchCostSnapshot<N> {}

impl<const N: usize> SwitchCostSnapshot<N> {
    /// Directed switch difficulty `SkE(a, b)`.
    #[inline]
    pub fn ske(&self, a: usize, b: usize) -> f32 {
        self.0.ske(a, b)
    }

    /// [`Self::ske`] gated on a warm-up floor.
    #[inline]
    pub fn ske_if_armed(&self, a: usize, b: usize, min_pair_trials: u32) -> Option<f32> {
        self.0.ske_if_armed(a, b, min_pair_trials)
    }

    /// Mean directed SkE along a mode sequence (paper Eq. 4).
    #[inline]
    pub fn sequence_entropy(&self, seq: &[usize]) -> f32 {
        self.0.sequence_entropy(seq)
    }

    /// Solo trials recorded for `mode`.
    #[inline]
    pub fn solo_trials(&self, mode: usize) -> u32 {
        self.0.solo_trials(mode)
    }

    /// Pair trials recorded for the ordered switch `a → b`.
    #[inline]
    pub fn pair_trials(&self, a: usize, b: usize) -> u32 {
        self.0.pair_trials(a, b)
    }
}

/// Factorized switch-difficulty table (paper Eq. 7).
///
/// `SkE(a, b) ≈ SkE(a, fam_b) · SkE(fam_a, b)` — a leave-cost × land-cost
/// product through mode families. O(N·F) pair counters instead of O(N²);
/// real switch telemetry `record_switch(a, b)` is routed into both the
/// leave cell `(a, family_of[b])` and the land cell `(family_of[a], b)`.
///
/// Not `Pod` (the `family_of` map is `[usize; N]`); the exact
/// [`SwitchCostTable`] is the POD snapshot form for small mode sets.
#[derive(Clone, Copy, Debug)]
pub struct FactorizedSwitchCost<const N: usize, const F: usize> {
    family_of: [usize; N],
    solo_success: [u32; N],
    solo_trials: [u32; N],
    /// Family-aggregated solo counters (maintained on `record_solo`).
    fam_solo_success: [u32; F],
    fam_solo_trials: [u32; F],
    /// `[a][f]`: any member of family `f` attempted after `a` (leave side).
    leave_success: [[u32; F]; N],
    leave_trials: [[u32; F]; N],
    /// `[f][b]`: `b` attempted after any member of family `f` (land side).
    land_success: [[u32; N]; F],
    land_trials: [[u32; N]; F],
    alpha: f32,
}

impl<const N: usize, const F: usize> FactorizedSwitchCost<N, F> {
    /// Empty factorized table. `family_of[mode] < F` for every mode is a
    /// constructor contract (callers build it from a static enum mapping).
    pub fn new(family_of: [usize; N], alpha: f32) -> Self {
        debug_assert!(alpha > 0.0);
        for f in family_of.iter() {
            debug_assert!(*f < F, "family id out of range");
        }
        Self {
            family_of,
            solo_success: [0; N],
            solo_trials: [0; N],
            fam_solo_success: [0; F],
            fam_solo_trials: [0; F],
            leave_success: [[0; F]; N],
            leave_trials: [[0; F]; N],
            land_success: [[0; N]; F],
            land_trials: [[0; N]; F],
            alpha,
        }
    }

    /// Record one solo trial of `mode` (also feeds the family aggregate).
    #[inline]
    pub fn record_solo(&mut self, mode: usize, success: bool) {
        debug_assert!(mode < N);
        self.solo_trials[mode] = self.solo_trials[mode].saturating_add(1);
        let fam = self.family_of[mode];
        self.fam_solo_trials[fam] = self.fam_solo_trials[fam].saturating_add(1);
        if success {
            self.solo_success[mode] = self.solo_success[mode].saturating_add(1);
            self.fam_solo_success[fam] = self.fam_solo_success[fam].saturating_add(1);
        }
    }

    /// Record one pair trial, routed into the leave and land family cells.
    #[inline]
    pub fn record_switch(&mut self, a: usize, b: usize, success: bool) {
        debug_assert!(a < N && b < N);
        let (fa, fb) = (self.family_of[a], self.family_of[b]);
        self.leave_trials[a][fb] = self.leave_trials[a][fb].saturating_add(1);
        self.land_trials[fa][b] = self.land_trials[fa][b].saturating_add(1);
        if success {
            self.leave_success[a][fb] = self.leave_success[a][fb].saturating_add(1);
            self.land_success[fa][b] = self.land_success[fa][b].saturating_add(1);
        }
    }

    /// Laplace smoothing in use.
    #[inline]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Family-aggregated solo success rate of family `f`.
    #[inline]
    pub fn family_accuracy(&self, f: usize) -> f32 {
        debug_assert!(f < F);
        accuracy(self.fam_solo_success[f], self.fam_solo_trials[f])
    }

    /// Factorized estimate `SkE(a, b) ≈ SkE(a, fam_b) · SkE(fam_a, b)`.
    ///
    /// Cold table evaluates to exactly `1.0` (both factors neutral), matching
    /// the exact table's cold semantics.
    #[inline]
    pub fn ske(&self, a: usize, b: usize) -> f32 {
        debug_assert!(a < N && b < N);
        let (fa, fb) = (self.family_of[a], self.family_of[b]);
        let leave = ske_from(
            self.solo_accuracy(a),
            self.family_accuracy(fb),
            accuracy(self.leave_success[a][fb], self.leave_trials[a][fb]),
            self.alpha,
        );
        let land = ske_from(
            self.family_accuracy(fa),
            self.solo_accuracy(b),
            accuracy(self.land_success[fa][b], self.land_trials[fa][b]),
            self.alpha,
        );
        leave * land
    }

    /// Mean factorized SkE along a mode sequence (consecutive pairs).
    #[inline]
    pub fn sequence_entropy(&self, seq: &[usize]) -> f32 {
        mean_pairwise(|a, b| self.ske(a, b), seq)
    }

    #[inline]
    fn solo_accuracy(&self, mode: usize) -> f32 {
        accuracy(self.solo_success[mode], self.solo_trials[mode])
    }
}

/// Empirical-CDF rank of `value` within `sample` (paper §4 reward math).
///
/// Returns the fraction of samples `≤ value` — the scale-free rank the
/// skill-entropy reward (`r_ent = 1 − |ρ̂ − ρ★|`) consumes. An empty sample
/// returns [`NEUTRAL_ACC`]. NaN samples never compare `≤`, so they rank low;
/// callers feeding entropies (finite by construction) are unaffected.
pub fn cdf_rank(value: f32, sample: &[f32]) -> f32 {
    if sample.is_empty() {
        return NEUTRAL_ACC;
    }
    let le = sample.iter().filter(|&&x| x <= value).count();
    le as f32 / sample.len() as f32
}

/// Raw success rate with the neutral prior at zero trials.
#[inline]
fn accuracy(success: u32, trials: u32) -> f32 {
    if trials == 0 {
        NEUTRAL_ACC
    } else {
        success as f32 / trials as f32
    }
}

/// SkE from precomputed accuracies (shared by the exact and factorized paths).
#[inline]
fn ske_from(acc_a: f32, acc_b: f32, acc_pair: f32, alpha: f32) -> f32 {
    (0.5 * acc_a + 0.5 * acc_b + alpha) / (acc_pair + alpha)
}

/// Mean pairwise cost over consecutive sequence elements; 1.0 if no pairs.
#[inline]
fn mean_pairwise<F: Fn(usize, usize) -> f32>(cost: F, seq: &[usize]) -> f32 {
    if seq.len() < 2 {
        return 1.0;
    }
    let mut sum = 0.0f32;
    for w in seq.windows(2) {
        sum += cost(w[0], w[1]);
    }
    sum / (seq.len() - 1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    /// Hand-computed fixture: solos 0.8/0.2/0.5, pair 0→1 at 0.1, pair 1→0 at 0.8.
    fn fixture() -> SwitchCostTable<3> {
        let mut t = SwitchCostTable::new(DEFAULT_ALPHA);
        for i in 0..10 {
            t.record_solo(0, i < 8);
            t.record_solo(1, i < 2);
            t.record_solo(2, i < 5);
        }
        for i in 0..10 {
            t.record_switch(0, 1, i < 1); // Acc(0,1) = 0.1
            t.record_switch(1, 0, i < 8); // Acc(1,0) = 0.8
        }
        t
    }

    #[test]
    fn cold_table_is_exactly_neutral() {
        let t = SwitchCostTable::<4>::default();
        for a in 0..4 {
            for b in 0..4 {
                assert_eq!(t.ske(a, b), 1.0, "cold ske({a},{b})");
            }
        }
        assert_eq!(t.sequence_entropy(&[]), 1.0);
        assert_eq!(t.sequence_entropy(&[2]), 1.0);
        assert_eq!(t.sequence_entropy(&[0, 3]), 1.0);
    }

    #[test]
    fn formula_matches_hand_computation() {
        let t = fixture();
        // SkE(0,1) = (0.5·0.8 + 0.5·0.2 + 0.1)/(0.1 + 0.1) = 0.6/0.2 = 3.0
        assert!((t.ske(0, 1) - 3.0).abs() < TOL, "{}", t.ske(0, 1));
        // SkE(1,0) = (0.5·0.2 + 0.5·0.8 + 0.1)/(0.8 + 0.1) = 0.6/0.9
        assert!((t.ske(1, 0) - 0.6 / 0.9).abs() < TOL, "{}", t.ske(1, 0));
    }

    #[test]
    fn directionality_is_constructible_and_pinned() {
        let t = fixture();
        assert!((t.ske(0, 1) - t.ske(1, 0)).abs() > 1.0);
        assert!(t.ske(0, 1) > t.ske(1, 0));
    }

    #[test]
    fn sequence_entropy_is_mean_over_consecutive_pairs() {
        let t = fixture();
        let want = (t.ske(0, 1) + t.ske(1, 0)) / 2.0;
        assert!((t.sequence_entropy(&[0, 1, 0]) - want).abs() < TOL);
        // Repeated mode: SkE(a,a) participates like any pair.
        let seq = [0, 0, 1];
        let want2 = (t.ske(0, 0) + t.ske(0, 1)) / 2.0;
        assert!((t.sequence_entropy(&seq) - want2).abs() < TOL);
    }

    #[test]
    fn record_order_does_not_change_results_bit_identically() {
        use fastrand::Rng;

        let mut x = SwitchCostTable::<3>::new(DEFAULT_ALPHA);
        let mut y = SwitchCostTable::<3>::new(DEFAULT_ALPHA);
        let mut rng = Rng::with_seed(42);
        let mut events = Vec::with_capacity(500);
        for _ in 0..500 {
            let a = rng.usize(0..3);
            let b = rng.usize(0..3);
            let ok = rng.bool();
            events.push((a, b, ok));
            x.record_switch(a, b, ok);
        }
        // Replay in reverse order into y — u32 counters commute exactly.
        for &(a, b, ok) in events.iter().rev() {
            y.record_switch(a, b, ok);
        }
        for _ in 0..200 {
            let m = rng.usize(0..3);
            let ok = rng.bool();
            x.record_solo(m, ok);
            y.record_solo(m, ok); // same order for solos; pairs already cover order-independence
        }
        let snap_x = x.snapshot();
        let snap_y = y.snapshot();
        for a in 0..3 {
            for b in 0..3 {
                assert_eq!(snap_x.ske(a, b).to_bits(), snap_y.ske(a, b).to_bits());
            }
            assert_eq!(snap_x.solo_trials(a), snap_y.solo_trials(a));
        }
        assert_eq!(
            x.sequence_entropy(&[0, 1, 2, 1, 0]).to_bits(),
            y.sequence_entropy(&[0, 1, 2, 1, 0]).to_bits()
        );
    }

    #[test]
    fn ske_grows_monotonically_as_pair_failures_accumulate() {
        let mut t = SwitchCostTable::<2>::new(DEFAULT_ALPHA);
        for i in 0..10 {
            t.record_solo(0, i < 8);
            t.record_solo(1, i < 8);
        }
        let mut prev = t.ske(0, 1); // cold 1.0
        for k in 0..20 {
            t.record_switch(0, 1, false);
            let cur = t.ske(0, 1);
            assert!(cur >= prev, "step {k}: {cur} < {prev}");
            prev = cur;
        }
        assert!(prev > 1.0);
    }

    #[test]
    fn ske_if_armed_respects_warmup_floor() {
        let mut t = SwitchCostTable::<2>::new(DEFAULT_ALPHA);
        assert!(t.ske_if_armed(0, 1, 1).is_none());
        t.record_switch(0, 1, true);
        assert!(t.ske_if_armed(0, 1, 1).is_some());
        assert!(t.ske_if_armed(0, 1, 2).is_none());
        assert_eq!(t.snapshot().ske_if_armed(0, 1, 1).unwrap(), t.ske(0, 1));
    }

    #[test]
    fn snapshot_is_pod_and_layout_stable() {
        let t = fixture();
        let snap = t.snapshot();
        let bytes = bytemuck::bytes_of(&snap);
        assert_eq!(bytes.len(), std::mem::size_of::<SwitchCostTable<3>>());
        let thawed: SwitchCostSnapshot<3> = *bytemuck::from_bytes(bytes);
        for a in 0..3 {
            for b in 0..3 {
                assert_eq!(thawed.ske(a, b).to_bits(), snap.ske(a, b).to_bits());
            }
        }
        assert_eq!(
            thawed.sequence_entropy(&[0, 1, 2]).to_bits(),
            snap.sequence_entropy(&[0, 1, 2]).to_bits()
        );
    }

    #[test]
    fn factorized_cold_is_neutral_and_routes_correctly() {
        let mut f = FactorizedSwitchCost::<4, 2>::new([0, 0, 1, 1], DEFAULT_ALPHA);
        for a in 0..4 {
            for b in 0..4 {
                assert_eq!(f.ske(a, b), 1.0);
            }
        }
        f.record_switch(0, 2, false); // leave cell (0, fam 1), land cell (fam 0, 2)
        f.record_solo(2, true);
        assert_eq!(f.family_accuracy(1), 1.0);
        assert!(f.ske(0, 2) > 1.0, "one failure on the only pair trial");
    }

    #[test]
    fn factorized_known_value_product() {
        let mut f = FactorizedSwitchCost::<2, 2>::new([0, 1], DEFAULT_ALPHA);
        for i in 0..10 {
            f.record_solo(0, i < 8); // Acc(0)=0.8
            f.record_solo(1, i < 4); // Acc(1)=0.4, fam1 Acc=0.4
        }
        // Both family cells see the same 20 routed switches (15/20 = 0.75):
        //   leave factor = (½·Acc(0) + ½·Acc(fam1) + α)/(0.75 + α) = 0.7/0.85
        //   land  factor = (½·Acc(fam0) + ½·Acc(1) + α)/(0.75 + α) = 0.7/0.85
        // (fam0 accuracy = Acc(0) = 0.8 — mode 0 is fam0's only member.)
        for i in 0..10 {
            f.record_switch(0, 1, i < 5);
            f.record_switch(0, 1, true);
        }
        let factor = (0.5 * 0.8 + 0.5 * 0.4 + DEFAULT_ALPHA) / (0.75 + DEFAULT_ALPHA);
        let want = factor * factor;
        assert!(
            (f.ske(0, 1) - want).abs() < TOL,
            "{} vs {want}",
            f.ske(0, 1)
        );
    }

    #[test]
    fn cdf_rank_known_values() {
        let sample = [0.1, 0.2, 0.3, 0.4];
        assert!((cdf_rank(0.0, &sample) - 0.0).abs() < TOL);
        assert!((cdf_rank(0.25, &sample) - 0.5).abs() < TOL);
        assert!((cdf_rank(0.4, &sample) - 1.0).abs() < TOL);
        assert!((cdf_rank(9.0, &sample) - 1.0).abs() < TOL);
        assert_eq!(cdf_rank(0.3, &[]), NEUTRAL_ACC);
    }
}
