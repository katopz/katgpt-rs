//! Lemma-1 incremental decode entmax-1.5
//! (Issue 747 P3, Research 549 — arXiv:2506.16640 Lemma 3.1 / App. D.1).
//!
//! The paper's non-vanishing-attention lemma, turned into a decode-path
//! cost primitive: adding a candidate whose score is at or below the current
//! entmax threshold leaves every existing probability **exactly** unchanged
//! — so a decode step that appends one candidate costs O(1) (one compare +
//! one zero write) instead of the O(n log n) full re-sort of
//! [`super::entmax::entmax_1p5`], recomputing only on **support-entry
//! events** (new score above τ).
//!
//! # Why dropping below-τ scores is exact (not approximate)
//!
//! The threshold of our Peters-style scan is **monotone non-decreasing
//! under additions**: a below-τ entrant fails the sorted scan (weighted-
//! average threshold `t_{k+1} = (k·t_k + s)/(k+1)` sits between `t_k` and
//! `s ≤ t_k`, so the entrant never passes) and leaves `(τ, support)` and
//! every probability bit-identical; a high entrant can only pull τ up.
//! Once a score has fallen at-or-below τ it can therefore NEVER re-enter
//! the support — the structure stores only the current support candidates,
//! and a rescan over just those candidates reproduces the full re-sort's
//! `(τ, support)` and probabilities **bit-for-bit** (same sorted prefix
//! values in the same order ⇒ same `t_k` arithmetic; dropped entries sit
//! strictly after every candidate in sort order and provably fail).
//!
//! Note the rescan can SHRINK the support (the `[5, 5]` + `6` case: τ rises
//! above the old plateau and former support members drop to exactly 0.0) —
//! handled by rebuilding `probs` from the candidate list, not by patching.
//!
//! Bit-exactness vs [`super::entmax::entmax_1p5`] on the same history is
//! the G1 gate (`tests/asentmax_p3_incremental_g1.rs`), including
//! threshold-brushing adversarial streams.
//!
//! # Cost shape
//!
//! - below-τ push: O(1), zero allocation at capacity (G4-gated).
//! - support-entry push: O(candidates) rescan + O(len) probs rebuild —
//!   amortized over realistic decode streams (events are rare once a
//!   stable support forms) this is the O(candidates)-per-step decode the
//!   issue's G2 axis measures, vs O(n log n) per step for the naive
//!   full resort.
//!
//! Scores are expected finite (routing logits); `push` debug-asserts it —
//! the full re-sort variant's NaN placement is `total_cmp`-dependent and
//! not a consumed behavior.
//!
//! Feature gate: `asentmax_schedule` (the Issue 747 family flag). Opt-in.

/// Incremental entmax-1.5 over a grow-by-one candidate stream.
///
/// Maintains the exact normalized probability vector of
/// [`super::entmax::entmax_1p5`] applied to the full push history (see
/// module docs for the bit-exactness argument).
pub struct IncrementalEntmax1p5 {
    /// Scores strictly above the current τ, sorted descending (stable index
    /// order among ties — parity with `entmax_1p5_into`'s stable sort).
    /// Exactly the support candidates; at-or-below-τ pushes are dropped.
    candidates: Vec<(usize, f32)>,
    /// Current threshold in the `entmax_1p5_into` convention (support ⟺
    /// `score > tau`). `−∞` before the first push — every finite first
    /// score is a support entry (`entmax_1p5`'s empty-row convention is
    /// `0.0`; parity holds from the first push on).
    tau: f32,
    support_size: usize,
    /// Normalized probabilities, indexed by push order (the row
    /// `entmax_1p5` would return for the full history).
    probs: Vec<f32>,
}

impl IncrementalEntmax1p5 {
    /// New state pre-sized for `capacity` candidates. `probs` allocates
    /// once up front; `candidates` starts small (the support is typically
    /// ≪ n — that is the point) and grows only on support growth.
    pub fn new(capacity: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(64),
            tau: f32::NEG_INFINITY,
            support_size: 0,
            probs: Vec::with_capacity(capacity),
        }
    }

    /// Append the next candidate score (push order = index order).
    ///
    /// Returns `true` on a **support-entry event** (score above τ; the
    /// distribution was recomputed), `false` when Lemma 1 applied (score
    /// at-or-below τ; existing probabilities are untouched bit-for-bit and
    /// the entrant is exactly `0.0`).
    pub fn push(&mut self, score: f32) -> bool {
        debug_assert!(score.is_finite(), "incremental entmax expects finite scores");
        if score <= self.tau {
            // Lemma 1 fast path: x + 0.0 = x keeps the normalization sum
            // bit-identical, so the old normalized probs stay exact.
            self.probs.push(0.0);
            return false;
        }
        let idx = self.probs.len();
        // Insert AFTER equal scores — stable-sort index order among ties.
        let pos = self.candidates.partition_point(|&(_, s)| s >= score);
        self.candidates.insert(pos, (idx, score));
        self.rescan(idx + 1);
        true
    }

    /// Re-derive `(τ, support)` from the candidate list and rebuild `probs`
    /// — the same arithmetic sequence as `entmax_1p5_into` over the full
    /// history (identical sorted prefix ⇒ identical `t_k`; dropped scores
    /// sort after every candidate and provably fail the scan), so the
    /// result is bit-identical to the full re-sort.
    fn rescan(&mut self, len: usize) {
        let mut cumsum = 0.0f32;
        let mut tau = 0.0f32;
        let mut support = 0usize;
        for (k, &(_, score)) in self.candidates.iter().enumerate() {
            cumsum += score;
            let t = (cumsum - 1.0) / ((k + 1) as f32);
            if score > t {
                tau = t;
                support = k + 1;
            }
        }
        self.tau = tau;
        self.support_size = support;

        // Rebuild in the entmax_1p5_into shape: zero-fill → support writes
        // (hoisted half-τ, same operands) → index-ordered sum → reciprocal
        // normalize. Same values in the same order ⇒ same bits.
        self.probs.resize(len, 0.0);
        self.probs[..len].fill(0.0);
        let half_tau = 0.5 * tau;
        for &(orig_idx, score) in self.candidates.iter().take(support) {
            let v = 0.5 * score - half_tau;
            self.probs[orig_idx] = v * v;
        }
        let sum: f32 = self.probs[..len].iter().sum();
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for p in self.probs[..len].iter_mut() {
                *p *= inv_sum;
            }
        }
    }

    /// Current normalized probabilities (index = push order). The empty
    /// slice before the first push.
    #[inline]
    pub fn probs(&self) -> &[f32] {
        &self.probs
    }

    /// Current threshold (`entmax_1p5_into` convention: support ⟺
    /// `score > tau`). `−∞` before the first push.
    #[inline]
    pub fn tau(&self) -> f32 {
        self.tau
    }

    /// Current support size (number of strictly-positive probabilities).
    #[inline]
    pub fn support_size(&self) -> usize {
        self.support_size
    }

    /// Number of candidates pushed so far.
    #[inline]
    pub fn len(&self) -> usize {
        self.probs.len()
    }

    /// True when nothing has been pushed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.probs.is_empty()
    }
}

impl Default for IncrementalEntmax1p5 {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::entmax::entmax_1p5;
    use super::*;

    fn assert_parity(history: &[f32], inc: &IncrementalEntmax1p5) {
        let (probs, tau) = entmax_1p5(history);
        assert_eq!(inc.len(), history.len());
        assert_eq!(
            inc.tau().to_bits(),
            tau.to_bits(),
            "tau bits diverge at n={}",
            history.len()
        );
        for (i, (&a, &b)) in inc.probs().iter().zip(probs.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "probs[{i}] bits diverge");
        }
    }

    #[test]
    fn empty_state() {
        let inc = IncrementalEntmax1p5::default();
        assert!(inc.is_empty());
        assert_eq!(inc.support_size(), 0);
        assert!(inc.probs().is_empty());
    }

    #[test]
    fn first_push_is_always_an_event() {
        let mut inc = IncrementalEntmax1p5::new(8);
        assert!(inc.push(3.0));
        assert_eq!(inc.support_size(), 1);
        assert!((inc.probs()[0] - 1.0).abs() < 1e-6);
        assert_parity(&[3.0], &inc);
    }

    #[test]
    fn below_threshold_push_is_noop() {
        let mut inc = IncrementalEntmax1p5::new(8);
        assert!(inc.push(3.0));
        let before = inc.probs().to_vec();
        let tau = inc.tau();
        assert!(!inc.push(tau)); // exactly at τ
        assert!(!inc.push(tau - 1.0)); // below τ
        assert!(!inc.push(-1e30)); // hugely negative (finite) — same fast path
        assert_eq!(&inc.probs()[..before.len()], before.as_slice());
        assert_eq!(inc.probs()[1], 0.0);
        assert_eq!(inc.probs()[2], 0.0);
        assert_eq!(inc.probs()[3], 0.0);
        assert_parity(&[3.0, tau, tau - 1.0, -1e30], &inc);
    }

    #[test]
    fn spike_can_shrink_support() {
        // The [5,5]+6 case: the spike raises τ above the old plateau and
        // the two 5s drop to EXACTLY zero.
        let mut inc = IncrementalEntmax1p5::new(8);
        assert!(inc.push(5.0));
        assert!(inc.push(5.0));
        assert_eq!(inc.support_size(), 2);
        assert!(inc.push(6.0));
        assert_eq!(inc.support_size(), 1);
        assert_eq!(inc.probs()[0], 0.0);
        assert_eq!(inc.probs()[1], 0.0);
        assert!((inc.probs()[2] - 1.0).abs() < 1e-6);
        assert_parity(&[5.0, 5.0, 6.0], &inc);
    }

    #[test]
    fn all_equal_stream_every_push_events_but_stays_exact() {
        // Worst case for amortization: every equal-score push enters the
        // support. Parity must still hold bit-for-bit.
        let mut inc = IncrementalEntmax1p5::new(512);
        let history: Vec<f32> = vec![2.0; 512];
        for &s in &history {
            inc.push(s);
        }
        assert_eq!(inc.support_size(), 512);
        assert_parity(&history, &inc);
    }

    #[test]
    fn random_stream_parity_at_checkpoints() {
        // splitmix64 → uniform in [-4, 8): mixed below/above-τ pushes.
        let mut state = 0xC0FFEEu64;
        let mut inc = IncrementalEntmax1p5::new(1024);
        let mut history = Vec::with_capacity(1024);
        let mut next_checkpoint = 64;
        while history.len() < 1024 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let s = ((state >> 40) as f32 / ((1u64 << 24) as f32)) * 12.0 - 4.0;
            history.push(s);
            inc.push(s);
            if history.len() == next_checkpoint {
                assert_parity(&history, &inc);
                next_checkpoint *= 2;
            }
        }
        assert_parity(&history, &inc);
    }

    #[test]
    fn probs_sum_to_one_with_exact_zeros() {
        let mut state = 0xABCDEFu64;
        let mut inc = IncrementalEntmax1p5::new(256);
        for _ in 0..256 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let s = ((state >> 40) as f32 / ((1u64 << 24) as f32)) * 20.0 - 10.0;
            inc.push(s);
        }
        let sum: f32 = inc.probs().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum {sum}");
        let zeros = inc
            .probs()
            .iter()
            .filter(|&&p| p == 0.0)
            .count();
        assert_eq!(zeros + inc.support_size(), inc.len());
        for (i, &p) in inc.probs().iter().enumerate() {
            assert!(p >= 0.0, "negative prob at {i}");
        }
    }
}
