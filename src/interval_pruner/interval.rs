//! Interval-preserving token mask operations (arXiv:2503.13663).
//!
//! An interval-closed mask has contiguous valid regions — no gaps.
//! If tokens i..k are valid, every token j between them must also be valid.
//! This eliminates "Swiss cheese" patterns where scattered tokens are rejected
//! while in-between tokens are valid, reducing branching waste in DDTree.

#[cfg(feature = "interval_pruner")]
use katgpt_core::ConstraintPruner;

// ---------------------------------------------------------------------------
// IntervalMask
// ---------------------------------------------------------------------------

/// Boolean validity mask over a token vocabulary.
///
/// Invariant: `mask.len()` equals the vocabulary size for the current decode step.
/// `mask[i] = true` means token `i` is valid (not pruned).
#[cfg(feature = "interval_pruner")]
#[derive(Clone, Debug)]
pub struct IntervalMask {
    mask: Vec<bool>,
}

#[cfg(feature = "interval_pruner")]
impl IntervalMask {
    /// Construct from a raw boolean vec.
    #[inline]
    pub fn from_vec(bools: Vec<bool>) -> Self {
        Self { mask: bools }
    }

    /// Check that valid regions form contiguous intervals (no gaps).
    ///
    /// An interval-closed mask satisfies: for any valid i < j < k,
    /// if i and k are valid then j is valid. O(n) single-pass scan.
    #[inline]
    pub fn is_interval_closed(&self) -> bool {
        self.gap_count() == 0
    }

    /// Merge nearby valid ranges. If the gap between two valid ranges is
    /// ≤ `gap_threshold`, fill the gap (make them one contiguous range).
    /// Returns a new mask.
    ///
    /// `gap_threshold = 0` means no merging — returns a clone.
    pub fn close_intervals(&self, gap_threshold: usize) -> Self {
        if gap_threshold == 0 { return self.clone() }

        let intervals = self.valid_intervals();
        match intervals.len() {
            0 | 1 => return self.clone(),
            _ => {}
        }

        let mut result = self.mask.clone();
        close_intervals_inplace(&mut result, &intervals, gap_threshold);
        Self { mask: result }
    }

    /// Count "Swiss cheese" gaps: transitions valid→invalid→valid.
    ///
    /// Each gap corresponds to a region of invalid tokens sandwiched between
    /// two valid regions. An interval-closed mask has `gap_count() == 0`.
    #[inline]
    pub fn gap_count(&self) -> usize {
        let mut count = 0usize;
        let mut state = State::LeadingInvalid;

        for &v in &self.mask {
            state = match (state, v) {
                (State::LeadingInvalid, true) => State::InValid,
                (State::LeadingInvalid, false) => State::LeadingInvalid,
                (State::InValid, true) => State::InValid,
                (State::InValid, false) => State::InGap,
                (State::InGap, true) => {
                    count += 1;
                    State::InValid
                }
                (State::InGap, false) => State::InGap,
            };
        }

        count
    }

    /// Return list of `(start, end)` for contiguous valid ranges.
    ///
    /// `end` is exclusive (one past the last valid index).
    pub fn valid_intervals(&self) -> Vec<(usize, usize)> {
        valid_intervals_from_slice(&self.mask)
    }

    #[inline]
    pub fn as_slice(&self) -> &[bool] {
        &self.mask
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// Check that valid regions form contiguous intervals using adaptive routing.
    ///
    /// Selects between scalar and SIMD backends based on `config.interval_simd_threshold`.
    /// For masks smaller than the threshold, scalar is faster (no SIMD setup overhead).
    #[inline]
    pub fn is_interval_closed_adaptive(&self, config: &crate::interval_pruner::AdaptiveConfig) -> bool {
        config.is_interval_closed(&self.mask)
    }

    /// Merge nearby valid ranges using adaptive routing.
    ///
    /// Selects between scalar and SIMD backends based on `config.interval_simd_threshold`.
    /// Returns a new mask with gaps filled.
    pub fn close_intervals_adaptive(&self, gap_threshold: usize, config: &crate::interval_pruner::AdaptiveConfig) -> Self {
        let result = config.close_intervals(&self.mask, gap_threshold);
        Self { mask: result }
    }
}

// ---------------------------------------------------------------------------
// Free helpers (zero-allocation interval ops on borrowed slices)
// ---------------------------------------------------------------------------

/// Return list of `(start, end)` for contiguous valid ranges in `mask`.
///
/// `end` is exclusive. Shared by [`IntervalMask::valid_intervals`] and the
/// in-place batch path so callers don't need to wrap a slice in an owned
/// `IntervalMask` just to compute intervals.
fn valid_intervals_from_slice(mask: &[bool]) -> Vec<(usize, usize)> {
    let n = mask.len();
    if n == 0 {
        return Vec::new();
    }

    let mut intervals = Vec::with_capacity(4);
    let mut i = 0;

    while i < n {
        // skip invalid
        while i < n && !mask[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let start = i;
        // scan valid run
        while i < n && mask[i] {
            i += 1;
        }
        intervals.push((start, i));
    }

    intervals
}

/// Fill gaps ≤ `gap_threshold` between adjacent intervals directly into
/// `mask`, in place. Walks adjacent interval pairs.
#[inline]
fn close_intervals_inplace(mask: &mut [bool], intervals: &[(usize, usize)], gap_threshold: usize) {
    for w in intervals.windows(2) {
        let (_a_start, a_end) = w[0];
        let (b_start, _b_end) = w[1];
        let gap = b_start - a_end; // exclusive end → exclusive start distance
        if gap <= gap_threshold {
            // fill everything from a_end to b_start (exclusive)
            mask[a_end..b_start].fill(true);
        }
    }
}

/// State machine for gap counting.
#[cfg(feature = "interval_pruner")]
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum State {
    LeadingInvalid,
    InValid,
    InGap,
}

// ---------------------------------------------------------------------------
// IntervalPruner<P>
// ---------------------------------------------------------------------------

/// Wraps any [`ConstraintPruner`] and enforces interval closure on batch output.
///
/// After the inner pruner produces a validity mask, `IntervalPruner` merges
/// nearby valid ranges whose gap ≤ `gap_threshold`, then overrides filled-in
/// tokens to valid. This reduces "Swiss cheese" branching waste.
#[cfg(feature = "interval_pruner")]
#[derive(Debug)]
pub struct IntervalPruner<P> {
    inner: P,
    gap_threshold: usize,
}

#[cfg(feature = "interval_pruner")]
impl<P> IntervalPruner<P> {
    /// Create a new interval-enforcing wrapper.
    ///
    /// * `inner` — the underlying pruner.
    /// * `gap_threshold` — max gap size to merge. 0 = no merging, just detection.
    #[inline]
    pub fn new(inner: P, gap_threshold: usize) -> Self {
        Self {
            inner,
            gap_threshold,
        }
    }

    /// Access the inner pruner.
    #[inline]
    pub fn inner(&self) -> &P {
        &self.inner
    }
}

#[cfg(feature = "interval_pruner")]
impl<P: ConstraintPruner> ConstraintPruner for IntervalPruner<P> {
    #[inline]
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        self.inner.is_valid(depth, token_idx, parent_tokens)
    }

    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        // 1. Delegate to inner pruner — fills results in-place.
        self.inner
            .batch_is_valid(depth, candidates, parent_tokens, results);

        // 2. Apply interval closure on the result batch.
        //    Build an IntervalMask, close gaps, write back to results.
        let len = candidates.len().min(results.len());
        if len == 0 {
            return;
        }

        // Build mask only over the batch slice.
        // Compute intervals directly from `results` (no Vec<bool> allocation,
        // no IntervalMask wrapper), then fill gaps in place.
        match self.gap_threshold {
            0 => { /* no merging */ }
            _ => {
                let intervals = valid_intervals_from_slice(&results[..len]);
                match intervals.len() {
                    0 | 1 => {}
                    _ => close_intervals_inplace(&mut results[..len], &intervals, self.gap_threshold),
                }
            }
        }
    }

    fn propagate(&mut self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        self.inner.propagate(depth, token_idx, parent_tokens)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(feature = "interval_pruner")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_is_interval_closed() {
        let mask = IntervalMask::from_vec(vec![false, true, true, true, false]);
        assert!(mask.is_interval_closed());
        assert_eq!(mask.gap_count(), 0);
    }

    #[test]
    fn test_swiss_cheese_not_interval_closed() {
        let mask = IntervalMask::from_vec(vec![true, false, true, false, true]);
        assert!(!mask.is_interval_closed());
        assert_eq!(mask.gap_count(), 2);
    }

    #[test]
    fn test_close_intervals_merges_small_gaps() {
        let mask = IntervalMask::from_vec(vec![true, false, false, true]);
        let closed = mask.close_intervals(2);
        assert_eq!(closed.as_slice(), &[true, true, true, true]);
        assert!(closed.is_interval_closed());
    }

    #[test]
    fn test_close_intervals_preserves_large_gaps() {
        let mask = IntervalMask::from_vec(vec![true, false, false, false, true]);
        let closed = mask.close_intervals(1);
        assert_eq!(closed.as_slice(), &[true, false, false, false, true]);
        assert!(!closed.is_interval_closed());
    }

    #[test]
    fn test_valid_intervals() {
        let mask = IntervalMask::from_vec(vec![false, true, true, false, true, false]);
        let intervals = mask.valid_intervals();
        assert_eq!(intervals, vec![(1, 3), (4, 5)]);
    }

    #[test]
    fn test_nopruner_all_valid() {
        let pruner = IntervalPruner::new(katgpt_core::NoPruner, 1);
        let candidates = vec![0usize, 1, 2, 3, 4];
        let mut results = vec![false; 5];
        pruner.batch_is_valid(0, &candidates, &[], &mut results);
        assert!(results.iter().all(|&v| v));
    }

    /// Simulated inner pruner that rejects specific tokens.
    struct SelectivePruner {
        rejected: Vec<usize>,
    }

    impl ConstraintPruner for SelectivePruner {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            !self.rejected.contains(&token_idx)
        }
    }

    #[test]
    fn test_scattered_rejects_get_filled() {
        // Inner pruner accepts tokens 0,1,2,4,5,6,8,9 (rejects 3 and 7).
        let inner = SelectivePruner {
            rejected: vec![3, 7],
        };
        let pruner = IntervalPruner::new(inner, 2);

        let candidates: Vec<usize> = (0..10).collect();
        let mut results = vec![false; 10];
        pruner.batch_is_valid(0, &candidates, &[], &mut results);

        // Before closure: valid at 0,1,2, _gap(3)_, 4,5,6, _gap(7)_, 8,9
        // Gaps are size 1 and 1, both ≤ threshold 2 → fill.
        // After closure: all 10 tokens valid.
        assert!(
            results.iter().all(|&v| v),
            "all tokens should be valid after closure"
        );
    }

    #[test]
    fn test_close_intervals_zero_threshold() {
        let mask = IntervalMask::from_vec(vec![true, false, true]);
        let closed = mask.close_intervals(0);
        assert_eq!(closed.as_slice(), &[true, false, true]);
    }

    #[test]
    fn test_empty_mask() {
        let mask = IntervalMask::from_vec(vec![]);
        assert!(mask.is_interval_closed());
        assert_eq!(mask.gap_count(), 0);
        assert!(mask.valid_intervals().is_empty());
    }

    #[test]
    fn test_all_valid() {
        let mask = IntervalMask::from_vec(vec![true, true, true]);
        assert!(mask.is_interval_closed());
        assert_eq!(mask.valid_intervals(), vec![(0, 3)]);
    }

    #[test]
    fn test_all_invalid() {
        let mask = IntervalMask::from_vec(vec![false, false, false]);
        assert!(mask.is_interval_closed());
        assert!(mask.valid_intervals().is_empty());
    }

    #[test]
    fn test_single_valid() {
        let mask = IntervalMask::from_vec(vec![false, true, false]);
        assert!(mask.is_interval_closed());
        assert_eq!(mask.valid_intervals(), vec![(1, 2)]);
    }

    // -----------------------------------------------------------------------
    // T29: Adaptive routing tests on IntervalMask
    // -----------------------------------------------------------------------

    #[test]
    fn test_interval_mask_adaptive_is_interval_closed() {
        use crate::interval_pruner::AdaptiveConfig;
        let config = AdaptiveConfig::default();

        // Small mask (below threshold) → scalar path.
        let small = IntervalMask::from_vec(vec![true, false, true]);
        assert!(!small.is_interval_closed_adaptive(&config));

        // Contiguous small mask.
        let small_closed = IntervalMask::from_vec(vec![true, true, false]);
        assert!(small_closed.is_interval_closed_adaptive(&config));
    }

    #[test]
    fn test_interval_mask_adaptive_close_intervals() {
        use crate::interval_pruner::AdaptiveConfig;
        let config = AdaptiveConfig::default();

        let mask = IntervalMask::from_vec(vec![true, false, false, true]);
        let closed = mask.close_intervals_adaptive(3, &config);
        assert_eq!(closed.as_slice(), &[true, true, true, true]);
    }

    #[test]
    fn test_interval_mask_adaptive_matches_non_adaptive() {
        use crate::interval_pruner::AdaptiveConfig;
        let config = AdaptiveConfig::default();

        // Small mask.
        let small = IntervalMask::from_vec(vec![true, false, true]);
        assert_eq!(
            small.is_interval_closed_adaptive(&config),
            small.is_interval_closed()
        );

        // Large mask.
        let mut large = vec![false; 512];
        large[100..200].fill(true);
        large[203..300].fill(true);
        let mask = IntervalMask::from_vec(large);
        assert_eq!(
            mask.is_interval_closed_adaptive(&config),
            mask.is_interval_closed()
        );

        let closed_adaptive = mask.close_intervals_adaptive(5, &config);
        let closed_plain = mask.close_intervals(5);
        assert_eq!(closed_adaptive.as_slice(), closed_plain.as_slice());
    }
}
