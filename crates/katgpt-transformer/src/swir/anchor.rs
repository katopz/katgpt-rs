//! Anchor-exit boundary set — Plan 626 T0.1.
//!
//! The anchor arm exits a latent/soft-embedding span when the implicit argmax
//! readout lands in a derived **boundary-token set** after a minimum dwell.
//! This module owns the set itself: a fixed-capacity, sorted `u32` array with
//! binary-search membership — zero-alloc after construction, O(log |B|) on the
//! hot path, |B| capped well under the ~50 the plan budgets (cap 64).
//!
//! # The DEFAULT arm is MINIMAL (the plan's over-firing defence)
//!
//! `{</think>, EOS, <end_of_turn>}` only. Newline/period-class tokens are a
//! SEPARATE arm: they are frequent ambient argmaxes in math reasoning, and a
//! set carrying them collapses the anchor arm into a fixed min-dwell cap.
//! Tokens are disqualified from the set with T0.2's measured per-token ambient
//! frequency, never by guesswork — the host builds an arm via
//! [`BoundarySet::from_ids`] and the telemetry decides.
//!
//! Token ids are model-specific: the host resolves them from its tokenizer
//! (the same posture as [`crate::thinking_cot::ControlTokenIds`]). Nothing
//! here hardcodes a vocabulary.
//!
//! # Determinism
//!
//! Construction sorts + dedups, so the binary search's probe order is
//! independent of the host's id order; membership is a pure function of the
//! sorted content. Equal ids collapse — a set never contains duplicates.

/// Fixed capacity. The plan budgets |B| ≤ ~50; 64 gives headroom without
/// moving the struct off the stack (256 B payload).
pub const BOUNDARY_SET_CAP: usize = 64;

/// A sorted, deduplicated, fixed-capacity set of boundary token ids.
///
/// Construct via [`BoundarySet::minimal`] (the plan's DEFAULT arm) or
/// [`BoundarySet::from_ids`] (a host-derived arm — the config override).
/// Over-capacity input is HONESTLY truncated: [`BoundarySet::was_truncated`]
/// reports it, and the T0.2 telemetry surface reads that flag — a silently
/// dropped boundary token would change membership semantics, which is the
/// one thing this module must not do quietly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundarySet {
    ids: [u32; BOUNDARY_SET_CAP],
    len: usize,
    truncated: bool,
}

impl BoundarySet {
    /// The plan's DEFAULT minimal arm: `{</think>, EOS, <end_of_turn>}`.
    ///
    /// Ids are host-resolved (tokenizer lookup); duplicates collapse (e.g. a
    /// tokenizer whose EOS *is* `<end_of_turn>` yields a 2-element set).
    #[inline]
    pub fn minimal(think_close: u32, eos: u32, end_of_turn: u32) -> Self {
        Self::from_ids([think_close, eos, end_of_turn])
    }

    /// Build from any id sequence: sorts, dedups. Over-capacity input sets
    /// the [`BoundarySet::was_truncated`] flag instead of panicking — the
    /// constructor sits off the hot path, but honesty beats either a panic
    /// an arm can't catch or a silent membership change.
    pub fn from_ids<I>(ids: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        let mut set = Self {
            ids: [0; BOUNDARY_SET_CAP],
            len: 0,
            truncated: false,
        };
        for id in ids {
            if set.contains(id) {
                continue; // dedup (also keeps the array sorted-valid on insert)
            }
            if set.len == BOUNDARY_SET_CAP {
                set.truncated = true;
                continue;
            }
            // Insertion sort into the sorted tail — |B| ≤ 64, O(n²) worst
            // case is 4096 comparisons at construction, never on the hot path.
            let mut pos = set.len;
            while pos > 0 && set.ids[pos - 1] > id {
                set.ids[pos] = set.ids[pos - 1];
                pos -= 1;
            }
            set.ids[pos] = id;
            set.len += 1;
        }
        set
    }

    /// Number of distinct ids in the set.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the set is empty — membership is then `false` for every id,
    /// and the anchor arm cannot fire (the T1.5 empty-set byte-identity
    /// posture builds on exactly this).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// O(log |B|) membership on the hot path. Zero-alloc.
    #[inline]
    pub fn contains(&self, id: u32) -> bool {
        self.ids[..self.len].binary_search(&id).is_ok()
    }

    /// The sorted distinct ids (for telemetry / T0.2 reporting).
    pub fn ids(&self) -> &[u32] {
        &self.ids[..self.len]
    }

    /// True when construction dropped distinct ids for capacity. See the
    /// struct docs — honesty surface for the T0.2 telemetry, never silent.
    #[inline]
    pub fn was_truncated(&self) -> bool {
        self.truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_arm_holds_the_three_boundary_ids() {
        let set = BoundarySet::minimal(7, 1, 106);
        assert_eq!(set.len(), 3);
        assert!(set.contains(7)); // </think>
        assert!(set.contains(1)); // EOS
        assert!(set.contains(106)); // <end_of_turn>
        assert!(!set.contains(0));
        assert!(!set.contains(8));
        // Sorted regardless of argument order.
        assert_eq!(set.ids(), &[1, 7, 106]);
    }

    #[test]
    fn duplicates_collapse() {
        // A tokenizer whose EOS *is* <end_of_turn> — the minimal arm dedups.
        let set = BoundarySet::minimal(7, 106, 106);
        assert_eq!(set.len(), 2);
        assert_eq!(set.ids(), &[7, 106]);
        // Dedup also works on longer arms.
        let set = BoundarySet::from_ids([5, 5, 5, 2, 9, 2]);
        assert_eq!(set.ids(), &[2, 5, 9]);
    }

    #[test]
    fn membership_is_order_independent_and_sorted() {
        let a = BoundarySet::from_ids([30, 10, 20]);
        let b = BoundarySet::from_ids([10, 20, 30]);
        assert_eq!(a, b);
        for id in [10u32, 20, 30] {
            assert!(a.contains(id));
        }
        assert!(!a.contains(15));
        assert!(!a.contains(9));
        assert!(!a.contains(31));
    }

    #[test]
    fn empty_set_membership_is_always_false() {
        let set = BoundarySet::from_ids([]);
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(!set.contains(0));
        assert!(!set.contains(u32::MAX));
    }

    #[test]
    fn full_capacity_boundary_hits_and_misses() {
        // A dense arm covering a contiguous range — binary-search correctness
        // at both ends and in the middle.
        let ids: Vec<u32> = (100..100 + BOUNDARY_SET_CAP as u32).collect();
        let set = BoundarySet::from_ids(ids.iter().copied());
        assert_eq!(set.len(), BOUNDARY_SET_CAP);
        assert!(set.contains(100));
        assert!(set.contains(100 + BOUNDARY_SET_CAP as u32 - 1));
        assert!(set.contains(150));
        assert!(!set.contains(99));
        assert!(!set.contains(100 + BOUNDARY_SET_CAP as u32));
    }

    #[test]
    fn capacity_truncation_is_honest_not_silent() {
        // Over the cap: distinct ids beyond the cap are dropped, the flag
        // reports it, and membership NEVER reports an id that was not kept.
        let over: Vec<u32> = (0..BOUNDARY_SET_CAP as u32 + 10).collect();
        let set = BoundarySet::from_ids(over.iter().copied());
        assert_eq!(set.len(), BOUNDARY_SET_CAP);
        assert!(set.was_truncated(), "over-cap input must set the flag");
        for id in set.ids() {
            assert!(over.contains(id));
        }
        // Under the cap: the flag stays false (the minimal arm's steady state).
        assert!(!BoundarySet::minimal(7, 1, 106).was_truncated());
    }

    #[test]
    fn copy_semantics_keep_the_set_value_type() {
        let set = BoundarySet::minimal(7, 1, 106);
        let moved = set; // Copy — no borrow dance on the hot path
        assert_eq!(moved, set);
    }
}
