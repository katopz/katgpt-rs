//! The two-tier pairwise comparator + first-index champion scan, extracted
//! VERBATIM from the riir-scenelab scenario arena (Plan 631; the arena's
//! inline rule in `riir-game-sdk/crates/riir-scenelab/src/arena.rs` is the
//! extraction SPEC — adoption there must leave Elo tables and champion
//! choice byte-identical, pinned by the spec-match grid below).
//!
//! # The rule (exactly as the arena ships it — no third tier)
//!
//! Pairwise score between two variants over one shared scenario:
//!
//! 1. **Verdict class** — [`VerdictRank::PASS`] over
//!    [`VerdictRank::PARTIAL`] over [`VerdictRank::FAIL`] (Fail and
//!    Timeout are ONE class — the arena's
//!    rule; a timeout is a failed checklist, not a separate strength).
//!    Higher class wins (1.0), lower loses (0.0).
//! 2. Same class → **fired fraction** — the higher fraction wins; equal is
//!    a **draw** (0.5, the house tournament convention
//!    `katgpt_core::rating::update_scored` consumes directly).
//!
//! [`two_tier_score_with_ticks`] is the OPT-IN third tier (mean
//! budget-normalized terminal ticks, lower wins) for the runtime consumer
//! (riir-ai Plan 642) ONLY. **The arena never consumes it**: adding the
//! tier there would flip current draws to wins/losses and change champion
//! selection (sdk Plan 009's byte-identical promise forbids it).
//!
//! # The determinism law at this seam
//!
//! The comparator is a pure total function (no state, no ties broken
//! here); selection breaks ties by FIRST INDEX — [`champion_index`] is
//! that scan extracted beside the comparator it selects from (strictly
//! greater wins, so the lowest index holds on ties — never
//! HashMap/backing-map order).
//!
//! Pure modelless arithmetic — no deps, no allocs, no_std-compatible ops
//! only. Zero-cost-unless-invoked.
//!
//! # Toy wiring example
//!
//! A worker under a harness vector, an evaluator emitting (rank, fired)
//! pairs over shared scenarios, Elo folding the pairwise scores, the
//! scan picking the champion:
//!
//! ```
//! use katgpt_core::harness_evolution::{VerdictRank, two_tier_score, champion_index};
//! use katgpt_core::rating::{STANDARD_BASE, STANDARD_K, STANDARD_SCALE, update_scored};
//!
//! // The evaluator's read over one shared scenario: rank + fired fraction.
//! let workers: [(VerdictRank, f64); 3] = [
//!     (VerdictRank::PASS, 0.9),
//!     (VerdictRank::PARTIAL, 1.0),
//!     (VerdictRank::PASS, 0.9),
//! ];
//!
//! // Fixed pair order (i < j outer, scenario inner in a real arena) —
//! // deterministic Elo folding.
//! let mut ratings = [STANDARD_BASE; 3];
//! for i in 0..workers.len() {
//!     for j in (i + 1)..workers.len() {
//!         let (a, b) = (workers[i], workers[j]);
//!         let score = two_tier_score(a.0, b.0, a.1, b.1);
//!         let (na, nb) = update_scored(ratings[i], ratings[j], score, STANDARD_K, STANDARD_SCALE);
//!         ratings[i] = na;
//!         ratings[j] = nb;
//!     }
//! }
//!
//! // Class separation survives the fold: BOTH Pass workers finish above
//! // the Partial one. (The fold's fixed i<j pair order decides which
//! // Pass worker leads — a nominal tie between workers 0 and 2 is
//! // exactly the champion scan's tie case.)
//! assert!(ratings[0] > ratings[1]);
//! assert!(ratings[2] > ratings[1]);
//! let champ = champion_index(&ratings).expect("non-empty table");
//! assert!(champ == 0 || champ == 2, "the champion is a Pass worker");
//!
//! // The first-index law itself: identical ratings hold the LOWEST index.
//! assert_eq!(champion_index(&[1100.0, 1000.0, 1100.0]), Some(0));
//! ```

use std::cmp::Ordering;

/// Verdict class rank — the arena's ordering `PASS > PARTIAL > FAIL`
/// (Fail and Timeout share one class). Ordered by the inner `u8`; only
/// the three named constants are meaningful.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerdictRank(pub u8);

impl VerdictRank {
    /// {Fail, Timeout} — one class (the arena's rule: a timeout is a
    /// failed checklist, not a separate strength).
    pub const FAIL: Self = Self(0);
    /// Partial — some checklist nodes fired.
    pub const PARTIAL: Self = Self(1);
    /// Pass — every checklist node fired in window.
    pub const PASS: Self = Self(2);
}

/// The arena's two-tier pairwise score for `a` vs `b` over one shared
/// scenario: verdict class first, fired fraction second, identical is a
/// draw. Returns the score for `a` (`1.0` win / `0.5` draw / `0.0` loss).
///
/// The expression tree is VERBATIM the arena's `elo_table` match
/// (riir-game-sdk `riir-scenelab/src/arena.rs`, Proposal 029 P4) — the
/// spec-match grid test below pins bit-identity over the full rank ×
/// fraction cross product, so arena adoption is byte-identical by
/// construction.
#[must_use]
pub fn two_tier_score(a: VerdictRank, b: VerdictRank, a_fired: f64, b_fired: f64) -> f64 {
    match a.cmp(&b) {
        Ordering::Greater => 1.0,
        Ordering::Less => 0.0,
        Ordering::Equal => {
            if a_fired > b_fired {
                1.0
            } else if a_fired < b_fired {
                0.0
            } else {
                0.5
            }
        }
    }
}

/// The OPT-IN third tier (runtime consumer riir-ai Plan 642 only — the
/// arena NEVER consumes it, see the module doc): verdict class → fired
/// fraction → **budget-normalized terminal ticks (LOWER wins)** → draw.
///
/// The tick axis is the `mean_ticks_norm` arena outcome (terminal tick
/// over the scenario budget): a variant that reached the same checklist
/// state in fewer ticks is the faster harness. Only reached when class
/// AND fired fraction tie — the two shipped tiers still dominate.
#[must_use]
pub fn two_tier_score_with_ticks(
    a: VerdictRank,
    b: VerdictRank,
    a_fired: f64,
    b_fired: f64,
    a_ticks_norm: f64,
    b_ticks_norm: f64,
) -> f64 {
    let base = two_tier_score(a, b, a_fired, b_fired);
    if base != 0.5 {
        return base;
    }
    if a_ticks_norm < b_ticks_norm {
        1.0
    } else if a_ticks_norm > b_ticks_norm {
        0.0
    } else {
        0.5
    }
}

/// The champion scan over a rating table: strictly-greater wins, so the
/// LOWEST index holds on ties (the determinism law's first-index rule).
/// `None` when the table is empty (the arena maps that to its
/// `CHAMPION_NONE` sentinel at its own boundary).
#[must_use]
pub fn champion_index(ratings: &[f64]) -> Option<usize> {
    let mut best = 0usize;
    for (i, r) in ratings.iter().enumerate() {
        if *r > ratings[best] {
            best = i;
        }
    }
    (!ratings.is_empty()).then_some(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// VERBATIM the arena's inline rule (riir-scenelab `arena.rs`
    /// `elo_table` score match @ sdk commit `2fe99ad`) — the extraction
    /// SPEC. Bit-identity against this reference over the full cross
    /// product is Plan 631 T1.1's adoption criterion.
    fn arena_reference(a_rank: VerdictRank, b_rank: VerdictRank, fa: f64, fb: f64) -> f64 {
        match a_rank.cmp(&b_rank) {
            Ordering::Greater => 1.0,
            Ordering::Less => 0.0,
            Ordering::Equal => {
                if fa > fb {
                    1.0
                } else if fa < fb {
                    0.0
                } else {
                    0.5
                }
            }
        }
    }

    #[test]
    fn spec_match_grid_bit_identical_to_the_arena_rule() {
        let ranks = [VerdictRank::FAIL, VerdictRank::PARTIAL, VerdictRank::PASS];
        // Fraction pairs incl. the exact-draw case and near-ties on both
        // sides (the epsilon-less `>` / `<` comparisons must classify
        // exactly as the arena's do).
        let fracs = [0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 0.999, 1.0 - 1e-12];
        for &ra in &ranks {
            for &rb in &ranks {
                for &fa in &fracs {
                    for &fb in &fracs {
                        assert_eq!(
                            two_tier_score(ra, rb, fa, fb).to_bits(),
                            arena_reference(ra, rb, fa, fb).to_bits(),
                            "diverged at rank {ra:?} vs {rb:?}, fired {fa} vs {fb}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn class_order_and_draw_boundaries() {
        // Class dominates fraction on BOTH directions.
        assert_eq!(
            two_tier_score(VerdictRank::PASS, VerdictRank::PARTIAL, 0.0, 1.0),
            1.0
        );
        assert_eq!(
            two_tier_score(VerdictRank::FAIL, VerdictRank::PASS, 1.0, 0.0),
            0.0
        );
        // Fail vs Partial is a class step (NOT a same-class fraction
        // fight) — and Fail vs Timeout-class is a DRAW at equal fraction
        // (one class).
        assert_eq!(
            two_tier_score(VerdictRank::FAIL, VerdictRank::PARTIAL, 1.0, 0.0),
            0.0
        );
        assert_eq!(
            two_tier_score(VerdictRank::FAIL, VerdictRank::FAIL, 0.5, 0.5),
            0.5
        );
        // Same class, fraction decides; exact equality draws.
        assert_eq!(
            two_tier_score(VerdictRank::PARTIAL, VerdictRank::PARTIAL, 0.75, 0.25),
            1.0
        );
        assert_eq!(
            two_tier_score(VerdictRank::PARTIAL, VerdictRank::PARTIAL, 0.25, 0.75),
            0.0
        );
        assert_eq!(
            two_tier_score(VerdictRank::PASS, VerdictRank::PASS, 0.9, 0.9),
            0.5
        );
        // Negated inputs flip the sign symmetrically (the pairwise
        // antisymmetry the Elo fold relies on).
        for &(ra, rb, fa, fb) in &[
            (VerdictRank::PASS, VerdictRank::FAIL, 0.1, 0.9),
            (VerdictRank::PARTIAL, VerdictRank::PARTIAL, 0.3, 0.6),
        ] {
            assert_eq!(
                two_tier_score(ra, rb, fa, fb) + two_tier_score(rb, ra, fb, fa),
                1.0
            );
        }
    }

    #[test]
    fn third_tier_only_breaks_exact_two_tier_draws() {
        // A class or fraction decision is NEVER overridden by ticks.
        assert_eq!(
            two_tier_score_with_ticks(VerdictRank::PASS, VerdictRank::FAIL, 0.0, 1.0, 1.0, 0.0),
            1.0
        );
        assert_eq!(
            two_tier_score_with_ticks(VerdictRank::FAIL, VerdictRank::PASS, 1.0, 0.0, 0.0, 1.0),
            0.0
        );
        // Exact two-tier draw → ticks decide, LOWER wins.
        assert_eq!(
            two_tier_score_with_ticks(
                VerdictRank::PARTIAL,
                VerdictRank::PARTIAL,
                0.5,
                0.5,
                0.4,
                0.6
            ),
            1.0
        );
        assert_eq!(
            two_tier_score_with_ticks(
                VerdictRank::PARTIAL,
                VerdictRank::PARTIAL,
                0.5,
                0.5,
                0.6,
                0.4
            ),
            0.0
        );
        // Full tie stays a draw.
        assert_eq!(
            two_tier_score_with_ticks(
                VerdictRank::PARTIAL,
                VerdictRank::PARTIAL,
                0.5,
                0.5,
                0.4,
                0.4
            ),
            0.5
        );
    }

    #[test]
    fn champion_scan_is_first_index_on_ties_and_none_on_empty() {
        assert_eq!(champion_index(&[]), None);
        assert_eq!(champion_index(&[1000.0]), Some(0));
        // Ties hold the FIRST index (strictly-greater scan).
        assert_eq!(champion_index(&[1100.0, 1100.0, 1000.0]), Some(0));
        assert_eq!(champion_index(&[1000.0, 1100.0, 1100.0]), Some(1));
        // -0.0 vs 0.0 compare EQUAL in IEEE (`0.0 > -0.0` is false), so
        // neither displaces the first index — the scan is a total-order
        // read, no sign-of-zero surprises in the arena's finite-rating
        // domain.
        assert_eq!(champion_index(&[0.0, -0.0]), Some(0));
        assert_eq!(champion_index(&[-0.0, 0.0]), Some(0));
    }
}
