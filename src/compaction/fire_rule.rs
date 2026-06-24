//! The `FireRule` enum — Boolean combination of predicate verdicts.
//!
//! A fire rule maps a [`RubricVerdict`](super::RubricVerdict)'s `Yes`/`No`
//! pattern to a single `bool`: should the gate COMPRESS? The combinators are
//! the paper's `And` / `Or` / `Not` / `Box` (compose) constructors
//! (SelfCompact, arXiv:2606.23525, Appendix A/B).
//!
//! All combinators operate on the `u8` bitmask produced by
//! [`RubricVerdict::yes_mask`](super::RubricVerdict::yes_mask). The `And` /
//! `Or` variants take a **mask** selecting which predicate indices participate
//! (bit `i` set iff predicate `i` is in the combination). This matches the
//! research-note sketch (`FireRule::And(0b1111)` = "all four predicates").
//!
//! **Zero-allocation contract**: [`FireRule::evaluate`] walks the (small,
//! config-time-constructed) tree without allocating. The `Box` variant uses
//! heap storage for the sub-rules, but that storage is allocated once at
//! construction and never touched on the hot path.

use super::rubric::RubricVerdict;

/// Binary combine op for the [`FireRule::Box`] variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CombineOp {
    /// Left AND Right.
    And,
    /// Left OR Right.
    Or,
}

/// Boolean combination of predicate results.
///
/// Built once at gate configuration time; evaluated many times on the hot
/// path. The `evaluate` method is total and branch-light.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FireRule {
    /// COMPRESS iff **all** predicates named in `mask` are `Yes`.
    ///
    /// `mask & yes_mask == mask` (the Yes bits must be a superset of the
    /// required bits). Predicates not in `mask` are ignored.
    And(u8),
    /// COMPRESS iff **any** predicate named in `mask` is `Yes`.
    ///
    /// `mask & yes_mask != 0`.
    Or(u8),
    /// COMPRESS iff predicate `index` is `No`.
    ///
    /// The paper's `¬N1` ("not stuck") is `Not(1 << 3)` for the N1 slot.
    /// `!(yes_mask & (1 << index) != 0)`.
    Not(u8),
    /// Compose two sub-rules with a binary op. The paper's math rule
    /// `Q1 ∨ (Q2 ∧ Q3)` is `Box(Or, Or(Q1_mask), And(Q2_mask | Q3_mask))`.
    /// The paper's search rule `C1 ∧ C2 ∧ C3 ∧ ¬N1` is
    /// `Box(And, And(C1|C2|C3), Not(N1))`.
    ///
    /// `Box` here is the *constructor name* (boxing the sub-rules), not
    /// the op; the op is the explicit [`CombineOp`] argument.
    Box(CombineOp, Box<Self>, Box<Self>),
}

impl FireRule {
    /// Evaluate the rule against a packed `yes_mask`.
    ///
    /// `yes_mask` bit `i` is set iff predicate `i` is `Yes` (see
    /// [`RubricVerdict::yes_mask`](super::RubricVerdict::yes_mask)).
    #[inline]
    #[must_use]
    pub fn evaluate_mask(&self, yes_mask: u8) -> bool {
        match self {
            Self::And(mask) => *mask & yes_mask == *mask,
            Self::Or(mask) => *mask & yes_mask != 0,
            // `Not` takes a *single-bit* index mask (1 << index). If a caller
            // passes a multi-bit mask, only the lowest set bit is negated —
            // debug-assert so the misuse is caught.
            Self::Not(bit) => {
                debug_assert!(
                    (*bit).is_power_of_two() || *bit == 0,
                    "FireRule::Not expects a single-bit index mask (1 << i); got {bit:#010b}"
                );
                yes_mask & bit == 0
            }
            Self::Box(op, a, b) => {
                let lhs = a.evaluate_mask(yes_mask);
                let rhs = b.evaluate_mask(yes_mask);
                match op {
                    CombineOp::And => lhs && rhs,
                    CombineOp::Or => lhs || rhs,
                }
            }
        }
    }

    /// Convenience: evaluate against a full [`RubricVerdict`].
    ///
    /// Debug-asserts `N <= 8` (the bitmask width).
    #[inline]
    #[must_use]
    pub fn evaluate<const N: usize>(&self, verdict: &RubricVerdict<N>) -> bool {
        self.evaluate_mask(verdict.yes_mask())
    }

    /// Construct the paper's search rule: `C1 ∧ C2 ∧ C3 ∧ ¬N1` over a
    /// 4-predicate rubric. Indices: C1=0, C2=1, C3=2, N1=3.
    ///
    /// `Box(And, And(C1|C2|C3), Not(N1))`: COMPRESS iff (all of C1,C2,C3
    /// are Yes) AND (N1 is No).
    #[must_use]
    pub fn search_rule_4() -> Self {
        // C1|C2|C3 mask = 0b0111. N1 bit = 0b1000.
        Self::Box(
            CombineOp::And,
            Box::new(Self::And(0b0111)),
            Box::new(Self::Not(0b1000)),
        )
    }

    /// Construct the paper's math rule: `Q1 ∨ (Q2 ∧ Q3)` over a 3-predicate
    /// rubric. Indices: Q1=0, Q2=1, Q3=2.
    ///
    /// `Box(Or, Or(Q1), And(Q2|Q3))`: COMPRESS iff Q1 is Yes OR (Q2 and Q3
    /// both Yes).
    #[must_use]
    pub fn math_rule_3() -> Self {
        // Q1 mask = 0b001. Q2|Q3 mask = 0b110.
        Self::Box(
            CombineOp::Or,
            Box::new(Self::Or(0b001)),
            Box::new(Self::And(0b110)),
        )
    }

    /// Construct the shard-freeze rule (G7 isomorphism target): `P0 ∧ P1`
    /// over a 2-predicate rubric. Indices: P0 (input_sufficient)=0,
    /// P1 (output_converged)=1.
    ///
    /// `And(0b0011)`: COMPRESS iff both predicates are Yes. This mirrors
    /// `ConsolidationPipeline::can_freeze`'s `input_sufficient &&
    /// output_converged`.
    #[must_use]
    pub fn shard_freeze_rule_2() -> Self {
        Self::And(0b0011)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::rubric::{PredicateReason, PredicateResult, RubricVerdict};

    fn yes(_i: usize) -> PredicateResult {
        PredicateResult::Yes {
            quote_start: 0,
            quote_len: 1,
        }
    }
    fn no() -> PredicateResult {
        PredicateResult::No {
            reason: PredicateReason::Unset,
        }
    }

    #[test]
    fn and_all_four_predicates_yes_fires() {
        // And(0b1111) over 4 predicates: fires iff all four Yes.
        let rule = FireRule::And(0b1111);
        let all_yes = RubricVerdict::<4>::new([yes(0), yes(1), yes(2), yes(3)]);
        assert!(rule.evaluate(&all_yes));
    }

    #[test]
    fn and_missing_one_predicate_does_not_fire() {
        let rule = FireRule::And(0b1111);
        let missing_one = RubricVerdict::<4>::new([yes(0), no(), yes(2), yes(3)]);
        assert!(!rule.evaluate(&missing_one));
    }

    #[test]
    fn or_single_bit_fires_on_that_predicate() {
        // Or(0b0001): fires iff predicate 0 is Yes.
        let rule = FireRule::Or(0b0001);
        let p0_yes = RubricVerdict::<4>::new([yes(0), no(), no(), no()]);
        assert!(rule.evaluate(&p0_yes));

        let p0_no = RubricVerdict::<4>::new([no(), yes(1), yes(2), yes(3)]);
        assert!(!rule.evaluate(&p0_no));
    }

    #[test]
    fn not_fires_when_predicate_is_no() {
        // Not(0): fires iff predicate 0 is No.
        let rule = FireRule::Not(0b0001);
        let p0_no = RubricVerdict::<4>::new([no(), yes(1), yes(2), yes(3)]);
        assert!(rule.evaluate(&p0_no));

        let p0_yes = RubricVerdict::<4>::new([yes(0), no(), no(), no()]);
        assert!(!rule.evaluate(&p0_yes));
    }

    #[test]
    fn box_compose_or_of_two_subrules() {
        // Box(Or, Or(0b001), And(0b110)) — paper math rule.
        let rule = FireRule::math_rule_3();

        // Q1 Yes → fires regardless of Q2/Q3.
        let q1_only = RubricVerdict::<3>::new([yes(0), no(), no()]);
        assert!(rule.evaluate(&q1_only));

        // Q2 ∧ Q3 (Q1 No) → fires.
        let q23 = RubricVerdict::<3>::new([no(), yes(1), yes(2)]);
        assert!(rule.evaluate(&q23));

        // Q2 only → does not fire.
        let q2_only = RubricVerdict::<3>::new([no(), yes(1), no()]);
        assert!(!rule.evaluate(&q2_only));

        // All No → does not fire.
        let none = RubricVerdict::<3>::all_no();
        assert!(!rule.evaluate(&none));
    }

    #[test]
    fn box_compose_and_of_two_subrules() {
        // Box(And, And(0b0111), Not(0b1000)) — paper search rule. This is the
        // critical test: the AND-combine means BOTH sub-rules must hold.
        let rule = FireRule::search_rule_4();

        // All of C1,C2,C3 Yes AND N1 No → fires.
        let fire = RubricVerdict::<4>::new([yes(0), yes(1), yes(2), no()]);
        assert!(rule.evaluate(&fire));

        // N1 Yes → Not(N1) fails → AND-combine fails → does NOT fire.
        let n1_yes = RubricVerdict::<4>::new([yes(0), yes(1), yes(2), yes(3)]);
        assert!(!rule.evaluate(&n1_yes), "AND-combine must require ¬N1");

        // All No (empty trajectory case): And(0b0111) fails (C1..C3 not Yes),
        // so AND-combine fails even though Not(N1) alone holds.
        let all_no = RubricVerdict::<4>::all_no();
        assert!(
            !rule.evaluate(&all_no),
            "AND-combine must require C1∧C2∧C3 too"
        );
    }

    #[test]
    fn search_rule_4_fires_only_when_c1_c2_c3_yes_and_n1_no() {
        let rule = FireRule::search_rule_4();

        // Canonical fire: C1,C2,C3 Yes, N1 No.
        let fire = RubricVerdict::<4>::new([yes(0), yes(1), yes(2), no()]);
        assert!(rule.evaluate(&fire), "C1∧C2∧C3∧¬N1 must fire");

        // N1 Yes → stuck, does not fire.
        let stuck = RubricVerdict::<4>::new([yes(0), yes(1), yes(2), yes(3)]);
        assert!(!rule.evaluate(&stuck), "¬N1 fails when N1 Yes");

        // C2 missing → not summarizable, does not fire.
        let not_summarizable = RubricVerdict::<4>::new([yes(0), no(), yes(2), no()]);
        assert!(!rule.evaluate(&not_summarizable), "C2 required");
    }

    #[test]
    fn shard_freeze_rule_2_is_strict_and() {
        let rule = FireRule::shard_freeze_rule_2();

        let both = RubricVerdict::<2>::new([yes(0), yes(1)]);
        assert!(rule.evaluate(&both), "P0 ∧ P1 fires");

        let only_p0 = RubricVerdict::<2>::new([yes(0), no()]);
        assert!(!rule.evaluate(&only_p0), "P1 required");

        let only_p1 = RubricVerdict::<2>::new([no(), yes(1)]);
        assert!(!rule.evaluate(&only_p1), "P0 required");

        let neither = RubricVerdict::<2>::all_no();
        assert!(!rule.evaluate(&neither));
    }

    #[test]
    fn evaluate_mask_and_or_equivalence() {
        // De Morgan spot-check via masks.
        // And(0b11) over mask 0b11 → true; over 0b10 → false.
        assert!(FireRule::And(0b11).evaluate_mask(0b11));
        assert!(!FireRule::And(0b11).evaluate_mask(0b10));
        // Or(0b01) over mask 0b10 → false; over 0b01 → true.
        assert!(!FireRule::Or(0b01).evaluate_mask(0b10));
        assert!(FireRule::Or(0b01).evaluate_mask(0b01));
    }
}
