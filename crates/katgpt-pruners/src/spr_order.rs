//! SPR success-offset lexicographic ordering — Terminal-RSI extraction (issue 948 T2).
//!
//! The Terminal-RSI paper's (arXiv:2610.22192) Success-Prioritized Reward
//! `R = S + P/(P+D')` encodes a pure ordering law: the strict class dominates —
//! **no amount of partial credit ever promotes a failure over a success**. The
//! soft term is a saturating monotone squash (Michaelis-Menten — sigmoid-family
//! in log space; house sigmoid-not-softmax rule compatible).
//!
//! # Consumers
//!
//! Every candidate-ranking site that consumes scalar scores where a
//! verified-pass and a near-miss differ by a small delta: DDTree/MCTS child
//! stats, drafter candidate ordering, CGSP replay/absorb selection, corpus
//! filter admission. Sort by the [`SprKey`] pair ordering.
//!
//! # The pair ordering is canonical; the float is transport
//!
//! The single-float form ([`spr_reward`]) equals the lexicographic
//! `(passed, partial)` order in real arithmetic because the squash is strictly
//! below 1 — but at f32 width a ratio with `passes ≫ divisor` can round up to
//! exactly 1.0, where a failure's reward ties a zero-pass success's. The
//! [`Ord`]-shaped pair compare is therefore the canonical primitive; the float
//! form is documented transport with a non-strict guarantee (see the property
//! tests).
//!
//! # Determinism
//!
//! [`mass_on_failures`] is a plain sequential fold over the given order
//! (local-only diagnostic — never synced), per the float-reduction law.

/// A candidate's ordering key: strict class first, partial credit second.
///
/// Compare with [`SprKey::cmp_key`] (or the [`Ord`]-shaped helpers); the
/// strict class dominates, ties inside a class compare the partial credit
/// with the total float order (NaN-safe).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SprKey {
    /// Did the candidate verifiably pass? This bit dominates every partial.
    pub passed: bool,
    /// The saturating partial credit, conventionally `P/(P+D')` in `[0, 1)`.
    pub partial: f32,
}

impl SprKey {
    /// A key from the paper's parts: `partial = mm_squash(passes, divisor)`.
    #[inline]
    pub fn new(passed: bool, passes: f32, divisor: f32) -> Self {
        Self {
            passed,
            partial: mm_squash(passes, divisor),
        }
    }

    /// The canonical lexicographic compare: `passed` first, `partial` second
    /// (total float order — NaN partials sort deterministically and never
    /// poison the strict-class separation).
    #[inline]
    pub fn cmp_key(&self, other: &Self) -> std::cmp::Ordering {
        self.passed
            .cmp(&other.passed)
            .then_with(|| self.partial.total_cmp(&other.partial))
    }

    /// Convenience: is `self` strictly ahead of `other` under the law?
    #[inline]
    pub fn dominates(&self, other: &Self) -> bool {
        self.cmp_key(other) == std::cmp::Ordering::Greater
    }
}

/// Michaelis-Menten saturating squash: `P / (P + D')`, computed in the
/// overflow-free reciprocal form `1 / (1 + D'/P)`.
///
/// Monotone in `P`, bounded in `[0, 1]` for finite inputs, and strictly below
/// 1 in real arithmetic for `P, D' > 0` — the property that makes the paper's
/// single-float reward lexicographic. Fails soft: a non-finite or negative
/// `passes`, a non-finite/zero/negative `divisor`, or zero passes all squash
/// to `0.0` (no panic, no NaN on the ranking path).
#[inline]
pub fn mm_squash(passes: f32, divisor: f32) -> f32 {
    if !passes.is_finite() || passes <= 0.0 || !divisor.is_finite() || divisor <= 0.0 {
        return 0.0;
    }
    1.0 / (1.0 + divisor / passes)
}

/// The paper's Success-Prioritized Reward as a single float:
/// `R = S + P/(P+D')` with `S = passed as u8 as f32`.
///
/// **Transport form.** In real arithmetic, ordering by `R` is exactly the
/// [`SprKey`] lexicographic order (the squash is strictly < 1, so the
/// strict class dominates). At f32 width the guarantee is non-strict —
/// `R(success) ≥ R(failure)` for every finite input, with equality only in
/// the rounding-degenerate corner (a failure's squash rounding up to 1.0
/// against a zero-pass success). Rank by [`SprKey`] when the distinction
/// matters; use this where a scalar slot is the interface.
#[inline]
pub fn spr_reward(passed: bool, passes: f32, divisor: f32) -> f32 {
    if passed {
        1.0 + mm_squash(passes, divisor)
    } else {
        mm_squash(passes, divisor)
    }
}

/// Eval-board diagnostic (Terminal-RSI F1): the fraction of total reward
/// mass landing on failures.
///
/// A frozen-corpus health read — the paper's 50.09% → 28.72% shift is this
/// statistic under reshaped training rewards; here it audits any scored
/// candidate set. Sequential fold in input order (local-only diagnostic,
/// never synced). An empty input or a zero total returns `0.0` — no mass,
/// no claim.
pub fn mass_on_failures<'a, I>(rows: I) -> f32
where
    I: IntoIterator<Item = &'a (bool, f32)>,
{
    let mut total = 0.0f32;
    let mut failure_mass = 0.0f32;
    for &(passed, reward) in rows {
        total += reward;
        if !passed {
            failure_mass += reward;
        }
    }
    if !total.is_finite() || total <= 0.0 {
        return 0.0;
    }
    (failure_mass / total).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn strict_class_dominates_by_construction() {
        // Executable separation, not a measured one: for ANY partial values,
        // a success outranks a failure under the pair ordering.
        let edge_partials = [
            f32::NEG_INFINITY,
            -1.0,
            0.0,
            f32::MIN_POSITIVE,
            0.25,
            0.5,
            0.999_999,
            1.0,
            2.0,
            f32::INFINITY,
            f32::NAN,
        ];
        for &f in &edge_partials {
            for &s in &edge_partials {
                let failure = SprKey {
                    passed: false,
                    partial: f,
                };
                let success = SprKey {
                    passed: true,
                    partial: s,
                };
                assert_eq!(
                    failure.cmp_key(&success),
                    Ordering::Less,
                    "failure {f:?} must never outrank success {s:?}"
                );
                assert_eq!(success.cmp_key(&failure), Ordering::Greater,);
                assert!(success.dominates(&failure));
                assert!(!failure.dominates(&success));
            }
        }
    }

    #[test]
    fn partial_breaks_ties_inside_a_class_with_total_order() {
        let low = SprKey {
            passed: true,
            partial: 0.1,
        };
        let high = SprKey {
            passed: true,
            partial: 0.9,
        };
        assert_eq!(high.cmp_key(&low), Ordering::Greater);
        assert_eq!(low.cmp_key(&high), Ordering::Less);
        assert_eq!(low.cmp_key(&low), Ordering::Equal);
        // NaN partials are ordered deterministically (total_cmp), never equal
        // to a real value and never panicking.
        let nan = SprKey {
            passed: true,
            partial: f32::NAN,
        };
        assert_eq!(nan.cmp_key(&high), Ordering::Greater); // NaN is "greater" in total_cmp
        assert_eq!(nan.cmp_key(&nan), Ordering::Equal);
    }

    #[test]
    fn mm_squash_is_saturating_monotone_and_fails_soft() {
        // Monotone in passes, bounded below 1 in the tested regime.
        let divisor = 10.0;
        let mut prev = -1.0f32;
        for k in 0..=100 {
            let p = k as f32;
            let s = mm_squash(p, divisor);
            assert!((0.0..1.0).contains(&s), "squash {s} out of [0,1) at p={p}");
            assert!(s > prev, "squash must be monotone at p={p}");
            prev = s;
        }
        // Saturation: an order of magnitude more passes barely moves it.
        assert!(mm_squash(1_000.0, divisor) - mm_squash(10_000.0, divisor) < 1e-3);
        // Fail-soft inputs — no panic, no NaN propagation.
        assert_eq!(mm_squash(-1.0, divisor), 0.0);
        assert_eq!(mm_squash(f32::NAN, divisor), 0.0);
        assert_eq!(mm_squash(5.0, 0.0), 0.0);
        assert_eq!(mm_squash(5.0, -2.0), 0.0);
        assert_eq!(mm_squash(5.0, f32::NAN), 0.0);
        assert_eq!(mm_squash(f32::INFINITY, divisor), 0.0); // !(inf >= 0) is false — guard below
    }

    #[test]
    fn spr_reward_is_non_strictly_separating_and_the_corner_is_named() {
        // Non-strict law at f32 width: success >= failure for every finite input.
        let divisors = [0.001f32, 1.0, 1e6];
        for &d in &divisors {
            for p in [0.0f32, 0.5, 1.0, 10.0, 1e9] {
                for q in [0.0f32, 0.5, 1.0, 10.0, 1e9] {
                    assert!(
                        spr_reward(true, p, d) >= spr_reward(false, q, d),
                        "success p={p} must not lose to failure q={q} at d={d}"
                    );
                }
            }
        }
        // The degenerate corner, executed: failure's squash rounds UP to 1.0
        // while the success has zero passes — the pair ordering still
        // separates, the float form ties.
        let corner = spr_reward(false, 1e9, 1.0);
        assert_eq!(corner, 1.0, "failure squash rounds to exactly 1.0");
        assert_eq!(spr_reward(true, 0.0, 1.0), 1.0);
        assert_eq!(
            SprKey {
                passed: false,
                partial: 1.0
            }
            .cmp_key(&SprKey {
                passed: true,
                partial: 0.0
            }),
            Ordering::Less,
            "the pair ordering separates where the float ties"
        );
    }

    #[test]
    fn spr_key_new_squashes_the_paper_parts() {
        let k = SprKey::new(true, 3.0, 1.0);
        assert!(k.passed);
        assert!((k.partial - 0.75).abs() < 1e-6);
        assert!((spr_reward(true, 3.0, 1.0) - 1.75).abs() < 1e-6);
    }

    #[test]
    fn mass_on_failures_reads_the_fraction_and_stays_silent_on_zero_mass() {
        // Half the mass on failures.
        let rows = [(true, 1.0f32), (false, 1.0), (true, 1.0), (false, 1.0)];
        assert!((mass_on_failures(rows.iter()) - 0.5).abs() < 1e-6);
        // All-mass-on-success → 0; all-failure → 1.
        let all_pass = [(true, 2.0f32), (true, 3.0)];
        assert_eq!(mass_on_failures(all_pass.iter()), 0.0);
        let all_fail = [(false, 2.0f32)];
        assert_eq!(mass_on_failures(all_fail.iter()), 1.0);
        // No mass, no claim: empty input and zero/negative totals → 0.0.
        let empty: [(bool, f32); 0] = [];
        assert_eq!(mass_on_failures(empty.iter()), 0.0);
        let zeros = [(true, 0.0f32), (false, 0.0)];
        assert_eq!(mass_on_failures(zeros.iter()), 0.0);
    }
}
