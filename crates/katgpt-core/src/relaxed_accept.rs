//! PLM relaxed-acceptance admit predicate (Issue 973, arXiv:2610.26693 Alg. 1).
//!
//! The lossy half of the relaxed rule `accept(d) ⇔ [d == x̃] ∨ [d ∈ ρ-nucleus(p)]
//! ∨ [p(d) ≥ ε]` — everything except the verifier-draw match term, which each
//! consumer owns (inverse-CDF draw in katgpt-forward's `NucleusAccept`, the keyed
//! Gumbel stream in riir-infer's keyed spec lane). Lifted here so both verify
//! paths share ONE nucleus definition (pinned first-index tie rule, sequential
//! ascending fold — Issue 940 determinism law).
//!
//! `(ρ = 0, ε > 1)` admits nothing: the consumer's rule collapses to its exact
//! match term. The calibration property (coverage ≥ ρ, minimality, rank-prefix)
//! and the planted-violation arms live with the katgpt-forward consumer tests.

/// Mass ranked strictly above `d` under descending-p order with the pinned
/// first-index tie rule (`p_t > p_d`, or `p_t == p_d ∧ t < d`) — one
/// sequential ascending fold, zero-alloc.
#[inline]
pub fn mass_above(p: &[f32], d: usize) -> f32 {
    let pd = p.get(d).copied().unwrap_or(0.0);
    let mut above = 0.0f32;
    for (t, &pt) in p.iter().enumerate() {
        if pt > pd || (pt == pd && t < d) {
            above += pt;
        }
    }
    above
}

/// ρ-nucleus membership from [`mass_above`]: `d` is in the smallest top-ranked
/// set with mass `≥ ρ` iff it carries mass and the mass ranked above it is
/// still short of `ρ`. `rho ≤ 0` ⇒ empty nucleus.
#[inline]
pub fn in_rho_nucleus(pd: f32, mass_above: f32, rho: f32) -> bool {
    pd > 0.0 && mass_above < rho
}

/// The relaxed admit terms: `p(d) > 0 ∧ (p(d) ≥ ε ∨ d ∈ ρ-nucleus(p))`.
///
/// Cost ladder: the ε term reads `p(d)` only, O(1); the ρ term's full-vocab
/// fold runs only when `rho > 0` and ε did not already admit.
#[inline]
pub fn relaxed_admit(p: &[f32], d: usize, rho: f32, eps: f32) -> bool {
    let pd = p.get(d).copied().unwrap_or(0.0);
    pd > 0.0 && (pd >= eps || (rho > 0.0 && in_rho_nucleus(pd, mass_above(p, d), rho)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eps_term_admits_at_threshold() {
        let p = [0.5f32, 0.3, 0.2];
        assert!(relaxed_admit(&p, 2, 0.0, 0.2));
        assert!(!relaxed_admit(&p, 2, 0.0, 0.21));
    }

    #[test]
    fn nucleus_is_the_smallest_top_prefix_with_tie_rule() {
        // Ties at 0.25: index 1 ranks above index 2 (first-index rule).
        let p = [0.5f32, 0.25, 0.25];
        assert!(relaxed_admit(&p, 0, 0.5, 2.0));
        assert!(!relaxed_admit(&p, 1, 0.5, 2.0)); // 0.5 above is not < 0.5
        assert!(relaxed_admit(&p, 1, 0.6, 2.0));
        assert!(!relaxed_admit(&p, 2, 0.6, 2.0)); // 0.75 above
        assert_eq!(mass_above(&p, 2), 0.75);
    }

    #[test]
    fn corner_admits_nothing_and_zero_mass_never_admits() {
        let p = [1.0f32, 0.0];
        assert!(!relaxed_admit(&p, 0, 0.0, 1.0 + f32::EPSILON));
        assert!(!relaxed_admit(&p, 1, 1.0, 0.0));
        assert!(!relaxed_admit(&p, 7, 1.0, 0.0)); // out of range = zero mass
    }
}
