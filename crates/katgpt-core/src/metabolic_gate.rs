//! Metabolic Gate — energy-coupled compute gating (Research 550, Plan 585
//! consumer, arXiv:2609.10817 "Tapes Together Strong: The Co-evolution of
//! Computation and Cooperation").
//!
//! Distilled from the paper's energy↔computation coupling: every op costs
//! energy and execution share = energy share, so cooperation self-stabilizes
//! WITHOUT memory, assortment, or punishment because lossy theft destroys the
//! energy a defector needs to finish its own computation (Theorem 1). This
//! file ships the LAW, not the ALife substrate:
//!
//! - [`MetabolicGate::depth_factor`] — `σ((stock − base)/scale)`: compute-depth
//!   multiplier from a metabolic stock (a sibling of `gain_cost_halt` in the
//!   value-of-computation family, gating on a stock instead of a utility flow).
//! - [`defector_starves`] — the closed-form starvation design law
//!   `2ε < L(1+(1−α)δ)`.
//! - [`metabolic_drag_threshold`] — break-even inefficiency `K(ε,L)` (Lemma 1:
//!   a defector whose `(1−α)δ` exceeds `K` is strictly slower than mutual
//!   cooperators even in its best case), bisection-solved.
//! - [`steal_lossy`] — lossy transfer: the system loses `(1−α)δ`.
//! - [`execution_share`] — lottery-share scheduling over metabolic stocks.
//!
//! # Latent vs Raw
//!
//! Energy stocks (`Stamina`, zone pools) are RAW physical-domain values —
//! synced, committed, replayed. `depth_factor` is a LOCAL latent projection
//! (sigmoid on a raw stock; one-way world → cognition, never synced). The
//! design-law functions are pure arithmetic on design constants — safe to
//! evaluate anywhere.
//!
//! # NaN contract
//!
//! Non-finite inputs never produce an actionable verdict: `depth_factor`
//! returns the neutral 0.5 on a non-finite stock, `defector_starves` returns
//! `false` (no starvation claim without valid constants), `steal_lossy` is a
//! no-op returning 0, `execution_share` returns 0.5, and
//! `metabolic_drag_threshold` returns NaN for a non-viable regime
//! (`eps <= repl_cost`) where the break-even concept is undefined.

#![allow(clippy::float_cmp)] // float comparisons in tests against exact constants

use crate::sigmoid;

/// Sigmoid-gated compute-depth selector over a metabolic stock.
///
/// `depth_factor(stock) = σ((stock − e_base) / e_scale)` maps an energy
/// stock to a compute-depth multiplier in `(0, 1)`: at `stock == e_base` the
/// depth is exactly 0.5 (σ(0)); depleted stocks dim cognition, abundant
/// stocks brighten it. Consumers multiply this onto a tier budget or depth
/// ladder — never onto raw synced state.
///
/// Construction contract: `e_base` finite, `e_scale` finite and `> 0`. The
/// struct is `Copy` (2 × f32, 8 bytes) and allocation-free by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetabolicGate {
    e_base: f32,
    e_scale: f32,
}

impl MetabolicGate {
    /// Construct from the stock anchor `e_base` (depth 0.5) and the sigmoid
    /// steepness scale `e_scale` (stock units per σ transition; must be `> 0`).
    pub const fn new(e_base: f32, e_scale: f32) -> Self {
        Self { e_base, e_scale }
    }

    /// Compute-depth multiplier `σ((stock − e_base)/e_scale)` ∈ `(0, 1)`.
    ///
    /// Monotone non-decreasing in `stock`. A non-finite stock returns the
    /// neutral 0.5 — garbage must neither maximally dim nor boost cognition
    /// (the saddle_escape "NaN never fires" convention).
    #[inline]
    #[must_use]
    pub fn depth_factor(&self, energy_stock: f32) -> f32 {
        if !energy_stock.is_finite() {
            return 0.5;
        }
        sigmoid((energy_stock - self.e_base) / self.e_scale)
    }

    /// The stock anchor (depth 0.5 point).
    #[inline]
    #[must_use]
    pub const fn e_base(&self) -> f32 {
        self.e_base
    }

    /// The sigmoid steepness scale (stock units per σ transition).
    #[inline]
    #[must_use]
    pub const fn e_scale(&self) -> f32 {
        self.e_scale
    }
}

/// Paper Theorem 1 — the starvation design law.
///
/// `2ε < L(1 + (1−α)δ)` ⇒ a defector exhausts the pool before completing its
/// L replication writes ⇒ cooperation locally favored in well-mixed,
/// memoryless populations. `ε` = per-agent energy grant rate, `L` =
/// replication (spawn) cost, `α` = theft retention fraction (`α → 1` recovers
/// zero-sum transfer — no starvation), `δ` = per-theft transfer amount
/// (`δ = 0` ⇒ `2ε < L`, false in any cooperator-viable regime `ε > L`).
///
/// Returns `false` on any non-finite input (no starvation claim on garbage).
#[inline]
#[must_use]
pub fn defector_starves(eps: f32, repl_cost: f32, alpha: f32, delta: f32) -> bool {
    if !eps.is_finite() || !repl_cost.is_finite() || !alpha.is_finite() || !delta.is_finite() {
        return false;
    }
    2.0 * eps < repl_cost * (1.0 + (1.0 - alpha) * delta)
}

/// Paper Lemma 1 — break-even inefficiency `K(ε, L)`, bisection-solved.
///
/// Solves the transcendental break-even equation
/// `ln(2ε/(2ε − L(1+K))) = (1+K) · ln(ε/(ε − L))`
/// for `K`: a defector whose inefficiency `(1−α)δ` EXCEEDS `K(ε, L)` is
/// strictly slower than mutual cooperators even in its best case.
///
/// Domain: requires the cooperator-viable regime `eps > repl_cost > 0`;
/// anything else (including the degenerate `repl_cost == 0`, where the
/// equation collapses to `0 = 0` and break-even is vacuous) returns NaN.
///
/// Bisection over `K ∈ [0, 2ε/L − 1)` — at `K = 0` the LHS sits below the
/// RHS (`ln(1/(1−L/2ε)) < ln(1/(1−L/ε))`), and the LHS diverges to +∞ as
/// `K → 2ε/L − 1`, so exactly one root brackets. 32 iterations pin the root
/// to f32 precision (bracket shrinks by 2³² ≈ 4×10⁹); each iteration is one
/// log pair — bounded, allocation-free.
#[must_use]
pub fn metabolic_drag_threshold(eps: f32, repl_cost: f32) -> f32 {
    if !eps.is_finite()
        || !repl_cost.is_finite()
        || repl_cost <= 0.0
        || eps <= repl_cost
    {
        return f32::NAN;
    }
    // repl_cost > 0 and eps > repl_cost here, so the bracket (0, 2ε/L − 1)
    // is non-empty and both log arguments are positive.
    let ln_pair = (eps / (eps - repl_cost)).ln();
    let mut lo = 0.0f32;
    let mut hi = 2.0 * eps / repl_cost - 1.0;
    for _ in 0..32 {
        let mid = 0.5 * (lo + hi);
        let denom = 2.0 * eps - repl_cost * (1.0 + mid);
        // f(mid) = ln(2ε/denom) − (1+mid)·ln(ε/(ε−L)); denom <= 0 ⇒ f = +∞
        // (the float edge of the divergence) ⇒ root is below mid.
        let f = if denom <= 0.0 {
            f32::INFINITY
        } else {
            (2.0 * eps / denom).ln() - (1.0 + mid) * ln_pair
        };
        if f < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Lossy transfer — theft that burns the commons.
///
/// The thief gains `α·actual`, the victim loses `actual`, and the SYSTEM
/// destroys `(1−α)·actual` (returned). `actual = min(delta, victim)` clamps
/// to the victim's available stock — stocks never go negative and the
/// conservation identity `thief_gain + destroyed == victim_loss` holds
/// exactly in f32 for the clamped amount.
///
/// No-op returning 0 on non-finite inputs, `delta <= 0`, or `alpha` outside
/// `[0, 1]` (the valid-domain contract; garbage never moves energy).
#[inline]
#[must_use = "the destroyed amount is the system drain — track it"]
pub fn steal_lossy(thief: &mut f32, victim: &mut f32, delta: f32, alpha: f32) -> f32 {
    if !thief.is_finite()
        || !victim.is_finite()
        || !delta.is_finite()
        || !alpha.is_finite()
        || delta <= 0.0
        || !(0.0..=1.0).contains(&alpha)
    {
        return 0.0;
    }
    let actual = delta.min(victim.max(0.0));
    let destroyed = (1.0 - alpha) * actual;
    *victim -= actual;
    *thief += alpha * actual;
    destroyed
}

/// Lottery-share scheduling probability over metabolic stocks (paper C.1).
///
/// `execution_share(e_i, e_j) = e_i / (e_i + e_j)`: the fraction of
/// execution bandwidth agent `i` receives when kinetics scale with energy
/// (`dt = 1/(E_i+E_j)`). Both-zero (or any non-positive sum, defensively)
/// returns 0.5 — equal split when neither agent holds energy.
#[inline]
#[must_use]
pub fn execution_share(e_i: f32, e_j: f32) -> f32 {
    let sum = e_i + e_j;
    if !sum.is_finite() || sum <= 0.0 {
        return 0.5;
    }
    e_i / sum
}

#[cfg(test)]
mod tests {
    use super::*;

    // The Plan 585 T0.2 regime tuple: cooperator-viable (2ε = 24 > 2L = 20)
    // AND defector-starving (24 < 10·(1+0.7·3) = 31).
    const EPS: f32 = 12.0;
    const L: f32 = 10.0;
    const ALPHA: f32 = 0.3;
    const DELTA: f32 = 3.0;

    #[test]
    fn depth_factor_is_bounded_and_anchored() {
        let gate = MetabolicGate::new(50.0, 10.0);
        assert_eq!(gate.depth_factor(50.0), 0.5); // σ(0) anchor
        assert!(gate.depth_factor(0.0) < 0.01); // 5σ below
        assert!(gate.depth_factor(100.0) > 0.99); // 5σ above
        assert!(gate.depth_factor(f32::NAN) == 0.5); // neutral on garbage
        assert!(gate.depth_factor(f32::INFINITY) == 0.5);
    }

    #[test]
    fn depth_factor_monotone_in_stock() {
        let gate = MetabolicGate::new(10.0, 3.0);
        let mut prev = gate.depth_factor(-100.0);
        for i in 0..200 {
            let stock = -100.0 + i as f32;
            let d = gate.depth_factor(stock);
            assert!(d >= prev, "not monotone at stock={stock}");
            assert!((0.0..=1.0).contains(&d));
            prev = d;
        }
    }

    #[test]
    fn starvation_law_regimes() {
        // T0.2 tuple: viable + starving (viability 2ε=24 > 2L=20 holds by
        // construction — asserted in the bench's G1 where it feeds a verdict).
        assert!(defector_starves(EPS, L, ALPHA, DELTA));
        // α → 1 recovers zero-sum: no loss ⇒ no starvation.
        assert!(!defector_starves(EPS, L, 1.0, DELTA));
        assert!(!defector_starves(EPS, L, 0.999, 0.0));
        // δ = 0 ⇒ bound is L; 2ε < L is false in any viable regime (ε > L).
        assert!(!defector_starves(EPS, L, ALPHA, 0.0));
        // Strict inequality at the exact boundary 2ε == L(1+(1−α)δ).
        assert!(!defector_starves(12.0, 8.0, 0.0, 2.0)); // 24 == 24
        // Non-finite ⇒ no claim.
        assert!(!defector_starves(f32::NAN, L, ALPHA, DELTA));
    }

    #[test]
    fn drag_threshold_solves_the_transcendental() {
        // Residual of the break-even equation at the returned K must be ~0.
        for &(eps, l) in &[(12.0, 10.0), (20.0, 10.0), (100.0, 10.0), (12.0, 11.0)] {
            let k = metabolic_drag_threshold(eps, l);
            assert!(k.is_finite() && k > 0.0, "eps={eps} l={l} k={k}");
            let lhs = (2.0 * eps / (2.0 * eps - l * (1.0 + k))).ln();
            let rhs = (1.0 + k) * (eps / (eps - l)).ln();
            assert!(
                (lhs - rhs).abs() < 1e-4,
                "residual {} at eps={eps} l={l} k={k}",
                (lhs - rhs).abs()
            );
        }
        // The T0.2 tuple's inefficiency (1−α)δ = 2.1 sits ABOVE break-even:
        // the starvation regime and the drag regime tell one coherent story.
        let k = metabolic_drag_threshold(EPS, L);
        assert!(k < 2.1, "K({EPS},{L})={k} not below (1−α)δ=2.1");
        // Undefined regime: cooperators cannot replicate ⇒ NaN.
        assert!(metabolic_drag_threshold(5.0, 10.0).is_nan());
        assert!(metabolic_drag_threshold(f32::NAN, 10.0).is_nan());
    }

    #[test]
    fn steal_lossy_conservation_identity() {
        // Full transfer, binary-fraction α for exact f32 arithmetic.
        let mut thief = 0.0f32;
        let mut victim = 40.0f32;
        let destroyed = steal_lossy(&mut thief, &mut victim, 10.0, 0.5);
        assert_eq!(destroyed, 5.0);
        assert_eq!(thief, 5.0);
        assert_eq!(victim, 30.0);
        // thief_gain + destroyed == victim_loss (exact).
        assert_eq!(thief - 0.0 + destroyed, 10.0);

        // Partial transfer: clamped to the victim's available stock.
        let mut t2 = 0.0f32;
        let mut v2 = 3.0f32;
        let d2 = steal_lossy(&mut t2, &mut v2, 10.0, 0.5);
        assert_eq!(d2, 1.5);
        assert_eq!(t2, 1.5);
        assert_eq!(v2, 0.0);

        // Empty victim ⇒ no-op.
        let mut t3 = 7.0f32;
        let mut v3 = 0.0f32;
        assert_eq!(steal_lossy(&mut t3, &mut v3, 10.0, 0.5), 0.0);
        assert_eq!((t3, v3), (7.0, 0.0));

        // Garbage / invalid-domain ⇒ no energy moves.
        let mut t4 = 1.0f32;
        let mut v4 = 1.0f32;
        assert_eq!(steal_lossy(&mut t4, &mut v4, f32::NAN, 0.5), 0.0);
        assert_eq!(steal_lossy(&mut t4, &mut v4, -1.0, 0.5), 0.0);
        assert_eq!(steal_lossy(&mut t4, &mut v4, 1.0, 1.5), 0.0);
        assert_eq!((t4, v4), (1.0, 1.0));

        // α → 1 is zero-sum: nothing destroyed (the no-starvation limit).
        let mut t5 = 0.0f32;
        let mut v5 = 10.0f32;
        assert_eq!(steal_lossy(&mut t5, &mut v5, 10.0, 1.0), 0.0);
        assert_eq!((t5, v5), (10.0, 0.0));
    }

    #[test]
    fn execution_share_symmetry_and_guards() {
        let a = execution_share(10.0, 30.0);
        assert!((a - 0.25).abs() < 1e-6);
        assert!((a + execution_share(30.0, 10.0) - 1.0).abs() < 1e-6);
        // Paper C.1: both-zero → ½.
        assert_eq!(execution_share(0.0, 0.0), 0.5);
        // Degenerate negative sum → neutral split (never a negative share).
        assert_eq!(execution_share(-1.0, 0.0), 0.5);
        assert_eq!(execution_share(f32::NAN, 1.0), 0.5);
        // Monotone in own energy.
        assert!(execution_share(3.0, 10.0) < execution_share(4.0, 10.0));
    }
}
