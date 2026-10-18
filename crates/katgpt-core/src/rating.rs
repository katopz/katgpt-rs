//! Elo + Plackett-Luce rating primitives (Issue 686 — promoted from four
//! in-stack copies).
//!
//! The one math four consumers shared as private copies until this module:
//!
//! | Consumer (former copy) | Precision | Convention |
//! |---|---|---|
//! | `katgpt-pruners` arena `EloCalculator` | f64 | K=32, base field 1000 (seed), scale 400, win/loss |
//! | `katgpt-pruners` proof `lambda_to_elo` | f64 | the PL→Elo curve, offset 1200 / scale 400 |
//! | `riir-ai` `riir-games` ruliology `ParadigmRanking` | f64 | K=32, start 1000, scale 400, win/loss/**draw** |
//! | `riir-clippy` `src/elo.rs` (Issue 039) | f32 | K=32, base 1200, scale 400, win/loss |
//!
//! All four delegate here now; the expression trees below are verbatim the
//! union of their forms so every delegation is **bit-identical** to the copy
//! it replaced (the GOAT G1 gate — pinned by tests here AND by each
//! consumer's unchanged expectations).
//!
//! ## The two rating systems, one curve
//!
//! Pairwise (standard) Elo — [`expected`] + [`update`]/[`update_scored`] —
//! and batch Plackett-Luce agree at the fixed point: a pair playing at win
//! rate `p` equilibrates where the expected score equals `p`, a rating gap
//! of `scale · log10(W/L)` — exactly [`elo_from_lambda`]'s curve. The
//! conversion is the bridge between the two systems, which is why it lives
//! beside the pairwise math (pinned by
//! [`equilibrium_sits_on_the_pl_curve`][self::tests] in the test module).
//!
//! ## Conventions
//!
//! [`STANDARD_K`] / [`STANDARD_BASE`] / [`STANDARD_SCALE`] are the
//! standard-chess values all four copies used for K / base / scale (the
//! base-1200 half is the PL conversion's `elo_offset`; the arena and
//! ruliology rankers seed players at 1000 — a per-consumer choice, carried
//! there by their own constants). Draw scoring (`0.5`) is the house
//! tournament convention (`katgpt-core/induced_cwm/tournament.rs`).
//!
//! Pure modelless arithmetic — no deps, no allocs, no_std-compatible ops
//! only (`powf`/`log10`/`max`). Zero-cost-unless-invoked.
//!
//! rating — Elo + Plackett-Luce rating primitives (Issue 686, promoted from
//! four in-stack copies: katgpt-pruners arena EloCalculator + proof
//! lambda_to_elo, riir-ai riir-games ruliology ParadigmRanking, riir-clippy
//! src/elo.rs). Pure modelless arithmetic; bit-identical expression trees;
//! f64 canonical + f32 twins (riir-clippy's persisted ratings keep their
//! exact numerics). Zero-cost-unless-invoked.

/// Standard-chess K-factor (per-match maximum movement) — every in-stack
/// copy used 32.
pub const STANDARD_K: f64 = 32.0;

/// Standard-chess base rating — the PL conversion's `elo_offset` and
/// riir-clippy 039's unrated default.
pub const STANDARD_BASE: f64 = 1200.0;

/// Standard-chess Elo scale (Elo per log10 unit).
pub const STANDARD_SCALE: f64 = 400.0;

/// Expected score of `a` vs `b`: `1 / (1 + 10^((b-a)/scale))`.
///
/// The verbatim expression tree of all four former copies (f64 half).
#[must_use]
pub fn expected(a: f64, b: f64, scale: f64) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf((b - a) / scale))
}

/// One win/loss match update — returns `(new_a, new_b)`.
///
/// Bit-identical to the arena `EloCalculator::update` expression tree it
/// replaced (the zero-sum property holds up to float rounding; pinned by
/// tests here and by the consumer suites).
#[must_use]
pub fn update(a: f64, b: f64, a_won: bool, k: f64, scale: f64) -> (f64, f64) {
    update_scored(a, b, if a_won { 1.0 } else { 0.0 }, k, scale)
}

/// One scored match update — `score_a` in `[0,1]` (1 = win, 0.5 = draw,
/// 0 = loss; the house tournament convention).
///
/// Covers [`update`]'s binary form exactly (`score_a ∈ {0,1}` makes
/// `1.0 - score_a` exact) and adds the draw case `riir-games` ruliology
/// needs. The expression tree matches both former copies.
#[must_use]
pub fn update_scored(a: f64, b: f64, score_a: f64, k: f64, scale: f64) -> (f64, f64) {
    let expected_a = expected(a, b, scale);
    let expected_b = 1.0 - expected_a;
    (
        a + k * (score_a - expected_a),
        b + k * ((1.0 - score_a) - expected_b),
    )
}

/// The Plackett-Luce → Elo conversion curve:
/// `elo = base + scale · log10(max(λ, 1e-10))`.
///
/// λ (the PL strength parameter, Gamma-posterior mean in the Gibbs rater)
/// maps onto the Elo scale at the SAME fixed point pairwise Elo converges
/// to — see the module doc. The `1e-10` clamp avoids `-inf` from `log10`
/// (verbatim from the former `katgpt-pruners` `lambda_to_elo`).
#[must_use]
pub fn elo_from_lambda(lambda_mean: f64, base: f64, scale: f64) -> f64 {
    let clamped = lambda_mean.max(1e-10);
    base + scale * clamped.log10()
}

/// f32 twin of [`expected`] — riir-clippy 039's persisted f32 ratings keep
/// their exact former numerics (single-precision throughout, no f64→f32
/// double rounding).
#[must_use]
pub fn expected_f32(a: f32, b: f32, scale: f32) -> f32 {
    1.0 / (1.0 + 10.0_f32.powf((b - a) / scale))
}

/// f32 twin of [`update`] — see [`expected_f32`] on precision.
#[must_use]
pub fn update_f32(a: f32, b: f32, a_won: bool, k: f32, scale: f32) -> (f32, f32) {
    let expected_a = expected_f32(a, b, scale);
    let expected_b = 1.0 - expected_a;
    let score_a = if a_won { 1.0 } else { 0.0 };
    (
        a + k * (score_a - expected_a),
        b + k * ((1.0 - score_a) - expected_b),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_half_at_equal_ratings() {
        // Exactly representable at any base — all four copies' symmetry
        // fixture.
        assert_eq!(expected(1000.0, 1000.0, 400.0), 0.5);
        assert_eq!(expected(1200.0, 1200.0, 400.0), 0.5);
        assert_eq!(expected_f32(1200.0, 1200.0, 400.0), 0.5);
        assert!(expected(1200.0, 800.0, 400.0) > 0.5);
        // The logistic curve at ±scale: a one-scale gap is 10:1 odds.
        let e = expected(1400.0, 1000.0, 400.0);
        assert!((e - 10.0 / 11.0).abs() < 1e-12);
    }

    #[test]
    #[allow(clippy::float_cmp)] // ±16 is exactly representable — equality IS the fixture
    fn one_match_from_base_moves_exactly_16() {
        // E = 0.5 exactly → ±K/2 with K=32 → ±16 exactly representable in
        // BOTH precisions (riir-clippy's hand fixture; holds at either base).
        let (a, b) = update(1200.0, 1200.0, true, STANDARD_K, 400.0);
        assert_eq!(a, 1216.0);
        assert_eq!(b, 1184.0);
        let (a, b) = update(1000.0, 1000.0, false, STANDARD_K, 400.0);
        assert_eq!(a, 984.0);
        assert_eq!(b, 1016.0);
        let (a, b) = update_f32(1200.0, 1200.0, true, 32.0, 400.0);
        assert_eq!(a, 1216.0);
        assert_eq!(b, 1184.0);
    }

    #[test]
    fn updates_conserve_total_up_to_rounding() {
        for &(ra, rb, won) in &[
            (1000.0, 1000.0, true),
            (1300.0, 900.0, false),
            (1180.0, 1260.0, true),
            (1216.0, 1184.0, false),
        ] {
            let (na, nb) = update(ra, rb, won, STANDARD_K, 400.0);
            assert!(
                (na + nb - (ra + rb)).abs() < 1e-9,
                "conservation violated: {ra}/{rb} -> {na}/{nb}"
            );
        }
        // Draws too, at any rating pair.
        for &(ra, rb) in &[(1000.0, 1000.0), (1200.0, 900.0)] {
            let (na, nb) = update_scored(ra, rb, 0.5, STANDARD_K, 400.0);
            assert!((na + nb - (ra + rb)).abs() < 1e-9);
        }
    }

    #[test]
    fn upset_moves_more_than_expected_result() {
        // The arena calculator's own property: the underdog gains more from
        // an upset than the favorite gains from an expected win.
        let (fav_gain, _) = update(1200.0, 800.0, true, STANDARD_K, 400.0);
        let (_, upset_gain) = update(800.0, 1200.0, true, STANDARD_K, 400.0);
        assert!(upset_gain - 800.0 > fav_gain - 1200.0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact-identity IS the draw fixture
    fn draw_preserves_ratings_at_equal_and_is_flat() {
        // At equal ratings a draw is a no-op exactly (E = 0.5 = score).
        let (a, b) = update_scored(1000.0, 1000.0, 0.5, STANDARD_K, 400.0);
        assert_eq!((a, b), (1000.0, 1000.0));
        // Away from equal, a draw pulls both toward the mean (score below
        // expectation for the favorite) — opposite, conserving, symmetric:
        // a 400-gap favorite (E = 10/11) drops K·(0.5 − 10/11) ≈ −13.09.
        let (a, b) = update_scored(1200.0, 800.0, 0.5, STANDARD_K, 400.0);
        assert!(
            (a - 1_186.909_090_909).abs() < 1e-6,
            "favorite drifts down: {a}"
        );
        assert!(
            (b - 813.090_909_091).abs() < 1e-6,
            "underdog drifts up: {b}"
        );
        assert!((a + b - 2000.0).abs() < 1e-9);
    }

    #[test]
    #[allow(clippy::float_cmp)] // binary == scored-extremes bit-identity IS the fixture
    fn update_binary_equals_scored_extremes() {
        for &(ra, rb) in &[(1000.0, 1000.0), (1216.0, 1184.0), (900.0, 1300.0)] {
            assert_eq!(
                update(ra, rb, true, 32.0, 400.0),
                update_scored(ra, rb, 1.0, 32.0, 400.0)
            );
            assert_eq!(
                update(ra, rb, false, 32.0, 400.0),
                update_scored(ra, rb, 0.0, 32.0, 400.0)
            );
        }
    }

    #[test]
    fn elo_from_lambda_fixtures() {
        // The PL conversion fixtures from the former katgpt-pruners tests.
        let (base, scale) = (1200.0, 400.0);
        // λ = 1 (no evidence) → exactly the base.
        assert_eq!(elo_from_lambda(1.0, base, scale), base);
        // λ = 10 → base + scale; λ = 0.1 → base − scale.
        assert_eq!(elo_from_lambda(10.0, base, scale), base + scale);
        assert_eq!(elo_from_lambda(0.1, base, scale), base - scale);
        // Clamp: λ below 1e-10 never reaches −inf.
        assert_eq!(
            elo_from_lambda(1e-12, base, scale),
            base + scale * 1e-10_f64.log10()
        );
        assert!(elo_from_lambda(0.0, base, scale).is_finite());
    }

    #[test]
    fn f32_f64_agreement_over_sequence() {
        // The "two math paths" gate (riir-clippy's arena cross-check,
        // promoted to the primitive's own test): same mixed sequence in
        // both precisions — the only drift is f32 rounding.
        let seq = [
            true, false, false, true, true, true, false, true, true, true, false, true,
        ];
        let (mut ra64, mut rb64) = (1200.0f64, 1200.0f64);
        let (mut ra32, mut rb32) = (1200.0f32, 1200.0f32);
        for &won in &seq {
            let (na, nb) = update(ra64, rb64, won, STANDARD_K, 400.0);
            ra64 = na;
            rb64 = nb;
            let (na, nb) = update_f32(ra32, rb32, won, 32.0, 400.0);
            ra32 = na;
            rb32 = nb;
            assert!((f64::from(ra32) - ra64).abs() < 0.01, "a: {ra32} vs {ra64}");
            assert!((f64::from(rb32) - rb64).abs() < 0.01, "b: {rb32} vs {rb64}");
        }
    }

    #[test]
    fn equilibrium_sits_on_the_pl_curve() {
        // Standard Elo's equilibrium IS elo_from_lambda's curve: a pair at
        // win rate p equilibrates at gap scale·log10(W/L). 100 cycles of
        // (1 loss, 3 wins) = 75% → gap 400·log10(3) ≈ 190.8. Discrete
        // fixed-order matches oscillate AROUND the fixed point (band
        // ~±K·E), so the gate is the band, not the point.
        let (mut ra, mut rb) = (1200.0f64, 1200.0f64);
        for _ in 0..100 {
            let (a, b) = update(ra, rb, false, STANDARD_K, 400.0);
            ra = a;
            rb = b;
            for _ in 0..3 {
                let (a, b) = update(ra, rb, true, STANDARD_K, 400.0);
                ra = a;
                rb = b;
            }
        }
        let gap = ra - rb;
        let want = 400.0 * (3.0_f64 / 1.0).log10();
        assert!(
            (gap - want).abs() < 30.0,
            "gap {gap} vs PL fixed point {want} (band ±K·E)"
        );
        // And the expected score at equilibrium is the win rate itself.
        let e = expected(ra, rb, 400.0);
        assert!((e - 0.75).abs() < 0.05, "expected {e} vs win rate 0.75");
        // The same point ON the curve: a λ ratio of 3 maps to the same gap.
        let on_curve = elo_from_lambda(3.0, STANDARD_BASE, STANDARD_SCALE)
            - elo_from_lambda(1.0, STANDARD_BASE, STANDARD_SCALE);
        assert!((on_curve - want).abs() < 1e-9);
    }
}
