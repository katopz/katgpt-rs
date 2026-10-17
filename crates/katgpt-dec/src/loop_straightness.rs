//! Loop-straightness trajectory diagnostics — the DEC-native stalemate signal
//! (Issue 922; LiFT appendix C, arXiv:2610.05538, modelless extraction).
//!
//! For any iterative refinement trajectory `x_0 → x_1 → … → x_K` with deltas
//! `Δ_k = x_{k+1} − x_k` on grid weights `δ_k > 0`:
//!
//! ```text
//! A_disc = ½·Σ_k ‖Δ_k‖²/δ_k  ≥  ½·‖x_K − x_0‖²/Σ_k δ_k   (Cauchy–Schwarz)
//! η      = ‖x_K − x_0‖² / (Σ_k ‖Δ_k‖²/δ_k · Σ_k δ_k)  ∈ (0, 1]
//! ```
//!
//! (The ½ cancels in the ratio; [`path_action`] reports the un-halved sum so
//! the floor reads directly off its two outputs.)
//!
//! - `η = 1` **iff** the path is straight at constant grid-speed
//!   (`Δ_k = c·δ_k·u` for a fixed unit direction `u`) — exact in `f32` on
//!   dyadic grids, which is the G1 straightness pin.
//! - `1 − η` is the exact wasted-motion fraction (backtracking, orbiting,
//!   A/B/A oscillation): motion that cancelled itself.
//! - **Grid-invariant**: the identity holds on ANY nonuniform positive grid
//!   (knot-wise minimizer of the weighted action), so a nonuniform
//!   deliberation schedule admits the same diagnostic — `η` normalizes by
//!   `Σδ` internally and never assumes the caller's weights sum to 1.
//! - **Boundary-vs-interior**: the floor reads only the two endpoints; the
//!   cost samples K interior knots. The summation axis is K, not the latent
//!   dim — no curse of dimensionality (the manifold-geometry rule).
//!
//! This is the PRIMITIVE half; consumers (riir-ai Issue 1037: deliberation
//! cadence, cgsp collapse bridge, refine self-evolve, reflex cascade) feed it
//! trajectory logs the runtime already traverses. The stalemate LAW stays the
//! consumer's: a signal, never a counter-forced action — `1 − η` feeds a
//! sigmoid drive there, it never gates here.
//!
//! Signal-diff (pre-implementation, recorded in Issue 922): every shipped
//! cousin is a COUNTER or an ENTROPY read, none is geometric —
//! `swarm/deliberation.rs` counts consecutive flee ticks (the gapped
//! oscillator), `cgsp_runtime` reads solver entropy collapse
//! (`EntropyCollapse`/`CollapseSignal`), `latent_functor` classifies crowd
//! regimes (`NoiseSustainedOscillation`), refine `self_evolve` reads
//! trajectory outcomes. No workspace code computes the action floor, η, or a
//! chord projection over a refinement trajectory.
//!
//! Zero-alloc by construction: caller-supplied slices only, no `Vec`, no
//! iteration-order dependence beyond the path's own order (which IS the
//! signal). NaN propagates (a consumer checking `η.is_finite()` is cheaper
//! than this module guessing a sentinel).

/// `(Σ_k ‖Δ_k‖²/δ_k, ‖x_K − x_0‖², Σ_k δ_k)` for a flattened trajectory.
///
/// `points` holds K+1 knots of `stride` `f32`s each; `weights` holds K grid
/// weights, one per delta. Weights MUST be positive and finite — a zero or
/// negative weight makes `‖Δ‖²/δ` meaningless; that is a caller-contract
/// violation, not a case this fn adjudicates (NaN/inf propagate on violation,
/// which is checkable from the output).
///
/// Returns the un-halved weighted action numerator, the squared chord norm,
/// and the weight sum, so both [`action_efficiency`] and any consumer that
/// wants `A_disc = ½·numerator` or the raw floor read from ONE pass.
pub fn path_action(points: &[f32], stride: usize, weights: &[f32]) -> (f32, f32, f32) {
    let knots = points.len() / stride;
    debug_assert!(weights.len() + 1 == knots, "one weight per delta");
    let mut weighted = 0.0f32;
    let mut weight_sum = 0.0f32;
    for (k, w) in weights.iter().enumerate().take(knots.saturating_sub(1)) {
        let base = k * stride;
        let mut delta_sq = 0.0f32;
        for d in 0..stride {
            let dx = points[base + stride + d] - points[base + d];
            delta_sq += dx * dx;
        }
        weighted += delta_sq / w;
        weight_sum += w;
    }
    let mut chord_sq = 0.0f32;
    if knots >= 2 {
        let last = (knots - 1) * stride;
        for d in 0..stride {
            let dx = points[last + d] - points[d];
            chord_sq += dx * dx;
        }
    }
    (weighted, chord_sq, weight_sum)
}

/// Action efficiency `η ∈ (0, 1]` — 1 iff straight at constant grid-speed.
///
/// `η = chord² / (Σ(‖Δ_k‖²/δ_k) · Σδ)` — the ratio the Cauchy–Schwarz floor
/// (`chord² ≤ Σ(‖Δ‖²/δ)·Σδ`) bounds by 1 on ANY positive grid; normalizing
/// by `Σδ` is what makes the read grid-invariant when the caller's weights
/// do not sum to 1. Degenerate shapes by pinned convention: a single knot
/// (K = 0, no motion) is **1.0** (nothing wasted); a closed loop (chord = 0)
/// is **0.0** (all motion cancelled — the A/B/A oscillator's exact score);
/// a fully degenerate zero-motion path is 1.0 (denominator 0, and the limit
/// is the straight one).
pub fn action_efficiency(points: &[f32], stride: usize, weights: &[f32]) -> f32 {
    let (weighted, chord_sq, weight_sum) = path_action(points, stride, weights);
    if weighted == 0.0 {
        return 1.0;
    }
    chord_sq / (weighted * weight_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dyadic nonuniform grid along +x: Δ_k = δ_k exactly → η bit-exact 1.0.
    #[test]
    fn straight_constant_grid_speed_is_one_bit_exact() {
        let points = [0.0f32, 0.5, 0.75, 0.875, 1.0];
        let weights = [0.5f32, 0.25, 0.125, 0.125];
        assert_eq!(action_efficiency(&points, 1, &weights), 1.0f32);
        // 2-D diagonal straight path, same dyadic split on BOTH coords
        // (5 knots × 2 dims — one weight per delta).
        let pts2 = [0.0f32, 0.0, 0.5, 0.5, 0.75, 0.75, 0.875, 0.875, 1.0, 1.0];
        assert_eq!(action_efficiency(&pts2, 2, &weights), 1.0f32);
    }

    /// A/B/A oscillator that RETURNS to start: chord = 0 → η = 0 exactly.
    /// (The OPEN form 0→1→0→1 ends where the last flight left it — chord ≠ 0
    /// — and its η = 1/9 is a different, also-correct pin, kept below.)
    #[test]
    fn closed_aba_loop_scores_zero() {
        let points = [0.0f32, 1.0, 0.0, 1.0, 0.0];
        let weights = [0.25f32; 4];
        assert_eq!(action_efficiency(&points, 1, &weights), 0.0f32);
    }

    /// The OPEN three-flight oscillator: 0→1→0→1, uniform δ. Hand-derived:
    /// Σ‖Δ‖²/δ = 3/(1/3) = 9, chord² = 1, Σδ = 1 → η = 1/9.
    #[test]
    fn open_aba_oscillator_scores_one_ninth() {
        let points = [0.0f32, 1.0, 0.0, 1.0];
        let weights = [1.0f32 / 3.0; 3];
        let eta = action_efficiency(&points, 1, &weights);
        assert!((eta - 1.0 / 9.0).abs() < 1e-6, "eta={eta}");
    }

    /// Partial backtrack 0 → 1 → 0.5, uniform δ: Σ = 1/δ + 0.25/δ = 2.5 at
    /// δ = ½, chord² = 0.25 → η = 0.25/2.5 = 0.1 (hand-derived; the naive
    /// 0.2 forgets the /δ doubling of BOTH terms).
    #[test]
    fn partial_backtrack_score_is_exact() {
        let points = [0.0f32, 1.0, 0.5];
        let weights = [0.5f32, 0.5];
        assert_eq!(action_efficiency(&points, 1, &weights), 0.1f32);
    }

    /// The Cauchy–Schwarz floor (η ≤ 1 + ε) holds over seeded random walks,
    /// uniform AND nonuniform grids, multiple dims — the G1 identity sweep.
    #[test]
    fn cauchy_schwarz_floor_holds_on_random_walks() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut lcg = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for case in 0..500 {
            let dim = 1 + (case % 4);
            let k = 2 + (case % 7);
            let mut points = vec![0.0f32; (k + 1) * dim];
            for knot in 1..=k {
                for d in 0..dim {
                    points[knot * dim + d] =
                        points[(knot - 1) * dim + d] + (lcg() % 97) as f32 / 97.0 - 0.5;
                }
            }
            let weights: Vec<f32> = (0..k)
                .map(|i| {
                    if case % 2 == 0 {
                        1.0
                    } else {
                        1.0 / (i as f32 + 1.0)
                    }
                })
                .collect();
            let eta = action_efficiency(&points, dim, &weights);
            assert!(eta <= 1.0 + 1e-5, "floor violated: eta={eta}");
            assert!(eta >= 0.0, "negative eta={eta}");
        }
    }

    /// Grid-invariance half: dyadic midpoint refinement of a straight path
    /// keeps η bit-exact 1.0 (the knot-wise minimizer property).
    #[test]
    fn dyadic_refinement_of_straight_path_stays_one() {
        let coarse = [0.0f32, 0.5, 1.0];
        let w_coarse = [0.5f32, 0.5];
        let fine = [0.0f32, 0.25, 0.5, 0.75, 1.0];
        let w_fine = [0.25f32; 4];
        assert_eq!(action_efficiency(&coarse, 1, &w_coarse), 1.0f32);
        assert_eq!(action_efficiency(&fine, 1, &w_fine), 1.0f32);
    }

    /// Degenerate shapes are pinned, not accidental.
    #[test]
    fn degenerate_shapes_are_pinned() {
        let single = [3.0f32, -1.0];
        assert_eq!(action_efficiency(&single, 2, &[]), 1.0f32);
        let frozen = [1.0f32, 1.0, 1.0];
        assert_eq!(action_efficiency(&frozen, 1, &[0.5, 0.5]), 1.0f32);
        // NaN propagates — the consumer's check, not our sentinel.
        let nan_path = [0.0f32, f32::NAN, 1.0];
        assert!(action_efficiency(&nan_path, 1, &[0.5, 0.5]).is_nan());
    }

    /// path_action's raw outputs: A_disc = ½·Σ and the floor ½·chord²/Σδ read
    /// off one call (the un-halved form the ratio cancels in).
    #[test]
    fn path_action_reports_floor_and_cost() {
        let points = [0.0f32, 1.0, 0.5];
        let (weighted, chord_sq, weight_sum) = path_action(&points, 1, &[0.5, 0.5]);
        assert_eq!(chord_sq, 0.25);
        assert_eq!(weight_sum, 1.0);
        assert_eq!(weighted, 2.5); // 1/0.5 + 0.25/0.5
        assert!(weighted * weight_sum >= chord_sq); // the floor, un-halved
    }
}
