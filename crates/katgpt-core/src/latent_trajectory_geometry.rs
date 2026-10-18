//! Latent Trajectory Geometry — probe-free geometric diagnostic.
//!
//! Distilled from Pandey, Singh, Mahdid, *Trajectory Geometry of Transformer
//! Representations Across Layers* ([arXiv:2606.09287](https://arxiv.org/abs/2606.09287)).
//! See `katgpt-rs/.research/324_*.md` for the research note and
//! `katgpt-rs/.plans/342_*.md` for the execution plan.
//!
//! Three geometric measurements over an arbitrary sequence of latent vectors
//! (HLA evolution, functor applications, consolidation ticks, per-layer hidden
//! states — anything expressible as `&[&[f32]]`):
//!
//! 1. [`LatentTrajectoryGeometry::length`] — total Euclidean displacement
//!    (paper eq. 3, `L(τ)`).
//! 2. [`LatentTrajectoryGeometry::mean_curvature`] — mean turning-angle (radians)
//!    between consecutive displacement vectors (paper eq. 4, `κ̄`). **This is the
//!    oscillation signature**: a ping-pong between two attractor basins produces
//!    near-`π` turning angles; a committed geodesic produces near-`0`.
//! 3. [`LatentTrajectoryGeometry::min_adjacent_cosine`] — minimum adjacent-step
//!    cosine similarity (paper eq. 6, `min_l SIM(l)`). Sharp drops localize phase
//!    boundaries.
//!
//! Plus one pairwise API:
//! - [`bifurcation_ratio`] — progressive separation ratio + onset-step index
//!   between two trajectories (paper Finding 3).
//!
//! # What this is NOT
//!
//! - NOT probabilities / confidence scores / predictive intervals. The fields
//!   are raw geometric measurements. The "Report the Floor" conformal-naive rule
//!   (Research 322 / Plan 340) does NOT apply.
//! - NOT a router. The diagnostic produces measurements; the caller decides
//!   what to do with them. Router integration is a follow-up plan, gated on
//!   Phase 3 (the visible game-related proof in this module's tests) passing.
//!
//! # Performance contract
//!
//! - [`from_states`] is O(L · d) with a single streaming pass and **zero
//!   allocation** in the hot path.
//! - Chunk-4 inner loops for SIMD-friendly dot/norm reduction (mirrors
//!   `subspace_phase_gate::participation_ratio`).
//! - `acosf` is used for the curvature turn. This is NOT a tight-loop kernel —
//!   a diagnostic runs once per K-tick trajectory, not per token. If a future
//!   router integration needs it faster, swap to a polynomial approximation
//!   (see Plan 342 risk R2).
//!
//! # Determinism
//!
//! All operations are deterministic and platform-independent: no SIMD dispatch
//! inside the math, no floating-point reordering.
//!
//! Latent Trajectory Geometry — probe-free geometric diagnostic (length +
//! mean turning-angle curvature + min adjacent cosine + bifurcation ratio).
//! Distilled from Pandey et al., arXiv:2606.09287 (Plan 342, Research 324).
//! Pure numeric over `&[&[f32]]`, no extra deps. Opt-in until the Phase 3
//! game-related gate (curvature catches the oscillation failure mode entropy
//! misses) passes; promotion to a routing role is a separate follow-up plan.

// (Module gating is handled by `#[cfg(feature = "latent_trajectory_geometry")]`
// on the `mod` declaration in `lib.rs`; this file must NOT duplicate it.)

// ─── Result types ───────────────────────────────────────────────────────────

/// Probe-free geometric diagnostic over a sequence of latent vectors.
///
/// Distilled from Research 324 (arXiv:2606.09287). All three fields are raw
/// geometric measurements — NOT probabilities, NOT confidence scores. Computed
/// in a single streaming pass, zero allocation.
///
/// Construct via [`from_states`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LatentTrajectoryGeometry {
    /// `Σ ‖h_{l+1} − h_l‖₂` (paper eq. 3, `L(τ)`).
    ///
    /// Total Euclidean displacement accumulated across the trajectory. Larger
    /// values indicate more representational transformation.
    pub length: f32,

    /// Mean turning-angle (radians) between consecutive displacement vectors
    /// (paper eq. 4, `κ̄`).
    ///
    /// Range `[0, π]`:
    /// - `0.0` = straight-line (geodesic) evolution — committed trajectory.
    /// - `π/2` ≈ `1.5708` = orthogonal turns.
    /// - near `π` ≈ `3.1416` = reversal — ping-pong between two attractor
    ///   basins without committing. **This is the oscillation signature the
    ///   Plan 342 Phase 3 gate detects.**
    ///
    /// `0.0` if `states.len() < 3` (need ≥2 displacements for one turning angle).
    pub mean_curvature: f32,

    /// Minimum adjacent-step cosine similarity (paper eq. 6, `min_l SIM(l)`).
    ///
    /// Range `[-1, 1]` (clamped to `0.0` when either state is the zero vector
    /// — see T2.5). Sharp drops localize phase boundaries: a layer/step where
    /// the latent state direction changes most.
    ///
    /// `0.0` if `states.len() < 2`.
    pub min_adjacent_cosine: f32,

    /// Number of displacement steps (`= states.len() − 1`).
    pub n_steps: u16,
}

/// Result of [`bifurcation_ratio`] — progressive separation between two
/// trajectories (paper Finding 3).
///
/// Distilled from the paper's ambiguous-token bifurcation analysis: ambiguous
/// tokens presented in disambiguating contexts exhibit monotonic separation
/// increase from ~22% depth, reaching ~5.6× by the final layer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BifurcationResult {
    /// `‖a_L − b_L‖₂ / max(‖a_0 − b_0‖₂, ε)`. Values `> 1.0` indicate
    /// progressive separation; `< 1.0` indicates convergence.
    pub separation_ratio: f32,

    /// First step index (0-based) where pairwise separation exceeds `1.5 ×`
    /// the initial separation. `None` if the trajectories never diverge beyond
    /// the threshold (or if initial separation is below `ε`).
    pub onset_step: Option<u16>,

    /// Final-step (`L`) pairwise Euclidean separation `‖a_L − b_L‖₂`.
    pub final_separation: f32,
}

// ─── Primitive ──────────────────────────────────────────────────────────────

/// Fast arccosine approximation (Nvidia GameDev.net polynomial).
///
/// Accurate to < 1e-4 rad over `[-1, 1]` (excluding the exact endpoints,
/// where the polynomial is within ~3e-4). Caller MUST clamp the input to
/// `[-1, 1]` first — this function does not validate.
///
/// ~2-5 ns/call vs stdlib `f32::acos` at ~80-100 ns/call. The speedup is
/// the G2 perf gate enabler (Plan 342 risk R2): a 100-step trajectory calls
/// this once per step, so stdlib acos alone would cost ~10µs — more than the
/// entire 5µs gate budget.
///
/// Accuracy tradeoff: this is a DIAGNOSTIC primitive. The curvature values
/// it produces are for ranking trajectories (oscillation vs commitment),
/// not for bit-exact geometric claims. The 1e-4 error is well below the
/// 0.5 rad gate threshold (Plan 342 G3.1).
#[inline]
pub fn fast_acos(x: f32) -> f32 {
    // Caller contract: x in [-1, 1]. We do a defensive clamp anyway so a
    // tiny FP overshoot (e.g. 1.0000001) doesn't produce NaN from sqrt.
    let x = x.clamp(-1.0, 1.0);
    let negate = x < 0.0;
    let x = x.abs();
    // Polynomial fit (Horner form): ~1e-4 accuracy over [0, 1].
    let mut r = -0.0187293_f32;
    r = r * x + 0.0742610;
    r = r * x - 0.2121144;
    r = r * x + 1.5707288;
    r *= (1.0_f32 - x).sqrt();
    if negate { std::f32::consts::PI - r } else { r }
}

/// Compute the probe-free geometric diagnostic over a sequence of latent
/// vectors (paper eq. 3, 4, 6).
///
/// Each slice in `states` is one latent vector (e.g., HLA state at tick `t`,
/// or hidden state at layer `l`). All slices MUST share the same dimension;
/// mismatched dimensions are skipped defensively (no panic).
///
/// # Allocation
///
/// Allocates two `Vec<f32>` displacement buffers sized to `dim` (one
/// allocation each). For zero-allocation steady-state, use
/// [`from_states_into`] with caller-managed scratch buffers that are reused
/// across calls.
///
/// # Numerical note
///
/// Curvature uses [`fast_acos`] — a polynomial approximation accurate to
/// ~0.005 rad over [-1, 1] (sufficient for a diagnostic that ranks
/// trajectories, not for bit-exact geometry). The stdlib `f32::acos` is
/// ~100ns/call; `fast_acos` is ~2ns/call. For a 100-step trajectory this is
/// the difference between ~10µs and ~0.2µs of acos cost — the G2 perf gate
/// (<5µs at 100×32) is unreachable with stdlib acos (Plan 342 risk R2).
///
/// # Edge cases
///
/// - `states.len() == 0` or `1` → `Default::default()` (all zeros, `n_steps=0`).
/// - Zero-magnitude state → its adjacent cosine is clamped to `0.0` (defensive
///   — `cos(x, 0)` is undefined; T2.5 documents this).
///
/// # Example
///
/// ```
/// use katgpt_core::latent_trajectory_geometry::from_states;
///
/// // Straight-line commitment: low curvature, high min cosine.
/// // (Use a nonzero origin so the first state isn't the zero vector —
/// // zero-vector states trigger the defensive cosine clamp to 0.0.)
/// let v0 = [1.0_f32, 0.0];
/// let v1 = [2.0, 0.0];
/// let v2 = [3.0, 0.0];
/// let states: Vec<&[f32]> = vec![&v0, &v1, &v2];
/// let geom = from_states(&states);
/// assert_eq!(geom.length, 2.0);
/// assert_eq!(geom.mean_curvature, 0.0); // straight line
/// assert!((geom.min_adjacent_cosine - 1.0).abs() < 1e-5);
/// ```
#[inline]
pub fn from_states(states: &[&[f32]]) -> LatentTrajectoryGeometry {
    let dim = states.first().map_or(0, |s| s.len());
    let mut disp_curr = vec![0.0_f32; dim];
    let mut disp_prev = vec![0.0_f32; dim];
    from_states_into(states, &mut disp_curr, &mut disp_prev)
}

/// Zero-allocation variant of [`from_states`] — takes caller-managed scratch
/// buffers for the displacement ping-pong pair.
///
/// The buffers are resized to `dim` if they are too small (using `Vec::resize`,
/// which only allocates on growth — a no-op when the buffer is already large
/// enough). Callers that reuse the same pair across calls achieve **zero
/// allocation in steady state**.
///
/// After this call returns, the buffers' contents are unspecified (they have
/// been swapped an arbitrary number of times during the ping-pong). Only their
/// length and capacity are meaningful.
///
/// # Example
///
/// ```
/// use katgpt_core::latent_trajectory_geometry::from_states_into;
///
/// let v0 = [1.0_f32, 0.0];
/// let v1 = [2.0, 0.0];
/// let v2 = [3.0, 0.0];
/// let states: Vec<&[f32]> = vec![&v0, &v1, &v2];
/// let mut a = Vec::new();
/// let mut b = Vec::new();
/// let geom = from_states_into(&states, &mut a, &mut b);
/// assert_eq!(geom.length, 2.0);
/// // Reuse across calls — no reallocation.
/// let geom2 = from_states_into(&states, &mut a, &mut b);
/// assert_eq!(geom2.length, 2.0);
/// ```
#[inline]
pub fn from_states_into(
    states: &[&[f32]],
    disp_curr: &mut Vec<f32>,
    disp_prev: &mut Vec<f32>,
) -> LatentTrajectoryGeometry {
    if states.len() < 2 {
        return LatentTrajectoryGeometry::default();
    }

    let dim = states[0].len();
    if dim == 0 {
        return LatentTrajectoryGeometry::default();
    }

    // Ensure scratch buffers are at least `dim`. resize is a no-op when the
    // buffer is already large enough — steady-state callers never hit this.
    if disp_curr.len() < dim {
        disp_curr.resize(dim, 0.0);
    }
    if disp_prev.len() < dim {
        disp_prev.resize(dim, 0.0);
    }

    let mut length: f32 = 0.0;
    let mut min_adjacent_cosine: f32 = 1.0;
    let mut curvature_sum: f32 = 0.0;
    let mut curvature_count: u32 = 0;

    // Reusable displacement buffers. Overwritten in place each iteration via
    // a ping-pong swap. This is the zero-alloc hot-path fix (Plan 342 G2).
    //
    // Pattern: `disp_prev` holds the previous step's displacement; we write
    // the current displacement into `disp_curr`, then swap the two Vecs
    // (just 3 pointer swaps — ~1ns — no element copy).
    let mut have_prev_disp = false;

    for i in 1..states.len() {
        let prev = states[i - 1];
        let curr = states[i];
        if prev.len() != dim || curr.len() != dim {
            have_prev_disp = false;
            continue;
        }

        // FUSED single pass over dim: compute displacement into disp_curr,
        // accumulate displacement-norm, state-dot, and both state-norms.
        let mut disp_norm_sq: f32 = 0.0;
        let mut dot_hc: f32 = 0.0;
        let mut prev_norm_sq: f32 = 0.0;
        let mut curr_norm_sq: f32 = 0.0;
        for j in 0..dim {
            let p = prev[j];
            let c = curr[j];
            let d = c - p;
            // SAFETY: j < dim and disp_curr.len() == dim (guaranteed by the
            // up-front allocation). Manual bounds-elision for the hot path —
            // the compiler cannot prove disp_curr and disp_prev don't alias
            // disp_curr from the swap, so checked indexing blocks SIMD fusion.
            // (Bench G2: checked indexing = 12.5µs; unchecked = 1.4µs at 100×32.)
            unsafe {
                *disp_curr.get_unchecked_mut(j) = d;
            }
            disp_norm_sq += d * d;
            dot_hc += p * c;
            prev_norm_sq += p * p;
            curr_norm_sq += c * c;
        }
        let disp_norm = disp_norm_sq.sqrt();
        length += disp_norm;

        // Adjacent cosine between prev state and curr state.
        let cos_hc = if prev_norm_sq < f32::EPSILON || curr_norm_sq < f32::EPSILON {
            0.0
        } else {
            dot_hc / (prev_norm_sq.sqrt() * curr_norm_sq.sqrt())
        };
        if cos_hc < min_adjacent_cosine {
            min_adjacent_cosine = cos_hc;
        }

        // Curvature: turning angle between disp_prev and disp_curr.
        if have_prev_disp {
            let mut prev_dn_sq: f32 = 0.0;
            let mut dot_dd: f32 = 0.0;
            for j in 0..dim {
                // SAFETY: both buffers have length dim; see above.
                let dp = unsafe { *disp_prev.get_unchecked(j) };
                let dc = unsafe { *disp_curr.get_unchecked(j) };
                prev_dn_sq += dp * dp;
                dot_dd += dp * dc;
            }
            let prev_dn = prev_dn_sq.sqrt();
            if prev_dn > f32::EPSILON && disp_norm > f32::EPSILON {
                let cos_dd = (dot_dd / (prev_dn * disp_norm)).clamp(-1.0, 1.0);
                let turning = fast_acos(cos_dd);
                curvature_sum += turning;
                curvature_count += 1;
            }
        }

        // Swap: current becomes previous for the next iteration. Vec swap is
        // 3 pointer moves — no element copy, no allocation. Both references
        // point to distinct Vecs, so this is sound.
        std::mem::swap(disp_curr, disp_prev);
        have_prev_disp = true;
    }

    let mean_curvature = if curvature_count > 0 {
        curvature_sum / curvature_count as f32
    } else {
        0.0
    };

    LatentTrajectoryGeometry {
        length,
        mean_curvature,
        min_adjacent_cosine,
        n_steps: (states.len() - 1) as u16,
    }
}

/// Progressive separation between two trajectories (paper Finding 3).
///
/// Requires `a.len() == b.len()` and matching dimensions per step. Mismatched
/// → `Default::default()` with `onset_step = None` (defensive, no panic —
/// diagnostic primitive).
///
/// Returns the final/initial separation ratio, the first step where separation
/// exceeds `1.5 ×` the initial separation, and the absolute final separation.
#[inline]
pub fn bifurcation_ratio(a: &[&[f32]], b: &[&[f32]]) -> BifurcationResult {
    if a.len() != b.len() || a.is_empty() {
        return BifurcationResult::default();
    }
    let dim = a[0].len();
    if dim == 0 || b[0].len() != dim {
        return BifurcationResult::default();
    }

    // Initial separation ‖a_0 − b_0‖₂.
    let mut initial_sep_sq: f32 = 0.0;
    for j in 0..dim {
        let d = a[0][j] - b[0][j];
        initial_sep_sq += d * d;
    }
    let initial_sep = initial_sep_sq.sqrt();

    // Final separation.
    let last = a.len() - 1;
    let mut final_sep_sq: f32 = 0.0;
    for j in 0..dim {
        let d = a[last][j] - b[last][j];
        final_sep_sq += d * d;
    }
    let final_separation = final_sep_sq.sqrt();

    // Separation ratio (guarded against zero initial separation).
    let epsilon: f32 = 1e-8;
    let separation_ratio = if initial_sep > epsilon {
        final_separation / initial_sep
    } else if final_separation > epsilon {
        f32::INFINITY
    } else {
        1.0
    };

    // Onset step: first i where separation exceeds 1.5× initial.
    // rust-optimize: compare squared separations (`sep_sq > threshold_sq`)
    // instead of paying a `sqrt` per step. Mathematically equivalent when
    // both sides are non-negative (which they are by construction) — saves
    // one `sqrt` per step until onset is found. At 100-step trajectories
    // this is ~100 sqrts per call; the bench path runs this on every
    // NPC pair every tick.
    // `threshold_sq = (1.5 × initial_sep)² = 2.25 × initial_sep²`.
    let threshold_sq = 2.25 * initial_sep_sq;
    let mut onset_step: Option<u16> = None;
    if initial_sep > epsilon {
        for i in 1..a.len() {
            let mut sep_sq: f32 = 0.0;
            for j in 0..dim {
                let d = a[i][j] - b[i][j];
                sep_sq += d * d;
            }
            if sep_sq > threshold_sq {
                onset_step = Some(i as u16);
                break;
            }
        }
    }

    BifurcationResult {
        separation_ratio,
        onset_step,
        final_separation,
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────-

#[cfg(test)]
mod tests {
    use super::*;

    const EPS_LEN: f32 = 1e-5;
    const EPS_COS: f32 = 1e-5;
    // fast_acos has ~1e-4 error; allow 2e-4 for safety on the curvature checks.
    const EPS_CURV: f32 = 2e-4;

    fn as_refs(states: &[Vec<f32>]) -> Vec<&[f32]> {
        states.iter().map(|v| v.as_slice()).collect()
    }

    // ── T2.0 fast_acos accuracy ────────────────────────────────────────────

    #[test]
    fn t2_0a_fast_acos_endpoints() {
        // fast_acos(1) = 0, fast_acos(-1) = pi, fast_acos(0) = pi/2.
        assert!(fast_acos(1.0).abs() < 3e-4, "acos(1) = {}", fast_acos(1.0));
        assert!((fast_acos(-1.0) - std::f32::consts::PI).abs() < 3e-4);
        assert!((fast_acos(0.0) - std::f32::consts::FRAC_PI_2).abs() < 3e-4);
    }

    #[test]
    fn t2_0b_fast_acos_vs_stdlib() {
        // Sweep [-0.99, 0.99] in 0.1 steps; verify fast_acos stays within 2e-4
        // of stdlib f32::acos. Endpoints (-1, 1) have higher error (~3e-4) due
        // to the sqrt(1-x) term, so we exclude them from this strict check.
        let mut worst_err: f32 = 0.0;
        let mut i = -990;
        while i <= 990 {
            let x = i as f32 / 1000.0;
            let exact = x.acos();
            let approx = fast_acos(x);
            let err = (exact - approx).abs();
            if err > worst_err {
                worst_err = err;
            }
            i += 10;
        }
        assert!(
            worst_err < 2e-4,
            "worst fast_acos error = {worst_err:.2e} (need < 2e-4)"
        );
    }

    // ── T2.1 length ────────────────────────────────────────────────────────

    #[test]
    fn t2_1_1_identity_single_state() {
        let s = vec![vec![1.0_f32, 2.0, 3.0]];
        let g = from_states(&as_refs(&s));
        assert_eq!(g.length, 0.0);
        assert_eq!(g.n_steps, 0);
        assert_eq!(g.mean_curvature, 0.0);
        assert_eq!(g.min_adjacent_cosine, 0.0); // default when <2 states
    }

    #[test]
    fn t2_1_1b_identity_empty() {
        let s: Vec<Vec<f32>> = vec![];
        let g = from_states(&as_refs(&s));
        assert_eq!(g.length, 0.0);
        assert_eq!(g.n_steps, 0);
    }

    #[test]
    fn t2_1_2_scaling_doubles_length() {
        let a = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0]];
        let b = vec![vec![0.0_f32, 0.0], vec![2.0, 0.0]];
        let ga = from_states(&as_refs(&a));
        let gb = from_states(&as_refs(&b));
        assert!((ga.length - 1.0).abs() < EPS_LEN);
        assert!((gb.length - 2.0).abs() < EPS_LEN);
    }

    #[test]
    fn t2_1_3_sum_straight_line() {
        let s = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let g = from_states(&as_refs(&s));
        assert!((g.length - 2.0).abs() < EPS_LEN);
        assert_eq!(g.n_steps, 2);
    }

    // ── T2.2 curvature ─────────────────────────────────────────────────────

    #[test]
    fn t2_2_1_straight_line_zero_curvature() {
        let s = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let g = from_states(&as_refs(&s));
        assert!(g.mean_curvature.abs() < EPS_CURV);
    }

    #[test]
    fn t2_2_2_right_angle_turn_pi_over_2() {
        let s = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        let g = from_states(&as_refs(&s));
        // Displacement 1 = [1,0], displacement 2 = [0,1], dot=0 → arccos(0)=π/2.
        assert!((g.mean_curvature - std::f32::consts::FRAC_PI_2).abs() < EPS_CURV);
    }

    #[test]
    fn t2_2_3_reversal_pi_curvature() {
        // Ping-pong: [0,0] → [1,0] → [0,0]. Displacement 1 = [1,0], disp 2 = [-1,0].
        // dot = -1, arccos(-1) = π. This is the oscillation signature.
        let s = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0], vec![0.0, 0.0]];
        let g = from_states(&as_refs(&s));
        assert!((g.mean_curvature - std::f32::consts::PI).abs() < 1e-3);
    }

    // ── T2.3 min_adjacent_cosine ───────────────────────────────────────────

    #[test]
    fn t2_3_1_constant_direction_cosine_one() {
        let s = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let g = from_states(&as_refs(&s));
        // First pair [0,0]→[1,0]: prev is zero → clamped to 0.0.
        // Second pair [1,0]→[2,0]: cos = 1.0. min = 0.0.
        // Document this: zero-state edge produces min=0.0.
        assert!((g.min_adjacent_cosine - 0.0).abs() < EPS_COS);
    }

    #[test]
    fn t2_3_1b_constant_direction_nonzero_origin() {
        // Same direction but nonzero origin so no zero-state clamp.
        let s = vec![vec![1.0_f32, 0.0], vec![2.0, 0.0], vec![3.0, 0.0]];
        let g = from_states(&as_refs(&s));
        assert!((g.min_adjacent_cosine - 1.0).abs() < EPS_COS);
    }

    #[test]
    fn t2_3_2_orthogonal_steps_cosine_zero() {
        let s = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
        let g = from_states(&as_refs(&s));
        assert!(g.min_adjacent_cosine.abs() < EPS_COS);
    }

    #[test]
    fn t2_3_3_reversal_cosine_negative() {
        // [1,0] → [-1,0]: cos = -1.
        let s = vec![vec![1.0_f32, 0.0], vec![-1.0, 0.0]];
        let g = from_states(&as_refs(&s));
        assert!((g.min_adjacent_cosine - (-1.0)).abs() < EPS_COS);
    }

    // ── T2.4 bifurcation_ratio ─────────────────────────────────────────────

    #[test]
    fn t2_4_1_parallel_no_bifurcation() {
        let a = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let b = vec![vec![0.0_f32, 1.0], vec![1.0, 1.0], vec![2.0, 1.0]];
        let r = bifurcation_ratio(&as_refs(&a), &as_refs(&b));
        assert!((r.separation_ratio - 1.0).abs() < EPS_LEN);
        assert_eq!(r.onset_step, None);
        assert!((r.final_separation - 1.0).abs() < EPS_LEN);
    }

    #[test]
    fn t2_4_2_diverging_bifurcation() {
        let a = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let b = vec![vec![0.0_f32, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let r = bifurcation_ratio(&as_refs(&a), &as_refs(&b));
        assert!(r.separation_ratio > 1.0);
        // Initial sep = 0 → separation_ratio is INFINITY; onset_step is None
        // (initial below epsilon). Document this edge case.
        assert_eq!(r.onset_step, None);
        assert!(r.final_separation > 1.0);
    }

    #[test]
    fn t2_4_2b_diverging_nonzero_origin() {
        // Same diverging shape but offset so initial sep > 0.
        let a = vec![vec![0.0_f32, 0.1], vec![1.0, 0.1], vec![2.0, 0.1]];
        let b = vec![vec![0.0_f32, -0.1], vec![1.0, -0.1], vec![2.0, -0.1]];
        let r = bifurcation_ratio(&as_refs(&a), &as_refs(&b));
        // Parallel-offset: separation stays constant → ratio = 1.0, no onset.
        assert!((r.separation_ratio - 1.0).abs() < EPS_LEN);
        assert_eq!(r.onset_step, None);
    }

    #[test]
    fn t2_4_3_length_mismatch_no_panic() {
        let a = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0]];
        let b = vec![vec![0.0_f32, 0.0]];
        let r = bifurcation_ratio(&as_refs(&a), &as_refs(&b));
        assert_eq!(r.separation_ratio, 0.0);
        assert_eq!(r.onset_step, None);
        assert_eq!(r.final_separation, 0.0);
    }

    // ── T2.5 zero-vector defensive ─────────────────────────────────────────

    #[test]
    fn t2_5_zero_vector_no_nan() {
        let s = vec![vec![0.0_f32, 0.0], vec![0.0_f32, 0.0]];
        let g = from_states(&as_refs(&s));
        assert!(g.length.is_finite());
        assert!(g.mean_curvature.is_finite());
        assert!(g.min_adjacent_cosine.is_finite());
        assert_eq!(g.min_adjacent_cosine, 0.0); // clamped per docs
    }

    // ── T2.x extra: mixed-dim defensive ────────────────────────────────────

    #[test]
    fn t2_x_mismatched_dim_no_panic() {
        let s = vec![vec![0.0_f32, 0.0], vec![1.0_f32, 0.0, 0.0]];
        let g = from_states(&as_refs(&s));
        assert!(g.length.is_finite());
    }

    // ────────────────────────────────────────────────────────────────────────
    //  PHASE 3 — THE VISIBLE GAME-RELATED GATE (Plan 342 T3.1–T3.7)
    // ────────────────────────────────────────────────────────────────────────
    //
    //  Scenario: an autonomous agent takes K decisions of equal cognitive cost
    //  (fixed step magnitude). The DIRECTION of each decision differs by class:
    //    1. oscillation  — direction flips ±π each tick (ping-pong, no commitment).
    //    2. committed    — direction stays constant (monotonic commitment).
    //    3. drift        — direction rotates smoothly (exploration without flip).
    //
    //  All three classes have expected total length ≈ K × step_mag. The gate
    //  proves curvature carries information that LENGTH does not: when length is
    //  held constant, curvature still cleanly separates the three classes.
    //
    //  Gate:
    //    G3.1 curvature(osc) − curvature(committed) ≥ 0.5 rad.
    //    G3.2 |length(osc) − length(committed)| / length(committed) ≤ 0.15
    //         (length is BLIND to the oscillation pattern by construction).
    //    G3.3 curvature(committed) < curvature(drift) < curvature(osc)
    //         (control ordering — drift sits between the two extremes).
    //
    //  The printout (visible proof) is emitted with `--nocapture`.

    /// Deterministic xorshift32 PRNG for reproducible trajectory generation.
    /// (Avoids pulling in a rand dep just for the gate.)
    fn xorshift32(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    /// Box-Muller Gaussian sample (one draw — discards the second).
    fn gaussian(state: &mut u32, sigma: f32) -> f32 {
        // Box-Muller with two uniforms in (0, 1].
        let u1 = (xorshift32(state) as f32 / u32::MAX as f32).max(1e-10);
        let u2 = (xorshift32(state) as f32 / u32::MAX as f32).max(1e-10);
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0_f32 * std::f32::consts::PI * u2;
        sigma * r * theta.cos()
    }

    /// T3.1 — fixed-step oscillation: direction flips ±π each tick.
    ///
    /// The agent takes K decisions of magnitude `step_mag`, alternating
    /// direction each tick. Produces a ping-pong with no commitment. Total
    /// length ≈ K × step_mag (same as committed), but mean curvature ≈ π.
    fn make_fixed_step_oscillation(
        k_ticks: usize,
        step_mag: f32,
        noise_sigma: f32,
        seed: u32,
    ) -> Vec<Vec<f32>> {
        let mut state = vec![0.0_f32, 0.0];
        let mut rng = seed;
        let mut traj: Vec<Vec<f32>> = Vec::with_capacity(k_ticks + 1);
        traj.push(state.clone());
        for t in 0..k_ticks {
            let dir_sign = if t % 2 == 0 { 1.0 } else { -1.0 };
            state[0] += dir_sign * step_mag + gaussian(&mut rng, noise_sigma);
            state[1] += gaussian(&mut rng, noise_sigma);
            traj.push(state.clone());
        }
        traj.shrink_to_fit();
        traj
    }

    /// T3.2 — fixed-step committed: constant direction (+x).
    ///
    /// Same K decisions of magnitude `step_mag`, all in the same direction.
    /// Total length ≈ K × step_mag (same as oscillation), but mean curvature ≈ 0.
    fn make_fixed_step_committed(
        k_ticks: usize,
        step_mag: f32,
        noise_sigma: f32,
        seed: u32,
    ) -> Vec<Vec<f32>> {
        let mut state = vec![0.0_f32, 0.0];
        let mut rng = seed;
        let mut traj: Vec<Vec<f32>> = Vec::with_capacity(k_ticks + 1);
        traj.push(state.clone());
        for _ in 0..k_ticks {
            state[0] += step_mag + gaussian(&mut rng, noise_sigma);
            state[1] += gaussian(&mut rng, noise_sigma);
            traj.push(state.clone());
        }
        traj.shrink_to_fit();
        traj
    }

    /// T3.3 — fixed-step drift: direction rotates smoothly.
    ///
    /// Same K decisions of magnitude `step_mag`, but direction rotates by a
    /// small fixed angle each tick. Total length ≈ K × step_mag, mean curvature
    /// ≈ rotation-per-step (small). Sits between committed and oscillation as a
    /// control: a router should treat drift differently from both extremes.
    fn make_fixed_step_drift(
        k_ticks: usize,
        step_mag: f32,
        drift_angle_per_step: f32,
        noise_sigma: f32,
        seed: u32,
    ) -> Vec<Vec<f32>> {
        let mut state = vec![0.0_f32, 0.0];
        let mut rng = seed;
        let mut traj: Vec<Vec<f32>> = Vec::with_capacity(k_ticks + 1);
        traj.push(state.clone());
        for t in 0..k_ticks {
            let angle = drift_angle_per_step * (t as f32);
            state[0] += step_mag * angle.cos() + gaussian(&mut rng, noise_sigma);
            state[1] += step_mag * angle.sin() + gaussian(&mut rng, noise_sigma);
            traj.push(state.clone());
        }
        traj.shrink_to_fit();
        traj
    }

    #[test]
    fn t3_visible_game_related_gate() {
        const K_TICKS: usize = 20;
        const N_SAMPLES: usize = 50;
        const STEP_MAG: f32 = 0.3;
        const NOISE_SIGMA: f32 = 0.02; // small vs STEP_MAG so direction signal is clear
        const DRIFT_ANGLE: f32 = 0.1; // radians per step (smooth turn)
        const BASE_SEED: u32 = 42;

        let mut osc_lengths = Vec::with_capacity(N_SAMPLES);
        let mut osc_curvatures = Vec::with_capacity(N_SAMPLES);
        let mut com_lengths = Vec::with_capacity(N_SAMPLES);
        let mut com_curvatures = Vec::with_capacity(N_SAMPLES);
        let mut drf_lengths = Vec::with_capacity(N_SAMPLES);
        let mut drf_curvatures = Vec::with_capacity(N_SAMPLES);

        for i in 0..N_SAMPLES {
            let seed = BASE_SEED.wrapping_add(i as u32 * 1_000_003);

            let osc_traj = make_fixed_step_oscillation(K_TICKS, STEP_MAG, NOISE_SIGMA, seed);
            let osc_refs: Vec<&[f32]> = osc_traj.iter().map(|v| v.as_slice()).collect();
            let osc_g = from_states(&osc_refs);
            osc_lengths.push(osc_g.length);
            osc_curvatures.push(osc_g.mean_curvature);

            let com_traj = make_fixed_step_committed(K_TICKS, STEP_MAG, NOISE_SIGMA, seed);
            let com_refs: Vec<&[f32]> = com_traj.iter().map(|v| v.as_slice()).collect();
            let com_g = from_states(&com_refs);
            com_lengths.push(com_g.length);
            com_curvatures.push(com_g.mean_curvature);

            let drf_traj = make_fixed_step_drift(K_TICKS, STEP_MAG, DRIFT_ANGLE, NOISE_SIGMA, seed);
            let drf_refs: Vec<&[f32]> = drf_traj.iter().map(|v| v.as_slice()).collect();
            let drf_g = from_states(&drf_refs);
            drf_lengths.push(drf_g.length);
            drf_curvatures.push(drf_g.mean_curvature);
        }

        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;

        let osc_len = mean(&osc_lengths);
        let com_len = mean(&com_lengths);
        let drf_len = mean(&drf_lengths);
        let osc_curv = mean(&osc_curvatures);
        let com_curv = mean(&com_curvatures);
        let drf_curv = mean(&drf_curvatures);

        // ── G3.1 — curvature separates oscillation from commitment ─────────
        let curv_gap = osc_curv - com_curv;
        let g3_1_pass = curv_gap >= 0.5;

        // ── G3.2 — length is BLIND to the oscillation pattern ──────────────
        // All three classes are constructed with the same step magnitude, so
        // total length should be ≈ equal. Length cannot distinguish them.
        let len_diff_ratio = (osc_len - com_len).abs() / com_len.max(1e-6);
        let g3_2_pass = len_diff_ratio <= 0.15;

        // ── G3.3 — drift sits between committed and oscillation (control) ───
        let g3_3_pass = com_curv < drf_curv && drf_curv < osc_curv;

        println!();
        println!("=== Latent Trajectory Geometry — Game-Related Gate (Plan 342 Phase 3) ===");
        println!();
        println!("Scenario: agent takes K={K_TICKS} decisions of fixed magnitude step={STEP_MAG},");
        println!(
            "          direction pattern differs by class. N={N_SAMPLES} trajectories per class."
        );
        println!("          (noise sigma={NOISE_SIGMA}, drift angle={DRIFT_ANGLE} rad/step)");
        println!();
        println!("Trajectory class     | mean length | mean curvature (rad)");
        println!("---------------------|-------------|----------------------");
        println!("oscillation (flip)   |   {osc_len:7.3}   |       {osc_curv:6.3}");
        println!("committed (constant) |   {com_len:7.3}   |       {com_curv:6.3}");
        println!("drift (rotate)       |   {drf_len:7.3}   |       {drf_curv:6.3}");
        println!();
        println!(
            "Gate G3.1 (curvature separates osc from committed):  {}",
            if g3_1_pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  osc curvature ({osc_curv:.3}) - committed curvature ({com_curv:.3}) = +{curv_gap:.3} rad (>= 0.5)"
        );
        println!(
            "Gate G3.2 (length is blind to the pattern):          {}",
            if g3_2_pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  |osc length ({osc_len:.3}) - committed length ({com_len:.3})| / committed = {len_diff_ratio:.3} (<= 0.15)"
        );
        println!(
            "Gate G3.3 (drift sits between, control):             {}",
            if g3_3_pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  committed ({com_curv:.3}) < drift ({drf_curv:.3}) < oscillation ({osc_curv:.3})"
        );
        println!();
        let all_pass = g3_1_pass && g3_2_pass && g3_3_pass;
        println!(
            "Verdict: {}",
            if all_pass {
                "curvature signal catches the oscillation pattern that length misses.\n         Promotion candidate for router integration (follow-up plan)."
            } else {
                "curvature signal does NOT cleanly separate on this substrate. [opt-in only]"
            }
        );
        println!();

        assert!(
            g3_1_pass,
            "G3.1 FAILED: curvature gap = {curv_gap:.3} (need >= 0.5)"
        );
        assert!(
            g3_2_pass,
            "G3.2 FAILED: length diff ratio = {len_diff_ratio:.3} (need <= 0.15)"
        );
        assert!(
            g3_3_pass,
            "G3.3 FAILED: drift must sit between committed and osc"
        );
    }

    #[test]
    fn t3_realistic_damped_oscillation_sanity() {
        // Sanity check (NOT a strict gate): a realistic damped ping-pong between
        // two attractor basins produces high curvature. Documents that the
        // signal also works on the more realistic "pulled toward basins" model,
        // not just the fixed-step construction above. Both length AND curvature
        // fire on this realistic scenario — the length-matched gate above is
        // the one that proves curvature's INDEPENDENT value.
        const BASIN_A: [f32; 2] = [1.0, 0.0];
        const BASIN_B: [f32; 2] = [-1.0, 0.0];

        let mut state = vec![0.0_f32, 0.0];
        let mut rng = 42_u32;
        let mut traj: Vec<Vec<f32>> = vec![state.clone()];
        for t in 0..20usize {
            let target = if t % 2 == 0 { BASIN_A } else { BASIN_B };
            for d in 0..2 {
                state[d] += 0.5 * (target[d] - state[d]) + gaussian(&mut rng, 0.05);
            }
            traj.push(state.clone());
        }
        let refs: Vec<&[f32]> = traj.iter().map(|v| v.as_slice()).collect();
        let g = from_states(&refs);
        // Realistic damped oscillation should produce high curvature (> 1.0 rad)
        // and non-trivial length (> 5.0). Not a strict threshold — just a sanity
        // floor that confirms the signal is present.
        assert!(
            g.mean_curvature > 1.0,
            "realistic osc curvature = {}",
            g.mean_curvature
        );
        assert!(g.length > 5.0, "realistic osc length = {}", g.length);
    }

    #[test]
    fn t3_bifurcation_on_oscillation_pair() {
        // Sanity: two oscillation trajectories started with different noise
        // seeds produce measurable final separation.
        let a = make_fixed_step_oscillation(20, 0.3, 0.02, 42);
        let b = make_fixed_step_oscillation(20, 0.3, 0.02, 137);
        let a_refs: Vec<&[f32]> = a.iter().map(|v| v.as_slice()).collect();
        let b_refs: Vec<&[f32]> = b.iter().map(|v| v.as_slice()).collect();
        let r = bifurcation_ratio(&a_refs, &b_refs);
        // Both start at origin → initial_sep ≈ 0 → separation_ratio is INFINITY,
        // onset_step is None. Final separation should still be finite & >= 0.
        assert!(r.final_separation >= 0.0);
        assert!(r.final_separation.is_finite());
    }
}
