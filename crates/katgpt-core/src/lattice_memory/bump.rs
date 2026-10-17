//! Per-axis 3-tap bump weights over the fractional coordinate (Plan 619 T1.3).
//!
//! DESIGN DECISION (recorded — the plan leaves the kernel shape open at T1.3
//! and pins only its tests: LUT parity ≤ 1e-6, per-axis sum = 1, zero mass
//! outside the support): the 3 taps are the cells
//! `{round(c)−1, round(c), round(c)+1}` around the primary `m = round(c)`,
//! weighted by a **radius-1.5 tent centered at the query** `c = m + δ` —
//! raw weight `max(0, 1 − u/1.5)` at each tap's distance `u ∈ {1+δ, |δ|,
//! 1−δ}` — then normalized to a per-axis partition of unity. Closed form
//! for `δ ∈ [−0.5, 0.5]`, `D = 5 − 2|δ|`:
//!
//! ```text
//! w_prev = (1 − 2δ)/D    w_center = (3 − 2|δ|)/D    w_next = (1 + 2δ)/D
//! ```
//!
//! Properties (each gated below): sums to 1 exactly; direction-sensitive
//! (the nearer side tap outweighs the farther — δ < 0 weights `m−1` more);
//! the far tap decays to 0 exactly at the ±0.5 boundary where the primary
//! itself switches (support is exactly ±1 cell around `m` — no weight ever
//! reaches two cells away); exact hits (δ = 0) read `(0.2, 0.6, 0.2)` — the
//! 3-tap form's deliberate graded-recall dilution, the contrast T1.7's G5
//! measures against Engram's hash cliff. The cos² alternative is T1.8's
//! A/B, not this module's concern.
//!
//! The LUT is the plan's 2⁸ quantization over the signed domain; the hot
//! path is one table read — no division on the hot path (the denominator is
//! folded into the table).

/// LUT resolution over `δ ∈ [−0.5, 0.5]` — the plan's 2⁸ quantization.
const LUT_N: usize = 256;

/// `(w_prev, w_center, w_next)` for one axis.
pub type AxisWeights = (f32, f32, f32);

/// The 2⁸-entry bump table over the signed fractional coordinate.
pub type BumpLut = [AxisWeights; LUT_N];

/// Build the table once at construction — O(1) reads beat re-deriving the
/// closed form (with its division) per query.
#[must_use]
pub fn build_lut() -> BumpLut {
    let mut lut = [(0.0_f32, 0.0_f32, 0.0_f32); LUT_N];
    for (i, entry) in lut.iter_mut().enumerate() {
        // Bin CENTER, not bin edge: the quantization error stays ≤ half a
        // bin and the parity gate below sweeps bin centers against the
        // direct form.
        let delta = -0.5 + (i as f32 + 0.5) * (1.0 / LUT_N as f32);
        *entry = weights_direct(delta);
    }
    lut
}

/// The DIRECT closed form — the LUT's parity oracle.
///
/// `δ` outside `[-0.5, 0.5]` cannot arise from `round(c)` (asserted by the
/// caller contract); the clamp is defensive and pins abuse to the boundary.
#[must_use]
pub fn weights_direct(delta: f32) -> AxisWeights {
    let d = delta.clamp(-0.5, 0.5);
    let abs = d.abs();
    let denom = 5.0 - 2.0 * abs;
    (
        (1.0 - 2.0 * d) / denom,
        (3.0 - 2.0 * abs) / denom,
        (1.0 + 2.0 * d) / denom,
    )
}

/// One axis's 3-tap weights from the LUT: `(w(m−1), w(m), w(m+1))` for the
/// query at `m + δ`.
#[inline]
#[must_use]
pub fn axis_weights(lut: &BumpLut, delta: f32) -> AxisWeights {
    // Map δ ∈ [-0.5, 0.5] → bin index; the `.clamp` after the cast guards
    // the round-up at the top edge (δ = 0.5 must index LUT_N−1, not LUT_N).
    let scaled = (delta + 0.5) * LUT_N as f32;
    let idx = if scaled <= 0.0 {
        0
    } else {
        (scaled as usize).min(LUT_N - 1)
    };
    lut[idx]
}

/// The cos² 3-tap alternative — the Plan 619 T1.8 A/B arm.
///
/// `g(o; δ) = cos²(π(δ−o)/4)` on the same `{−1, 0, +1}` taps, normalized
/// per axis. It exists in the SDM lineage for SGD differentiability, which
/// this modelless primitive does not have — it is measured, not assumed:
/// the T1.8 bench compares near-miss recovery and exact-recall dilution
/// against the tent and pins the winner in `BumpKernel`. Two shape facts
/// the A/B reads (both gated below): the truncated kernel is WIDER (exact
/// hit reads `(0.25, 0.5, 0.25)` — more dilution than the tent's
/// `(0.2, 0.6, 0.2)`), and truncation leaks mass through the ±0.5 boundary
/// (the far tap keeps ~0.08 at the primary switch, where the tent is
/// exactly 0 — the support is no longer ±1 cell).
#[must_use]
pub fn axis_weights_cos2(delta: f32) -> AxisWeights {
    let d = delta.clamp(-0.5, 0.5);
    let g = |off: f32| {
        let t = core::f32::consts::FRAC_PI_4 * (d - off);
        t.cos() * t.cos()
    };
    let (g0, g1, g2) = (g(-1.0), g(0.0), g(1.0));
    let s = g0 + g1 + g2;
    (g0 / s, g1 / s, g2 / s)
}

#[cfg(test)]
mod tests {
    use super::{LUT_N, axis_weights, build_lut, weights_direct};

    #[test]
    fn lut_matches_direct_within_1e6() {
        // The plan's parity bar: every bin center against the closed form.
        let lut = build_lut();
        for (i, &(qp, qc, qn)) in lut.iter().enumerate() {
            let delta = -0.5 + (i as f32 + 0.5) * (1.0 / LUT_N as f32);
            let (dp, dc, dn) = weights_direct(delta);
            assert!((qp - dp).abs() <= 1e-6, "prev @ bin {i}: {qp} vs {dp}");
            assert!((qc - dc).abs() <= 1e-6, "center @ bin {i}: {qc} vs {dc}");
            assert!((qn - dn).abs() <= 1e-6, "next @ bin {i}: {qn} vs {dn}");
        }
    }

    #[test]
    fn per_axis_weights_sum_to_one() {
        // The partition-of-unity law — LUT bins and the direct form both.
        let lut = build_lut();
        for &(a, b, c) in lut.iter() {
            assert!(
                (a + b + c - 1.0).abs() <= 1e-6,
                "lut sum {} at ({a},{b},{c})",
                a + b + c
            );
        }
        let steps = 4096;
        for s in 0..=steps {
            let delta = -0.5 + s as f32 / steps as f32;
            let (a, b, c) = weights_direct(delta);
            assert!((a + b + c - 1.0).abs() <= 1e-6, "direct sum at δ={delta}");
        }
    }

    #[test]
    fn zero_mass_outside_support() {
        // "Support exactly ±1 cell": the far tap hits ZERO exactly at the
        // boundary where the primary switches, and never below zero.
        let (p, c, n) = weights_direct(0.5);
        assert_eq!((p, c, n), (0.0, 0.5, 0.5));
        let (p, c, n) = weights_direct(-0.5);
        assert_eq!((p, c, n), (0.5, 0.5, 0.0));
        // Interior: every weight stays positive, and for δ > 0 the FAR tap
        // (prev) decays monotonically toward its boundary zero.
        let mut prev_far = f32::INFINITY;
        for s in 1..256 {
            let delta = s as f32 / 512.0; // (0, 0.5)
            let (p, _, n) = weights_direct(delta);
            assert!(
                p > 0.0 && n > 0.0,
                "interior taps must stay positive at δ={delta}"
            );
            assert!(p < prev_far, "far-tap mass must decay toward the boundary");
            prev_far = p;
        }
    }

    #[test]
    fn direction_sensitive_and_monotone() {
        // δ < 0 weights the previous cell more; δ > 0 the next; the center
        // is maximal at δ = 0 and decays symmetrically.
        let lut = build_lut();
        let (pm, _, nm) = axis_weights(&lut, -0.3);
        assert!(pm > nm, "δ<0 must weight the previous cell more");
        let (pp, _, np) = axis_weights(&lut, 0.3);
        assert!(np > pp, "δ>0 must weight the next cell more");
        let (_, c0, _) = weights_direct(0.0);
        let (_, ch, _) = weights_direct(0.25);
        assert!(
            c0 > ch,
            "center weight must decay away from the cell center"
        );
    }

    #[test]
    fn known_closed_form_values_pinned() {
        // Hand-derived pins: δ=0 → (0.2, 0.6, 0.2); δ=0.5 → (0, 0.5, 0.5).
        let (p, c, n) = weights_direct(0.0);
        assert!((p - 0.2).abs() < 1e-7 && (c - 0.6).abs() < 1e-7 && (n - 0.2).abs() < 1e-7);
        let (p, c, n) = weights_direct(0.5);
        assert!(p.abs() < 1e-7 && (c - 0.5).abs() < 1e-7 && (n - 0.5).abs() < 1e-7);
    }

    #[test]
    fn cos2_ab_arm_sums_to_one_and_matches_known_shapes() {
        use super::axis_weights_cos2;
        // Partition of unity across the domain.
        for s in 0..=512 {
            let delta = -0.5 + s as f32 / 512.0;
            let (a, b, c) = axis_weights_cos2(delta);
            assert!((a + b + c - 1.0).abs() <= 1e-6, "cos² sum at δ={delta}");
            assert!(a >= 0.0 && b >= 0.0 && c >= 0.0);
        }
        // Exact hit: the wider kernel's (0.25, 0.5, 0.25).
        let (a, b, c) = axis_weights_cos2(0.0);
        assert!((a - 0.25).abs() < 1e-6 && (b - 0.5).abs() < 1e-6 && (c - 0.25).abs() < 1e-6);
        // Truncation leak: the far tap retains mass at the primary switch
        // (cos²(3π/8) normalized ≈ 0.078); the near tap never falls below
        // the center (equal is the truncation's ceiling at the boundary).
        let (far, center, near) = axis_weights_cos2(0.5);
        assert!(
            far > 0.05 && far < 0.12,
            "far-tap leak {far} outside the documented band"
        );
        assert!(
            near >= center - 1e-6,
            "near tap {near} below center {center} at δ=+0.5"
        );
    }
}
