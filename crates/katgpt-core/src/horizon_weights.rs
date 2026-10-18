//! Remaining-horizon weighting — the (T−t) Fubini accumulation law and the
//! PFD closed-form `w(t)` schedule (Issue 875 T1 / Research 582).
//!
//! Source: *Probability-Flow Distillation: Exact Wasserstein Gradient Flow
//! for High-Fidelity 3D Generation* (arXiv:2605.09071), Theorem 1: under
//! linear drift `f(x,t) = a(t)·x` with stop-gradient, the expected
//! discrepancy of a uniformly-sampled-`t` partial integration equals the
//! Wasserstein gradient of a time-averaged functional weighted by
//!
//! ```text
//! w(t) = ½ (T−t) · g(t)² · c(t,0)²,    c(t,0) = exp(−∫₀ᵗ a(s) ds)
//! ```
//!
//! The `(T−t)` factor is a Fubini swap on the triangular integration
//! domain: averaging observations taken at uniformly sampled `t` over
//! `[0,T]` gives each partial integral effective weight `(T−s)` — MAXIMUM
//! at low noise (`t → 0`), decaying linearly to EXACTLY ZERO at `t = T`.
//! Corollary encoded here: skipping `t` near `T` is first-order lossless
//! (`w(T) = 0` — the zero-terminal-weight truncation).
//!
//! # Mechanism-distinct neighbors (the substrate-first vocabulary check)
//!
//! - [`crate::tether::horizon_decay`] (and the `hint_regret::memory`
//!   `r̂·σ(−λ·Δt)` family) is PAST-looking staleness fading — weight by
//!   time SINCE an observation. This module is FUTURE-looking — weight by
//!   the REMAINING horizon of an accumulation. Same word, opposite sign of
//!   information.
//! - `set_diffusion_schedule::PositionOffsetSchedule` decides WHERE (which
//!   position) to reveal next; this module decides HOW MUCH an observation
//!   at time `t` contributes to a time-averaged accumulation.
//! - `renoise_ce` averages its `k_draws` FLAT (`sum / k`); Issue 875 T2
//!   wires these weights into that average.
//!
//! # What ships
//!
//! - [`remaining_horizon_weight`] / [`remaining_horizon_weights`]: the
//!   normalized `(T−t)/T` generic law (pure arithmetic, any horizon).
//! - [`remaining_horizon_t_sample`]: the SAMPLING realization of the same
//!   law — the exact inverse CDF of the density `∝ (T−t)` on a sub-range
//!   (Issue 875 T2's draw schedule; one `sqrt`, zero alloc).
//! - [`TimeAnnealRange`] + [`terminal_truncation_ceiling`] /
//!   [`truncated_w_mass_fraction`]: the Issue 875 T3 schedule layer —
//!   iteration-indexed annealing of a SAMPLING RANGE toward low noise
//!   (the paper's `[0.02T, 0.98T] → [0.02T, 0.70T]` late-iteration shape;
//!   state-indexed switching composes with it), plus the zero-terminal-
//!   weight truncation predicate in closed form (`t_cut = T − (T−t_min)·√ε`
//!   — skipping the top of the range discards a known, squared-root-small
//!   fraction of the law's mass).
//! - [`pfd_horizon_weight_at`] / [`pfd_horizon_weights`]: the exact closed
//!   form over a discrete uniform grid (cumulative trapezoid of `a`, one
//!   `exp` per grid point — the offline computation).
//! - [`HorizonWeightTable`]: the closed form frozen into a fixed
//!   `[f32; HORIZON_WEIGHT_GRID]` with a BLAKE3 commitment (the
//!   `katgpt-attn/static_cal.rs` `StaticCalTable` pattern — except this
//!   table is exact closed form, no calibration pass): O(1) nearest-grid
//!   lookup, cross-process reproducible, tamper-evident.
//!
//! # Gates (GOAT, per Issue 875 T5)
//!
//! - **G1** table vs closed-form recompute: bit-identical (`to_bits`) —
//!   pinned in-module (against both the grid fn and an independent f64
//!   oracle within tolerance) and e2e in the root gate
//!   `tests/bench_875_horizon_weights_goat.rs`.
//! - **G2** O(1) lookup vs per-call `exp` recompute (the STRONG baseline:
//!   caller-held cumulative integral) — the root gate via the shared
//!   `ab_timing` harness, median lookup/recompute ≤ 0.5 (≥ 2× faster).
//! - **G3** no-regression: opt-in, default-off — no default-path surface.
//! - **G4** fixed-size table, zero allocation on build and lookup (asserted
//!   under `debug_assertions`/`alloc_tracking` via `crate::alloc`).
//!
//! OPT-IN per the no-default-consumer rule: promotion to default waits for
//! the T2 consumer GOAT ((T−t)-weighted `renoise_ce` averaging).
//!
//! NaN policy: NaN inputs propagate through the pure functions (except
//! [`HorizonWeightTable::w_at_unit`], where the grid clamp sends NaN's
//! saturating cast to index 0 — documented there). This is a weight
//! schedule, not a sync-boundary value; callers on the sync path feed
//! clamped time fractions.
//!
//! horizon_weights (Issue 875 T1 / Research 582, arXiv:2605.09071): the PFD
//! (T−t) Fubini accumulation law + the exact w(t) = ½(T−t)g²c² closed form
//! (c = exp(−∫a)), with the BLAKE3-committed fixed-grid table form (the
//! static_cal pattern, exact — no calibration pass). Future-looking
//! remaining-horizon weighting — mechanism-distinct from
//! tether::horizon_decay's past-looking staleness fading. T2 consumer
//! shipped as the combined-gate renoise_ce_score_horizon (Bench 877);
//! the feature itself stays OPT-IN per the T5 promotion rule.

/// Fixed grid density for [`HorizonWeightTable`]. 64 points covers the
/// schedule shapes the PFD paper anneals over (`[0.02T, 0.98T] → [0.02T,
/// 0.70T]`) with nearest-grid lookup error well under the trapezoid
/// discretization error at those smoothness classes; consumers needing
/// other densities use the slice-based pure functions.
pub const HORIZON_WEIGHT_GRID: usize = 64;

/// Normalized remaining-horizon weight `(T−t)/T`, clamped to `[0, 1]`.
///
/// The pure Fubini law: how much an observation at time `t` contributes to
/// a time-averaged accumulation over `[0, T]` when `t` is sampled
/// uniformly. `t = 0` → 1.0 (max weight, low noise), `t = T` → exactly 0.
#[inline]
pub fn remaining_horizon_weight(t: f32, horizon: f32) -> f32 {
    debug_assert!(horizon > 0.0, "horizon must be positive");
    ((horizon - t) / horizon).clamp(0.0, 1.0)
}

/// Fill `out` with the normalized `(T−t)/T` weights over a UNIFORM grid
/// `t_i = T·i/(n−1)`, `i = 0..n` (the horizon itself cancels — the output
/// is dimensionless).
///
/// `out[n−1]` is EXACTLY `0.0` (the zero-terminal-weight corollary).
/// Requires `n ≥ 2`: a single-point grid has no interval to weight over.
pub fn remaining_horizon_weights(out: &mut [f32]) {
    let n = out.len();
    assert!(n >= 2, "remaining_horizon_weights needs n >= 2, got {n}");
    let last = (n - 1) as f32;
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = 1.0 - i as f32 / last;
    }
    // Exact-zero terminal weight (+0.0, never −0.0): the truncation
    // corollary's anchor.
    out[n - 1] = 0.0;
}

/// Inverse-CDF sample from the remaining-horizon law (Issue 875 T2): given
/// a uniform `u ∈ [0, 1]`, return `t` drawn from the density
/// `p(t) ∝ (horizon − t)` on `[t_min, t_max]` — the sampling realization of
/// the law [`remaining_horizon_weight`] weights by. Low-noise draws are
/// the most likely; the terminal point the least (`w(T) = 0`).
///
/// Exact inverse of the law's CDF
/// `F(t) = [(T−t_min)² − (T−t)²] / [(T−t_min)² − (T−t_max)²]`:
///
/// ```text
/// t = T − sqrt((1−u)·(T−t_min)² + u·(T−t_max)²)
/// ```
///
/// One `sqrt` plus a handful of multiplies — zero allocation, deterministic
/// under the caller's RNG stream. `u = 0` returns `t_min` (the density
/// maximum), `u = 1` returns `t_max`.
///
/// Degenerate inputs fall back to uniform interpolation (total function,
/// never NaN): a collapsed/inverted range (`t_max ≤ t_min`), or a horizon
/// that does not strictly contain the range (`horizon < t_max`), lerp at
/// `u`. Non-finite `u` is treated as `0.0` (→ `t_min`); finite `u` outside
/// `[0, 1]` clamps. Same NaN policy as the weight fns: this is a schedule,
/// not a sync-boundary value.
#[inline]
pub fn remaining_horizon_t_sample(u: f32, t_min: f32, t_max: f32, horizon: f32) -> f32 {
    let u = if u.is_finite() {
        u.clamp(0.0, 1.0)
    } else {
        0.0
    };
    // Fallback (uniform lerp) on any degenerate/incomparable input — the
    // partial_cmp form keeps the function total under NaN (None → fallback).
    let range_ok = matches!(
        t_max.partial_cmp(&t_min),
        Some(core::cmp::Ordering::Greater)
    );
    let horizon_ok = matches!(
        horizon.partial_cmp(&t_max),
        Some(core::cmp::Ordering::Greater) | Some(core::cmp::Ordering::Equal)
    );
    if !range_ok || !horizon_ok {
        return t_min + u * (t_max - t_min);
    }
    let lo = horizon - t_min;
    let hi = horizon - t_max;
    let t = horizon - ((1.0 - u) * lo * lo + u * hi * hi).sqrt();
    // Rounding at the endpoints can land 1 ulp outside; clamp home.
    t.clamp(t_min, t_max)
}

// ─────────────────────────────────────────────────────────────────────────
// Issue 875 T3: time-annealed sampling ranges + terminal truncation
// ─────────────────────────────────────────────────────────────────────────

/// Zero-terminal-weight truncation predicate, MASS form: the fraction of
/// the (T−t) law's mass on `[t_min, T]` that lives at or above `t_cut`.
///
/// The law integrates to `(T−t)²/2` over any suffix, so the ratio is the
/// SQUARE of the remaining-horizon ratio:
///
/// ```text
/// ε(t_cut) = ((T − t_cut) / (T − t_min))²
/// ```
///
/// `t_cut = T` → exactly `0.0` (the corollary's anchor, `w(T) = 0`);
/// `t_cut ≤ t_min` → `1.0` (nothing above the floor is excluded). A
/// degenerate horizon (`T ≤ t_min`) or non-finite `t_cut` reads as `1.0` —
/// the conservative answer, "no truncation certifiable" (the
/// [`remaining_horizon_t_sample`] NaN policy).
#[inline]
pub fn truncated_w_mass_fraction(t_cut: f32, t_min: f32, horizon: f32) -> f32 {
    let denom = horizon - t_min;
    if !denom.is_finite() || denom <= 0.0 {
        return 1.0;
    }
    let above = horizon - t_cut;
    if !above.is_finite() {
        return 1.0;
    }
    let frac = (above / denom).clamp(0.0, 1.0);
    frac * frac
}

/// Zero-terminal-weight truncation ceiling: the largest `t_cut` whose
/// at-or-above mass on `[t_min, T]` is at most `eps_mass` — the closed-form
/// inverse of [`truncated_w_mass_fraction`]:
///
/// ```text
/// t_cut = T − (T − t_min)·√ε
/// ```
///
/// The paper's late-iteration target `[0.02T, 0.70T]` is this predicate at
/// `ε = ((1−0.70)/(1−0.02))² ≈ 0.0937` on the plain law: skipping the top
/// 30% of the range discards ≤ 9.4% of the (T−t) mass, and that mass sits
/// where the law is weakest — first-order lossless in a time-averaged
/// estimator. `eps_mass ≤ 0` or non-finite → `T` (skip nothing); `≥ 1` →
/// `t_min` (everything above the floor); the result is additionally
/// clamped into `[t_min, T]` against rounding.
#[inline]
pub fn terminal_truncation_ceiling(eps_mass: f32, t_min: f32, horizon: f32) -> f32 {
    let denom = horizon - t_min;
    if !eps_mass.is_finite() || !denom.is_finite() || denom <= 0.0 || eps_mass <= 0.0 {
        return horizon;
    }
    let eps = eps_mass.min(1.0);
    (horizon - denom * eps.sqrt()).clamp(t_min, horizon)
}

/// Iteration-indexed time-anneal schedule for SOLVER SAMPLING RANGES
/// (Issue 875 T3 / Research 582): the paper's `[0.02T, 0.98T] →
/// [0.02T, 0.70T]` late-iteration shape, generalized.
///
/// The range ceiling holds at `ceil_start_frac` for the first
/// `(1 − anneal_frac)` of iterations, then eases LINEARLY down to
/// `ceil_end_frac` over the last `anneal_frac`. The floor never moves —
/// low-noise observations carry the law's maximum weight
/// ([`remaining_horizon_weight`]) and are never traded away. Orthogonal to
/// `dllm_solver`'s entropy-triggered switching: that one is STATE-indexed
/// (which step is critical), this one is TIME-indexed (how far into the
/// run) — the two compose.
///
/// All fields are fractions of the horizon T (dimensionless). Invalid
/// fields fall back to their `DEFAULT` value at use, and the final tuple
/// is clamped into order (the `RenoiseCeHorizon` pattern) — the schedule
/// is total, never NaN, deterministic.
///
/// `range_at(iter, total)` pins its endpoints exactly: pre-anneal
/// iterations return exactly `ceil_start_frac`, the final iteration
/// exactly `ceil_end_frac` (the eased sum `a + (b−a)` is NOT relied on —
/// it does not round back to `b` in f32).
///
/// # Recommended consumption
///
/// Keep the estimator's total weight FIXED while concentrating placement:
/// renormalize over the truncated range by the closed-form mass ratio
/// ([`truncated_w_mass_fraction`] of the ceiling, exact — no quadrature).
/// The plain truncated integral is the corollary's own cost (≈ ε of the
/// mass, a slightly smaller effective step) — the caller's trade to make;
/// the cross-repo quality gate (Bench 883) measures both postures.
#[derive(Clone, Copy, Debug)]
pub struct TimeAnnealRange {
    /// Range floor as a fraction of T — FIXED through the schedule.
    pub floor_frac: f32,
    /// Pre-anneal ceiling as a fraction of T.
    pub ceil_start_frac: f32,
    /// Fully-annealed ceiling as a fraction of T (the late-iteration end).
    pub ceil_end_frac: f32,
    /// The fraction of the FINAL iterations over which the ceiling eases
    /// from `ceil_start_frac` down to `ceil_end_frac`.
    pub anneal_frac: f32,
}

impl TimeAnnealRange {
    /// The paper's schedule: `[0.02, 0.98] → [0.02, 0.70]` over the last
    /// 30% of iterations.
    pub const DEFAULT: Self = Self {
        floor_frac: 0.02,
        ceil_start_frac: 0.98,
        ceil_end_frac: 0.70,
        anneal_frac: 0.30,
    };

    /// The `(t_min, t_max)` sampling range (fractions of T) at iteration
    /// `iter` of `total` (0-indexed, `total ≥ 1`).
    ///
    /// `iter ≥ total` returns the terminal posture `(floor, ceil_end)` — a
    /// safe monotone extension for callers that overrun. A `total` too
    /// small to contain a gradual anneal window (`(total−1) ≤
    /// (1−anneal_frac)·total`) never anneals: no integer iteration falls
    /// inside the window, so the ceiling stays at `ceil_start` — there is
    /// no schedule to traverse (documented, deterministic).
    pub fn range_at(&self, iter: usize, total: usize) -> (f32, f32) {
        assert!(
            total >= 1,
            "TimeAnnealRange::range_at needs total >= 1, got {total}"
        );
        let def = Self::DEFAULT;
        let floor = match self.floor_frac {
            v if v.is_finite() && v > 0.0 && v < 1.0 => v,
            _ => def.floor_frac,
        };
        let ceil_start = match self.ceil_start_frac {
            v if v.is_finite() && v > floor && v <= 1.0 => v,
            _ => def.ceil_start_frac,
        };
        let ceil_end = match self.ceil_end_frac {
            v if v.is_finite() && v >= floor && v <= ceil_start => v,
            _ => def.ceil_end_frac,
        };
        let anneal_frac = match self.anneal_frac {
            v if v.is_finite() && v > 0.0 && v < 1.0 => v,
            _ => def.anneal_frac,
        };
        // Ordered by construction (each fallback is validated against the
        // effective floor/ceil_start; the min() closes the mixed-default
        // corner that cannot arise but is pinned anyway).
        let ceil_end = ceil_end.min(ceil_start);

        let total_f = total as f32;
        let anneal_start = total_f * (1.0 - anneal_frac);
        let denom = (total_f - 1.0) - anneal_start;
        let p = if denom > 0.0 {
            ((iter as f32 - anneal_start) / denom).clamp(0.0, 1.0)
        } else if iter as f32 >= anneal_start && anneal_start < total_f {
            1.0
        } else {
            0.0
        };
        // Endpoint pins (f32 a+(b−a) does not round to b).
        let ceiling = if p >= 1.0 {
            ceil_end
        } else if p <= 0.0 {
            ceil_start
        } else {
            ceil_start + p * (ceil_end - ceil_start)
        };
        (floor, ceiling)
    }
}

/// The PFD closed form at a single point, given the caller-held cumulative
/// drift integral `A(t) = ∫₀ᵗ a(s) ds`:
///
/// ```text
/// w(t) = ½ (T−t) · g(t)² · exp(−2·A(t))      (c(t,0)² = exp(−2A))
/// ```
///
/// This is the STRONG per-call baseline the table's G2 gate races against:
/// one `exp` plus a handful of multiplies, with the integral already in
/// hand. Hot-path callers should hold the [`HorizonWeightTable`] instead.
///
/// The evaluation order here is pinned — [`pfd_horizon_weights`] calls this
/// function so the grid fill and the point form agree bit-for-bit.
#[inline]
pub fn pfd_horizon_weight_at(t: f32, horizon: f32, g_t: f32, a_integral_t: f32) -> f32 {
    0.5 * (horizon - t) * g_t * g_t * (-2.0 * a_integral_t).exp()
}

/// Fill `out` with the PFD closed form over a UNIFORM grid
/// `t_i = T·i/(n−1)`, from drift-diffusion schedules sampled at the same
/// grid points (`g[i] = g(t_i)`, `a[i] = a(t_i)`).
///
/// The cumulative drift integral is the left-to-right trapezoid (exact for
/// constant `a`; the deterministic accumulation order is pinned by the G1
/// bit-match test). `out[n−1]` is EXACTLY `0.0` — the `(T−t)` factor, not
/// a clamp. Requires `g.len() == a.len() == out.len()` and `n ≥ 2`.
pub fn pfd_horizon_weights(g: &[f32], a: &[f32], horizon: f32, out: &mut [f32]) {
    let n = out.len();
    assert!(n >= 2, "pfd_horizon_weights needs n >= 2, got {n}");
    assert_eq!(g.len(), n, "g schedule must match the output grid");
    assert_eq!(a.len(), n, "a schedule must match the output grid");
    let dt = horizon / (n - 1) as f32;
    let mut a_int = 0.0f32;
    for i in 0..n {
        if i > 0 {
            a_int += 0.5 * (a[i - 1] + a[i]) * dt;
        }
        let t = horizon * i as f32 / (n - 1) as f32;
        out[i] = pfd_horizon_weight_at(t, horizon, g[i], a_int);
    }
    out[n - 1] = 0.0;
}

/// The PFD closed-form schedule frozen into a fixed-size committed table
/// (the `StaticCalTable` pattern: exact closed form instead of a
/// calibration pass). Build once offline, [`verify`](Self::verify) on
/// load, O(1) nearest-grid lookups on the hot path.
///
/// G4: `[f32; HORIZON_WEIGHT_GRID]` — zero allocation on build and lookup.
///
/// ⛔ The commitment is the identity of the TABLE, not of the schedules:
/// `verify` proves the frozen weights were not tampered with, not that
/// they were built from the right `g`/`a` (mirrors
/// `StaticCalTable::commitment` being the table's identity, not the
/// weights'). Bind build provenance at the caller if that matters.
#[derive(Clone, Debug)]
pub struct HorizonWeightTable {
    w: [f32; HORIZON_WEIGHT_GRID],
    /// `1/horizon`, stored so lookup is multiply-round-index (no divide).
    inv_horizon: f32,
    commitment: [u8; 32],
}

impl HorizonWeightTable {
    /// Build from schedules sampled at the grid points and commit.
    ///
    /// Bit-identical to [`pfd_horizon_weights`] on the same inputs (G1:
    /// the build delegates to the pure function; the test pins it).
    pub fn build(
        g: &[f32; HORIZON_WEIGHT_GRID],
        a: &[f32; HORIZON_WEIGHT_GRID],
        horizon: f32,
    ) -> Self {
        assert!(
            horizon > 0.0 && horizon.is_finite(),
            "horizon must be positive and finite, got {horizon}"
        );
        let mut w = [0.0f32; HORIZON_WEIGHT_GRID];
        pfd_horizon_weights(g, a, horizon, &mut w);
        let mut table = Self {
            w,
            inv_horizon: 1.0 / horizon,
            commitment: [0u8; 32],
        };
        table.commit();
        table
    }

    /// Recompute the BLAKE3 commitment over the stored weights + grid
    /// scale. Deterministic, zero-alloc.
    pub fn commit(&mut self) {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.inv_horizon.to_bits().to_le_bytes());
        hasher.update(bytemuck::cast_slice(&self.w));
        self.commitment
            .copy_from_slice(hasher.finalize().as_bytes());
    }

    /// Verify the stored commitment against the stored weights.
    pub fn verify(&self) -> bool {
        let mut probe = Self {
            w: self.w,
            inv_horizon: self.inv_horizon,
            commitment: [0u8; 32],
        };
        probe.commit();
        probe.commitment == self.commitment
    }

    /// O(1) nearest-grid lookup at absolute time `t` (same units the table
    /// was built with).
    #[inline]
    pub fn w_at(&self, t: f32) -> f32 {
        self.w_at_unit(t * self.inv_horizon)
    }

    /// O(1) nearest-grid lookup at normalized time `x = t/T ∈ [0, 1]`
    /// (the form zone-graph / per-tick consumers want: pass
    /// `tick / total_ticks`).
    ///
    /// NaN input saturates to index 0 (NaN fails the clamp's ordering and
    /// the saturating `as usize` cast lands at 0) — the max-weight end.
    /// Callers feeding unclamped external time should sanitize first.
    #[inline]
    pub fn w_at_unit(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        let idx = (x * (HORIZON_WEIGHT_GRID - 1) as f32 + 0.5) as usize;
        self.w[idx.min(HORIZON_WEIGHT_GRID - 1)]
    }

    /// The frozen weights; grid point `i` ↔ `t_i = T·i/(N−1)`.
    pub fn weights(&self) -> &[f32; HORIZON_WEIGHT_GRID] {
        &self.w
    }

    /// The BLAKE3 commitment over the frozen weights + grid scale.
    pub fn commitment(&self) -> &[u8; 32] {
        &self.commitment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: f32 = 10.0;

    /// VE-flavored deterministic schedules: g increasing in t, positive
    /// drift a with a mild slope — the regime where all three factors of
    /// w(t) move.
    fn schedules(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut g = Vec::with_capacity(n);
        let mut a = Vec::with_capacity(n);
        for i in 0..n {
            let t = T * i as f32 / (n - 1) as f32;
            g.push(0.5 + 0.08 * t);
            a.push(0.25 + 0.01 * t);
        }
        (g, a)
    }

    #[test]
    fn generic_law_endpoints_monotone_mean_half() {
        let mut w = [0.0f32; 16];
        remaining_horizon_weights(&mut w);
        assert_eq!(w[0], 1.0);
        assert_eq!(
            w[15].to_bits(),
            0.0f32.to_bits(),
            "terminal weight is exact +0.0"
        );
        for i in 1..16 {
            assert!(
                w[i] <= w[i - 1],
                "generic law must be non-increasing, w[{i}]={}",
                w[i]
            );
        }
        // Uniform-grid mean of (1 − x) is exactly 1/2 up to f32 rounding.
        let mean = w.iter().sum::<f32>() / w.len() as f32;
        assert!((mean - 0.5).abs() < 1e-3, "Fubini mean {mean}");
        // Point form agrees with the grid form.
        for (i, &wi) in w.iter().enumerate() {
            let t = T * i as f32 / 15.0;
            let point = remaining_horizon_weight(t, T);
            assert!((point - wi).abs() <= 1e-6);
        }
    }

    #[test]
    fn zero_drift_reduces_to_half_remaining_horizon_g_squared() {
        let (g, _) = schedules(32);
        let a_zero = vec![0.0f32; 32];
        let mut w = [0.0f32; 32];
        pfd_horizon_weights(&g, &a_zero, T, &mut w);
        for i in 0..32 {
            let t = T * i as f32 / 31.0;
            // c(t,0) = exp(0) = 1 exactly; exp(-2*0.0) == 1.0 exactly, and
            // x*1.0 == x bit-exactly, so this is bit-equality.
            let expect = 0.5 * (T - t) * g[i] * g[i];
            assert_eq!(w[i].to_bits(), expect.to_bits(), "i={i}");
        }
    }

    #[test]
    fn terminal_weight_is_exactly_zero() {
        let (g, a) = schedules(64);
        let mut w = [0.0f32; 64];
        pfd_horizon_weights(&g, &a, T, &mut w);
        assert_eq!(w[63].to_bits(), 0.0f32.to_bits());
        // Not near-zero: EXACTLY zero, even with g/a nonzero at t=T.
        assert!(g[63] > 0.0 && a[63] > 0.0);
    }

    #[test]
    fn f64_oracle_agreement() {
        // Independent recomputation in f64 (different precision, different
        // accumulation shape) — the correctness check that is NOT
        // satisfied by construction.
        let n = 64usize;
        let (g, a) = schedules(n);
        let mut w = [0.0f32; 64];
        pfd_horizon_weights(&g, &a, T, &mut w);
        let dt = f64::from(T) / (n - 1) as f64;
        let mut a_int = 0.0f64;
        for (i, &wi) in w.iter().enumerate() {
            if i > 0 {
                a_int += 0.5 * (f64::from(a[i - 1]) + f64::from(a[i])) * dt;
            }
            let t = f64::from(T) * i as f64 / (n - 1) as f64;
            let oracle = 0.5 * (f64::from(T) - t) * f64::from(g[i]).powi(2) * (-2.0 * a_int).exp();
            if i == n - 1 {
                assert_eq!(oracle, 0.0);
            } else {
                let rel = (f64::from(wi) - oracle).abs() / oracle.abs().max(1e-12);
                assert!(rel < 1e-5, "i={i} f32 {wi} vs oracle {oracle} (rel {rel})");
            }
        }
    }

    #[test]
    fn trapezoid_is_exact_for_constant_drift() {
        // For constant a the trapezoid integral is a·t up to the dt
        // rounding; check against the closed form fed the analytic
        // integral A(t) = a·t.
        let n = 64usize;
        let g = vec![1.0f32; n];
        let a_const = 0.7f32;
        let a = vec![a_const; n];
        let mut w = [0.0f32; 64];
        pfd_horizon_weights(&g, &a, T, &mut w);
        for (i, &wi) in w.iter().enumerate() {
            let t = T * i as f32 / (n - 1) as f32;
            let analytic = pfd_horizon_weight_at(t, T, 1.0, a_const * t);
            let rel = (wi - analytic).abs() / analytic.abs().max(1e-12);
            assert!(rel < 1e-5, "i={i} rel {rel}");
        }
    }

    #[test]
    fn table_matches_pure_fn_bit_for_bit() {
        let (g, a) = schedules(HORIZON_WEIGHT_GRID);
        let gt: &[f32; HORIZON_WEIGHT_GRID] = g.as_slice().try_into().unwrap();
        let at: &[f32; HORIZON_WEIGHT_GRID] = a.as_slice().try_into().unwrap();
        let table = HorizonWeightTable::build(gt, at, T);
        let mut direct = [0.0f32; HORIZON_WEIGHT_GRID];
        pfd_horizon_weights(&g, &a, T, &mut direct);
        for (i, (&tw, &dw)) in table.weights().iter().zip(direct.iter()).enumerate() {
            assert_eq!(
                tw.to_bits(),
                dw.to_bits(),
                "G1 bit-match failed at grid point {i}"
            );
        }
    }

    #[test]
    fn commitment_roundtrip_and_tamper_detection() {
        let (g, a) = schedules(HORIZON_WEIGHT_GRID);
        let gt: &[f32; HORIZON_WEIGHT_GRID] = g.as_slice().try_into().unwrap();
        let at: &[f32; HORIZON_WEIGHT_GRID] = a.as_slice().try_into().unwrap();
        let mut table = HorizonWeightTable::build(gt, at, T);
        assert!(table.verify());
        assert_ne!(
            table.commitment(),
            &[0u8; 32],
            "commitment must be set at build"
        );
        // Tamper with one weight → verify fails.
        table.w_mut_for_test()[7] += 0.001;
        assert!(!table.verify());
        // Re-commit → verifies again (the table's own identity).
        table.commit();
        assert!(table.verify());
    }

    /// Test-only tamper seam (keeps `w` private).
    impl HorizonWeightTable {
        fn w_mut_for_test(&mut self) -> &mut [f32; HORIZON_WEIGHT_GRID] {
            &mut self.w
        }
    }

    #[test]
    fn w_at_rounds_to_nearest_grid_point() {
        let (g, a) = schedules(HORIZON_WEIGHT_GRID);
        let gt: &[f32; HORIZON_WEIGHT_GRID] = g.as_slice().try_into().unwrap();
        let at: &[f32; HORIZON_WEIGHT_GRID] = a.as_slice().try_into().unwrap();
        let table = HorizonWeightTable::build(gt, at, T);
        assert_eq!(table.w_at_unit(0.0).to_bits(), table.weights()[0].to_bits());
        assert_eq!(
            table.w_at_unit(1.0).to_bits(),
            table.weights()[HORIZON_WEIGHT_GRID - 1].to_bits()
        );
        // x = 32.5/63 rounds up to 33; x = 31.6/63 rounds down to 32.
        let inv = (HORIZON_WEIGHT_GRID - 1) as f32;
        let x_up = 32.5 / inv;
        assert_eq!(
            table.w_at_unit(x_up).to_bits(),
            table.weights()[33].to_bits(),
            "midpoint rounds up"
        );
        let x_down = 31.6 / inv;
        assert_eq!(
            table.w_at_unit(x_down).to_bits(),
            table.weights()[32].to_bits(),
            "below midpoint rounds down"
        );
        // Absolute-time form scales through the stored inv_horizon.
        let t_mid = T * 32.0 / inv;
        assert_eq!(table.w_at(t_mid).to_bits(), table.weights()[32].to_bits());
        // Out-of-range clamps.
        assert_eq!(table.w_at(-1.0).to_bits(), table.weights()[0].to_bits());
        assert_eq!(
            table.w_at(2.0 * T).to_bits(),
            table.weights()[HORIZON_WEIGHT_GRID - 1].to_bits()
        );
    }

    #[cfg(all(test, any(debug_assertions, feature = "alloc_tracking")))]
    #[test]
    fn build_and_lookup_are_alloc_free() {
        use std::hint::black_box;
        let (g, a) = schedules(HORIZON_WEIGHT_GRID);
        let gt: &[f32; HORIZON_WEIGHT_GRID] = g.as_slice().try_into().unwrap();
        let at: &[f32; HORIZON_WEIGHT_GRID] = a.as_slice().try_into().unwrap();
        crate::alloc::reset_alloc_stats();
        let table = HorizonWeightTable::build(gt, at, T);
        let mut sink = 0.0f32;
        for i in 0..1024usize {
            sink += black_box(table.w_at(black_box(i as f32) * 0.01));
        }
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(count, 0, "G4: build + 1024 lookups allocated {count} times");
        assert!(sink.is_finite(), "sink must be consumed: {sink}");
    }

    // ---- remaining_horizon_t_sample (Issue 875 T2) ----

    const T_SAMPLE_MIN: f32 = 0.02 * T;
    const T_SAMPLE_MAX: f32 = 0.98 * T;

    #[test]
    fn t_sample_endpoints_and_monotone() {
        let lo = remaining_horizon_t_sample(0.0, T_SAMPLE_MIN, T_SAMPLE_MAX, T);
        let hi = remaining_horizon_t_sample(1.0, T_SAMPLE_MIN, T_SAMPLE_MAX, T);
        assert!(
            (lo - T_SAMPLE_MIN).abs() < 1e-4 * T,
            "u=0 -> t_min, got {lo}"
        );
        assert!(
            (hi - T_SAMPLE_MAX).abs() < 1e-4 * T,
            "u=1 -> t_max, got {hi}"
        );
        // Monotone in u across a dense sweep (non-decreasing).
        let mut prev = f32::NEG_INFINITY;
        for i in 0..=256 {
            let u = i as f32 / 256.0;
            let t = remaining_horizon_t_sample(u, T_SAMPLE_MIN, T_SAMPLE_MAX, T);
            assert!(t >= prev, "not monotone at u={u}: {t} < {prev}");
            prev = t;
        }
        // All draws land inside the range.
        for i in 0..=64 {
            let u = i as f32 / 64.0;
            let t = remaining_horizon_t_sample(u, T_SAMPLE_MIN, T_SAMPLE_MAX, T);
            assert!(
                (T_SAMPLE_MIN..=T_SAMPLE_MAX).contains(&t),
                "out of range: {t}"
            );
        }
    }

    #[test]
    fn t_sample_empirical_distribution_matches_cdf() {
        // Chi-square-style bucket check: 10 equal-CDF buckets, 8192 draws,
        // each bucket within ±20% of its expected count. Deterministic seed.
        let mut rng = fastrand::Rng::with_seed(8752);
        let n = 8192usize;
        let buckets = 10usize;
        let mut counts = [0usize; 10];
        for _ in 0..n {
            let u = rng.f32();
            let t = remaining_horizon_t_sample(u, T_SAMPLE_MIN, T_SAMPLE_MAX, T);
            // Invert through the CDF to find the bucket.
            let lo = T - T_SAMPLE_MIN;
            let hi2 = T - T_SAMPLE_MAX;
            let f = (lo * lo - (T - t) * (T - t)) / (lo * lo - hi2 * hi2);
            let b = (f * buckets as f32).floor() as usize;
            let b = b.min(buckets - 1);
            counts[b] += 1;
        }
        let expected = n / buckets;
        for (b, &c) in counts.iter().enumerate() {
            let dev = (c as isize - expected as isize).abs() as f32 / expected as f32;
            assert!(
                dev < 0.20,
                "bucket {b}: {c} vs expected {expected} (dev {:.3})",
                dev
            );
        }
        // And the law's shape: the bottom decile of the RANGE (low noise)
        // must hold more mass than the top decile.
        let mut low = 0usize;
        let mut high = 0usize;
        for _ in 0..n {
            let u = rng.f32();
            let t = remaining_horizon_t_sample(u, T_SAMPLE_MIN, T_SAMPLE_MAX, T);
            let span = T_SAMPLE_MAX - T_SAMPLE_MIN;
            if t < T_SAMPLE_MIN + 0.1 * span {
                low += 1;
            }
            if t > T_SAMPLE_MAX - 0.1 * span {
                high += 1;
            }
        }
        assert!(
            low > high,
            "remaining-horizon law must favor low noise: low decile {low} vs top decile {high}"
        );
    }

    #[test]
    fn t_sample_degenerate_fallbacks() {
        // Collapsed range -> constant t_min.
        let t = remaining_horizon_t_sample(0.7, 0.5, 0.5, 1.0);
        assert_eq!(t, 0.5);
        // Inverted range -> lerp can go below t_min (documented total-fn
        // behavior; callers validate).
        let t = remaining_horizon_t_sample(0.5, 0.8, 0.2, 1.0);
        assert_eq!(t, 0.5); // lerp midpoint of [0.2, 0.8]
        // Horizon not containing the range -> lerp.
        let t = remaining_horizon_t_sample(0.5, 0.3, 0.8, 0.7);
        assert_eq!(t, 0.55); // lerp midpoint
        // Horizon exactly t_max is legal (zero terminal weight).
        let t = remaining_horizon_t_sample(1.0, 0.1, 0.8, 0.8);
        assert!((t - 0.8).abs() < 1e-6);
        // Non-finite u -> t_min (density maximum, the safe low-noise end;
        // endpoints land within 1 ulp of the exact inverse — the clamp
        // home can round either way).
        let t = remaining_horizon_t_sample(f32::NAN, 0.1, 0.8, 1.0);
        assert!((t - 0.1).abs() < 1e-6);
        // Out-of-range u clamps.
        let t = remaining_horizon_t_sample(-5.0, 0.1, 0.8, 1.0);
        assert!((t - 0.1).abs() < 1e-6);
        let t = remaining_horizon_t_sample(5.0, 0.1, 0.8, 1.0);
        assert!((t - 0.8).abs() < 1e-6);
    }

    // ---- Issue 875 T3: truncation predicate + time-anneal schedule ----

    #[test]
    fn truncation_round_trip_and_endpoints() {
        for &eps in &[0.01f32, 0.05, 0.0936829, 0.25, 0.5, 0.9] {
            for &t_min in &[0.0f32, 0.02 * T, 0.1 * T] {
                let cut = terminal_truncation_ceiling(eps, t_min, T);
                let back = truncated_w_mass_fraction(cut, t_min, T);
                let rel = ((back - eps) / eps).abs();
                assert!(rel < 1e-5, "eps={eps} t_min={t_min}: back={back} rel={rel}");
            }
        }
        // Endpoints: eps=0 -> skip nothing (exactly T; x - 0.0 == x).
        assert_eq!(
            terminal_truncation_ceiling(0.0, 0.02 * T, T).to_bits(),
            T.to_bits()
        );
        // eps=1 -> t_min (subtraction rounding; tolerance, not bits).
        let cut_min = terminal_truncation_ceiling(1.0, 0.02 * T, T);
        assert!((cut_min - 0.02 * T).abs() < 1e-4);
        // Monotone: larger eps -> lower (or equal) ceiling.
        let mut prev = T;
        for i in 0..=64 {
            let eps = i as f32 / 64.0;
            let cut = terminal_truncation_ceiling(eps, 0.02 * T, T);
            assert!(
                cut <= prev,
                "ceiling must be non-increasing in eps: {cut} > {prev}"
            );
            prev = cut;
        }
        // Mass form endpoints + NaN/degenerate policy (conservative 1.0).
        assert_eq!(
            truncated_w_mass_fraction(T, 0.02 * T, T).to_bits(),
            0.0f32.to_bits(),
            "w(T)=0 anchor"
        );
        assert_eq!(truncated_w_mass_fraction(0.0, 0.02 * T, T), 1.0);
        assert_eq!(truncated_w_mass_fraction(f32::NAN, 0.02 * T, T), 1.0);
        assert_eq!(
            truncated_w_mass_fraction(0.5 * T, T, T),
            1.0,
            "T <= t_min degenerate"
        );
        assert_eq!(terminal_truncation_ceiling(f32::NAN, 0.02 * T, T), T);
        assert_eq!(terminal_truncation_ceiling(-1.0, 0.02 * T, T), T);
    }

    #[test]
    fn truncation_paper_alignment() {
        // The paper's [0.02T, 0.98T] -> [0.02T, 0.70T] is this predicate:
        // eps = ((1-0.70)/(1-0.02))^2 ~= 0.093683.
        let t_min = 0.02f32;
        let eps = truncated_w_mass_fraction(0.70, t_min, 1.0);
        let expect = (0.3f32 / 0.98).powi(2);
        assert!((eps - expect).abs() < 1e-6, "{eps} vs {expect}");
        assert!((eps - 0.0937).abs() < 5e-4, "{eps}");
        let cut = terminal_truncation_ceiling(expect, t_min, 1.0);
        assert!((cut - 0.70).abs() < 1e-6, "{cut}");
    }

    #[test]
    fn anneal_shape_law() {
        let sch = TimeAnnealRange::DEFAULT;
        let total = 200usize;
        // Pre-anneal: EXACTLY the flat posture (endpoint pin).
        for &iter in &[0usize, 1, 50, 100, 139] {
            let (lo, hi) = sch.range_at(iter, total);
            assert_eq!(lo.to_bits(), 0.02f32.to_bits());
            assert_eq!(hi.to_bits(), 0.98f32.to_bits(), "iter={iter}");
        }
        // Final iteration: EXACTLY the annealed end.
        let (lo, hi) = sch.range_at(total - 1, total);
        assert_eq!(lo.to_bits(), 0.02f32.to_bits());
        assert_eq!(hi.to_bits(), 0.70f32.to_bits());
        // Fixed floor, ordered range, non-increasing ceiling throughout.
        let mut prev_hi = f32::INFINITY;
        for iter in 0..total {
            let (lo, hi) = sch.range_at(iter, total);
            assert_eq!(
                lo.to_bits(),
                0.02f32.to_bits(),
                "floor moves at iter={iter}"
            );
            assert!(
                hi <= prev_hi,
                "ceiling rises at iter={iter}: {hi} > {prev_hi}"
            );
            assert!(hi >= lo, "range inverts at iter={iter}");
            prev_hi = hi;
        }
        // Linear ease: the midpoint of the anneal window sits at the
        // midpoint of the eased ceilings (f32 rounding only).
        let a = 0.7f32 * total as f32;
        let mid_iter = (a + ((total - 1) as f32 - a) / 2.0) as usize;
        let (_, hi) = sch.range_at(mid_iter, total);
        let want = 0.98 + ((mid_iter as f32 - a) / ((total - 1) as f32 - a)) * (0.70 - 0.98);
        assert!((hi - want).abs() < 1e-6, "mid-anneal {hi} vs {want}");
        // Determinism: repeated calls bit-equal.
        for &iter in &[0usize, 150, total - 1] {
            let x = sch.range_at(iter, total);
            let y = sch.range_at(iter, total);
            assert_eq!(x.0.to_bits(), y.0.to_bits());
            assert_eq!(x.1.to_bits(), y.1.to_bits());
        }
    }

    #[test]
    fn anneal_degenerate_and_fallbacks() {
        let sch = TimeAnnealRange::DEFAULT;
        // total = 1: the anneal window holds no iteration — flat posture.
        let (lo, hi) = sch.range_at(0, 1);
        assert_eq!((lo, hi), (0.02, 0.98));
        // total = 2: window [1.4, 2) holds no integer iteration either.
        let (_, hi2) = sch.range_at(1, 2);
        assert_eq!(hi2, 0.98);
        // Overrun: terminal posture (monotone extension).
        let (lo, hi) = sch.range_at(999, 100);
        assert_eq!((lo, hi), (0.02, 0.70));
        // Invalid fields fall back per-field to DEFAULT, validated against
        // the EFFECTIVE (post-fallback) fields.
        let bad = TimeAnnealRange {
            floor_frac: f32::NAN,
            ceil_start_frac: 2.0,
            ceil_end_frac: 0.95,
            anneal_frac: 0.0,
        };
        // floor NaN -> 0.02; ceil_start 2.0 (>1) -> 0.98; ceil_end 0.95 is
        // within [floor, 0.98] -> stays; anneal_frac 0 -> 0.30.
        let (_, hi_end) = bad.range_at(9, 10);
        assert!((hi_end - 0.95).abs() < 1e-6, "{hi_end}");
        let (_, hi_start) = bad.range_at(0, 10);
        assert_eq!(hi_start, 0.98);
        // ceil_end above a VALID ceil_start never takes effect: it falls
        // back to DEFAULT (0.70) and the ordering min() respects the
        // caller's ceiling — the terminal posture is the caller's 0.5,
        // never the invalid 0.9.
        let bad2 = TimeAnnealRange {
            ceil_start_frac: 0.5,
            ceil_end_frac: 0.9,
            ..TimeAnnealRange::DEFAULT
        };
        let (_, hi0) = bad2.range_at(0, 10);
        assert_eq!(hi0, 0.5, "pre-anneal uses the valid caller ceiling");
        let (_, hi2) = bad2.range_at(9, 10);
        assert_eq!(hi2, 0.5, "ceil_end must not exceed ceil_start: {hi2}");
        // total = 0 asserts (caller bug).
        let boomed = std::panic::catch_unwind(|| sch.range_at(0, 0));
        assert!(boomed.is_err(), "total=0 must assert");
    }

    #[test]
    fn anneal_f64_oracle() {
        let sch = TimeAnnealRange::DEFAULT;
        let total = 101usize;
        for iter in 0..total {
            let (_, hi) = sch.range_at(iter, total);
            let a = 0.7 * total as f64;
            let denom = (total as f64 - 1.0) - a;
            let p = ((iter as f64 - a) / denom).clamp(0.0, 1.0);
            let want = 0.98 + p * (0.70 - 0.98);
            assert!(
                (f64::from(hi) - want).abs() < 1e-6,
                "iter={iter}: {hi} vs {want}"
            );
        }
    }

    #[cfg(all(test, any(debug_assertions, feature = "alloc_tracking")))]
    #[test]
    fn anneal_and_truncation_are_alloc_free() {
        use std::hint::black_box;
        let sch = TimeAnnealRange::DEFAULT;
        crate::alloc::reset_alloc_stats();
        let mut sink = 0.0f32;
        for i in 0..1024usize {
            let (lo, hi) = sch.range_at(black_box(i), black_box(1024));
            sink += black_box(lo) + black_box(hi);
            sink += black_box(terminal_truncation_ceiling(
                black_box(i as f32) * 0.001,
                0.02,
                1.0,
            ));
            sink += black_box(truncated_w_mass_fraction(black_box(hi), 0.02, 1.0));
        }
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(
            count, 0,
            "G4: 3x1024 schedule calls allocated {count} times"
        );
        assert!(sink.is_finite(), "sink must be consumed: {sink}");
    }

    #[test]
    fn t_sample_f64_oracle() {
        // Independent f64 recomputation of the inverse CDF, u sweep.
        for i in 0..=128 {
            let u = i as f32 / 128.0;
            let got = remaining_horizon_t_sample(u, T_SAMPLE_MIN, T_SAMPLE_MAX, T) as f64;
            let lo = (T as f64) - T_SAMPLE_MIN as f64;
            let hi = (T as f64) - T_SAMPLE_MAX as f64;
            let want = (T as f64) - ((1.0 - u as f64) * lo * lo + u as f64 * hi * hi).sqrt();
            assert!(
                (got - want).abs() < 1e-5 * T as f64,
                "u={u}: {got} vs {want}"
            );
        }
    }
}
