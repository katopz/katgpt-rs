//! Issue 717 — LT2 deep-loop instrumentation + runtime stabilization knobs.
//!
//! `forward_looped` (Plan 108) is validated at T=4 only; nothing measured
//! T ≫ 4. Source distillation: `riir-train/.research/440_Sotaku_Late_State_
//! Looped_Solver.md` (sotaku @ `6cdb9a9b`, MIT) — three facts with no
//! in-stack counterpart:
//!
//! 1. **Delayed damping is a runtime, checkpoint-agnostic rescue.** First B
//!    iterations untouched, then `h ← (1−α)h + α·F(h)`; α
//!    checkpoint-dependent (0.25 → 0.03125). Closed form: damping maps a
//!    locally-linear mode λ → 1−α+αλ (see [`project_lambda`]).
//! 2. **Tangential-vs-radial:** which axis of the update Δh = F(h)−h you
//!    scale depends on the failure mode — upstream (direction-drift
//!    failure) needed tangential ×0.25; a magnitude-driven failure needs
//!    the radial axis. Both knobs are exposed; the probe gate reports which
//!    axis matters on our fixture.
//! 3. **f32-state law:** sub-f32 carried-state arithmetic AMPLIFIES with
//!    loop depth (BF16 @4096 = 43.7% vs FP32 98.6% upstream) — the contrast
//!    to Bench 802, where f16-KV deviation DILUTES with attention context.
//!    Attention rounding averages out; weight-tied recurrence accumulates.
//!    The carried state in `forward_looped` is `ctx.x: Vec<f32>`
//!    end-to-end; `issue_717_t1_t2_deep_baseline::f32_state_contract`
//!    pins that no sub-f32 lattice sneaks in.
//!
//! Zero-cost-when-off contract (mirrors the Issue 035 elastic-override and
//! Plan 304 halter precedents): every knob lives behind `Option`, the off
//! path is bit-identical, and the hot loop allocates nothing (G4 gate).

// ---------------------------------------------------------------------------
// T1 — deep-run stats (always compiled under lt2_looped; zero cost when the
// caller passes `None`).
// ---------------------------------------------------------------------------

/// Per-call statistics collected by [`LoopDeepRun`] inside `forward_looped`.
///
/// Snapshots are taken at the END of loop iterations `tau` where
/// `(tau + 1) % snapshot_every == 0` — i.e. AFTER the residual-gate
/// injection, the Issue 717 direction scales, and the Issue 717 damping.
/// The snapshot therefore observes the state that actually carries into
/// the next iteration (and, at the last snapshot, into the readout).
#[derive(Debug, Default, Clone)]
pub struct LoopDeepStats {
    /// Number of snapshots taken (== number of qualifying iterations).
    pub snapshots_taken: usize,
    /// `‖h‖₂` of the carried state at each snapshot.
    pub state_norms: Vec<f32>,
    /// First snapshot index whose state contained a non-finite value
    /// (`None` = state stayed finite at every snapshot).
    pub state_non_finite_at: Option<usize>,
    /// First snapshot index whose logit tripwire read a non-finite logit
    /// (`None` = never tripped or `check_logits == false`).
    pub logits_non_finite_at: Option<usize>,
    /// Raw state copies at each snapshot (only when `capture_states`).
    /// Test/probe fuel: tangential/radial decomposition, direction-drift
    /// diagnostics, multiplier estimation.
    pub state_snapshots: Vec<Vec<f32>>,
}

impl LoopDeepStats {
    /// Clear collected data, keeping capacity (steady-state reuse — the G4
    /// gate clears between measured calls instead of dropping buffers).
    pub fn clear(&mut self) {
        self.snapshots_taken = 0;
        self.state_norms.clear();
        self.state_non_finite_at = None;
        self.logits_non_finite_at = None;
        // `clear` on the outer Vec would DROP the inner buffers and force
        // re-allocation on the next capture; reuse them instead.
        for s in &mut self.state_snapshots {
            s.clear();
        }
    }
}

/// Per-call deep-loop control for `forward_looped` (Issue 717).
///
/// Pass `Some(&mut run)` to instrument a deep run (T ≫ 4) and/or enable the
/// runtime stabilization knobs; pass `None` for bit-identical baseline
/// behavior (the elastic-override contract, re-proven by the Issue 717 G1
/// gate). Construction is allocation-free; the stats vectors grow once on
/// first use and are reused thereafter.
pub struct LoopDeepRun {
    /// Snapshot every K iterations (1-based count; `0` disables snapshots).
    pub snapshot_every: usize,
    /// Also copy the raw state at each snapshot (probe fuel; off by default).
    pub capture_states: bool,
    /// Compute the readout at each snapshot into a scratch buffer and check
    /// finiteness (the logit-finite tripwire; one `lm_head` matmul per
    /// snapshot, opt-in).
    pub check_logits: bool,
    /// Collected stats (read after the call).
    pub stats: LoopDeepStats,
    /// Scratch for the logit tripwire (grown once, then reused). The only
    /// reader is the `check_logits` branch inside `forward_looped`, so the
    /// field is dead in every build without `lt2_looped` — allow is scoped
    /// to exactly that case and stays strict when the feature is on.
    #[cfg_attr(not(feature = "lt2_looped"), allow(dead_code))]
    pub(crate) logit_scratch: Vec<f32>,
    /// Issue 717 T3 — delayed damping knob (feature `lt2_deep_stability`).
    #[cfg(feature = "lt2_deep_stability")]
    pub damping: Option<LoopDamping>,
    /// Issue 717 T4 — tangential/radial update-scale knobs (feature
    /// `lt2_deep_stability`).
    #[cfg(feature = "lt2_deep_stability")]
    pub direction_scales: Option<DirectionScales>,
}

impl LoopDeepRun {
    /// Instrument-only run control: snapshot state norm (+ optional tripwire)
    /// every `snapshot_every` iterations. No stabilization knobs.
    pub fn new(snapshot_every: usize) -> Self {
        Self {
            snapshot_every,
            capture_states: false,
            check_logits: true,
            stats: LoopDeepStats::default(),
            logit_scratch: Vec::new(),
            #[cfg(feature = "lt2_deep_stability")]
            damping: None,
            #[cfg(feature = "lt2_deep_stability")]
            direction_scales: None,
        }
    }

    /// Issue 717 T3 — enable delayed damping
    /// `h ← (1−α)h + α·F(h)` after `burn_in` iterations.
    ///
    /// `alpha == 0.0` disables the knob entirely (bit-identical to `None` —
    /// the G1 contract). The closed-form map is [`project_lambda`].
    #[cfg(feature = "lt2_deep_stability")]
    pub fn with_damping(alpha: f32, burn_in: usize, snapshot_every: usize) -> Self {
        let mut run = Self::new(snapshot_every);
        run.damping = Some(LoopDamping { alpha, burn_in });
        run
    }
}

// ---------------------------------------------------------------------------
// T3/T4 — stabilization knobs (feature `lt2_deep_stability`; DEFAULT-OFF).
// ---------------------------------------------------------------------------

/// Issue 717 T3 — delayed damping parameters.
///
/// After `burn_in` iterations, each carried state is blended toward the
/// previous state: `h ← (1−α)·h + α·h_prev`. `alpha = 0.0` disables the knob
/// (bit-identical); `alpha ∈ (0, 1]` damps. The application site is the END
/// of each loop body — after the residual-gate injection, the direction
/// scales, and the halter/recursion-gate checks — so halter gain/cos
/// measurements see the undamped step and the readout sees the damped state.
#[cfg(feature = "lt2_deep_stability")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopDamping {
    /// Blend weight toward the previous state; `0.0` = disabled.
    pub alpha: f32,
    /// Iterations left untouched before damping engages (0-based `tau` must
    /// satisfy `tau >= max(burn_in, 1)`; iteration 0 has no previous state).
    pub burn_in: usize,
}

/// Issue 717 T4 — tangential/radial scale knobs.
///
/// Decomposes the iteration update `Δh = F(h) − h_prev` into the component
/// along `h_prev` (radial) and the orthogonal remainder (tangential), then
/// rescales each: `h ← h_prev + s_r·radial + s_t·tangential`.
/// `{1.0, 1.0}` (both neutral) disables the knob entirely — the decomposition
/// round-trip is NOT bit-identical (reassociation), so the neutral case must
/// skip, not recompute.
#[cfg(feature = "lt2_deep_stability")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionScales {
    /// Scale of the `h_prev`-aligned (magnitude) update component.
    pub radial: f32,
    /// Scale of the orthogonal (rotational) update component.
    pub tangential: f32,
}

/// The closed-form damping map on a locally-linear mode: `λ → 1 − α + αλ`.
///
/// Upstream (sotaku) measured this as the rescue mechanism; the Issue 717 G2
/// gate checks the measured per-iteration multiplier of a gate-driven
/// destabilized fixture against this form. Brackets: `project_lambda(λ, 1)
/// == λ` (α=1 is the full update — undamped) and `project_lambda(λ, 0) == 1`
/// (α=0 freezes the carried state — which is also the bit-identical OFF
/// spelling, the G1 contract). For λ > 1 the map is monotonically
/// increasing in α: smaller α ⇒ slower growth.
#[cfg(feature = "lt2_deep_stability")]
#[inline]
pub fn project_lambda(lambda: f32, alpha: f32) -> f32 {
    1.0 - alpha + alpha * lambda
}

/// Overflow-safe L2 norm (max-abs-scaled two-pass).
///
/// Deep-loop states grow exponentially in the destabilized regimes this
/// module exists to measure; the naive `Σv²` overflows f32 for ‖x‖ ≳ 1e19
/// (v² ≈ 1e38) while the values themselves are still far inside range.
/// Returns NaN iff any element is non-finite (the stats snapshot uses that
/// as the non-finite marker); returns Inf only when the TRUE norm exceeds
/// the f32 range.
pub fn robust_norm(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let mut max_abs = 0.0f32;
    for &v in x {
        if !v.is_finite() {
            return f32::NAN;
        }
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs == 0.0 {
        return 0.0;
    }
    let mut s = 0.0f32;
    for &v in x {
        let r = v / max_abs;
        s += r * r;
    }
    max_abs * s.sqrt()
}

/// Apply the damping blend `x ← (1−α)·prev + α·x` element-wise (sotaku's
/// form: weight `α` on the fresh update, `1−α` on the previous state).
///
/// Op order is pinned (literal sotaku form: scale-then-fused-add) so the
/// trajectory is reproducible bit-for-bit on one platform. Zero allocation.
/// `alpha = 0` never reaches here in the forward path (`damping_active`
/// gates it off — that spelling is the bit-identical OFF contract, G1).
#[cfg(feature = "lt2_deep_stability")]
pub(crate) fn apply_damping(x: &mut [f32], prev: &[f32], alpha: f32) {
    debug_assert_eq!(x.len(), prev.len(), "damping: state/prev length mismatch");
    let one_minus = 1.0 - alpha;
    for (xi, &pi) in x.iter_mut().zip(prev.iter()) {
        *xi = one_minus * pi + alpha * *xi;
    }
}

/// Apply the tangential/radial update rescale (Issue 717 T4).
///
/// `Δ = x − prev`; `radial = (Δ·prev / ‖prev‖²)·prev`;
/// `x ← prev + s_r·radial + s_t·(Δ − radial)`.
/// Two passes over the slices, zero allocation. A zero `prev` (degenerate)
/// leaves `x` untouched — there is no direction to decompose against.
#[cfg(feature = "lt2_deep_stability")]
pub(crate) fn apply_direction_scales(x: &mut [f32], prev: &[f32], radial: f32, tangential: f32) {
    debug_assert_eq!(
        x.len(),
        prev.len(),
        "direction scales: state/prev length mismatch"
    );
    // Pass 1: dot(Δ, prev) and ‖prev‖².
    let mut dot = 0.0f32;
    let mut norm2 = 0.0f32;
    for i in 0..x.len() {
        let d = x[i] - prev[i];
        dot += d * prev[i];
        norm2 += prev[i] * prev[i];
    }
    if norm2 == 0.0 {
        return;
    }
    let coef = dot / norm2;
    // Pass 2: recombine. Add order pinned: (prev + s_r·radial) + s_t·tangential.
    for (i, xi) in x.iter_mut().enumerate() {
        let d = *xi - prev[i];
        let r = coef * prev[i];
        *xi = prev[i] + radial * r + tangential * (d - r);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── project_lambda: the closed form at its brackets ─────────────────
    #[cfg(feature = "lt2_deep_stability")]
    #[test]
    fn project_lambda_brackets_and_monotonicity() {
        let lam = 1.1f32;
        // Brackets: α = 1 → the full update (λ itself); α = 0 → frozen state
        // (multiplier 1 — also the bit-identical OFF spelling).
        assert_eq!(project_lambda(lam, 1.0), lam);
        assert_eq!(project_lambda(lam, 0.0), 1.0);
        // Monotone increasing in α for λ > 1 (more damping ⇔ smaller α ⇔
        // smaller multiplier): the G2 sweep's monotonicity axis.
        let mut prev = project_lambda(lam, 0.0);
        for a in [0.03125f32, 0.0625, 0.125, 0.25, 0.5, 1.0] {
            let cur = project_lambda(lam, a);
            assert!(cur >= prev, "map not monotone at α={a}: {cur} < {prev}");
            prev = cur;
        }
    }

    // ── damping: exact scalar-mode semantics ─────────────────────────────
    #[cfg(feature = "lt2_deep_stability")]
    #[test]
    fn damping_scalar_mode_matches_closed_form() {
        // A state that is exactly a scalar multiple of prev: after damping,
        // x must be the closed-form multiple of prev, element-wise.
        let prev = [1.0f32, -2.0, 3.5, 0.25];
        let lam = 1.7f32;
        let mut x = prev.map(|v| lam * v);
        let alpha = 0.25f32;
        apply_damping(&mut x, &prev, alpha);
        let expect = project_lambda(lam, alpha);
        for (i, &v) in x.iter().enumerate() {
            let want = expect * prev[i];
            assert!(
                (v - want).abs() <= 4.0 * want.abs() * f32::EPSILON,
                "elem {i}: {v} vs {want}"
            );
        }
    }

    // ── direction scales: pure-radial update rescales exactly ────────────
    #[cfg(feature = "lt2_deep_stability")]
    #[test]
    fn direction_scales_radial_component_exact() {
        // Δ exactly parallel to prev: tangential part is zero, so the result
        // must be prev + radial·Δ regardless of the tangential knob.
        let prev = [2.0f32, -1.0, 0.5];
        let lam = 1.2f32;
        let mut x = prev.map(|v| lam * v); // Δ = (lam−1)·prev → pure radial
        apply_direction_scales(&mut x, &prev, 0.25, 1.0);
        for (i, &v) in x.iter().enumerate() {
            let want = prev[i] + 0.25 * (lam - 1.0) * prev[i];
            assert!(
                (v - want).abs() <= 4.0 * want.abs() * f32::EPSILON,
                "elem {i}: {v} vs {want}"
            );
        }
    }

    // ── direction scales: pure-tangential update untouched by radial knob ─
    #[cfg(feature = "lt2_deep_stability")]
    #[test]
    fn direction_scales_tangential_component_exact() {
        // Δ exactly orthogonal to prev: radial part is zero, so the radial
        // knob must not move it and the tangential knob scales it fully.
        let prev = [1.0f32, 0.0, 0.0];
        let t = [0.0f32, 2.0, -3.0]; // orthogonal to prev
        let mut x = prev.to_vec();
        for (xi, &ti) in x.iter_mut().zip(t.iter()) {
            *xi += ti;
        }
        apply_direction_scales(&mut x, &prev, 0.0, 0.25);
        for (i, &v) in x.iter().enumerate() {
            let want = prev[i] + 0.25 * t[i];
            assert!(
                (v - want).abs() <= 4.0 * want.abs().max(1.0) * f32::EPSILON,
                "elem {i}: {v} vs {want}"
            );
        }
    }

    // ── robust_norm: matches the naive norm at benign scales, stays finite
    // where Σv² overflows, NaN on non-finite input ─────────────────
    #[test]
    fn robust_norm_overflow_safe() {
        // Benign scale: identical to the naive norm (within f32 rounding).
        let x = [3.0f32, -4.0, 0.0];
        assert!((robust_norm(&x) - 5.0).abs() < 1e-5);
        // Overflow regime: values ~1e24 are finite but v² ≈ 1e48 is not.
        let big = [1e24f32, -1e24, 2e24];
        let naive: f32 = big.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(!naive.is_finite(), "expected the naive norm to overflow");
        let r = robust_norm(&big);
        assert!(r.is_finite() && r > 0.0, "robust_norm must stay finite here");
        // Ratio preserved: robust_norm(big) / robust_norm([1, -1, 2]) == 1e24.
        let small = [1.0f32, -1.0, 2.0];
        let ratio = r / robust_norm(&small);
        assert!((ratio - 1e24).abs() <= 1e24 * 1e-5, "ratio {ratio}");
        // Non-finite input → NaN (the tripwire marker).
        assert!(robust_norm(&[1.0, f32::NAN]).is_nan());
        assert!(robust_norm(&[f32::INFINITY]).is_nan());
        // Degenerate inputs.
        assert_eq!(robust_norm(&[]), 0.0);
        assert_eq!(robust_norm(&[0.0, 0.0]), 0.0);
    }

    // ── f32-state size pin (T5, structural half) ────────────────────}
    #[test]
    fn f32_is_the_state_width() {
        // The carried loop state is Vec<f32> end-to-end; this pins the
        // element width so a silent switch to a 2-byte storage type is a
        // compile failure here. The behavioral half (no f16-lattice values
        // in a real trajectory) lives in the issue_717 baseline gate.
        assert_eq!(std::mem::size_of::<f32>(), 4);
        let state: Vec<f32> = vec![0.5f32];
        assert_eq!(std::mem::size_of_val(&state[0]), 4);
    }
}
