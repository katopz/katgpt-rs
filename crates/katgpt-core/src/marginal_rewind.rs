//! `marginal_rewind` — calibrated noise-backtrack for interpolant flow
//! states (Issue 746 Row 2; arXiv:2609.11801 "Thinking with Looped Flows",
//! Eq 18 minus the denoiser).
//!
//! # The math
//!
//! A flow state lives on the interpolant path `x_t = t·x* + (1−t)·ε`
//! (signal weight `t`, prior/noise weight `1−t`; `t = 1` is fully
//! committed). **Rewinding** from level `t` to an earlier level `s < t`
//! transports the CURRENT state — without knowing the clean point `x*` —
//! onto the exact level-`s` marginal:
//!
//! ```text
//! a      = s / t                        (equivalently the γ-dial: a = clip(1−γΔt), s = a·t)
//! x̄_s    = a·x_t + sqrt((1−s)² − (a−s)²) · ε'
//! ```
//!
//! The marginal-preservation identity (why `s = a·t` is forced):
//!
//! ```text
//! a²·(1−t)²  +  (1−s)² − (a−s)²  =  (1−s)²      ⟺  a·(1−t) = a − s  ⟺  s = a·t
//! ```
//!
//! i.e. the noise surviving the shrink (`a·(1−t) = a−s`) plus the fresh
//! injection sums to EXACTLY the level-`s` marginal variance `(1−s)²`.
//! The companion forward schedule (Euler-as-lerp with
//! `r_i = Δt/(1−t_i)` on a uniform grid) annihilates the initial prior's
//! weight to exactly 0 at `t = 1` (telescoping `∏(1−r_j) = (1−t_n)/(1−t₀)`)
//! — that schedule is exercised by the PoC harness
//! (`tests/marginal_rewind_poc.rs`) and deliberately NOT shipped here
//! (Issue 746 Row 1 verdict: no consumer with a time-grid need).
//!
//! # What this is / is NOT (signal-diff vs shipped cousins)
//!
//! - NOT [`crate::renoise_ce`] (feature `renoise_ce`): that perturbs a
//!   COMPLETED state with uncalibrated additive noise and scores the
//!   re-resolution drift — the candidate is returned unchanged. This
//!   operator transports a state to an earlier confidence level at the
//!   exact marginal variance, for recovery-by-re-integration.
//! - NOT `dllm_solver::q_sample_step` (feature `q_sample_solver`): the
//!   DDPM q-sample re-noises a CLEAN estimate `x0_hat` at an ᾱ level —
//!   it needs the denoiser's clean point. This operator rewinds the
//!   CURRENT noisy state with no clean-point knowledge (that is the
//!   Eq 18 contribution).
//! - NOT [`crate::saddle_escape::apply_kick`] (feature `saddle_escape`):
//!   that fires an uncalibrated `eps·u` escape kick whose magnitude comes
//!   from config + geometric decay, not from a marginal law. The PoC
//!   benchmarks the two head-to-head at equal injected budget.
//! - Composes with `cgsp` collapse detection (feature `cgsp`): detection
//!   is cgsp's job; this is the reactive recovery operator detection
//!   currently lacks (cgsp's dual-pool routing is proactive only).
//!
//! # Numeric contract
//!
//! - `rewind_noise_std` clamps the radicand at 0 (`s ≈ t` in f32 can go
//!   a hair negative) — a `s == t` rewind is an exact identity.
//! - Contract violations (out-of-domain `t`/`s`, length mismatch) panic
//!   with named messages (the `variable_rank_domain_expert` convention).
//! - Non-finite `t`/`s` propagate NaN through the arithmetic — callers
//!   gate recovery on finite observables (the `saddle_escape` NaN
//!   contract: never act on a non-finite signal).
//!
//! # Hot-path design
//!
//! - `rewind_into` is a zero-allocation lerp: caller supplies the state
//!   slice, the fresh-noise slice, and the output buffer (G4 gate:
//!   `tests/marginal_rewind_alloc_check.rs`).
//! - RNG policy stays with the CALLER — `eps` is standard-normal draws
//!   from whatever determinism contract the consumer runs (BLAKE3-seeded
//!   streams, fastrand, test LCGs). The module itself is RNG-free.
//!
//! # Opt-in
//!
//! Feature `marginal_rewind` (Issue 746). Quality claims are gated on the
//! defend-wrong PoC (`tests/marginal_rewind_poc.rs`: calibrated backtrack
//! vs additive renoise at equal budget vs naive full-variance resample vs
//! hard restart, on a stuck-attractor synthetic); promotion to default is
//! owner-gated (the `saddle_escape`/525 precedent).

/// Rewind plan solved from `(t, s)` — the pre-computed operator constants.
///
/// Building once and applying many times keeps per-application cost at one
/// multiply-add per element (hot-loop rule: compute once per recovery,
/// not per element).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewindPlan {
    /// Target level `s` (equals `a·t` by construction).
    pub s: f32,
    /// State shrink factor `a = s / t`.
    pub a: f32,
    /// Fresh-noise std `sqrt((1−s)² − (a−s)²)` (≥ 0; 0 iff `s == t`).
    pub sigma: f32,
}

/// Solve the rewind `(t → s)` operator constants.
///
/// # Panics
/// - `!(0.0 < s && s <= t && t <= 1.0)` (a rewind goes to an earlier
///   level on the open-closed path `(0, 1]`).
pub fn rewind_plan(t: f32, s: f32) -> RewindPlan {
    assert!(
        t > 0.0 && t <= 1.0,
        "rewind level t must be in (0, 1], got {t}"
    );
    assert!(
        s > 0.0 && s <= t,
        "rewind target s must be in (0, t] = (0, {t}], got {s}"
    );
    let a = s / t;
    // a − s = a·(1 − t) is the noise scale surviving the shrink; the fresh
    // injection tops the total up to the exact level-s marginal (1 − s).
    let residual = a - s;
    let radicand = (1.0 - s) * (1.0 - s) - residual * residual;
    RewindPlan {
        s,
        a,
        // f32 rounding can push the radicand a hair under 0 near s == t.
        sigma: radicand.max(0.0).sqrt(),
    }
}

/// Solve the γ-dial form (Eq 18 literal): `a = clip(1 − γ·Δt)`, `s = a·t`.
///
/// `γ = 0` is the identity (`a = 1`, `s = t`, `sigma = 0`); larger `γ`
/// (at fixed `Δt`) rewinds deeper; `γ·Δt ≥ 1` clamps to the full rewind
/// `a = 0`, `s = 0` — but note `s = 0` is outside the path contract, so
/// the clamp lands at `a = 0` with the caller treating the output as a
/// fresh prior draw. Callers wanting an in-path rewind should keep
/// `γ·Δt < 1 − t₀`.
///
/// # Panics
/// - `gamma < 0.0` or `dt <= 0.0` (the dial only rewinds).
/// - `t` outside `(0, 1]` (delegates to [`rewind_plan`]'s contract).
pub fn gamma_dial_plan(t: f32, gamma: f32, dt: f32) -> RewindPlan {
    assert!(gamma >= 0.0, "gamma dial is rewind-only, got {gamma}");
    assert!(dt > 0.0, "dt must be positive, got {dt}");
    let a = (1.0 - gamma * dt).clamp(0.0, 1.0);
    let s = a * t;
    if s == 0.0 {
        // Full rewind: a = 0 annihilates the state; sigma is the full
        // prior std. Expressed as a plan with s clamped onto the path
        // floor would misreport the marginal, so surface it directly.
        return RewindPlan {
            s: 0.0,
            a: 0.0,
            sigma: 1.0,
        };
    }
    rewind_plan(t, s)
}

/// Apply a solved rewind: `out = a·x + sigma·eps`.
///
/// `eps` is CALLER-SUPPLIED standard-normal draws (`N(0,1)` per element)
/// — the module is RNG-free by design (determinism contracts belong to
/// the consumer).
///
/// # Panics
/// - `x.len() != out.len()` or `eps.len() != out.len()`.
pub fn rewind_into(x: &[f32], plan: RewindPlan, eps: &[f32], out: &mut [f32]) {
    assert!(
        x.len() == out.len() && eps.len() == out.len(),
        "rewind_into length mismatch: x={}, eps={}, out={}",
        x.len(),
        eps.len(),
        out.len()
    );
    let a = plan.a;
    let sigma = plan.sigma;
    for i in 0..out.len() {
        out[i] = a * x[i] + sigma * eps[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only splitmix64 (test code may define inline helpers — the
    /// substrate-first exemption; the `saddle_escape_poc` convention).
    fn mix64(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Standard-normal draw via Box-Muller from two uniforms.
    fn normal(seed: u64) -> f32 {
        let u1 = (mix64(seed) >> 11) as f64 / (1u64 << 53) as f64;
        let u2 = (mix64(seed ^ 0xDEAD_BEEF_CAFE_F00D) >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }

    #[test]
    fn identity_when_s_equals_t() {
        let plan = rewind_plan(0.7, 0.7);
        assert_eq!(plan.a, 1.0);
        assert_eq!(plan.sigma, 0.0);
        let x = [0.25, -1.5, 3.0];
        let eps = [9.0, 9.0, 9.0]; // sigma = 0 ⇒ eps must not matter
        let mut out = [0.0; 3];
        rewind_into(&x, plan, &eps, &mut out);
        assert_eq!(out, x);
    }

    #[test]
    fn gamma_zero_is_identity_and_deeper_gamma_rewinds_more() {
        let id = gamma_dial_plan(0.7, 0.0, 0.1);
        assert_eq!((id.a, id.sigma), (1.0, 0.0));
        let mut prev_s = 0.7_f32;
        for gamma in [0.25_f32, 0.5, 1.0, 2.0, 4.0] {
            let p = gamma_dial_plan(0.7, gamma, 0.1);
            assert!(
                p.s < prev_s || p.a == 0.0,
                "gamma {gamma} must rewind deeper"
            );
            prev_s = p.s.min(prev_s);
            // The dial form must land exactly on the s = a·t identity.
            assert_eq!(p.s, p.a * 0.7);
        }
        // Full-rewind clamp: gamma·dt ≥ 1 ⇒ a = 0, sigma = 1 (fresh prior).
        let full = gamma_dial_plan(0.7, 100.0, 0.1);
        assert_eq!((full.a, full.s, full.sigma), (0.0, 0.0, 1.0));
    }

    #[test]
    fn marginal_variance_preserved() {
        // x_t = t·x* + (1−t)·ε; rewind to s; Var(out − s·x*) must equal
        // (1−s)² — the identity this module exists to guarantee.
        let (t, s, x_star) = (0.7_f32, 0.25_f32, 2.0_f32);
        let plan = rewind_plan(t, s);
        let k = 24_576_usize;
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for i in 0..k {
            let eps_t = normal(i as u64);
            let x_t = t * x_star + (1.0 - t) * eps_t;
            let eps_new = normal(i as u64 ^ 0xA5A5_5A5A);
            let out = plan.a * x_t + plan.sigma * eps_new;
            let centered = f64::from(out) - f64::from(s * x_star);
            sum += centered;
            sum_sq += centered * centered;
        }
        let mean = sum / k as f64;
        let var = sum_sq / k as f64 - mean * mean;
        let target = f64::from((1.0 - s) * (1.0 - s));
        // 5 standard errors of a chi-square(1)-scaled variance estimate.
        let se = target * (2.0 / k as f64).sqrt() * 5.0;
        assert!(
            (var - target).abs() < se,
            "marginal variance drifted: {var:.6} vs target {target:.6} (±{se:.6})"
        );
        // And the mean lands on the scaled signal, not on the old level.
        assert!((mean - 0.0).abs() < 5.0 * target.sqrt() / (k as f64).sqrt());
    }

    #[test]
    fn sigma_grows_monotonically_with_depth() {
        let t = 0.7_f32;
        let mut prev = 0.0_f32;
        for s in [0.65_f32, 0.5, 0.35, 0.2, 0.05] {
            let p = rewind_plan(t, s);
            assert!(p.sigma > prev, "deeper rewind must inject more noise");
            prev = p.sigma;
        }
    }

    #[test]
    fn no_nan_near_the_s_equals_t_boundary() {
        // f32 rounding must not produce a negative radicand → NaN.
        for k in 0..200 {
            let s = 0.7_f32 - k as f32 * 1e-4;
            let p = rewind_plan(0.7, s);
            assert!(p.sigma.is_finite());
        }
    }

    #[test]
    #[should_panic(expected = "rewind target s must be in (0, t]")]
    fn s_above_t_panics() {
        let _ = rewind_plan(0.5, 0.6);
    }

    #[test]
    #[should_panic(expected = "rewind level t must be in (0, 1]")]
    fn t_above_one_panics() {
        let _ = rewind_plan(1.2, 0.5);
    }

    #[test]
    #[should_panic(expected = "gamma dial is rewind-only")]
    fn negative_gamma_panics() {
        let _ = gamma_dial_plan(0.7, -0.1, 0.1);
    }

    #[test]
    #[should_panic(expected = "rewind_into length mismatch")]
    fn length_mismatch_panics() {
        let mut out = [0.0_f32; 2];
        rewind_into(&[0.0; 3], rewind_plan(0.7, 0.3), &[0.0; 3], &mut out);
    }
}
