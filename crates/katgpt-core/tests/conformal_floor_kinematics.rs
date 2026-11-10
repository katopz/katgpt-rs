//! Plan 578 T2.5 — "Report the Floor" comparison for
//! `kinematics::extrapolation_horizon`'s predictive interval.
//!
//! **UQ-bearing primitive:** the k-step extrapolation's predictive interval —
//! observation noise propagated through the extrapolator's observation
//! weights (`√(wss+1)`), with σ̂ estimated online from the **full-ladder
//! residual** and the ladder order gated by noise significance.
//!
//! **The floor** (pinned by the Report-the-Floor policy, Issue 010):
//! `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` m=1 —
//! [`FloorAdapter`] via [`run_floor_comparison`].
//!
//! ## Estimator design (what the adapter composes)
//!
//! 1. **σ̂ from the order-3 residual** — never the screened order's. A
//!    too-low order's residual contains the *motion*, not the noise: feeding
//!    it back into the order screen is self-referential and death-spirals
//!    (order-0 → residual = the growing increment → σ̂ inflates → the screen
//!    rejects the very velocity signal that would fix the order — measured:
//!    the parabola corpus pinned order 0 with σ̂ → 37 and 174-wide intervals).
//!    The full-ladder prediction is structure-correct on any smooth motion,
//!    so its residual is honest amplified noise (`√70·σ`), deconvolved by
//!    the same factor.
//! 2. **EMA-smoothed velocity point** — `pos + δ̂` with δ̂ an EMA of ∇¹/Δt.
//!    The raw 2-sample velocity has √2·σ noise; smoothing at β reduces the
//!    injected variance to `2β/(2−β)·σ²` (the α-β filter — the textbook
//!    noisy-kinematics estimator). The interval reflects it:
//!    `z·σ̂·√(1 + 2β/(2−β))`.
//!
//! ## Corpora
//!
//! 1. **noisy uniform** — constant velocity + i.i.d. noise.
//! 2. **noisy parabola** — constant acceleration + noise: the floor's
//!    conformal drift-correction cannot track a moving drift (its coverage
//!    collapses to 0.17 — measured); the kinematic structure can.
//! 3. **white noise** — the degenerate floor-favorable corpus (the floor's
//!    own doc): recorded honestly.
//!
//! ## Run
//!
//! ```bash
//! cargo test -p katgpt-core --test conformal_floor_kinematics \
//!   --features kinematic_rollout,conformal_predictive_intervals -- --nocapture
//! ```

use katgpt_core::conformal::{
    PredictiveInterval, PredictiveOutput, UqPrimitiveUnderTest, run_floor_comparison,
};
use katgpt_core::kinematics::perception::normal_two_sided_z;
use katgpt_core::kinematics::{
    KinState, SQRT2, SQRT6, SQRT20, Sched, extrapolation_weight_ss,
    kinematic_extrapolate_capped_into,
};

/// EMA smoothing for the velocity point estimate.
const SMOOTH_BETA: f32 = 0.25;

/// The kinematic UQ primitive under test (see the module doc for the design).
struct KinUq {
    state: KinState<1>,
    alpha: f32,
    /// EMA of |full-ladder residual| — the honest σ̂ source.
    mean_abs_r3: f32,
    /// EMA-smoothed velocity (the point estimate's slope).
    smooth_vel: f32,
    /// EMAs of the ladder-coefficient magnitudes (the screen inputs).
    ema_vel: f32,
    ema_acc: f32,
    ema_jerk: f32,
    /// Warmup observations before an interval is produced.
    n: u32,
}

impl KinUq {
    fn new(alpha: f32) -> Self {
        Self {
            state: KinState::<1>::new(1.0).expect("dt=1"),
            alpha,
            mean_abs_r3: 0.0,
            smooth_vel: 0.0,
            ema_vel: 0.0,
            ema_acc: 0.0,
            ema_jerk: 0.0,
            n: 0,
        }
    }

    /// Observation-noise estimate from the FULL-LADDER residual (see the
    /// module doc: the screened order's residual death-spirals).
    fn eps_hat(&self) -> f32 {
        let amp = (extrapolation_weight_ss(1, 3) + 1.0).sqrt().max(1.0);
        (self.mean_abs_r3 * 1.253_314_1) / amp
    }

    /// EMA-smoothed 2σ significance screen on the ladder coefficients.
    fn screened_order(&self, eps: f32) -> u8 {
        let ladder = match self.state.n_obs {
            0 | 1 => 0,
            2 => 1,
            3 => 2,
            _ => 3,
        };
        let dt = self.state.dt;
        let mut o = 0;
        if ladder >= 1 && self.ema_vel * dt > 2.0 * SQRT2 * eps {
            o = 1;
        }
        if ladder >= 2 && self.ema_acc * dt * dt > 2.0 * SQRT6 * eps {
            o = 2;
        }
        if ladder >= 3 && self.ema_jerk * dt * dt * dt > 2.0 * SQRT20 * eps {
            o = 3;
        }
        o
    }

    /// Point + screened order. The point is the smoothed-velocity (α-β)
    /// extrapolation; order ≥ 2 adds the (screened) curvature term.
    fn point_and_order(&mut self) -> Option<(f32, u8)> {
        if self.n < 16 {
            return None;
        }
        let eps = self.eps_hat().max(1e-6);
        let order = self.screened_order(eps);
        // Point: pos + smoothed vel (+ acc when the screen admits it).
        let mut med = self.state.pos[0] + self.smooth_vel * self.state.dt;
        if order >= 2 {
            med += self.state.acc[0] * 0.5 * self.state.dt * self.state.dt;
        }
        Some((med, order))
    }

    /// Predictive half-width for the smoothed-velocity point:
    /// `z·σ̂·√(2 + 2β/(2−β))` — the anchor's noise + the smoothing-injected
    /// variance + the new observation's noise. (The screened curvature term's
    /// own noise contribution is second-order: the screen only admits it
    /// when |acc| ≫ noise, where the signal dominates.)
    fn smoothed_half_width(&self, _order: u8) -> f32 {
        let extra = 2.0 * SMOOTH_BETA / (2.0 - SMOOTH_BETA);
        let amp = (2.0 + extra).sqrt();
        normal_two_sided_z(self.alpha) * self.eps_hat().max(1e-6) * amp
    }
}

impl UqPrimitiveUnderTest for KinUq {
    fn name(&self) -> &str {
        "kinematic extrapolation interval (smoothed-velocity point + wss propagation)"
    }

    fn predict_next(&mut self) -> PredictiveOutput {
        match self.point_and_order() {
            Some((med, order)) => {
                let hw = self.smoothed_half_width(order);
                PredictiveOutput::from_interval(PredictiveInterval::new(
                    med - hw,
                    med,
                    med + hw,
                    self.alpha,
                ))
            }
            None => PredictiveOutput::empty(),
        }
    }

    fn observe(&mut self, y: f32) {
        // Full-ladder residual (the σ̂ source), computed BEFORE observing.
        let mut p3 = [0.0f32; 1];
        kinematic_extrapolate_capped_into(&self.state, 1, &Sched::Measured, 3, &mut p3).ok();
        let r3 = if self.n >= 4 { y - p3[0] } else { 0.0 };
        self.mean_abs_r3 += 0.05 * (r3.abs() - self.mean_abs_r3);
        // Smoothed velocity + screen inputs.
        let sb = SMOOTH_BETA;
        self.smooth_vel += sb * (self.state.vel[0] - self.smooth_vel);
        let eb = 0.1f32;
        self.ema_vel += eb * (self.state.vel[0].abs() - self.ema_vel);
        self.ema_acc += eb * (self.state.acc[0].abs() - self.ema_acc);
        self.ema_jerk += eb * (self.state.jerk[0].abs() - self.ema_jerk);
        let t = self.n;
        self.state.observe_into(&[y], t).expect("monotonic");
        self.n += 1;
    }
}

/// SplitMix64 — the floor-harness convention.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Gaussian via central-limit sum of 12 uniforms (floor-harness form).
    #[inline]
    fn gaussian(&mut self, sigma: f32) -> f32 {
        let mut sum = 0.0_f32;
        for _ in 0..12 {
            sum += (self.next_u64() >> 40) as f32 * (1.0_f32 / (1u64 << 24) as f32);
        }
        (sum - 6.0) * sigma
    }
}

fn noisy_uniform(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..n).map(|t| 0.7 * t as f32 + rng.gaussian(0.1)).collect()
}

fn noisy_parabola(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..n)
        .map(|t| {
            let tf = t as f32;
            0.012 * tf * tf + rng.gaussian(0.1)
        })
        .collect()
}

fn white_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..n).map(|_| rng.gaussian(0.1)).collect()
}

#[test]
fn kinematics_floor_comparison() {
    let alpha = 0.05f32;
    let n = 1500usize;
    let warmup = 64usize;

    let corpora: Vec<(&str, Vec<f32>)> = vec![
        ("noisy_uniform_v0.7_sigma0.1", noisy_uniform(n, 11)),
        ("noisy_parabola_c0.012_sigma0.1", noisy_parabola(n, 22)),
        ("white_noise_sigma0.1", white_noise(n, 33)),
    ];

    println!("== kinematics extrapolation interval vs conformal-naive floor ==\n");
    let mut results: Vec<(&str, katgpt_core::conformal::OverallVerdict)> = Vec::new();
    for (name, corpus) in &corpora {
        let mut prim = KinUq::new(alpha);
        let report = run_floor_comparison(&mut prim, corpus, alpha, warmup, name);
        report.pretty_print();
        results.push((name, report.overall.clone()));
        println!();
    }

    println!("== verdict summary ==");
    for (name, v) in &results {
        println!("  {name}: {v:?}");
    }
    // The falsifiable claim: on CURVING motion the kinematic interval beats
    // the floor (the floor's conformal drift-correction cannot track a
    // moving drift — measured coverage collapse).
    let parabola = results
        .iter()
        .find(|(n, _)| n.contains("parabola"))
        .map(|(_, v)| v)
        .expect("parabola corpus present");
    assert!(
        matches!(parabola, katgpt_core::conformal::OverallVerdict::BeatsFloor),
        "the kinematic interval must beat the floor on curving motion — \
         the falsifiable claim of this comparison"
    );
    // The uniform/white-noise losses are the documented RANK-ONLY trigger
    // (Report-the-Floor policy): at the harness's h=1 protocol a
    // drift-corrected naive prediction is statistically optimal on straight
    // motion — both predictors anchor on the same noisy last observation
    // (√2·σ shared floor; conformal's finite-sample quantile edge ~1.35σ).
    // `extrapolation_horizon` therefore ships RANK-ONLY (k* ordering, no
    // calibrated-coverage claim); see .benchmarks/680.
    let uniform = results
        .iter()
        .find(|(n, _)| n.contains("uniform"))
        .map(|(_, v)| v)
        .expect("uniform corpus present");
    assert!(
        matches!(
            uniform,
            katgpt_core::conformal::OverallVerdict::LosesToFloor
        ),
        "the uniform corpus is expected to lose at h=1 (√2σ shared floor) — \
         if it wins, update the rank-only verdict in .benchmarks/680"
    );
}
