//! Issue 1000 T2 — the kernel half of the contraction-certificate fixture kit
//! (Research 667 §5, arXiv:2611.09217): F1 contraction, F2 invariance, F3's
//! consistency arm, F4 disturbance-fade settling — all against the REAL
//! `evolve_belief_additive` kernel. F3's primary (ℓ = 1) arm and F5's
//! stability edge live beside the primitive they exercise
//! (`katgpt-types/src/belief_contract.rs` tests) — they are properties of the
//! derived laws on synthetic cells, not of this kernel.
//!
//! Every fixture FAILS on its falsification arm — a fixture that cannot fail
//! is not landed (issue 1000 T2 contract). Honest scope notes:
//! - F2's "telemetry" surface on the additive path IS the clamp (there is no
//!   separate telemetry emit on this kernel); the garbage arm asserts the
//!   out-of-ball thawed state is corrected into the ball within one tick.
//!   NaN propagation is OUT of the kernel's contract (a NaN belief is a
//!   thaw-layer refusal, not a kernel clamp case) — never asserted here.
//! - Determinism: all rollouts are fixed-sequence f32 (no RNG), per the
//!   Issue-940 determinism law (nothing here crosses a sync boundary, but
//!   the fixtures pin their numbers anyway).

use katgpt_sense::reconstruction::{ReconstructionConfig, ReconstructionState};
use katgpt_types::belief_contract::{
    ADDITIVE_BALL_RADIUS, contraction_rate, disturbance_fade_ticks, settling_ticks,
};
use katgpt_types::simd::fast_sigmoid;

/// Drive for a kind activation at the shipped `additive_drive_scale = 4.0` —
/// the same `2σ(η·k)−1` the kernel computes, so fixture expectations are
/// derived from the kernel's own arithmetic.
fn drive(activation: f32) -> f32 {
    2.0 * fast_sigmoid(4.0 * activation) - 1.0
}

fn state_with_retention(alpha: f32, belief: [f32; 8]) -> ReconstructionState {
    ReconstructionState::with_config(
        belief,
        ReconstructionConfig {
            belief_retention: [alpha; 6],
            ..ReconstructionConfig::default()
        },
    )
}

/// F1 — contraction: paired rollouts from distinct inits under identical
/// drives decay as `α^t` per dim (the drives are equal, so the pair
/// difference obeys `Δh_{t+1} = α·Δh_t` exactly while interior), and the
/// regression-fitted rate matches `−ln α`. Falsification arm: planted
/// near-unity `α = 1−1e-6` shows NON-contraction at the same tick horizon
/// (the pair barely closes), and the certificate refuses `α = 1` outright.
#[test]
fn f1_contraction_pair_decay_matches_neg_ln_alpha() {
    let alpha = 0.9f32;
    let ticks = 64usize;
    let mut a = state_with_retention(alpha, [0.5; 8]);
    let mut b = state_with_retention(alpha, [-0.5; 8]);
    let activations = [0.5f32; 6]; // constant → identical drives both sides

    let d0 = 1.0f32; // |0.5 − (−0.5)| per dim
    let mut log_dist: Vec<(usize, f32)> = Vec::with_capacity(ticks);
    for t in 0..=ticks {
        a.evolve_belief_additive(&activations);
        b.evolve_belief_additive(&activations);
        let d = (0..8)
            .map(|i| (a.belief()[i] - b.belief()[i]).abs())
            .fold(0.0f32, f32::max);
        // Per-dim decay ≤ α^t (+ f32 accumulation headroom), never a clamp
        // (steady state 0.762 < 1 — the pair stays interior).
        assert!(
            d <= d0 * alpha.powi(t as i32) * (1.0 + 1e-3) + 1e-6,
            "tick {t}: distance {d} exceeds alpha^t envelope"
        );
        if t > 0 {
            log_dist.push((t, d.ln()));
        }
        if d <= 1e-6 {
            break; // converged — no more signal to fit
        }
    }

    // Least-squares slope of ln d over t = the fitted contraction rate.
    let n = log_dist.len() as f32;
    let t_bar: f32 = log_dist.iter().map(|(t, _)| *t as f32).sum::<f32>() / n;
    let y_bar: f32 = log_dist.iter().map(|(_, y)| *y).sum::<f32>() / n;
    let mut s_xy = 0.0f32;
    let mut s_xx = 0.0f32;
    for (t, y) in &log_dist {
        s_xy += (*t as f32 - t_bar) * (y - y_bar);
        s_xx += (*t as f32 - t_bar) * (*t as f32 - t_bar);
    }
    let kappa_fit = -s_xy / s_xx;
    let kappa_exact = contraction_rate(alpha).expect("0.9 in (0,1)").kappa;
    assert!(
        (kappa_fit - kappa_exact).abs() < 2e-3,
        "fitted rate {kappa_fit} vs exact −ln(0.9) {kappa_exact}"
    );
}

/// F1 falsification arm — near-unity retention: at the SAME horizon the pair
/// has NOT contracted (realistic tick budgets cannot wait out α→1), the
/// derived settling horizon explodes, and `α = 1` is refused by name.
#[test]
fn f1_near_unity_alpha_does_not_contract_at_realistic_horizons() {
    let alpha = 1.0 - 1e-6;
    let mut a = state_with_retention(alpha, [0.5; 8]);
    let mut b = state_with_retention(alpha, [-0.5; 8]);
    let activations = [0.5f32; 6];
    for _ in 0..64 {
        a.evolve_belief_additive(&activations);
        b.evolve_belief_additive(&activations);
    }
    let d = (0..8)
        .map(|i| (a.belief()[i] - b.belief()[i]).abs())
        .fold(0.0f32, f32::max);
    assert!(
        d >= 0.995,
        "near-unity pair must NOT close in 64 ticks: d={d}"
    );
    // The schedule names the cost: ~4.04e6 ticks to settle.
    let horizon = settling_ticks(alpha, 0.1, ADDITIVE_BALL_RADIUS).expect("α ∈ (0,1)");
    assert!(horizon > 1_000_000, "near-unity settling horizon {horizon}");
    // α = 1 is not a contraction — refused at the certificate.
    assert!(matches!(
        contraction_rate(1.0),
        Err(katgpt_types::belief_contract::BeliefContractError::RetentionOutOfRange { .. })
    ));
}

/// F2 — invariance: boundary-seeded rollouts stay in the clamp ball for the
/// whole horizon (drive saturating outward holds the boundary; drive 0 pulls
/// inward geometrically), and a thawed out-of-ball state is corrected into
/// the ball within ONE tick (the clamp is this kernel's telemetry surface).
#[test]
fn f2_invariance_boundary_seeds_and_thawed_garbage_stay_in_ball() {
    let activations_hold = [2.0f32; 6]; // drive ≈ 0.9997 — saturating outward
    let activations_rest = [0.0f32; 6]; // drive 0 — inward decay

    // Boundary hold: seeded AT +1, pushed outward — never leaves the ball.
    let mut s = state_with_retention(0.9, [1.0; 8]);
    for _ in 0..100 {
        s.evolve_belief_additive(&activations_hold);
        for &v in s.belief() {
            assert!((-1.0..=1.0).contains(&v), "boundary hold diverged: {v}");
        }
    }
    // Inward pull from both boundaries — geometric, still in the ball.
    for seed in [[1.0f32; 8], [-1.0f32; 8]] {
        let mut s = state_with_retention(0.9, seed);
        for _ in 0..100 {
            s.evolve_belief_additive(&activations_rest);
            for &v in s.belief() {
                assert!((-1.0..=1.0).contains(&v), "inward pull diverged: {v}");
            }
        }
    }
    // Thawed garbage: out-of-ball finite state corrected within ONE tick.
    let garbage = [3.0f32, -4.0, 1.5, -1.5, 2.0, -2.5, 3.0, -4.0];
    let mut s = state_with_retention(0.9, garbage);
    s.evolve_belief_additive(&activations_rest);
    for &v in s.belief() {
        assert!(
            (-1.0..=1.0).contains(&v),
            "garbage not clamped in one tick: {v}"
        );
    }
    for _ in 0..50 {
        s.evolve_belief_additive(&activations_rest);
        for &v in s.belief() {
            assert!((-1.0..=1.0).contains(&v));
        }
    }
}

/// F3 CONSISTENCY ARM — the additive kernel at ℓ = 0: the steady-state noise
/// floor is FLAT in κ (no rise), so the paper's κ^ℓ floor law is never
/// misapplied to the identity readout. With a biased readout the fixed point
/// shifts by EXACTLY the drive delta for EVERY α (`h* = d`), so the floor
/// equals the drive delta across the whole κ sweep. (The floor may FALL
/// under zero-mean noise as slow κ averages more samples — the
/// pre-registered guard is the ABSENCE of a rise; the biased arm asserts the
/// exact flat form.)
#[test]
fn f3_consistency_additive_floor_is_flat_in_kappa() {
    let activations_clean = [0.5f32; 6];
    let activations_noisy = [0.6f32; 6];
    let delta_drive = drive(0.6) - drive(0.5); // the readout bias δ_drive

    for alpha in [0.95f32, 0.5, 0.2, 0.05] {
        let mut clean = state_with_retention(alpha, [0.0; 8]);
        let mut noisy = state_with_retention(alpha, [0.0; 8]);
        for _ in 0..2_000 {
            clean.evolve_belief_additive(&activations_clean);
            noisy.evolve_belief_additive(&activations_noisy);
        }
        let floor = (0..8)
            .map(|i| (noisy.belief()[i] - clean.belief()[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            (floor - delta_drive.abs()).abs() < 1e-3,
            "α={alpha}: floor {floor} vs δ_drive {} — floor must be FLAT in κ (identity readout)",
            delta_drive.abs()
        );
    }
    // And across the sweep the floor does NOT rise with κ (0.95 → κ≈0.05,
    // 0.05 → κ≈3.0): the largest floor sits at the SLOWEST rate or is flat.
    let floor_at = |alpha: f32| {
        let mut clean = state_with_retention(alpha, [0.0; 8]);
        let mut noisy = state_with_retention(alpha, [0.0; 8]);
        for _ in 0..2_000 {
            clean.evolve_belief_additive(&activations_clean);
            noisy.evolve_belief_additive(&activations_noisy);
        }
        (0..8)
            .map(|i| (noisy.belief()[i] - clean.belief()[i]).abs())
            .fold(0.0f32, f32::max)
    };
    assert!(
        floor_at(0.05) <= floor_at(0.95) + 1e-3,
        "ℓ=0 floor rose with κ — the floor law must not fire on the identity readout"
    );
}

/// F1δ — the measured observation-noise surrogate (plan 637 T1, Issue 1000
/// T3's noise-input unblock): hold a bounded disturbance stream on ONE kind
/// channel against a clean twin and read the STEADY-STATE residual per dim —
/// the filter's own δ_m, normalized to the ball radius. At ℓ = 0 the floor
/// is FLAT in κ (F3's consistency arm), so the surrogate is a measured
/// number, not a derived law: δ_m = max-dim steady |noisy − clean| after a
/// warm horizon, under the SAME alternating worst-case disturbance shape F3
/// primary uses (deterministic stand-in for a bounded noise process, per the
/// Issue-940 determinism law — no RNG anywhere). NOTE the kernel's
/// positivity guard (`k = max(0, activation)`) makes the alternation
/// `+d / 0` — a HALF-cycle, whose two-cycle offset is correspondingly
/// larger than a signed ± cycle; the analytic record below uses the same
/// guard, so the fit is verified against the kernel's REAL arithmetic.
///
/// Returns the per-channel surrogate and asserts its two load-bearing
/// properties IN-FIXTURE (a number nobody bounds is not a surrogate):
/// (a) IDEMPOTENCE — re-running the fit reproduces the same bytes (the fit
/// is a function of the stream, not of timing);
/// (b) MONOTONE BOUND — a disturbance at 2× the measured bound produces a
/// steady residual above the surrogate (the bound is load-bearing), and a
/// disturbance at half the bound settles BELOW it (the surrogate is not
/// under-fit). Falsification arm: the fit against a stream at HALF the
/// surrogate's own disturbance must NOT reproduce the surrogate (a fit that
/// ignores its input magnitude is decorative).
///
/// The digest pin rides T1's landing commit (637 T1), never a fixture-time
/// stdout print — the value is ASSERTED against, not merely reported.
fn fit_delta_surrogate(alpha: f32, drive_disturbance: f32, warm: usize) -> f32 {
    // Disturbance rides kind channel 0 only — per-channel surrogate shape.
    let clean_drive = [0.0f32; 6];
    let mut noisy_drive = [0.0f32; 6];
    noisy_drive[0] = drive_disturbance;

    let mut clean = state_with_retention(alpha, [0.0; 8]);
    let mut noisy = state_with_retention(alpha, [0.0; 8]);

    // Alternating ±d on the disturbed channel (zero-mean worst-case shape);
    // the twin sees the clean stream. Sign alternation keeps the steady
    // state OFF the clamp wall so the residual reads contraction, not
    // saturation.
    let mut residual = 0.0f32;
    for t in 0..warm {
        let sign = if t % 2 == 0 { 1.0 } else { -1.0 };
        let mut d = noisy_drive;
        d[0] = drive_disturbance * sign;
        clean.evolve_belief_additive(&clean_drive);
        noisy.evolve_belief_additive(&d);
        if t >= warm / 2 {
            residual = residual.max(
                (0..8)
                    .map(|i| (noisy.belief()[i] - clean.belief()[i]).abs())
                    .fold(0.0f32, f32::max),
            );
        }
    }
    residual
}

#[test]
fn f1_delta_surrogate_is_measured_load_bearing_and_deterministic() {
    let alpha = 0.9f32;
    let disturbance = 0.5f32;
    let warm = 2_000usize;

    // The surrogate fit — measured, not derived (ℓ = 0 floor is flat in κ).
    let delta_fit = fit_delta_surrogate(alpha, disturbance, warm);
    assert!(
        delta_fit > 0.0 && delta_fit.is_finite(),
        "surrogate must be a finite positive number: {delta_fit}"
    );
    // The fit lives in the ball and well under the radius — it is a
    // per-channel disturbance residual, not a saturation artifact.
    assert!(
        delta_fit < ADDITIVE_BALL_RADIUS,
        "surrogate {delta_fit} at/over the ball radius — the fit read saturation, not contraction"
    );

    // (b) load-bearing: at half the disturbance the residual settles BELOW
    // the surrogate (scaled input ⇒ scaled floor, roughly linearly at ℓ=0 —
    // the additive map is affine in the drive); at double it is ABOVE.
    let delta_half = fit_delta_surrogate(alpha, disturbance / 2.0, warm);
    let delta_double = fit_delta_surrogate(alpha, disturbance * 2.0, warm);
    assert!(
        delta_half < delta_fit,
        "half-disturbance residual {delta_half} must sit under the surrogate {delta_fit}"
    );
    assert!(
        delta_double > delta_fit,
        "double-disturbance residual {delta_double} must sit over the surrogate {delta_fit}"
    );

    // Falsification arm: a fit that ignored its input magnitude would
    // reproduce the same number for both scales — refuse that shape by
    // asserting the ratio is strictly ordered (never within f32 equality).
    assert!(
        (delta_half - delta_fit).abs() > 1e-6 && (delta_double - delta_fit).abs() > 1e-6,
        "surrogate did not move with the input magnitude — the fit is decorative"
    );

    // (a) determinism: the identical call reproduces the same bytes.
    let delta_again = fit_delta_surrogate(alpha, disturbance, warm);
    assert_eq!(
        delta_fit.to_bits(),
        delta_again.to_bits(),
        "surrogate fit not deterministic"
    );

    // The T1 record: the surrogate at the shipped pins has its ANALYTIC
    // closed form under the kernel's own positivity guard (`k = max(0, a)`
    // ⇒ the alternating disturbance rides +d / 0, a HALF-cycle): the
    // two-cycle offsets are x₁ = 0.40084, x₂ = 0.36076 at α = 0.9,
    // d = 0.5 — the recurrence solved in-fixture from the guard-shaped
    // drive pair (+0.7616 / 0.0) — asserted at 1% tolerance, so the
    // fit is a VERIFIED measurement of the kernel's REAL arithmetic, not
    // an unbounded read.
    let dp = fast_sigmoid(4.0 * disturbance); // σ(η·d) = 0.8808 → drive 0.7616
    let dp_drive = 2.0 * dp - 1.0;
    let zero_drive = 2.0 * fast_sigmoid(0.0) - 1.0; // the guard's 0-activation arm
    let x1 =
        (alpha * (1.0 - alpha) * zero_drive + (1.0 - alpha) * dp_drive) / (1.0 - alpha * alpha);
    let x2 =
        (alpha * (1.0 - alpha) * dp_drive + (1.0 - alpha) * zero_drive) / (1.0 - alpha * alpha);
    let expect = x1.abs().max(x2.abs());
    assert!(
        (delta_fit - expect).abs() < 0.01 * expect,
        "surrogate {delta_fit} vs analytic two-cycle offset {expect} — the fit drifted off the kernel's own arithmetic"
    );
}

/// F4 — disturbance-fade settling: after a disturbance episode ends and clean
/// observations resume, the estimate error crosses ε at the S2-predicted
/// tick (the additive kernel fades at the FULL κ_m — the exact-linear rate).
/// Falsification arm: three ticks EARLIER the error is still above ε — the
/// law is load-bearing, not decorative. The generic κ/2 schedule holds as the
/// upper bound. (The ± tolerance exists because C2's √8 radius is
/// conservative — here the prediction uses the MEASURED residual at
/// resumption, so the tolerance covers f32 drift only.)
#[test]
fn f4_disturbance_fade_crosses_eps_at_predicted_tick() {
    let alpha = 0.9f32;
    let kappa_m = contraction_rate(alpha).expect("0.9 in (0,1)").kappa;
    let eps = 0.05f32;
    let episode_ticks = 30usize;

    let clean = [0.0f32; 6];
    let disturbed = [0.5f32; 6];
    let mut a = state_with_retention(alpha, [0.0; 8]);
    let mut b = state_with_retention(alpha, [0.0; 8]);

    let err = |a: &ReconstructionState, b: &ReconstructionState| {
        (0..8)
            .map(|i| (a.belief()[i] - b.belief()[i]).abs())
            .fold(0.0f32, f32::max)
    };

    // Disturbance episode: A's drive pulled toward 0.762, B resting at 0.
    for _ in 0..episode_ticks {
        a.evolve_belief_additive(&disturbed);
        b.evolve_belief_additive(&clean);
    }
    let e_resume = err(&a, &b); // the residual at the resumption tick

    // Clean observations resume — the pair difference now decays as α^k.
    let k_star = disturbance_fade_ticks(kappa_m, e_resume, eps).expect("valid inputs (full κ_m)");
    for k in 0..k_star {
        a.evolve_belief_additive(&clean);
        b.evolve_belief_additive(&clean);
        let e = err(&a, &b);
        if k + 1 == k_star.saturating_sub(3) {
            // Falsification arm: releasing 3 ticks early still shows error.
            assert!(
                e > eps,
                "error already ≤ ε at k={} — the fade law is decorative",
                k + 1
            );
        }
    }
    let e_star = err(&a, &b);
    assert!(
        e_star <= eps + 1e-3,
        "crossing at k*={k_star}: error {e_star} > ε"
    );

    // Generic nonlinear bound (κ/2) is the conservative UPPER schedule — it
    // must also hold (the exact-linear fade is faster than it predicts).
    let k_half = disturbance_fade_ticks(kappa_m / 2.0, e_resume, eps).expect("valid");
    assert!(k_half >= k_star);
    for _ in k_star..k_half {
        a.evolve_belief_additive(&clean);
        b.evolve_belief_additive(&clean);
    }
    assert!(
        err(&a, &b) <= eps + 1e-3,
        "κ/2 upper-bound schedule violated"
    );
}
