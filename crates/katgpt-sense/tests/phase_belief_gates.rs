//! Issue 939 — phase-carrying belief kernel gates (P1/P2/P4 of the POC; P3
//! the FSM real-boot test is the PROMOTION-deciding gate and waits for the
//! named consumer pull — defer-until-pull).
//!
//! The `#![cfg]` below protects the TEST COUNT (feature off → empty binary,
//! cargo prints `ok. 0 passed`); the `[[test]] required-features` row in
//! Cargo.toml protects the READER (the target is skipped loudly, not silently
//! green). Both are needed — see katgpt-rs AGENTS.md §"cfg-gated targets".
//!
//! Run: `cargo test -p katgpt-sense --features phase_belief --test phase_belief_gates`

#![cfg(feature = "phase_belief")]

use katgpt_sense::phase_belief::{DampedPhaseBlock, ModularPhaseCounter, PhaseBeliefError};

const TAU: f32 = core::f32::consts::TAU;
const FRAC_PI_2: f32 = core::f32::consts::FRAC_PI_2;

// ─── P1a: integer counter exactness over 10⁶ random events ─────────────

#[test]
fn integer_counter_is_exact_over_a_million_random_events() {
    // P1 gate: "integer counter exact over 10⁶ random streams". The
    // reference count is a separate i64 accumulator reduced mod m by the
    // CPU — two independent spellings must agree (to f32 LUT rounding) at
    // every checkpoint, for every m in a sweep.
    let mut rng = fastrand::Rng::with_seed(0x939_u64);
    for m in [2u32, 3, 5, 7, 16, 60, 255, 256] {
        let mut ctr = ModularPhaseCounter::new(m).expect("m in range");
        let mut reference: i64 = 0;
        // ~10⁶ events total per modulus, in random bursts (the "streams").
        while reference < 1_000_000 {
            let burst = rng.u32(..=64);
            ctr.observe_n(burst);
            reference += burst as i64;
            if reference % 9973 == 0 {
                let expect_c = reference.rem_euclid(m as i64) as usize;
                let phase = TAU * expect_c as f32 / m as f32;
                assert_eq!(ctr.count(), expect_c as u32, "m={m} at n={reference}");
                let pair = ctr.readout_pair();
                assert!(
                    (pair[0] - phase.cos()).abs() < 1e-6 && (pair[1] - phase.sin()).abs() < 1e-6,
                    "m={m} n={reference}: pair {pair:?} vs cos/sin of {phase}"
                );
            }
        }
        assert_eq!(ctr.count(), (reference % m as i64) as u32);
        // Per-event observes agree with the bulk form.
        ctr.observe();
        reference += 1;
        assert_eq!(ctr.count(), (reference % m as i64) as u32);
    }
}

// ─── P1a-b: the justification — f32 phase accumulation DRIFTS ──────────

#[test]
fn f32_phase_accumulation_drifts_where_the_integer_form_does_not() {
    // The verdict-round-1 premise: `φ += 2π/m` in f32 accumulates rounding
    // over 10⁶ events (|φ| reaches 2·10⁶ where the f32 ulp is 0.125) and
    // the readout decorrelates from the exact code. Mean |cos error| over
    // 1000 checkpoints concentrates far from zero; this pins WHY the
    // integer form is the deliverable (and guards a future "simplification"
    // back to f32 accumulation).
    let m = 3u32;
    let mut ctr = ModularPhaseCounter::new(m).unwrap();
    let mut phi = 0.0f32;
    let step = TAU / m as f32;
    let mut err_sum = 0.0f64;
    let mut checks = 0u64;
    for k in 1..=1_000_000u32 {
        ctr.observe();
        phi += step;
        if k % 1000 == 0 {
            let exact = ctr.readout(); // integer side never drifts
            err_sum += (phi.cos() - exact).abs() as f64;
            checks += 1;
        }
    }
    let mean_err = (err_sum / checks as f64) as f32;
    assert_eq!(
        ctr.count(),
        1,
        "1_000_000 mod 3 — the integer side is exact"
    );
    assert!(
        mean_err > 0.2,
        "f32 accumulation should decorrelate by 10⁶ events (mean |Δcos| = {mean_err}); \
         if this ever drops below ~1e-2 the integer form's justification is gone"
    );
}

// ─── P1b: damped block free-run decay matches ρⁿ·R(nθ) ─────────────────

#[test]
fn damped_block_free_run_matches_the_closed_form() {
    // P1 gate: "damped block's phase/magnitude decay matches the closed form
    // ρⁿ·R(nθ) to f32 tolerance" — zero writes, several (ρ, θ) shapes. All
    // ρⁿ stay in the normal f32 range for the n here (no subnormal noise).
    for &(rho, theta, n_max) in &[
        (0.97f32, 0.7f32, 200u32),
        (0.9, FRAC_PI_2, 200),
        (0.75, TAU / 8.0, 150),
        (1.0, 0.3, 200),
    ] {
        let mut blk = DampedPhaseBlock::new(rho, theta).unwrap();
        blk.step([0.8, -0.5]); // seed a non-trivial state
        let z0 = blk.state();
        let mut m = z0[0].hypot(z0[1]);
        let mut phi = z0[1].atan2(z0[0]);
        for n in 1..=n_max {
            blk.step([0.0, 0.0]);
            // Reference recurrence in the closed form's own order:
            // magnitude scales by ρ, angle adds θ, per step.
            m *= rho;
            phi += theta;
            // Relative magnitude tolerance (the state can decay hard);
            // f32 per-step rounding is ~4 ops · 2⁻²⁴, n-fold accumulated.
            let got_m = blk.magnitude();
            let tol_m = (1e-3 * m).max(1e-30);
            assert!(
                (got_m - m).abs() <= tol_m,
                "rho={rho} theta={theta} n={n}: |z|={got_m} vs ρⁿ|z₀|={m}"
            );
            // Phase chord on the unit circle (wrap-safe).
            let chord = (blk.phase_cos() - phi.cos()).hypot(blk.phase_sin() - phi.sin());
            let tol_phi = 2e-4 + 1e-5 * n as f32;
            assert!(
                chord <= tol_phi,
                "rho={rho} theta={theta} n={n}: phase chord={chord}"
            );
        }
    }
}

// ─── P1c: damped block with constant writes matches the geometric sum ──

/// `(a+bi)^n` by binary exponentiation — test-only complex power.
fn pow_ci(re: f64, im: f64, n: u32) -> (f64, f64) {
    let (mut acc_re, mut acc_im) = (1.0f64, 0.0f64);
    let (mut base_re, mut base_im) = (re, im);
    let mut k = n;
    while k > 0 {
        if k & 1 == 1 {
            let t = acc_re * base_re - acc_im * base_im;
            acc_im = acc_re * base_im + acc_im * base_re;
            acc_re = t;
        }
        let t = base_re * base_re - base_im * base_im;
        base_im = base_re * base_im + base_im * base_re;
        base_re = t;
        k >>= 1;
    }
    (acc_re, acc_im)
}

#[test]
fn damped_block_constant_write_matches_the_geometric_series() {
    // From the recursion z_{n+1} = ρR(θ)z_n + u (u = [1, 0]):
    // z_n = Σ_{j<n} (ρe^{iθ})^j = (1−wⁿ)/(1−w), w = ρe^{iθ} — computed here
    // in f64 as the independent reference for the block's f32 iterate.
    let (rho, theta) = (0.9f32, 0.7f32);
    let mut blk = DampedPhaseBlock::new(rho, theta).unwrap();
    let wr = rho as f64;
    let (s0, c0) = theta.sin_cos();
    let (w_re, w_im) = (wr * c0 as f64, wr * s0 as f64);
    for n in 1..=96u32 {
        blk.step([1.0, 0.0]);
        let (wn_re, wn_im) = pow_ci(w_re, w_im, n);
        let (a, b) = (1.0 - wn_re, -wn_im); // 1 − wⁿ
        let (den_re, den_im) = (1.0 - w_re, -w_im); // 1 − w
        let inv = 1.0 / (den_re * den_re + den_im * den_im);
        let want_re = (a * den_re + b * den_im) * inv;
        let want_im = (b * den_re - a * den_im) * inv;
        let got = blk.state();
        let tol = (1e-3 * (1.0 + n as f32 * 0.05)) as f64;
        assert!(
            (got[0] as f64 - want_re).abs() < tol && (got[1] as f64 - want_im).abs() < tol,
            "n={n}: got ({}, {}), want ({want_re:.5}, {want_im:.5})",
            got[0],
            got[1]
        );
    }
}

// ─── P2a: parity — the counter classifies what a leaky kernel cannot ───

#[test]
fn parity_is_exact_in_the_counter_and_invisible_to_a_leaky_kernel() {
    // Odd-count stream (3 clustered events) vs even-count stream (4 spread
    // events), read at a common tick T. A magnitude-only leaky integrator
    // ORDERS THEM BACKWARDS (3 recent > 4 stale), so no threshold of its
    // scalar classifies parity — while the counter's phase code is exact
    // regardless of timing.
    let alpha = 0.9f32;
    let big_t = 32u32;
    let leaky = |ticks: &[u32]| -> f32 {
        ticks
            .iter()
            .map(|&t| alpha.powi((big_t - t) as i32))
            .sum::<f32>()
    };
    let odd = leaky(&[big_t - 3, big_t - 2, big_t - 1]); // 3 events, clustered
    let even = leaky(&[0, big_t / 3, 2 * big_t / 3, big_t - 1]); // 4 events, spread
    assert!(
        odd > even,
        "leaky must order the ODD stream above the EVEN one (odd={odd}, even={even}) \
         — that ordering is exactly the parity-blindness"
    );

    // The counter: parity is ±1 regardless of when the events landed.
    let mut a = ModularPhaseCounter::new(2).unwrap();
    for _ in 0..3 {
        a.observe();
    }
    let mut b = ModularPhaseCounter::new(2).unwrap();
    b.observe_n(4);
    assert!((a.readout() - (-1.0)).abs() < 1e-6, "3 mod 2 → cos π = −1");
    assert!((b.readout() - 1.0).abs() < 1e-6, "4 mod 2 → cos 2π = +1");
    assert!(a.readout() < b.readout(), "counter orders parity CORRECTLY");

    // The drive readout separates them on the sigmoid (the FSM-facing
    // scalar), for any gain.
    for gain in [1.0f32, 4.0, 16.0] {
        assert!(a.drive_readout(gain) < b.drive_readout(gain));
    }
}

// ─── P2b: mod-3 — the PAIR is the code; cos alone is degenerate ────────

#[test]
fn mod_three_needs_both_quadratures_and_the_pair_delivers() {
    let mut ctr = ModularPhaseCounter::new(3).unwrap();
    // cos degeneracy first: counts 1 and 2 share a cosine.
    ctr.observe(); // count 1
    let cos1 = ctr.readout();
    ctr.observe(); // count 2
    let cos2 = ctr.readout();
    assert!(
        (cos1 - cos2).abs() < 1e-6,
        "cos(2π/3) == cos(4π/3) — the m=3 degeneracy must hold for this pin"
    );
    // The pair separates them: count 2 sits at 240° = −120°.
    let p2 = ctr.readout_pair();
    let ang2 = p2[1].atan2(p2[0]);
    assert!(
        (ang2 + 2.0 * core::f32::consts::FRAC_PI_3).abs() < 1e-5,
        "count 2 must read at −2π/3, got {ang2}"
    );
    // Full sweep: after three more observes the pair visits every residue —
    // counts 0, 1, 2 at angles 0, +2π/3, −2π/3 (atan2's branch).
    let mut angles = [0.0f32; 3];
    for slot in &mut angles {
        ctr.observe();
        let p = ctr.readout_pair();
        *slot = p[1].atan2(p[0]);
    }
    assert!(angles[0].abs() < 1e-5, "count 0 at angle 0");
    assert!(
        (angles[1] - 2.0 * core::f32::consts::FRAC_PI_3).abs() < 1e-5,
        "count 1 at +2π/3"
    );
    assert!(
        (angles[2] + 2.0 * core::f32::consts::FRAC_PI_3).abs() < 1e-5,
        "count 2 at −2π/3"
    );
}

// ─── P2c: phase and magnitude fade TOGETHER (the "when"-memory smoke) ──

#[test]
fn damped_block_still_counts_time_while_its_confidence_fades() {
    // A leaky scalar fades to a fixed direction — it answers only "how
    // much". The damped block's phase KEEPS ROTATING on a free run while
    // the magnitude decays: after 22 undisturbed ticks at θ = π/2 the
    // channel points at 11 half-turns (≈ [−1, 0]) with under 10% of its
    // initial confidence. Fading and timing are the same mechanism.
    let (rho, theta) = (0.9f32, FRAC_PI_2);
    let mut blk = DampedPhaseBlock::new(rho, theta).unwrap();
    blk.step([1.0, 0.0]); // one event at t = 0
    for _ in 0..22 {
        blk.step([0.0, 0.0]); // free run
    }
    let want_mag = rho.powi(22); // 0.9²² ≈ 0.098
    let got_mag = blk.magnitude();
    assert!(
        got_mag < 0.105 && (got_mag - want_mag).abs() < 1e-3,
        "magnitude must have faded to ρ²² = {want_mag}, got {got_mag}"
    );
    // 11 half-turns → pointing at [−1, 0] (well inside the faded envelope).
    let chord = (blk.phase_cos() + 1.0).hypot(blk.phase_sin());
    assert!(
        chord < 5e-3,
        "22 ticks at θ=π/2 must read as phase π (chord={chord})"
    );
    // And the closed-form predictors agree with both (the scheduled
    // re-observation surface: read confidence/phase without stepping).
    assert!((blk.magnitude_after(0) - got_mag).abs() < 1e-6);
    // 2 more ticks: 24 quarter-turns = 6 full turns → phase back to 0, cos +1;
    // magnitude × ρ². The predictor reads both without stepping.
    assert!((blk.phase_cos_after(2) - 1.0).abs() < 1e-2);
    assert!((blk.magnitude_after(2) - want_mag * rho * rho).abs() < 1e-3);
}

// ─── P4: the C.3 steady-state bound + the θ-dependence budget ──────────

#[test]
fn constant_write_converges_to_the_c3_steady_state_bound() {
    // ‖z_∞‖ = ‖u‖/|1−ρe^{iθ}| — converge a constant write and meet it.
    for &(rho, theta) in &[
        (0.9f32, 0.7f32),
        (0.75, TAU / 6.0),
        (0.999, 2.0 * core::f32::consts::FRAC_PI_3),
    ] {
        let blk = DampedPhaseBlock::new(rho, theta).unwrap();
        let bound = blk.steady_state_bound(1.0);
        // ρ→1 check folded in: the θ-only form 1/(2·sin(θ/2)) must agree
        // with the full form to O(1−ρ).
        let theta_only = 1.0 / (2.0 * (theta / 2.0).sin());
        let rel = (bound - theta_only).abs() / theta_only;
        let agree = if rho > 0.99 { rel < 5e-3 } else { rel < 1.0 };
        assert!(agree, "rho={rho}: bound {bound} vs θ-only {theta_only}");

        let mut b = DampedPhaseBlock::new(rho, theta).unwrap();
        let steps = if rho > 0.99 { 400_000 } else { 4_000 };
        for _ in 0..steps {
            b.step([1.0, 0.0]);
        }
        let got = b.magnitude();
        assert!(
            (got - bound).abs() < 5e-3 * bound.max(1.0),
            "rho={rho} theta={theta}: ‖z_{steps}‖={got} vs bound {bound}"
        );
    }
}

#[test]
fn write_budget_sizes_writes_to_a_target_bound() {
    // P4 as a design constraint: budget(ε)·(1/|1−ρe^{iθ}|) == ε, and a
    // constant write AT the budget settles at ≈ ε.
    let (rho, theta, eps) = (0.85f32, 1.1f32, 2.0f32);
    let blk = DampedPhaseBlock::new(rho, theta).unwrap();
    let budget = blk.write_budget_for(eps);
    assert!((blk.steady_state_bound(budget) - eps).abs() < 1e-4);
    let mut b = DampedPhaseBlock::new(rho, theta).unwrap();
    for _ in 0..4_000 {
        b.step([budget, 0.0]); // ‖u‖ = budget
    }
    assert!(
        (b.magnitude() - eps).abs() < 5e-3 * eps,
        "write at budget should settle at ε={eps}, got {}",
        b.magnitude()
    );
    // Small-θ honesty (P4 reviewer note): as θ→0 the budget collapses to the
    // leaky kernel's own (1−ρ) scale — the rotation buys nothing there.
    let small = DampedPhaseBlock::new(0.999, 1e-3).unwrap();
    let leaky_scale = 1.0 - 0.999f32; // |1−ρe^{iθ}| ≈ √((1−ρ)²+θ²) ≥ 1−ρ
    assert!(small.write_budget_for(1.0) >= leaky_scale * 0.99);
    assert!(
        small.write_budget_for(1.0) < 0.05,
        "θ=1e-3 keeps the budget tiny"
    );
}

// ─── Constructor guards + sigmoid readout shape ─────────────────────────

#[test]
fn constructors_refuse_out_of_domain_inputs() {
    assert_eq!(
        ModularPhaseCounter::new(0),
        Err(PhaseBeliefError::ModulusOutOfRange)
    );
    assert_eq!(
        ModularPhaseCounter::new(1),
        Err(PhaseBeliefError::ModulusOutOfRange)
    );
    assert_eq!(
        ModularPhaseCounter::new(257),
        Err(PhaseBeliefError::ModulusOutOfRange)
    );
    assert_eq!(
        DampedPhaseBlock::new(0.0, 0.5),
        Err(PhaseBeliefError::RhoOutOfRange)
    );
    assert_eq!(
        DampedPhaseBlock::new(1.5, 0.5),
        Err(PhaseBeliefError::RhoOutOfRange)
    );
    assert_eq!(
        DampedPhaseBlock::new(f32::NAN, 0.5),
        Err(PhaseBeliefError::RhoOutOfRange)
    );
    assert_eq!(
        DampedPhaseBlock::new(0.9, f32::NAN),
        Err(PhaseBeliefError::ThetaNotFinite)
    );
    assert_eq!(
        DampedPhaseBlock::new(0.9, f32::INFINITY),
        Err(PhaseBeliefError::ThetaNotFinite)
    );
}

#[test]
fn drive_readouts_are_sigmoid_shaped_never_softmax() {
    // (0.5, 1) bounded for positive gain, monotone in gain, and the
    // zero-state convention reads as count 0 (+1 phase → same readout).
    let ctr = ModularPhaseCounter::new(2).unwrap(); // fresh: count 0, cos +1
    let blk = DampedPhaseBlock::new(0.9, 0.7).unwrap(); // fresh: zero state
    for gain in [0.5f32, 2.0, 8.0] {
        let a = ctr.drive_readout(gain);
        let b = blk.drive_readout(gain);
        assert!(
            a > 0.5 && a < 1.0,
            "count-0 sigmoid readout must exceed 0.5"
        );
        assert!(
            (a - b).abs() < 1e-6,
            "zero-state convention matches count 0"
        );
        assert!(ctr.drive_readout(gain * 2.0) > a, "monotone in gain");
    }
}
