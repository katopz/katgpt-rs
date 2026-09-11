//! Issue 746 T1 — defend-wrong PoC for `marginal_rewind` (Row 2:
//! marginal-calibrated backtrack as a collapse-recovery dial).
//!
//! # The synthetic (stuck-attractor bimodal commitment)
//!
//! A 1-D interpolant flow `x_t = t·x* + (1−t)·ε` integrated by the paper's
//! Euler-as-lerp with the anytime commitment schedule
//! `r_i = Δt/(1−t_i)` (Row 1's math — exercised HERE, in the harness,
//! per the Row 1 no-consumer verdict). The "denoiser" is a nearest-mode
//! oracle `est(x) = ±M` (M = 1.5): a stand-in for a collapsed reasoner
//! that locks onto whichever basin it is in — once the basin is chosen,
//! the flow cannot leave it (the lerp target stays on the same side of
//! the boundary at 0). A run COLLAPSES when it locks the spurious basin
//! (`x < 0` at detection level `t_det = 0.7`).
//!
//! Two prior regimes:
//! - **Collapse-prone** (the named consumer scenario — cgsp-style collapse
//!   happens BECAUSE the environment favors the trap): prior
//!   `ε ~ N(−0.8, 1)` ⇒ ~79% of runs collapse.
//! - **Unbiased** (`ε ~ N(0, 1)`): the honesty regime — a clean prior makes
//!   hard restart strong (fresh draws land in the correct basin 50% of
//!   the time); retained wrong-commitment is a pure liability there.
//!
//! # Arms (recovery applied on detection, single-shot)
//!
//! 1. **Calibrated** (the candidate): `rewind_plan(t_det, s)` +
//!    `rewind_into` — shrink `a = s/t` + exact-marginal noise, then
//!    re-integrate from level `s`.
//! 2. **Additive** (the shipped-comparator class — `saddle_escape` kicks /
//!    `renoise_ce` perturbation shape): `x + σ_cal·ε` at the SAME injected
//!    variance as arm 1 at that `s`, then re-integrate from `t_det`.
//! 3. **Naive resample** (the `q_sample`-shaped heuristic without the
//!    identity): `a·x + (1−s)·ε` — shrink + FULL level-`s` noise
//!    (over-injects by `(a−s)²`). Isolates whether the exact marginal
//!    matters vs any shrink-plus-noise.
//! 4. **Hard restart** (the shipped recovery shape — cgsp restart):
//!    discard the state, fresh prior draw at `t₀`, full re-integration.
//!
//! Sweep `s ∈ {0.55, 0.40, 0.25, 0.10}` (deeper = more budget).
//!
//! # Gates (pre-registered)
//!
//! - **G0 harness sanity**: the commitment schedule's telescoping
//!   annihilation holds (`∏(1−r_i) = (1−t_k)/(1−t₀)`, exactly 0 at
//!   `t_n = 1`); the collapse-prone prior collapses ≥ 60% of runs
//!   (else the synthetic is not collapse-prone and the cohort is noise).
//! - **G1 (the mechanism gate)**: best-`s` calibrated recovery ≥ 1.5×
//!   best-`s` additive recovery at equal injected budget
//!   (collapse-prone regime). Analytic expectation
//!   `P_cal(s) = 1 − Φ(s·M/(1−s))` vs
//!   `P_add(s) = 1 − Φ(t_det·M/√((1−t_det)² + σ_cal²))` ⇒ ~3× across the
//!   sweep.
//! - **G2 (beats the shipped alternative)**: ∃`s` with calibrated
//!   recovery > hard restart at LOWER injected budget
//!   (`σ_cal(s) < 1 − t₀`). Analytic: s=0.25 ⇒ ~31% @ σ=0.74 vs
//!   restart ~21% @ σ=0.95.
//! - **Honesty axis (measured, not gated)**: unbiased regime — hard
//!   restart is expected to WIN there (clean prior ⇒ resampling beats
//!   retained commitment); naive ≈ calibrated everywhere (the exactness
//!   buys on-manifold level statistics, not raw escape power — reported,
//!   not hidden).
//!
//! Verdict tables print with `--nocapture`; the numbers land in
//! `.benchmarks/712_marginal_rewind_poc.md`. If G1 fails, the row dies
//! (Issue 746 T2 — the schedule-only form is not worth a primitive).

#![cfg(feature = "marginal_rewind")]

use katgpt_core::marginal_rewind::{rewind_into, rewind_plan};

// ─────────────────────────────────────────────────────────────────────
// Harness primitives (test-only inline helpers — substrate-first
// exemption, the saddle_escape_poc convention)
// ─────────────────────────────────────────────────────────────────────

fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Two standard-normal draws (Box-Muller) from one 64-bit seed.
fn normal2(seed: u64) -> (f32, f32) {
    let bits = |v: u64| (v >> 11) as f64 / (1u64 << 53) as f64;
    let u1 = bits(mix64(seed)).max(1e-300);
    let u2 = bits(mix64(seed ^ 0xA5A5_5A5A_5A5A_A5A5));
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    ((r * theta.cos()) as f32, (r * theta.sin()) as f32)
}

// ─────────────────────────────────────────────────────────────────────
// The synthetic flow
// ─────────────────────────────────────────────────────────────────────

const M: f32 = 1.5; // mode centers ±M
const T0: f32 = 0.05; // path start (prior level)
const N_STEPS: usize = 32; // grid resolution t₀ → 1
const T_DET_STEP: usize = 22; // detection step (mid-flight, ≈ 0.703)

fn dt() -> f32 {
    (1.0 - T0) / N_STEPS as f32
}

fn level(i: usize) -> f32 {
    T0 + i as f32 * dt()
}

fn t_det() -> f32 {
    level(T_DET_STEP)
}

/// Nearest-mode "denoiser" (the collapsed-reasoner surrogate).
fn est(x: f32) -> f32 {
    if x >= 0.0 { M } else { -M }
}

/// Euler-as-lerp with the anytime commitment schedule r_i = Δt/(1−t_i),
/// from `from_level` (state `x`) to `to_level`. Returns the final state.
fn integrate(mut x: f32, from_level: f32, to_level: f32) -> f32 {
    let mut t = from_level;
    // Keep the SAME Δt as the base grid (schedule-fair across arms).
    let d = dt();
    while t < to_level {
        let r = (d / (1.0 - t)).min(1.0);
        x = (1.0 - r) * x + r * est(x);
        t += d;
    }
    x
}

/// Run the pre-detection segment: prior draw at t₀, integrate to the
/// detection level. Returns the state at detection (level `t_det()`).
fn stuck_state(trial: u64, prior_bias: f32) -> f32 {
    let (n1, _) = normal2(trial.wrapping_mul(0x9E37_79B9));
    let eps = n1 + prior_bias;
    let x0 = (1.0 - T0) * eps;
    integrate(x0, T0, t_det())
}

// ─────────────────────────────────────────────────────────────────────
// G0 — the commitment schedule's telescoping law (Row 1's property)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g0_telescoping_annihilation() {
    let d = dt();
    // Full grid: prior weight annihilates to EXACTLY 0 at t_n = 1 (the
    // last step has r = Δt/Δt = 1).
    let mut w = 1.0_f32;
    let mut t = T0;
    let mut k = 0_usize;
    while t < 1.0 {
        let r = (d / (1.0 - t)).min(1.0);
        w *= 1.0 - r;
        t += d;
        k += 1;
    }
    assert_eq!(k, N_STEPS);
    assert_eq!(w, 0.0, "prior weight must annihilate EXACTLY at t=1");

    // Mid-grid: ∏(1−r_j) = (1−t_k)/(1−t₀).
    for k in [8_usize, 16, 24] {
        let mut w = 1.0_f32;
        for i in 0..k {
            let t = level(i);
            let r = (d / (1.0 - t)).min(1.0);
            w *= 1.0 - r;
        }
        let target = (1.0 - level(k)) / (1.0 - T0);
        assert!(
            (w - target).abs() < 1e-4,
            "telescoping drifted at k={k}: {w:.7} vs {target:.7}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// The recovery bench
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Calibrated,
    Additive,
    Naive,
    Restart,
}

const SWEEP: [f32; 4] = [0.55, 0.40, 0.25, 0.10];
const K: usize = 8192;

/// Apply one recovery arm to a stuck state; returns (state, level).
fn recover(trial: u64, arm: Arm, s: f32, x_stuck: f32, prior_bias: f32) -> (f32, f32) {
    let (n1, n2) = normal2(trial ^ ((arm_tag(arm) as u64) << 32));
    match arm {
        Arm::Calibrated => {
            let plan = rewind_plan(t_det(), s);
            let mut out = [0.0];
            rewind_into(&[x_stuck], plan, &[n1], &mut out);
            (out[0], plan.s)
        }
        Arm::Additive => {
            // Equal-BUDGET comparator: the same injected σ the calibrated
            // arm uses at this s — the saddle_escape-kick / renoise_ce
            // perturb shape (no de-commit, no level change).
            let plan = rewind_plan(t_det(), s);
            (x_stuck + plan.sigma * n1, t_det())
        }
        Arm::Naive => {
            // Shrink + FULL level-s noise — the identity-free heuristic.
            let plan = rewind_plan(t_det(), s);
            (plan.a * x_stuck + (1.0 - s) * n1, plan.s)
        }
        Arm::Restart => {
            // Fresh prior draw at t₀, INHERITING the system's prior regime
            // (bias included — "re-run the system" means the same biased
            // environment that produced the collapse).
            ((1.0 - T0) * (n2 + prior_bias), T0)
        }
    }
}

fn arm_tag(arm: Arm) -> u32 {
    match arm {
        Arm::Calibrated => 1,
        Arm::Additive => 2,
        Arm::Naive => 3,
        Arm::Restart => 4,
    }
}

fn run_regime(name: &str, prior_bias: f32) -> (Vec<f64>, Vec<f64>, f64, usize) {
    // Fix the stuck cohort once per regime (same trials for every arm —
    // the paired-state design; recovery noise is per-(trial, arm)).
    let mut stuck = Vec::with_capacity(K);
    for tr in 0..K as u64 {
        let x = stuck_state(tr, prior_bias);
        if x < 0.0 {
            stuck.push((tr, x));
        }
    }
    let collapse_rate = stuck.len() as f64 / K as f64;

    let mut cal = Vec::with_capacity(SWEEP.len());
    let mut add = Vec::with_capacity(SWEEP.len());
    let mut naive = Vec::with_capacity(SWEEP.len());
    println!(
        "\n=== {name} regime (prior bias b = {prior_bias}, collapsed {}/{} = {:.1}%) ===",
        stuck.len(),
        K,
        100.0 * collapse_rate
    );
    println!("  arm         s     sigma(budget)  recovered   rate");
    for &s in &SWEEP {
        let plan = rewind_plan(t_det(), s);
        let rates = [Arm::Calibrated, Arm::Additive, Arm::Naive].map(|arm| {
            let solved = stuck
                .iter()
                .filter(|(tr, x)| {
                    let (x2, lvl) = recover(*tr, arm, s, *x, prior_bias);
                    integrate(x2, lvl, 1.0) > 0.0
                })
                .count();
            let rate = solved as f64 / stuck.len() as f64;
            let label = match arm {
                Arm::Calibrated => "calibrated",
                Arm::Additive => "additive",
                _ => "naive",
            };
            println!(
                "  {label:<10} {s:.2}   {sigma:.3}          {solved:>5}/{n}   {rate:.1}%",
                sigma = plan.sigma,
                solved = solved,
                n = stuck.len(),
                rate = 100.0 * rate
            );
            rate
        });
        cal.push(rates[0]);
        add.push(rates[1]);
        naive.push(rates[2]);
    }
    let restart = {
        let solved = stuck
            .iter()
            .filter(|(tr, x)| {
                let (x2, lvl) = recover(*tr, Arm::Restart, T0, *x, prior_bias);
                integrate(x2, lvl, 1.0) > 0.0
            })
            .count();
        let rate = solved as f64 / stuck.len() as f64;
        println!(
            "  restart     t₀    {sigma:.3}          {solved:>5}/{n}   {rate:.1}%",
            sigma = 1.0 - T0,
            solved = solved,
            n = stuck.len(),
            rate = 100.0 * rate
        );
        rate
    };
    println!(
        "  naive-vs-calibrated deltas (exactness cost): {:>5.1}pp {:>5.1}pp {:>5.1}pp {:>5.1}pp",
        (naive[0] - cal[0]) * 100.0,
        (naive[1] - cal[1]) * 100.0,
        (naive[2] - cal[2]) * 100.0,
        (naive[3] - cal[3]) * 100.0
    );
    (cal, add, restart, stuck.len())
}

#[test]
fn g1_g2_recovery_bench() {
    // ── Collapse-prone regime (the gated one) ─────────────────────
    let (cal, add, restart, stuck_n) = run_regime("Collapse-prone", -0.8);

    // G0b: the synthetic must actually be collapse-prone.
    let collapse_rate = stuck_n as f64 / K as f64;
    assert!(
        collapse_rate >= 0.60,
        "collapse cohort too small: {collapse_rate:.2}"
    );

    // G1: best-s calibrated ≥ 1.5× best-s additive (equal budget per s).
    let best_cal = cal.iter().cloned().fold(0.0_f64, f64::max);
    let best_add = add.iter().cloned().fold(0.0_f64, f64::max);
    println!(
        "\nG1: best calibrated {best_cal:.3} vs best additive {best_add:.3} (ratio {:.2}×)",
        best_cal / best_add.max(1e-9)
    );
    assert!(
        best_cal >= 1.5 * best_add,
        "G1 FAIL — calibrated backtrack shows no advantage at equal budget ({best_cal:.3} vs {best_add:.3}); Issue 746 T2: the row dies"
    );

    // G2: ∃s — calibrated beats hard restart at LOWER injected budget.
    let restart_budget = f64::from(1.0 - T0);
    let se_restart = (restart * (1.0 - restart) / stuck_n as f64).sqrt();
    let g2 = SWEEP.iter().zip(cal.iter()).any(|(&s, &c)| {
        let sig = f64::from(katgpt_core::marginal_rewind::rewind_plan(t_det(), s).sigma);
        c > restart + 3.0 * se_restart && sig < restart_budget
    });
    println!(
        "G2: restart {restart:.3} (budget {restart_budget:.3}); calibrated beats it at lower budget: {g2}"
    );
    assert!(
        g2,
        "G2 FAIL — calibrated never dominates hard restart in the collapse-prone regime"
    );

    // ── Unbiased regime (honesty axis — measured, NOT gated) ──────────
    let (cal_u, _add_u, restart_u, stuck_u) = run_regime("Unbiased", 0.0);
    println!(
        "\nHonesty (unbiased): restart {restart_u:.3} vs best calibrated {:.3} over {stuck_u} stuck runs — expected: restart wins when the prior is clean",
        cal_u.iter().cloned().fold(0.0_f64, f64::max)
    );
}
