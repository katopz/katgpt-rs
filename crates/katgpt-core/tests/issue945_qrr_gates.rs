#![cfg(all(feature = "gain_cost_halt", feature = "qrr_convergence"))]
//! Issue 945 gates — QRR quotient-convergence extraction (Research 623,
//! arXiv:2610.18174). Labels are the PRE-REGISTERED oracle: the synthetic
//! trajectory families (settling flip, limit-cycle flip, settling spiral,
//! expanding spiral, still) are constructed with known moduli, so every
//! verdict here is checked against construction truth, never against the
//! instrument under test.
//!
//! Arms:
//! - **P1** — mode-fit recovery: damped-spiral channels recover the complex
//!   pair `(ρ, θ)`; alternating flips recover `S = −r` (through the ridge —
//!   single-mode windows are rank-deficient); an SNR sweep bounds the noise
//!   sensitivity; an exactly-still window is inert (`None`); a linear ramp
//!   honestly reports its unit root.
//! - **P2** — two-channel verdict in [`GainCostLoopHalter`]: settling
//!   flip/spiral NEVER halt with [`HaltReason::Oscillation`] (zero false
//!   halt-and-damps); a limit cycle halts at UNCHANGED latency (same loop
//!   index as the unarmed incumbent); `KATGPT_QRR_OSC_VERDICT=0` (the exact
//!   literal) restores byte-identity — verified in a CHILD PROCESS so the
//!   parent's env stays untouched.
//! - **P3** — budget calibration: predicted settling within ×2 of actual on
//!   ≥ 95% of the synthetic corpus (with the constant-budget baseline
//!   reported); the `r = 1` certificate is TWO-SIDED (never fires on
//!   settling members, always on `r = 1` members, never on a progressing
//!   complement).
//!
//! Zero soft asserts: every fixture's expected halt/no-halt is asserted
//! exactly; a NaN/inf corruption arm proves the safe (continue) direction.

use katgpt_core::gain_cost_halt::{GainCostLoopHalter, HaltDecision, HaltReason};
use katgpt_core::qrr_convergence::{
    ModeFitter, ModeKind, OscillationVerdict, R1Certificate, settling_steps, tolerance_for_budget,
};

/// Deterministic SplitMix64 — no global RNG (the workspace RNG-pinning law).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = self.0;
        let x = z ^ (z >> 30);
        let x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let x = x ^ (x >> 27);
        let x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        let x = x ^ (x >> 31);
        // Top 24 bits → [0, 1) → [−1, 1).
        ((x >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn noise(&mut self, sigma: f32) -> f32 {
        // Sum of two uniforms — cheap symmetric-ish bell, deterministic.
        (self.next_f32() + self.next_f32()) * 0.5 * sigma
    }
}

/// Verdict-patience used by every P2 gate: the fitter needs
/// MIN_FIT_SAMPLES = 6 fitted triples; all-flip streams form a triple per
/// loop from loop 3, so patience 10 guarantees a warmed estimate AT the
/// patience loop (the documented arming posture — patience ≥ the warmup).
const PATIENCE: u8 = 10;

/// Settling alternating flip: `Δh_t = (−r)^t·v` — `gain = r^t`, `cosθ = −1`.
fn flip_stream(r: f32, n: usize) -> Stream {
    (1..=n).map(|t| (r.powi(t as i32), 0.0, -1.0)).collect()
}

/// Spiral: `Δh_t = ρ^t·(cos tθ, sin tθ)` — `gain = ρ^t`, `cosθ = cos θ`
/// (the angle between successive updates is exactly θ).
fn spiral_stream(rho: f32, theta: f32, n: usize) -> Stream {
    (1..=n)
        .map(|t| (rho.powi(t as i32), 0.0, theta.cos()))
        .collect()
}

/// Run a stream through a halter, collecting decisions (loop_idx 1-based).
fn run(halter: &mut GainCostLoopHalter, stream: &Stream) -> Vec<(usize, HaltDecision)> {
    let mut out = Vec::with_capacity(stream.len());
    for (i, &(gain, cost, cos_theta)) in stream.iter().enumerate() {
        let d = halter.halt_decision(i + 1, gain, cost, cos_theta);
        out.push((i + 1, d));
        if matches!(d, HaltDecision::Halt { .. }) {
            break;
        }
    }
    out
}

/// A per-loop scalar stream the halter consumes: `(gain, cost, cosθ)`.
type Stream = Vec<(f32, f32, f32)>;

fn armed() -> GainCostLoopHalter {
    GainCostLoopHalter::new(1.0, PATIENCE, 1).with_mode_verdict()
}

fn unarmed() -> GainCostLoopHalter {
    GainCostLoopHalter::new(1.0, PATIENCE, 1)
}

// ─────────────────────────────────────────────────────────────────────────
// P1 — mode-fit recovery
// ─────────────────────────────────────────────────────────────────────────

/// A real channel of a damped spiral, `δ_t = ρ^t·cos(tθ)`, is an exact AR(2):
/// `a = 2ρ·cosθ`, `b = −ρ²` — the fit must recover the complex pair `(ρ, θ)`.
#[test]
fn p1_spiral_channel_recovers_the_complex_pair() {
    for (rho, theta) in [(0.9_f32, 0.7_f32), (0.8, 2.5), (0.95, 1.2)] {
        let mut f = ModeFitter::new();
        for t in 1..=48 {
            f.observe((t as f32 * theta).cos() * rho.powi(t));
        }
        let m = f.estimate().expect("exact AR(2) spiral must fit");
        assert_eq!(m.kind, ModeKind::ComplexPair, "rho {rho} theta {theta}");
        assert!(
            (m.r - rho).abs() < 5e-3,
            "rho {rho} theta {theta}: recovered r {}",
            m.r
        );
        let omega = m.omega.expect("complex pair carries omega");
        assert!(
            (omega - theta).abs() < 5e-3,
            "rho {rho} theta {theta}: recovered omega {omega}"
        );
    }
}

/// Alternating flip `δ_t = (−r)^t` — rank-deficient WITHOUT the ridge; the
/// ridge selects the min-norm solution whose dominant root recovers `−r`.
#[test]
fn p1_alternating_flip_recovers_s_minus_r_through_the_ridge() {
    for r in [0.6_f32, 0.85_f32, 0.97_f32] {
        let mut f = ModeFitter::new();
        let mut d = 1.0_f32;
        for _ in 0..48 {
            f.observe(d);
            d *= -r;
        }
        let m = f
            .estimate()
            .expect("flip window must fit through the ridge");
        assert_eq!(m.kind, ModeKind::RealFlip, "flip r {r}");
        assert!((m.r - r).abs() < 5e-3, "flip r {r}: recovered r {}", m.r);
    }
}

/// SNR sweep: RELATIVE noise on a spiral channel degrades the recovery
/// gracefully — tight at high SNR, bounded at low — never a kind flip to a
/// real root at the high-SNR end.
#[test]
fn p1_snr_sweep_bounds_the_noise_sensitivity() {
    for (snr_sigma, tol) in [(1e-3_f32, 5e-3_f32), (1e-2, 2e-2), (1e-1, 1e-1)] {
        for seed in [7_u64, 11, 23] {
            let mut rng = Lcg(seed);
            let mut f = ModeFitter::new();
            for t in 1..=48 {
                let delta = (t as f32 * 0.7_f32).cos() * 0.9_f32.powi(t);
                f.observe(delta + rng.noise(snr_sigma) * delta.abs());
            }
            let m = f
                .estimate()
                .unwrap_or_else(|| panic!("SNR {snr_sigma} seed {seed}: window must fit"));
            assert!(
                (m.r - 0.9).abs() < tol,
                "SNR {snr_sigma} seed {seed}: |r−0.9| = {} > {tol}",
                (m.r - 0.9).abs()
            );
        }
    }
}

/// A linear ramp (`δ_t = t`, aperiodic-but-moving) honestly reports its unit
/// positive root — the correct convergence statement (never converges).
/// Tolerance 5e-2, not 1e-3: the ramp's exact fit (a=2, b=−1) sits ON the
/// double-root boundary (disc = 0), where any ridge perturbation moves the
/// dominant root by O(√ridge-effect) — the verdict semantics (r ≈ 1 ⇒ a
/// bounded non-convergent mode) are what the gate pins.
#[test]
fn p1_ramp_reports_its_honest_unit_root() {
    let mut f = ModeFitter::new();
    for t in 1..48 {
        f.observe(t as f32);
    }
    let m = f.estimate().expect("a moving ramp window must fit");
    assert_eq!(m.kind, ModeKind::RealPositive);
    assert!((m.r - 1.0).abs() < 5e-2, "ramp root {}", m.r);
}

/// The budget law is invertible and consistent with the predictor.
#[test]
fn p3_budget_law_is_invertible() {
    let r = 0.8_f32;
    let delta = 0.01_f32;
    let steps = settling_steps(r, delta).expect("settling r<1");
    let back = tolerance_for_budget(r, steps).expect("tolerance r<1");
    assert!((back - delta).abs() < 1e-4, "δ_B round trip: {back}");
    assert_eq!(settling_steps(1.0, delta), None, "r=1 never converges");
    assert_eq!(settling_steps(0.0, delta), None);
    assert_eq!(tolerance_for_budget(1.5, 10.0), None);
}

// ─────────────────────────────────────────────────────────────────────────
// P2 — the halter verdict
// ─────────────────────────────────────────────────────────────────────────

/// Settling flip (r = 0.7): the incumbent detector false-halts at patience;
/// the armed halter NEVER emits `Oscillation` — zero false halt-and-damps.
#[test]
fn p2_settling_flip_never_halts_on_oscillation() {
    // Incumbent: halts (the false halt-and-damp the tension law forbids).
    let inc = run(&mut unarmed(), &flip_stream(0.7, 64));
    assert!(
        matches!(
            inc.last(),
            Some((
                _,
                HaltDecision::Halt {
                    reason: HaltReason::Oscillation
                }
            ))
        ),
        "incumbent must false-halt on a settling flip (the premise under repair)"
    );
    // Armed: never.
    let mut h = armed();
    let dec = run(&mut h, &flip_stream(0.7, 64));
    for (loop_idx, d) in &dec {
        assert!(
            !matches!(
                d,
                HaltDecision::Halt {
                    reason: HaltReason::Oscillation
                }
            ),
            "armed halter Oscillation-halted a settling flip at loop {loop_idx}"
        );
    }
    assert_eq!(
        h.last_oscillation_verdict(),
        Some(OscillationVerdict::OscillatingButSettling),
        "the observation surface must classify the settling flip"
    );
}

/// Settling spiral (ρ = 0.9, θ = 2.5 rad — reversals every loop): same zero
/// false-halt contract.
#[test]
fn p2_settling_spiral_never_halts_on_oscillation() {
    let mut h = armed();
    let dec = run(&mut h, &spiral_stream(0.9, 2.5, 64));
    for (loop_idx, d) in &dec {
        assert!(
            !matches!(
                d,
                HaltDecision::Halt {
                    reason: HaltReason::Oscillation
                }
            ),
            "armed halter Oscillation-halted a settling spiral at loop {loop_idx}"
        );
    }
    assert_eq!(
        h.last_oscillation_verdict(),
        Some(OscillationVerdict::OscillatingButSettling)
    );
}

/// Limit-cycle flip (r = 1.0): the armed halter keeps the incumbent halt at
/// UNCHANGED latency — same loop index, same reason.
#[test]
fn p2_limit_cycle_halts_at_unchanged_latency() {
    let stream = flip_stream(1.0, 64);
    let inc = run(&mut unarmed(), &stream);
    let arm = run(&mut armed(), &stream);
    let inc_halt = inc
        .iter()
        .find_map(|(i, d)| {
            matches!(
                d,
                HaltDecision::Halt {
                    reason: HaltReason::Oscillation
                }
            )
            .then_some(*i)
        })
        .expect("incumbent must halt a limit cycle");
    let arm_halt = arm
        .iter()
        .find_map(|(i, d)| {
            matches!(
                d,
                HaltDecision::Halt {
                    reason: HaltReason::Oscillation
                }
            )
            .then_some(*i)
        })
        .expect("armed halter must halt a limit cycle");
    assert_eq!(inc_halt, arm_halt, "true-halt latency must be maintained");
    let mut h = armed();
    let _ = run(&mut h, &stream);
    assert_eq!(
        h.last_oscillation_verdict(),
        Some(OscillationVerdict::LimitCycle)
    );
}

/// Expanding spiral (ρ = 1.1): a growth mode is a limit/exceeding cycle —
/// the armed halter must NOT suppress the halt.
#[test]
fn p2_expanding_spiral_still_halts() {
    let stream = spiral_stream(1.1, 2.5, 64);
    let dec = run(&mut armed(), &stream);
    assert!(
        dec.iter().any(|(_, d)| matches!(
            d,
            HaltDecision::Halt {
                reason: HaltReason::Oscillation
            }
        )),
        "an expanding spiral must still Oscillation-halt when armed"
    );
}

/// A NaN-corrupted gain must not fire anything and must not wipe the
/// fitter's verdict (the safe direction): the corrupt feed is DROPPED, the
/// estimate stays the last good one, and the loop continues.
#[test]
fn p2_nan_gain_is_safe() {
    let mut h = armed();
    // Warm past patience on a settling flip — every reversal is suppressed.
    for t in 1..=24 {
        let d = h.halt_decision(t, 0.7_f32.powi(t as i32), 0.0, -1.0);
        assert_eq!(
            d,
            HaltDecision::Continue,
            "settling flip must be suppressed at loop {t}"
        );
    }
    assert_eq!(
        h.last_oscillation_verdict(),
        Some(OscillationVerdict::OscillatingButSettling)
    );
    // Corrupt loop: NaN gain at a reversal — no panic, the dropped feed keeps
    // the last good verdict, and the decision stays the safe (continue) one.
    let d = h.halt_decision(25, f32::NAN, 0.0, -1.0);
    assert_eq!(d, HaltDecision::Continue, "NaN gain must not fire a halt");
    assert_eq!(
        h.last_oscillation_verdict(),
        Some(OscillationVerdict::OscillatingButSettling),
        "the dropped feed must preserve the last good verdict"
    );
    // And a healthy settling loop after the corruption still suppresses.
    let d = h.halt_decision(26, 0.7_f32.powi(26), 0.0, -1.0);
    assert_eq!(d, HaltDecision::Continue);
}

/// The kill switch (exact literal `KATGPT_QRR_OSC_VERDICT=0`): the CHILD
/// process proves armed == unarmed byte-identity under the env; the PARENT
/// (no env) proves arming actually changes the settling trajectories.
#[test]
fn p2_kill_switch_child_restores_byte_identity() {
    let exe = std::env::current_exe().expect("test exe");
    let out = std::process::Command::new(exe)
        .args(["qrr_stream_identity_child", "--exact", "--test-threads=1"])
        .env("KATGPT_QRR_OSC_VERDICT", "0")
        .output()
        .expect("spawn child");
    assert!(
        out.status.success(),
        "kill-switch child failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Runs in BOTH postures: under the parent's env the armed decisions must be
/// BYTE-IDENTICAL to the unarmed incumbent on every fixture; in the parent
/// (no env) arming must actually change the settling-fixture decisions.
#[test]
fn qrr_stream_identity_child() {
    let fixtures = [
        ("settling_flip", flip_stream(0.7, 64)),
        ("settling_spiral", spiral_stream(0.9, 2.5, 64)),
        ("limit_flip", flip_stream(1.0, 64)),
        ("expanding_spiral", spiral_stream(1.1, 2.5, 64)),
    ];
    let demoted = std::env::var("KATGPT_QRR_OSC_VERDICT").is_ok_and(|v| v == "0");
    let mut any_armed_difference = false;
    for (name, stream) in fixtures.iter() {
        let inc = run(&mut unarmed(), stream);
        let arm = run(&mut armed(), stream);
        let same = inc == arm;
        if demoted {
            assert_eq!(inc, arm, "{name}: kill switch must restore byte identity");
        } else if !same {
            any_armed_difference = true;
        }
    }
    if !demoted {
        assert!(
            any_armed_difference,
            "without the kill switch arming must change at least one settling fixture"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// P3 — budget calibration + certificate
// ─────────────────────────────────────────────────────────────────────────

/// One synthetic corpus run: feed the fitter the signed projection of a
/// decaying mode (spiral or flip) with noise; score the budget prediction at
/// the 2/3 mark against the TRUE remaining steps to relative tolerance δ.
fn calibration_run(rho: f32, theta: Option<f32>, seed: u64, sigma: f32) -> (f32, f32) {
    let mut rng = Lcg(seed);
    let mut f = ModeFitter::new();
    let n = 48_usize;
    let delta = 0.01_f32;
    let mut fitted_at_mark = None;
    for t in 1..=n {
        let gain = rho.powi(t as i32);
        let cos = match theta {
            Some(th) => th.cos(),
            None => -1.0, // flip
        };
        f.observe_signed(gain * cos + rng.noise(sigma) * gain.abs().max(1e-6), cos);
        if t == (n * 2) / 3
            && let Some(m) = f.estimate()
        {
            fitted_at_mark = Some(m.r);
        }
    }
    let fitted = fitted_at_mark.expect("mid-run fit must exist");
    let predicted = settling_steps(fitted, delta).unwrap_or(f32::INFINITY);
    // True remaining steps from the mark: ln(δ)/ln(ρ) relative decay.
    let actual = (delta.ln() / rho.ln()).abs();
    (predicted, actual)
}

/// Calibration gate: predicted vs actual settling within ×2 on ≥ 95% of the
/// synthetic corpus; the constant-budget baseline is printed beside it (the
/// floor the budget law beats: one budget must cover the WORST member).
#[test]
fn p3_budget_calibration_within_2x_on_95pct_of_corpus() {
    let mut corpus = Vec::new();
    for rho in [0.6_f32, 0.75, 0.9, 0.97] {
        for theta in [Some(0.7_f32), Some(2.5), None] {
            for seed in [7_u64, 11, 23] {
                corpus.push((rho, theta, seed));
            }
        }
    }
    let mut within = 0_usize;
    let mut worst_ratio = 0.0_f32;
    for &(rho, theta, seed) in &corpus {
        let (predicted, actual) = calibration_run(rho, theta, seed, 1e-2);
        if actual > 0.0 && predicted.is_finite() {
            let ratio = (predicted / actual).max(actual / predicted);
            worst_ratio = worst_ratio.max(ratio);
            if ratio <= 2.0 {
                within += 1;
            }
        }
    }
    let total = corpus.len();
    let share = within as f32 / total as f32;
    // The floor vs the constant-budget baseline: a FIXED budget must cover
    // the worst member (ρ = 0.97 → τ_0.01 ≈ 151 loops), while the predictor
    // pays each member only its own budget.
    let worst_member_budget = settling_steps(0.97, 0.01).unwrap();
    let mean_predicted: f32 = corpus
        .iter()
        .map(|&(rho, theta, seed)| calibration_run(rho, theta, seed, 1e-2).0)
        .filter(|p| p.is_finite())
        .sum::<f32>()
        / total as f32;
    println!("qrr p3 calibration: {within}/{total} within 2x (share {share:.3})");
    println!(
        "qrr p3 floor: constant budget {worst_member_budget:.1} vs mean predicted {mean_predicted:.1}"
    );
    assert!(
        share >= 0.95,
        "calibration {within}/{total} below the 95% bar (worst ratio {worst_ratio:.2})"
    );
}

/// The r = 1 certificate is TWO-SIDED: fires on `r = 1` members with a
/// stagnant complement, never on settling members (r < 1), and never when
/// only one signal is present.
#[test]
fn p3_certificate_is_two_sided() {
    let stagnant = true;
    let progressing = false;
    // r = 1 members (limit flip, pure rotation): near-one streak over the
    // per-loop estimates (the deployed usage — one estimate per reversal
    // loop), stagnant complement → FIRES.
    let mut limit = ModeFitter::new();
    for _ in 0..40 {
        limit.observe_signed(1.0, -1.0);
        let _ = limit.estimate();
    }
    assert!(
        limit.near_one_streak() >= 3,
        "unit flip must build a near-one streak"
    );
    let fires = R1Certificate {
        near_one: true,
        complement_stagnant: stagnant,
    }
    .fires();
    assert!(fires, "r=1 + stagnant complement must fire");

    // Settling member (ρ = 0.9 flip): the modulus half must FAIL even though
    // a raw-residual observer might call the complement stagnant.
    let mut settling = ModeFitter::new();
    let mut d = 1.0_f32;
    for _ in 0..40 {
        settling.observe_signed(d, -1.0);
        let _ = settling.estimate();
        d *= 0.9;
    }
    assert_eq!(
        settling.near_one_streak(),
        0,
        "a settling flip must never enter the near-one band"
    );
    assert!(
        !R1Certificate {
            near_one: false,
            complement_stagnant: stagnant
        }
        .fires(),
        "mode half missing → must not fire"
    );

    // Progressing complement (growing spiral): modulus half may hold but the
    // complement is moving → must not fire.
    assert!(
        !R1Certificate {
            near_one: true,
            complement_stagnant: progressing
        }
        .fires(),
        "complement half missing → must not fire"
    );
}
