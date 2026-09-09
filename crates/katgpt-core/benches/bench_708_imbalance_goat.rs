//! Issue 708 P2 — Two-channel imbalance collapse monitor GOAT gate.
//!
//! The falsifiable claim (Research 437 — arXiv:2608.29335 GenFirst
//! two-channel law, transferred to modelless runtime populations): a
//! conjunctive imbalance alarm (channel A = KL entropy FALLING while
//! channel B = directional concentration RISING, both in baseline σ)
//! detects population collapse EARLIER than the shipped absolute detector
//! (`h < τ_low`), because deviation-from-baseline crosses early while the
//! level must traverse healthy → collapsed. Kill condition: lead time ≈ 0
//! ⇒ the imbalance framing adds nothing over τ_low.
//!
//! Gates (run on the bench_681 fixture populations — gaussian d64,
//! bimodal-axis d64, lattice d8):
//!
//! - **G1 lead time** — on every planted mode-collapse arm: the imbalance
//!   alarm fires, the absolute detector fires, and the imbalance leads by
//!   ≥ 1 cycle.
//! - **G2 monotone in severity** — across the severity (collapse-rate)
//!   ladder, lead time is monotone non-increasing in rate (slower
//!   degradation ⇒ more lead cycles; deterministic seeds make the ordering
//!   exact).
//! - **G3 healthy specificity** — fresh-healthy populations for 40 cycles
//!   after warm-up: zero imbalance alarms, zero absolute crossings.
//! - **G4 sub-threshold detection** — degradation arms whose end-state
//!   severity never reaches the absolute threshold (`λ_max < λ_abs`) are
//!   detected by the imbalance channel alone (the operational win: early
//!   warning on sub-threshold drift the absolute detector cannot see).
//! - **G5 scope-boundary negative control** — ISOTROPIC σ-shrink: channel
//!   A falls hard (absolute fires) but channel B is scale-invariant and
//!   stays flat, so the conjunction correctly stays silent. The imbalance
//!   monitor's regime is DIRECTIONAL/mode collapse; isotropic drift belongs
//!   to the absolute/derivative channels. Pinned in-module too
//!   (`imbalance.rs::tests::isotropic_shrink_does_not_alarm`).
//! - **G6 determinism** — a planted arm re-run produces identical fire
//!   cycles (the module test pins bit-identical readings).
//!
//! # Trajectory + comparator (measured design, first bench run)
//!
//! The planted collapse is MODE INTERPOLATION — `p' = (1−λ)·p + λ·m`
//! toward a fixed off-center attractor `m` (‖m‖ = 3σ for the float
//! fixtures, the all-ones corner for the lattice): the issue's own game
//! reframe ("personality collapses into exploitation") — mass converging
//! onto a behavioral mode. The first run's variance-shrink trajectory
//! (minor-dim σ scaled down) was REFUTED as a lead-time fixture: on a
//! near-isotropic d64 population the cosine channel is scale-invariant and
//! correctly stays flat until λ ≳ 0.8 (d_eff = (Σv)²/Σv² moves 64 → 59 at
//! λ = 0.45), so the conjunct was B-starved and the gaussian lead
//! collapsed to [3, 1, 0]. Mode interpolation moves BOTH channels early on
//! every fixture structure.
//!
//! The absolute comparator is `τ_low = 5% of H₀` — the scale-free mirror
//! of the SHIPPED absolute detector's semantics (`CgspConfig::default()
//! .tau_low = 0.30` nats: a small fixed fraction of healthy entropy). A
//! ½·H₀ line was tried first and is REFUTED as unrepresentative — it lets
//! the absolute detector fire mid-descent and eats the measured lead.
//!
//! G4 (alloc-free) lives in the module's own test — the bench_681
//! convention (TrackingAllocator under cfg(test, debug_assertions), the
//! lib test binary installs it).
//!
//! std::time::Instant + harness=false (repo bench convention).
//!
//! Run: cargo bench -p katgpt-core --bench bench_708_imbalance_goat
//!      (or: cargo test --features imbalance_monitor
//!            --bench bench_708_imbalance_goat -- --nocapture)

use katgpt_core::data_probe::imbalance::{ImbalanceConfig, ImbalanceMonitor};
use katgpt_core::types::Rng;

const N: usize = 1024;
const D: usize = 64;
/// Lattice runs at a smaller population BY MEASUREMENT: d8 binary vectors
/// have only 2⁸ = 256 distinct points, so at n=1024 the mean cell occupancy
/// is 4 and 5-fold exact duplicates are common — P1's honest duplicate
/// contract (k-NN distance 0 ⇒ Ĥ = −∞) fires on the HEALTHY fixture (the
/// first run measured τ_low = −inf). At n=128 the mean occupancy is 0.5 and
/// 5-folds are essentially impossible; the fixture stays the bench_681
/// lattice, sized for the estimator's duplicate contract.
const N_LATTICE: usize = 128;
const WARMUP: usize = 12; // 2× the 2-sample minimum — stable Welford σ
const DEGRADE_CYCLES: usize = 60;
const HEALTHY_CYCLES: usize = 40;

/// Severity (collapse-rate) ladder — λ per cycle, clamped at `λ_max`.
const RATES: [f64; 3] = [0.025, 0.05, 0.1];

/// The GOAT's calibrated operating point (measured on the first bench run,
/// 2026-09-02): at d=64/n=1024 the KL estimator is variance-starved — the
/// healthy warm-up σ₀ measured ~12–15 nats on ~101 nats, so the default
/// 3σ_A margin needs a ~40-nat drop ≈ exactly where τ_low sits, eating the
/// lead (gaussian lead [4, 1, 0→negative] on run 1 while bimodal — σ₀ ~0.5
/// — showed the ideal [22, 9, 4]). Calibration: k = 8 (P1's documented
/// variance-reduction end of the band), warm-up 12, and k_a = 2.0 — the
/// conjunctive false-alarm rate stays ~3e-5/cycle (B is effectively
/// deterministic and gates the conjunction), pinned by the G3 specificity
/// gate. Channel B is untouched at 3σ.
fn goat_config() -> ImbalanceConfig {
    ImbalanceConfig {
        k_a: 2.0,
        k_b: 3.0,
        warmup_cycles: WARMUP,
        entropy_k: 8,
    }
}

// ── bench_681 fixture populations (test-side assembly, the bench_681
//    convention — the generators are defined in that bench file, not in the
//    library, so they are restated here verbatim) ─────────────────────────

fn gaussian_population(seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut out = vec![0.0f32; N * D];
    for v in out.iter_mut() {
        *v = rng.normal();
    }
    out
}

fn bimodal_axis_population(axis: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut out = vec![0.0f32; N * D];
    for i in 0..N {
        let sign = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
        for j in 0..D {
            out[i * D + j] = rng.normal();
        }
        out[i * D + axis] += sign * 3.0;
    }
    out
}

fn lattice_population_d8(seed: u64) -> (Vec<f32>, usize) {
    let d = 8;
    let mut rng = Rng::new(seed);
    let mut out = vec![0.0f32; N_LATTICE * d];
    for v in out.iter_mut() {
        *v = if rng.uniform() < 0.5 { 0.0 } else { 1.0 };
    }
    (out, d)
}

/// Normalized fixture: population + intrinsic dimension.
type Fixture = dyn Fn(u64) -> (Vec<f32>, usize);

fn gaussian_fx(seed: u64) -> (Vec<f32>, usize) {
    (gaussian_population(seed), D)
}

fn bimodal_fx(seed: u64) -> (Vec<f32>, usize) {
    (bimodal_axis_population(0, seed), D)
}

fn lattice_fx(seed: u64) -> (Vec<f32>, usize) {
    lattice_population_d8(seed)
}

/// Fixed off-center attractor for the mode-collapse trajectory: a 3σ-mode
/// direction for the float fixtures (`‖m‖ = 3`), the all-ones corner for
/// the d8 lattice (`‖m‖ = 2.83` vs its 1.41-norm centroid — a distinct
/// behavioral mode). Deterministic in `d`.
fn prototype_for(d: usize) -> Vec<f32> {
    let c = if d == 8 { 1.0 } else { 3.0 / (d as f64).sqrt() };
    vec![c as f32; d]
}

// ── degradation shapes ──────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum DegradeMode {
    /// Mode collapse: interpolate every point toward the fixed off-center
    /// attractor, `p' = (1−λ)·p + λ·m`. Channel A falls (the population
    /// contracts), channel B rises (pairwise alignment onto `m`).
    ModeCollapse,
    /// Isotropic contraction: scale EVERY dim by (1 − λ). Channel B flat.
    Isotropic,
}

fn degrade(base: &[f32], n: usize, d: usize, lam: f64, mode: DegradeMode) -> Vec<f32> {
    let keep = (1.0 - lam) as f32;
    let mut out = base.to_vec();
    match mode {
        DegradeMode::ModeCollapse => {
            let m = prototype_for(d);
            for i in 0..n {
                for j in 0..d {
                    out[i * d + j] =
                        (base[i * d + j] as f64 * keep as f64 + lam * m[j] as f64) as f32;
                }
            }
        }
        DegradeMode::Isotropic => {
            for v in out.iter_mut() {
                *v *= keep;
            }
        }
    }
    out
}

// ── arm runner ──────────────────────────────────────────────────────────

struct ArmResult {
    t_imb: Option<usize>,
    t_abs: Option<usize>,
    /// Healthy warm-up baseline (mean, σ) — reported per arm so the
    /// calibration is visible in the gate table.
    baseline: (f64, f64),
    /// Last post-warm-up reading (determinism comparison).
    last_reading: (u64, u64, u64, u64),
}

fn run_arm(
    fx: &Fixture,
    seed0: u64,
    rate: f64,
    lam_max: f64,
    mode: DegradeMode,
    degrade_cycles: usize,
) -> ArmResult {
    let verbose = std::env::var("B708_VERBOSE").is_ok();
    let mut mon = ImbalanceMonitor::new(goat_config());

    // Healthy warm-up: a FRESH population every cycle (natural estimator +
    // draw noise feeds the Welford baseline — a static population would give
    // zero variance and an infinitely jumpy z-score).
    let mut warm_entropies = Vec::with_capacity(WARMUP);
    let (pop0, d) = fx(seed0);
    let n = pop0.len() / d;
    for c in 0..WARMUP {
        let (pop, _) = fx(seed0 + c as u64);
        let r = mon.observe(&pop, n, d);
        warm_entropies.push(r.entropy);
    }
    let mean0 = warm_entropies.iter().sum::<f64>() / warm_entropies.len() as f64;
    let var0 = warm_entropies
        .iter()
        .map(|h| (h - mean0) * (h - mean0))
        .sum::<f64>()
        / (warm_entropies.len() - 1) as f64;
    let baseline = (mean0, var0.sqrt());
    // The absolute reference detector's threshold: 5% of the healthy level —
    // the scale-free mirror of the SHIPPED absolute detector (`CgspConfig
    // ::default().tau_low` = 0.30 nats, a small fixed fraction of healthy
    // entropy). The absolute channel fires only at a deep collapse
    // (λ_abs ≈ 0.78 on the d64 fixtures); the imbalance conjunct's claim is
    // that it fires well before that.
    let tau_low = mean0 * 0.05;

    let mut t_imb = None;
    let mut t_abs = None;
    let mut last_reading = (0u64, 0u64, 0u64, 0u64);
    for c in 0..degrade_cycles {
        let lam = if rate <= 0.0 {
            0.0
        } else {
            (rate * (c + 1) as f64).min(lam_max)
        };
        let (base, _) = fx(seed0 + WARMUP as u64 + c as u64);
        let pop = degrade(&base, n, d, lam, mode);
        let r = mon.observe(&pop, n, d);
        if verbose {
            eprintln!(
                "    c={c:<2} λ={lam:.3} h={:<8.2} a_z={:<8.2} conc={:.4} b_z={:<8.2} imb={}",
                r.entropy, r.a_z, r.concentration, r.b_z, r.alarmed
            );
        }
        if t_imb.is_none() && r.alarmed {
            t_imb = Some(c);
        }
        if t_abs.is_none() && r.entropy < tau_low {
            t_abs = Some(c);
        }
        last_reading = (
            r.entropy.to_bits(),
            r.concentration.to_bits(),
            r.a_z.to_bits(),
            r.b_z.to_bits(),
        );
        if t_imb.is_some() && t_abs.is_some() {
            break;
        }
    }
    ArmResult {
        t_imb,
        t_abs,
        baseline,
        last_reading,
    }
}

fn fmt_cycle(c: Option<usize>) -> String {
    match c {
        Some(v) => format!("{v:>3}"),
        None => "  —".to_string(),
    }
}

/// Determinism probe snapshot: (t_imb, t_abs, last-reading bits).
type DetProbe = (Option<usize>, Option<usize>, (u64, u64, u64, u64));

fn main() {
    let mut failures = 0usize;
    let t_start = std::time::Instant::now();

    println!("═══ Issue 708 P2 — imbalance collapse monitor GOAT ═══");
    println!(
        "fixtures: gaussian d{D} / bimodal-axis d{D} / lattice d8 · n={N} · \
         warmup={WARMUP} · τ_low = 5%·H₀ · rates {RATES:?} · degrade ≤ {DEGRADE_CYCLES} cycles"
    );
    println!();

    // ── G1 + G2: planted directional-collapse arms ───────────────────────
    println!("═══ G1/G2 — planted mode collapse: lead time ═══");
    println!("fixture   rate    λ_max  imb  abs  lead  H₀±σ₀(nats)");
    let fixtures: [(&str, &Fixture); 3] = [
        ("gaussian", &gaussian_fx),
        ("bimodal", &bimodal_fx),
        ("lattice", &lattice_fx),
    ];
    let mut leads: Vec<[usize; 3]> = Vec::with_capacity(fixtures.len());
    let mut det_probe: Option<DetProbe> = None;
    for (name, fx) in fixtures {
        let mut fixture_leads = [0usize; 3];
        for (ri, &rate) in RATES.iter().enumerate() {
            let arm = run_arm(
                fx,
                90_000 + (ri as u64) * 77,
                rate,
                1.0,
                DegradeMode::ModeCollapse,
                DEGRADE_CYCLES,
            );
            let lead_str = match (arm.t_imb, arm.t_abs) {
                (Some(a), Some(b)) if b >= a => format!("{}", b - a),
                (Some(a), Some(b)) => format!("NEG({})", b - a),
                _ => "—".to_string(),
            };
            println!(
                "{name:<9} {rate:<7} 1.0    {} {}  {:>6}  {:.1}±{:.1}",
                fmt_cycle(arm.t_imb),
                fmt_cycle(arm.t_abs),
                lead_str,
                arm.baseline.0,
                arm.baseline.1,
            );
            // G1: both channels must fire, imbalance strictly first.
            match (arm.t_imb, arm.t_abs) {
                (Some(ti), Some(ta)) if ta > ti => {
                    fixture_leads[ri] = ta - ti;
                }
                _ => {
                    failures += 1;
                    println!(
                        "  FAIL (G1): {name} rate {rate} — t_imb {:?} t_abs {:?} (need imb < abs, both fired)",
                        arm.t_imb, arm.t_abs
                    );
                }
            }
            if ri == 1 && name == "gaussian" {
                det_probe = Some((arm.t_imb, arm.t_abs, arm.last_reading));
            }
        }
        leads.push(fixture_leads);
    }

    // G2: lead monotone non-increasing across the rate ladder (slower
    // degradation ⇒ more lead cycles). Deterministic ⇒ exact ordering.
    println!();
    println!("═══ G2 — lead monotone in severity ═══");
    for ((name, _), l) in fixtures.iter().zip(&leads) {
        println!("{name:<9} lead by rate {RATES:?}: {l:?}");
        if !(l[0] >= l[1] && l[1] >= l[2]) {
            failures += 1;
            println!("  FAIL (G2): {name} lead not monotone non-increasing in rate");
        }
        if l[2] == 0 {
            failures += 1;
            println!("  FAIL (G2): {name} zero lead at the fastest rate — imbalance adds nothing");
        }
    }

    // ── G3: healthy specificity ──────────────────────────────────────────
    println!();
    println!("═══ G3 — healthy specificity ({HEALTHY_CYCLES} fresh-healthy cycles) ═══");
    for (name, fx) in fixtures {
        let arm = run_arm(
            fx,
            80_000,
            0.0,
            0.0,
            DegradeMode::ModeCollapse,
            HEALTHY_CYCLES,
        );
        let clean = arm.t_imb.is_none() && arm.t_abs.is_none();
        println!(
            "{name:<9} imb {} abs {} — {}",
            fmt_cycle(arm.t_imb),
            fmt_cycle(arm.t_abs),
            if clean { "CLEAN" } else { "FALSE ALARM" }
        );
        if !clean {
            failures += 1;
            println!("  FAIL (G3): {name} false-alarmed on healthy populations");
        }
    }

    // ── G4: sub-threshold severity — imbalance-only detection ────────────
    println!();
    println!("═══ G4 — sub-threshold severity (imbalance-only detection) ═══");
    // λ_max chosen between the measured imbalance alarm point (B crosses
    // k_b·σ_B around λ ≈ 0.3 on the gaussian mode-collapse trajectory) and
    // the absolute crossing (λ_abs ≈ 0.78 for τ = 5%·H₀): both arms sit
    // strictly BELOW the absolute threshold — the absolute detector
    // structurally cannot fire inside them — while remaining above the
    // imbalance alarm point. The falsifiable content: a severity the
    // absolute detector cannot see is detected early.
    for lam_max in [0.4f64, 0.6] {
        let arm = run_arm(
            &gaussian_fx,
            95_000,
            0.05,
            lam_max,
            DegradeMode::ModeCollapse,
            DEGRADE_CYCLES,
        );
        let imb_only = arm.t_imb.is_some() && arm.t_abs.is_none();
        println!(
            "gaussian λ_max={lam_max}  imb {} abs {} — {}",
            fmt_cycle(arm.t_imb),
            fmt_cycle(arm.t_abs),
            if imb_only { "IMB-ONLY (win)" } else { "MISSED" }
        );
        if !imb_only {
            failures += 1;
            println!(
                "  FAIL (G4): λ_max {lam_max} — expected imbalance-only detection \
                 (t_imb {:?}, t_abs {:?})",
                arm.t_imb, arm.t_abs
            );
        }
    }

    // ── G5: isotropic scope boundary ─────────────────────────────────────
    println!();
    println!("═══ G5 — isotropic σ-shrink scope boundary (negative control) ═══");
    {
        let arm = run_arm(
            &gaussian_fx,
            97_000,
            0.05,
            1.0,
            DegradeMode::Isotropic,
            DEGRADE_CYCLES,
        );
        let boundary = arm.t_abs.is_some() && arm.t_imb.is_none();
        println!(
            "gaussian isotropic  imb {} abs {} — {}",
            fmt_cycle(arm.t_imb),
            fmt_cycle(arm.t_abs),
            if boundary {
                "BOUNDARY HELD (absolute fires, imbalance correctly silent)"
            } else {
                "UNEXPECTED"
            }
        );
        if !boundary {
            failures += 1;
            println!(
                "  FAIL (G5): isotropic control — expected absolute-fire + imbalance-silent \
                 (t_imb {:?}, t_abs {:?})",
                arm.t_imb, arm.t_abs
            );
        }
    }

    // ── G6: determinism spot-check ───────────────────────────────────────
    println!();
    println!("═══ G6 — determinism spot-check (gaussian rate 0.05 re-run) ═══");
    {
        let arm = run_arm(
            &gaussian_fx,
            90_077,
            0.05,
            1.0,
            DegradeMode::ModeCollapse,
            DEGRADE_CYCLES,
        );
        if let Some((ti, ta, bits)) = det_probe {
            let same = arm.t_imb == ti && arm.t_abs == ta && arm.last_reading == bits;
            println!(
                "re-run: imb {} abs {} bits-match {}",
                fmt_cycle(arm.t_imb),
                fmt_cycle(arm.t_abs),
                arm.last_reading == bits
            );
            if !same {
                failures += 1;
                println!("  FAIL (G6): re-run diverged");
            }
        } else {
            failures += 1;
            println!("  FAIL (G6): determinism probe arm missing");
        }
    }

    println!();
    println!("elapsed: {:.1}s", t_start.elapsed().as_secs_f64());
    if failures == 0 {
        println!("Issue 708 P2 GOAT: ALL GATES PASS (opt-in; promotion deferred to a consumer)");
    } else {
        println!("Issue 708 P2 GOAT: {failures} FAILURES");
        std::process::exit(1);
    }
}
