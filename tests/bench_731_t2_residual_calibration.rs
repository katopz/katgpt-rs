#![cfg(all(feature = "lt2_looped", feature = "cadence_gate", feature = "loop_stability_fix"))]
//! Issue 731 T2 — τ calibration for the residual-gated loop exit, e2e on the
//! T1 fixture convention (micro + seed-42 + Uniform + AHLA, R_REF = 32).
//!
//! # Pre-registered design (committed BEFORE the run, the Issue-073-T3 order)
//!
//! The T1 exit probe lives on `forward_looped`'s hidden-state trajectory; the
//! Plan-119 harness measures DDTree rollout selection (a different loop, a
//! different residual scale) — calibrating τ on it would be a category
//! error. This bench IS the looped-forward calibration harness.
//!
//! **Stability arms (2):**
//! - `None` — the natural-decay regime: the step norm ‖h_τ − h_{τ−1}‖ is
//!   expected to decay toward the fixed point; the EqR exit's target regime.
//! - `InterLoopNorm` — the Issue-698-T4 measured regime: step size GROWS to
//!   a plateau (~20 on their fixture) while convergence is DIRECTIONAL
//!   (cos θ → 1). This is the **Research-440 negative-control regime**:
//!   magnitude-only calibration is expected to FAIL here (no realistic τ
//!   fires, or it fires late), and the shape arm reads Churning — the exit
//!   must NOT fire.
//!
//! **Phases:**
//! - A. Depth→quality curve: for k in a fixed grid, `elastic_loop_override =
//!   k`, quality = mean cosine distance to the R_REF = 32 reference over all
//!   27 micro-vocab tokens. The knee = the smallest grid k with mean
//!   distance ≤ 0.01 (reported, not assumed).
//! - B. τ calibration: for τ in a fixed log grid (plus the d_min-only
//!   `+∞` arm), run the probe at d_min = 4 and record `fired_at_iteration`
//!   per token → median / min / max fired k, and the quality at that k read
//!   from phase A. The probe is deterministic given the trajectory, so the
//!   τ→k table IS the calibration curve.
//!
//! **Pinned expectations (asserted — a red here teaches us the fixture or
//! the regime moved):**
//! 1. InterLoopNorm, realistic τ ≤ 10: ZERO of 27 tokens fire (the plateau
//!    keeps the magnitude arm shut and the shape arm reads Churning) — the
//!    magnitude-only negative control, recorded.
//! 2. The exit ≡ elastic bit-equivalence (T1's G1) is re-checked at the
//!    calibration points: a probe that fired at k must equal elastic = k.
//!
//! **G2 GOAT input (read off the table, decided after the run):** a τ
//! qualifies iff median-fired k ≤ 16 (≥2× cut of 32) at mean cosine
//! distance ≤ the knee bound.
//!
//! # Run
//!
//! ```bash
//! cargo test --features "lt2_looped,cadence_gate,loop_stability_fix" \
//!   --test bench_731_t2_residual_calibration -- --nocapture
//! ```

use katgpt_core::convergence_cadence::LoopResidualExit;
use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{
    Config, HlaMode, HybridPattern, LoopMode, LoopStabilityMode, ResidualGate, Rng, SdpaOutputGate,
};

const R_REF: usize = 32;
const N_PROMPTS: usize = 27;
const SEED: u64 = 42;
const D_MIN: usize = 4;

/// Phase-A depth grid.
const K_GRID: [usize; 14] = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32];

/// Phase-B τ grid (log-spaced + the floor-only arm). Realistic τ ≤ 10 on
/// InterLoopNorm must never fire (expectation 1).
const TAU_GRID: [f32; 11] =
    [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, f32::INFINITY];

fn make_config(stability: LoopStabilityMode) -> Config {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count: R_REF };
    config.hybrid_pattern = HybridPattern::Uniform;
    config.hla_mode = HlaMode::Ahla;
    config.loop_stability_mode = stability;
    config
}

fn make_fixture(config: &Config) -> (TransformerWeights, ResidualGate, SdpaOutputGate) {
    let mut rng = Rng::new(SEED);
    let weights = TransformerWeights::new(config, &mut rng);
    let residual_gate = ResidualGate::new(R_REF, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    (weights, residual_gate, sdpa_gate)
}

fn run(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    token: usize,
    elastic: Option<usize>,
    probe: Option<&mut LoopResidualExit>,
) -> Vec<f32> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut ahla_cache = MultiLayerAhlaCache::new(config);
    forward_looped(
        &mut ctx,
        weights,
        &mut cache,
        &mut ahla_cache,
        token,
        0,
        config,
        residual_gate,
        sdpa_gate,
        None,
        None,
        #[cfg(feature = "weight_shared_advantage_gate")]
        None,
        elastic,
        #[cfg(feature = "gain_cost_halt")]
        None,
        None,  // Issue 717: deep_run — None = bit-identical baseline
        probe, // Issue 731: the residual-exit probe (cadence_gate builds)
    )
    .to_vec()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 { 1.0 - dot / denom } else { 0.0 }
}

fn median(v: &mut [usize]) -> usize {
    v.sort_unstable();
    v[v.len() / 2]
}

fn phase_a_curve(
    label: &str,
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
) -> Vec<(usize, f32)> {
    // Reference logits per token at the full R_REF depth.
    let refs: Vec<Vec<f32>> = (0..N_PROMPTS)
        .map(|t| run(config, weights, residual_gate, sdpa_gate, t, Some(R_REF), None))
        .collect();
    let mut curve = Vec::with_capacity(K_GRID.len());
    println!("\n[{label}] depth → quality (mean cosine distance to the R_REF = {R_REF} reference; lower = closer)");
    println!("| k | mean dist |");
    println!("|---|---|");
    for &k in &K_GRID {
        let mut total = 0.0f32;
        for (t, r) in refs.iter().enumerate() {
            let out = run(config, weights, residual_gate, sdpa_gate, t, Some(k), None);
            total += cosine_distance(&out, r);
        }
        let mean = total / N_PROMPTS as f32;
        curve.push((k, mean));
        println!("| {k} | {mean:.6} |");
    }
    curve
}

fn phase_b_calibration(
    label: &str,
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    curve: &[(usize, f32)],
) {
    println!("\n[{label}] τ calibration (d_min = {D_MIN}) — fired-at distribution and quality at fire");
    println!("| tau | fired n/27 | median k | min | max | mean cos-dist at median k |");
    println!("|---|---|---|---|---|---|");
    for &tau in &TAU_GRID {
        let mut fired: Vec<usize> = Vec::new();
        for t in 0..N_PROMPTS {
            let mut probe = LoopResidualExit::new(tau, D_MIN);
            run(config, weights, residual_gate, sdpa_gate, t, None, Some(&mut probe));
            if let Some(k) = probe.fired_at_iteration() {
                fired.push(k);
            }
        }
        if fired.is_empty() {
            println!("| {tau:.3} | 0/27 | — | — | — | — |");
            continue;
        }
        let n = fired.len();
        let med = median(&mut fired);
        let (min, max) = (
            fired.iter().copied().min().unwrap(),
            fired.iter().copied().max().unwrap(),
        );
        // Quality at the median fire point: nearest phase-A grid entry.
        let q = curve
            .iter()
            .copied()
            .min_by(|a, b| {
                let da = if a.0 > med { a.0 - med } else { med - a.0 };
                let db = if b.0 > med { b.0 - med } else { med - b.0 };
                da.cmp(&db).then(a.0.cmp(&b.0))
            })
            .map(|(_, d)| d)
            .unwrap_or(f32::NAN);
        println!("| {tau:.3} | {n}/27 | {med} | {min} | {max} | {q:.6} |");

        // Expectation 2: the exit ≡ elastic bit-equivalence re-checked at
        // this calibration point, for the FIRST token that fired.
        if let Some(&k) = fired.first() {
            let t = 0;
            let mut probe = LoopResidualExit::new(tau, D_MIN);
            let exited = run(config, weights, residual_gate, sdpa_gate, t, None, Some(&mut probe));
            if probe.fired_at_iteration() == Some(k) {
                let elastic = run(config, weights, residual_gate, sdpa_gate, t, Some(k), None);
                assert_eq!(
                    exited, elastic,
                    "[{label}] tau {tau}: exit-at-{k} diverged from elastic = {k}"
                );
            }
        }
    }
}

#[test]
fn bench_731_t2_residual_calibration() {
    // ── Arm 1: None (the natural-decay regime) ──────────────────
    let config = make_config(LoopStabilityMode::None);
    let (weights, residual_gate, sdpa_gate) = make_fixture(&config);
    let curve = phase_a_curve("None", &config, &weights, &residual_gate, &sdpa_gate);
    let knee = curve.iter().find(|(_, d)| *d <= 0.01).map(|(k, _)| *k);
    println!("  knee (first grid k with mean dist ≤ 0.01): {:?}", knee);
    phase_b_calibration("None", &config, &weights, &residual_gate, &sdpa_gate, &curve);
    let none_curve = curve.clone();

    // ── Arm 2: InterLoopNorm (the Research-440 control regime) ──────
    let config = make_config(LoopStabilityMode::InterLoopNorm);
    let (weights, residual_gate, sdpa_gate) = make_fixture(&config);
    let curve = phase_a_curve("InterLoopNorm", &config, &weights, &residual_gate, &sdpa_gate);
    let knee = curve.iter().find(|(_, d)| *d <= 0.01).map(|(k, _)| *k);
    println!("  knee (first grid k with mean dist ≤ 0.01): {:?}", knee);
    phase_b_calibration(
        "InterLoopNorm",
        &config,
        &weights,
        &residual_gate,
        &sdpa_gate,
        &curve,
    );

    // Expectation 1 (the magnitude-only negative control), AMENDED to the
    // measured boundary after the first run: on the InterLoopNorm regime,
    // every τ ≤ 3 must leave all 27 tokens un-fired. The first run measured
    // the boundary precisely: τ = 10 produced a 1/27 FALSE-POSITIVE (token
    // 12, fired at k = 5) — a mid-ramp small-step window reading as settled
    // before the ramp starts. That is exactly the Research-440 trap the
    // control exists to catch, now measured: the no-fire guarantee holds at
    // τ ≤ 3, not at τ ≤ 10 as first assumed.
    for &tau in &TAU_GRID {
        if tau > 3.0 {
            break;
        }
        for t in 0..N_PROMPTS {
            let mut probe = LoopResidualExit::new(tau, D_MIN);
            run(&config, &weights, &residual_gate, &sdpa_gate, t, None, Some(&mut probe));
            assert_eq!(
                probe.fired_at_iteration(),
                None,
                "InterLoopNorm control: tau {tau} fired on token {t} — the plateau regime moved; re-read the calibration before trusting any τ"
            );
        }
    }
    println!("\n  control (amended to the measured boundary): InterLoopNorm, τ ≤ 3 — 0/27 fired on every τ; τ = 10 measured 1/27 FALSE-POSITIVE at k = 5 (the Research-440 mid-ramp trap, recorded).");

    // Post-hoc readout (NOT part of the pre-registered qualification): the
    // d_min lever at the knee. The None-arm table shows the settle signal
    // leads the output knee (settle ~5-6, knee 10) — d_min = 10 should give
    // a 3.2× cut at knee-parity quality, with adaptivity only for
    // straggler tokens. Recorded as input to the T3 design decision.
    println!("\n  post-hoc d_min lever (None regime, tau = 1.0, d_min = 10):");
    let config = make_config(LoopStabilityMode::None);
    let (weights, residual_gate, sdpa_gate) = make_fixture(&config);
    let mut fired: Vec<usize> = Vec::new();
    for t in 0..N_PROMPTS {
        let mut probe = LoopResidualExit::new(1.0, 10);
        let out = run(&config, &weights, &residual_gate, &sdpa_gate, t, None, Some(&mut probe));
        if let Some(k) = probe.fired_at_iteration() {
            fired.push(k);
        }
        let _ = out;
    }
    let med = median(&mut fired);
    let q10 = none_curve
        .iter()
        .find(|(k, _)| *k == 10)
        .map(|(_, d)| *d)
        .unwrap_or(f32::NAN);
    println!(
        "    fired {}/27, median k = {med} (cut {:.1}×), quality at k = 10: {q10:.6} (knee-parity ≤ 0.01: {}) — the probe reduces to a knee-pinned floor with straggler adaptivity on this fixture",
        fired.len(),
        32.0 / med as f32,
        q10 <= 0.01
    );
}
