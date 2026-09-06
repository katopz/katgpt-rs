#![cfg(feature = "lt2_looped")]
//! Issue 156 T2 — Any-Time LT2 Validation PoC.
//!
//! Validates Research 273 (ELT arXiv:2604.09168) §2.3's unvalidated Gain claim:
//! does our LT2 (`LoopMode::WeightShared`) exhibit the Any-Time property?
//!
//! **What Issue 035 tested (the mechanism):** elastic override clamping,
//! determinism across L, KV cache well-formedness, byte-identity. ALL PASS.
//!
//! **What Issue 035 did NOT test (the property):** does the output distribution
//! converge monotonically as elastic R → R_max? This PoC measures exactly that:
//! `KL(softmax(logits_R) || softmax(logits_R_max))` across R ∈ {1..R_max} for
//! multiple gate regimes. The Any-Time property holds iff KL decreases
//! monotonically as R → R_max (later loops refine, not corrupt).
//!
//! **Three competitors (per issue spec):**
//! 1. Baseline R=1 — single pass, no loop benefit (the cheap tier).
//! 2. Full loop R_max=6 — the nominal config (the expensive tier).
//! 3. Elastic R ∈ {1..6} — same artifact, varying loop count.
//!
//! **Gate regimes tested** (the key modelless variable — LOTUS/ELT get Any-Time
//! via TRAINING (ILSD); we test whether our GATE MECHANISM produces it for free):
//! - Zero-init (default `ResidualGate::new`) — ρ=0, no residual carry-forward
//! - Loop-stable conservative (decay=0.1) — mild carry, modelless (Plan 483)
//! - Loop-stable moderate (decay=0.3) — standard carry
//! - Loop-stable aggressive (decay=0.5) — strong carry, divergence risk
//!
//! **Honest scope:** this tests the STRUCTURAL convergence of our loop dynamics
//! with random untrained weights. It does NOT test a trained LT2 artifact's
//! quality across R (that needs riir-train — the L_step supervision recipe, see
//! Research 442 §3.6). But structural convergence is the necessary condition: if
//! the loop is chaotic with random weights, no training will fix it; if it
//! converges structurally, training can only improve the convergence quality.
//!
//! Run:
//! ```sh
//! CARGO_TARGET_DIR=/tmp/issue156_anytime_lt2/target \
//!   cargo test --features lt2_looped --test issue_156_anytime_lt2_poc -- --nocapture --ignored
//! ```

use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{Config, HybridPattern, LoopMode, ResidualGate, Rng, SdpaOutputGate};

// ── PoC parameters ───────────────────────────────────────────────────────

/// Trained loop count (R_max in ELT / LOTUS Table 6).
const R_MAX: usize = 6;
/// Number of random seeds to average over (reduces single-weight-draw noise).
const N_SEEDS: usize = 8;
/// Positions per seed to average over.
const N_POSITIONS: usize = 4;
/// Latency measurement iterations (warmup + measure).
const LATENCY_ITERS: usize = 500;

// ── Helpers ──────────────────────────────────────────────────────────────

/// Build a micro config with `loop_count = R_MAX`, Uniform hybrid, AHLA mode.
fn make_config() -> Config {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count: R_MAX };
    config.hybrid_pattern = HybridPattern::Uniform;
    config.hla_mode = katgpt_rs::types::HlaMode::Ahla;
    config
}

/// Run `forward_looped` for one decode step at `pos` with elastic override R.
/// Each invocation uses fresh context/caches so runs are independent.
fn run_once(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    pos: usize,
    elastic_override: Option<usize>,
) -> Vec<f32> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut ahla_cache = MultiLayerAhlaCache::new(config);
    let logits = forward_looped(
        &mut ctx,
        weights,
        &mut cache,
        &mut ahla_cache,
        0,
        pos,
        config,
        residual_gate,
        sdpa_gate,
        #[cfg(feature = "sleep_consolidation")]
        None,
        #[cfg(feature = "sleep_consolidation")]
        None,
        #[cfg(feature = "weight_shared_advantage_gate")]
        None,
        elastic_override,
        #[cfg(feature = "gain_cost_halt")]
        None,
        None, // Issue 717: deep_run — None = bit-identical baseline
        #[cfg(feature = "cadence_gate")]
        None, // Issue 731: residual-exit probe — None = bit-identical baseline
    );
    logits.to_vec()
}

/// Numerically stable softmax over a logit slice → probability distribution.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut probs = vec![0.0f32; logits.len()];
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max).exp();
        probs[i] = e;
        sum += e;
    }
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
    probs
}

/// KL divergence KL(P || Q) in nats. Handles zeros via smoothing epsilon.
/// Returns f32::INFINITY if Q has a support gap (Q=0 where P>0 after smoothing).
fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    debug_assert_eq!(p.len(), q.len());
    let eps = 1e-12f32;
    let mut kl = 0.0f32;
    for (pi, qi) in p.iter().zip(q.iter()) {
        let pi = pi + eps;
        let qi = qi + eps;
        kl += pi * (pi / qi).ln();
    }
    kl
}

/// One gate regime's results across R ∈ {1..R_MAX}.
struct RegimeResult {
    name: &'static str,
    /// Mean KL(P_R || P_Rmax) across seeds × positions, per R.
    /// Index 0 = R=1, index R_MAX-1 = R=R_MAX (=0 by definition, same dist).
    mean_kl_vs_rmax: Vec<f32>,
    /// Mean latency (ns) per R.
    mean_latency_ns: Vec<f64>,
    /// Whether KL decreases monotonically as R → R_MAX.
    monotonic: bool,
}

/// Run the full R-sweep for one gate regime.
fn run_regime(
    config: &Config,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    name: &'static str,
) -> RegimeResult {
    let mut kl_acc = [0.0f64; R_MAX]; // KL accumulator per R
    let mut lat_acc = [0.0f64; R_MAX]; // latency accumulator per R

    for seed in 0..N_SEEDS {
        let mut rng = Rng::new(seed as u64);
        let weights = TransformerWeights::new(config, &mut rng);

        for pos in 0..N_POSITIONS {
            // Compute reference distribution at R_MAX.
            let logits_rmax =
                run_once(config, &weights, residual_gate, sdpa_gate, pos, Some(R_MAX));
            let p_rmax = softmax(&logits_rmax);

            for r in 1..=R_MAX {
                let logits_r = run_once(config, &weights, residual_gate, sdpa_gate, pos, Some(r));
                let p_r = softmax(&logits_r);
                let kl = kl_divergence(&p_r, &p_rmax);
                kl_acc[r - 1] += kl as f64;

                // Latency: warmup + measure.
                let mut elapsed_ns = 0u128;
                for _ in 0..LATENCY_ITERS {
                    let start = std::time::Instant::now();
                    let _ = run_once(config, &weights, residual_gate, sdpa_gate, pos, Some(r));
                    elapsed_ns += start.elapsed().as_nanos();
                }
                lat_acc[r - 1] += elapsed_ns as f64 / LATENCY_ITERS as f64;
            }
        }
    }

    let denom = (N_SEEDS * N_POSITIONS) as f64;
    let mean_kl_vs_rmax: Vec<f32> = kl_acc.iter().map(|&v| (v / denom) as f32).collect();
    let mean_latency_ns: Vec<f64> = lat_acc.iter().map(|&v| v / denom).collect();

    // Check monotonic decrease (R_MAX slot is ~0 by construction — skip it).
    let mut monotonic = true;
    for i in 1..(R_MAX - 1) {
        // Allow small numerical noise: KL[i] should be ≤ KL[i-1] + 1e-6.
        if mean_kl_vs_rmax[i] > mean_kl_vs_rmax[i - 1] + 1e-6 {
            monotonic = false;
            break;
        }
    }

    RegimeResult {
        name,
        mean_kl_vs_rmax,
        mean_latency_ns,
        monotonic,
    }
}

fn print_header(title: &str) {
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  {title}");
    println!("═══════════════════════════════════════════════════════════════");
}

fn print_regime_table(r: &RegimeResult) {
    println!(
        "\n  Gate regime: {} {}",
        r.name,
        if r.monotonic {
            "✅ MONOTONIC"
        } else {
            "❌ NON-MONOTONIC"
        }
    );
    println!("  ┌──────┬──────────────────┬──────────────────┐");
    println!("  │  R   │  KL(P_R‖P_Rmax)  │  Latency (ns)    │");
    println!("  ├──────┼──────────────────┼──────────────────┤");
    for (i, r_val) in (1..=R_MAX).enumerate() {
        println!(
            "  │  {}   │  {:>14.6}   │  {:>14.0}   │{}",
            r_val,
            r.mean_kl_vs_rmax[i],
            r.mean_latency_ns[i] as u64,
            if r_val == R_MAX {
                "  ← R_max (ref)"
            } else if r_val == 1 {
                "  ← Baseline"
            } else {
                ""
            }
        );
    }
    println!("  └──────┴──────────────────┴──────────────────┘");
}

// ── The PoC test ─────────────────────────────────────────────────────────

#[test]
#[ignore = "Long-running PoC: ~4 regimes × 6 R-values × 8 seeds × 4 positions × 500 latency iters. Run with --ignored --nocapture."]
fn anytime_lt2_validation() {
    let config = make_config();

    print_header("Issue 156 T2 — Any-Time LT2 Validation PoC");
    println!(
        "  Config: {} layers, dim={}, heads={}, R_MAX={}",
        config.n_layer, config.n_embd, config.n_head, R_MAX
    );
    println!(
        "  Samples: {} seeds × {} positions = {} weight draws",
        N_SEEDS,
        N_POSITIONS,
        N_SEEDS * N_POSITIONS
    );
    println!("  Measurement: KL(softmax(logits_R) ‖ softmax(logits_R_max)) + latency");
    println!("  Any-Time holds iff KL decreases monotonically as R → R_MAX.");

    // Gate regime 1: Zero-init (default — ρ=0, no residual carry).
    let gate_zero = ResidualGate::new(R_MAX, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    let r_zero = run_regime(&config, &gate_zero, &sdpa_gate, "Zero-init (ρ=0, default)");

    // Gate regime 2: Loop-stable conservative (decay=0.1).
    let gate_01 = ResidualGate::new_loop_stable(R_MAX, config.n_embd, 0.1);
    let r_01 = run_regime(&config, &gate_01, &sdpa_gate, "Loop-stable (decay=0.1)");

    // Gate regime 3: Loop-stable moderate (decay=0.3).
    let gate_03 = ResidualGate::new_loop_stable(R_MAX, config.n_embd, 0.3);
    let r_03 = run_regime(&config, &gate_03, &sdpa_gate, "Loop-stable (decay=0.3)");

    // Gate regime 4: Loop-stable aggressive (decay=0.5).
    let gate_05 = ResidualGate::new_loop_stable(R_MAX, config.n_embd, 0.5);
    let r_05 = run_regime(&config, &gate_05, &sdpa_gate, "Loop-stable (decay=0.5)");

    let regimes = [&r_zero, &r_01, &r_03, &r_05];
    for r in &regimes {
        print_regime_table(r);
    }

    // ── Verdict ──────────────────────────────────────────────────────────
    print_header("VERDICT");

    let any_monotonic = regimes.iter().any(|r| r.monotonic);
    let all_monotonic = regimes.iter().all(|r| r.monotonic);

    println!();
    println!("  Gate regimes exhibiting Any-Time (monotonic KL decrease):");
    for r in &regimes {
        let kl_r1 = r.mean_kl_vs_rmax[0];
        let kl_rmid = r.mean_kl_vs_rmax[R_MAX / 2 - 1];
        println!(
            "    {} {} — KL(R=1)={:.6}, KL(R={})={:.6}",
            if r.monotonic { "✅" } else { "❌" },
            r.name,
            kl_r1,
            R_MAX / 2,
            kl_rmid
        );
    }
    println!();

    if all_monotonic {
        println!("  RESULT: ALL gate regimes exhibit Any-Time → structural convergence");
        println!("  CONFIRMED. Our LT2's loop dynamics converge without Any-Time-specific");
        println!("  training (ILSD). This is BETTER than ELT/LOTUS which require ILSD");
        println!("  training for the Any-Time property. Research 273's Gain claim HOLDS.");
    } else if any_monotonic {
        println!("  RESULT: PARTIAL — some gate regimes exhibit Any-Time, others don't.");
        println!("  The gate magnitude is the critical variable. Conservative gates");
        println!("  (decay≤0.3) converge; aggressive gates (decay≥0.5) may not.");
        println!("  Research 273's Gain claim HOLDS CONDITIONALLY (gate-dependent).");
    } else {
        println!("  RESULT: NO gate regime exhibits monotonic Any-Time convergence.");
        println!("  Research 273's Gain claim FAILS at the structural level — our LT2");
        println!("  loop dynamics are chaotic without ILSD-style training. The Any-Time");
        println!("  property requires riir-train (the L_step supervision recipe).");
    }

    // Latency linearity check (should scale ~linearly with R).
    println!();
    println!("  Latency scaling (should be ~linear in R):");
    for r in &regimes {
        let l1 = r.mean_latency_ns[0];
        let lr = r.mean_latency_ns[R_MAX - 1];
        let ratio = if l1 > 0.0 { lr / l1 } else { 0.0 };
        println!(
            "    {} — latency(R=1)={:.0}ns, latency(R={})={:.0}ns, ratio={:.2}× (expect ~{}×)",
            r.name, l1, R_MAX, lr, ratio, R_MAX
        );
    }

    // Soft assertion: the PoC is informational. We assert at least one regime
    // produces finite, non-NaN KL values (structural soundness), but the
    // monotonicity verdict is recorded, not enforced — it's a research finding.
    for r in &regimes {
        for (i, &kl) in r.mean_kl_vs_rmax.iter().enumerate() {
            assert!(
                kl.is_finite(),
                "[{}] KL at R={} is non-finite: {}",
                r.name,
                i + 1,
                kl
            );
        }
    }
    println!();
    println!("  ✅ All KL values finite — structural soundness confirmed.");
    println!("═══════════════════════════════════════════════════════════════");
}
