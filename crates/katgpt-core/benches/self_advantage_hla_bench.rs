//! Benchmark: Self-Advantage Gate on HLA Reconstruction (Plan 283 T5.1.3).
//!
//! Measures the GOAT gate criteria for wiring the advantage-margin gate
//! (arxiv:2511.16886 Eq. 18) into the HLA reconstruction loop as the 4th
//! early-stop criterion.
//!
//! # GOAT Gate (T5.1.4)
//!
//! | Gate | Criterion | Target |
//! |------|-----------|--------|
//! | G1 | Mean steps saved (baseline / gated) | ≥ 1.5× |
//! | G2 | Final-activations argmax match | ≥ 99% |
//! | G3 | Per-step latency overhead | < 100ns (vocab=6, sub-µs already) |
//!
//! If G1 + G2 pass → promote `advantage_margin_threshold` default from
//! `NaN` (disabled) to `0.01` (enabled).
//!
//! # Method
//!
//! No saved real reconstruction traces exist. We generate N diverse synthetic
//! brain configurations (varying HLA initial states + module confidences) and
//! replay each with gate disabled (baseline) vs enabled (threshold 0.01).
//! The argmax of the final activations is the quality metric — if the gate
//! halts early but the argmax matches the baseline, the halt was safe.
//!
//! ```bash
//! cargo run --release --bench self_advantage_hla_bench --features self_advantage_gate,sense_composition
//! ```

#![cfg(feature = "self_advantage_gate")]
#![cfg(feature = "sense_composition")]

use katgpt_core::sense::brain::NpcBrain;
use katgpt_core::sense::octree::{KgEmbedding, SenseOctreeBuilder};
use katgpt_core::sense::reconstruction::{
    ReconstructionConfig, ReconstructionState,
};
use katgpt_core::types::SenseKind;

const THRESHOLD: f32 = 0.01;

// Diverse HLA seeds — 8-dim, values in [-1, 1].
// Generated once to make the benchmark deterministic.
const HLA_SEEDS: [[f32; 8]; 10] = [
    [0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8],
    [0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4],
    [-0.5, 0.6, -0.3, 0.8, 0.1, -0.7, 0.4, 0.9],
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    [0.8, -0.6, 0.4, -0.2, 0.9, 0.3, -0.5, 0.7],
    [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
    [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2],
    [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.7, 0.3, 0.6, 0.4, 0.8, 0.2, 0.9, 0.1],
];

fn make_brain_with_6_modules(confidence_scale: f32) -> NpcBrain {
    let builder = SenseOctreeBuilder::new(3);
    let kinds = [
        SenseKind::CommonSense,
        SenseKind::FighterSense,
        SenseKind::GameTheorySense,
        SenseKind::SpatialSense,
        SenseKind::SocialSense,
        SenseKind::SkillSense,
    ];
    let modules: Vec<_> = kinds
        .iter()
        .enumerate()
        .map(|(i, &kind)| {
            let emb = KgEmbedding {
                entity_hash: kind as u64,
                relation_hash: kind as u64,
                embedding: [0.5; 8],
                sign: true,
                confidence: 1.0,
            };
            let mut m = builder.build(kind, &[emb]);
            m.confidence = (0.3 + 0.1 * i as f32) * confidence_scale;
            m.commit();
            m
        })
        .collect();

    NpcBrain::compose(modules)
    // HLA state set per-trace by caller
}

fn argmax6(v: &[f32; 6]) -> usize {
    let mut idx = 0;
    let mut max = v[0];
    for (i, &val) in v.iter().enumerate().skip(1) {
        if val > max {
            max = val;
            idx = i;
        }
    }
    idx
}

struct TraceResult {
    baseline_steps: u8,
    gated_steps: u8,
    baseline_argmax: usize,
    gated_argmax: usize,
}

fn run_trace(brain: &NpcBrain, hla: [f32; 8], baseline_config: ReconstructionConfig, gated_config: ReconstructionConfig) -> TraceResult {
    let mut state_b = ReconstructionState::with_config(hla, baseline_config);
    let baseline_acts = state_b.reconstruct(brain);

    let mut state_g = ReconstructionState::with_config(hla, gated_config);
    let gated_acts = state_g.reconstruct(brain);

    TraceResult {
        baseline_steps: state_b.step(),
        gated_steps: state_g.step(),
        baseline_argmax: argmax6(&baseline_acts),
        gated_argmax: argmax6(&gated_acts),
    }
}

fn bench_goat_gate() {
    // 10 HLA seeds × ~100 confidence variations = 1000 traces.
    // Use deterministic confidence scaling to avoid RNG.
    let confidence_scales: Vec<f32> = (0..100).map(|i| 0.5 + 0.01 * i as f32).collect();

    let baseline_config = ReconstructionConfig {
        max_steps: 5, // give room for the gate to save steps
        advantage_margin_threshold: f32::NAN, // disabled
        ..Default::default()
    };
    let gated_config = ReconstructionConfig {
        max_steps: 5,
        advantage_margin_threshold: THRESHOLD,
        ..Default::default()
    };

    let mut total_baseline_steps = 0u64;
    let mut total_gated_steps = 0u64;
    let mut argmax_matches = 0usize;
    let mut total_traces = 0usize;

    for &hla in &HLA_SEEDS {
        for &cs in &confidence_scales {
            let brain = make_brain_with_6_modules(cs);
            let r = run_trace(&brain, hla, baseline_config, gated_config);
            total_baseline_steps += r.baseline_steps as u64;
            total_gated_steps += r.gated_steps as u64;
            if r.baseline_argmax == r.gated_argmax {
                argmax_matches += 1;
            }
            total_traces += 1;
        }
    }

    let mean_baseline = total_baseline_steps as f64 / total_traces as f64;
    let mean_gated = total_gated_steps as f64 / total_traces as f64;
    let speedup = if mean_gated > 0.0 {
        mean_baseline / mean_gated
    } else {
        f64::INFINITY
    };
    let argmax_match_rate = argmax_matches as f64 / total_traces as f64;

    println!("── Plan 283 T5.1.3: Self-Advantage Gate on HLA Reconstruction ──");
    println!("Traces: {total_traces}, threshold: {THRESHOLD}, max_steps: 5");
    println!();
    println!("{:<35} {:>10.4}", "Mean baseline steps:", mean_baseline);
    println!("{:<35} {:>10.4}", "Mean gated steps:", mean_gated);
    println!();
    println!("── GOAT Gate ───────────────────────────────────────────────────");
    println!(
        "{:<35} {:>10.2}×   {}",
        "G1: Speedup (≥1.5× target):",
        speedup,
        if speedup >= 1.5 { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "{:<35} {:>9.2}%   {}",
        "G2: Argmax match (≥99% target):",
        argmax_match_rate * 100.0,
        if argmax_match_rate >= 0.99 { "✅ PASS" } else { "❌ FAIL" }
    );
}

fn bench_latency_overhead() {
    // G3: per-step latency overhead from the gate check.
    // Measure full reconstruction cycle with vs without gate.
    let brain = make_brain_with_6_modules(1.0);
    let hla = HLA_SEEDS[0];

    let baseline_config = ReconstructionConfig {
        max_steps: 5,
        advantage_margin_threshold: f32::NAN,
        ..Default::default()
    };
    let gated_config = ReconstructionConfig {
        max_steps: 5,
        advantage_margin_threshold: THRESHOLD,
        ..Default::default()
    };

    const ITERS: usize = 10_000;

    // Warmup
    for _ in 0..100 {
        let mut s = ReconstructionState::with_config(hla, baseline_config);
        let _ = s.reconstruct(&brain);
        let mut s = ReconstructionState::with_config(hla, gated_config);
        let _ = s.reconstruct(&brain);
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut s = ReconstructionState::with_config(hla, baseline_config);
        let _ = s.reconstruct(&brain);
        std::hint::black_box(&s);
    }
    let baseline_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut s = ReconstructionState::with_config(hla, gated_config);
        let _ = s.reconstruct(&brain);
        std::hint::black_box(&s);
    }
    let gated_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

    let overhead_ns = (gated_ns - baseline_ns).max(0.0);

    println!();
    println!("── G3: Latency ────────────────────────────────────────────────");
    println!("{:<35} {:>8.1} ns", "Baseline reconstruct cycle:", baseline_ns);
    println!("{:<35} {:>8.1} ns", "Gated reconstruct cycle:", gated_ns);
    println!("{:<35} {:>8.1} ns", "Overhead (per cycle):", overhead_ns);
    println!(
        "{:<35} {:>8.1} ns   {}",
        "G3: Overhead (<100ns target):",
        overhead_ns,
        if overhead_ns < 100.0 { "✅ PASS" } else { "❌ FAIL" }
    );
}

fn main() {
    bench_goat_gate();
    bench_latency_overhead();
}
