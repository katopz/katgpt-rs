//! Benchmark: OctreeCTC Reconstructive Navigation (Plan 248).
//!
//! Measures per-cycle latency for 3-step reconstruction (scalar vs SIMD vs matvec vs batch).
//! GOAT Gate: <200ns per 3-step reconstruction cycle.
//!
//! Measures:
//!   - Scalar `reconstruct()` — baseline
//!   - SIMD `reconstruct_simd()` — SIMD evolve (historically slower at 6×8)
//!   - Matvec `reconstruct_matvec()` — pre-computed weight matrix, single matvec expand
//!   - Multi-entity batch — N NPCs × same brain config, amortized SIMD
//!   - Per-step breakdown: expand → route → accumulate → evolve_hla
//!
//! Key finding: At 6 modules × 8-dim HLA, scalar wins for single entity.
//! SIMD only wins when batched across N ≥ 4 entities (48N f32 ops amortize NEON setup).

use katgpt_core::sense::brain::NpcBrain;
use katgpt_core::sense::octree::{KgEmbedding, SenseOctreeBuilder};
use katgpt_core::sense::reconstruction::{
    BatchProjectionWeights, ProjectionWeights, ReconstructionConfig, ReconstructionState,
};
use katgpt_core::types::SenseKind;

const ITERS: usize = 10_000;

fn make_brain_with_6_modules() -> NpcBrain {
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
            let m = builder.build(kind, &[emb]);
            // Vary confidence per module
            let mut m = m;
            m.confidence = 0.3 + 0.1 * i as f32;
            m.commit();
            m
        })
        .collect();

    let mut brain = NpcBrain::compose(modules);
    brain.hla_state = [0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8];
    brain
}

// ── Full Cycle Benchmarks ────────────────────────────────────────

fn bench_reconstruct_scalar(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct(brain);
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct(brain);
        std::hint::black_box(&state);
    }
    start.elapsed().as_nanos() as f64 / ITERS as f64
}

fn bench_reconstruct_simd(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct_simd(brain);
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct_simd(brain);
        std::hint::black_box(&state);
    }
    start.elapsed().as_nanos() as f64 / ITERS as f64
}

fn bench_reconstruct_matvec(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    // Pre-compute weights ONCE (production path — weights survive across ticks)
    let weights = ProjectionWeights::from_brain(brain);

    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct_with_weights(&weights);
    }

    // Benchmark: state creation is cheap, weights are pre-computed
    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct_with_weights(&weights);
        std::hint::black_box(&state);
    }
    start.elapsed().as_nanos() as f64 / ITERS as f64
}

// ── Per-Step Breakdown ───────────────────────────────────────────

fn bench_step_scalar(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand(brain);
        let selected = state.route(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla();
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand(brain);
        let selected = state.route(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla();
        std::hint::black_box(&state);
    }
    start.elapsed().as_nanos() as f64 / ITERS as f64
}

fn bench_step_matvec(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    let weights = ProjectionWeights::from_brain(brain);

    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand_with_weights(&weights);
        let selected = state.route(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla();
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand_with_weights(&weights);
        let selected = state.route(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla();
        std::hint::black_box(&state);
    }
    start.elapsed().as_nanos() as f64 / ITERS as f64
}

// ── Multi-Entity Batch ───────────────────────────────────────────

/// Benchmark batch expand across N entities.
/// Measures throughput (ns per entity) for different batch sizes.
fn bench_batch_expand(brain: &NpcBrain, n_entities: usize) -> f64 {
    let batch_weights = BatchProjectionWeights::new(brain, n_entities);

    // Prepare N varied HLA states
    let mut hla_batch = vec![0.0f32; n_entities * 8];
    for e in 0..n_entities {
        let off = e * 8;
        let base = 0.1 * (e + 1) as f32;
        hla_batch[off..off + 8].copy_from_slice(&[
            base,
            base + 0.2,
            base + 0.1,
            base + 0.3,
            base + 0.15,
            base + 0.05,
            base + 0.25,
            base + 0.35,
        ]);
    }
    let mut activations_out = vec![0.0f32; n_entities * 6];

    // Warmup
    for _ in 0..100 {
        batch_weights.expand_batch(&hla_batch, &mut activations_out);
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        batch_weights.expand_batch(&hla_batch, &mut activations_out);
        std::hint::black_box(&activations_out);
    }
    let total_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;
    total_ns / n_entities as f64 // per-entity cost
}

/// Benchmark: scalar expand per entity (baseline for batch comparison).
fn bench_scalar_expand_per_entity(brain: &NpcBrain, n_entities: usize) -> f64 {
    let mut hla_batch = vec![0.0f32; n_entities * 8];
    for e in 0..n_entities {
        let off = e * 8;
        let base = 0.1 * (e + 1) as f32;
        hla_batch[off..off + 8].copy_from_slice(&[
            base,
            base + 0.2,
            base + 0.1,
            base + 0.3,
            base + 0.15,
            base + 0.05,
            base + 0.25,
            base + 0.35,
        ]);
    }

    // Warmup
    for _ in 0..100 {
        for e in 0..n_entities {
            let off = e * 8;
            let hla: &[f32; 8] = unsafe { &*(&hla_batch[off] as *const f32 as *const [f32; 8]) };
            for module in &brain.modules {
                let _ = std::hint::black_box(module.project(hla));
            }
        }
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        for e in 0..n_entities {
            let off = e * 8;
            let hla: &[f32; 8] = unsafe { &*(&hla_batch[off] as *const f32 as *const [f32; 8]) };
            for module in &brain.modules {
                let _ = std::hint::black_box(module.project(hla));
            }
        }
    }
    let total_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;
    total_ns / n_entities as f64
}

/// Benchmark: matvec expand per entity (single-entity pre-computed matrix).
fn bench_matvec_expand_per_entity(brain: &NpcBrain, n_entities: usize) -> f64 {
    let weights = ProjectionWeights::from_brain(brain);

    let mut hla_batch = vec![0.0f32; n_entities * 8];
    for e in 0..n_entities {
        let off = e * 8;
        let base = 0.1 * (e + 1) as f32;
        hla_batch[off..off + 8].copy_from_slice(&[
            base,
            base + 0.2,
            base + 0.1,
            base + 0.3,
            base + 0.15,
            base + 0.05,
            base + 0.25,
            base + 0.35,
        ]);
    }

    // Warmup
    for _ in 0..100 {
        for e in 0..n_entities {
            let off = e * 8;
            let mut dots = [0.0f32; 6];
            katgpt_core::simd::simd_matmul_rows(
                &mut dots,
                &weights.matrix,
                &hla_batch[off..off + 8],
                6,
                8,
            );
            std::hint::black_box(dots);
        }
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        for e in 0..n_entities {
            let off = e * 8;
            let mut dots = [0.0f32; 6];
            katgpt_core::simd::simd_matmul_rows(
                &mut dots,
                &weights.matrix,
                &hla_batch[off..off + 8],
                6,
                8,
            );
            std::hint::black_box(dots);
        }
    }
    let total_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;
    total_ns / n_entities as f64
}

// ── Main ─────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 248: OctreeCTC Reconstruction Benchmark ===\n");

    let brain = make_brain_with_6_modules();
    let config = ReconstructionConfig::default(); // 3 steps

    // Report SIMD level
    let level = katgpt_core::simd::simd_level();
    println!("SIMD level: {level:?}");

    println!(
        "Config: max_steps={}, lr={}",
        config.max_steps, config.hla_learning_rate
    );
    println!("Modules: {}", brain.modules.len());
    println!("Iterations: {ITERS}\n");

    // ── Full 3-Step Cycle ──
    println!("=== Full 3-Step Cycle ===");
    let scalar_ns = bench_reconstruct_scalar(&brain, config);
    let simd_ns = bench_reconstruct_simd(&brain, config);
    let matvec_ns = bench_reconstruct_matvec(&brain, config);

    println!("Scalar:            {scalar_ns:>8.1} ns/cycle");
    println!(
        "SIMD (evolve):     {simd_ns:>8.1} ns/cycle  ({:.2}×)",
        scalar_ns / simd_ns
    );
    println!(
        "Matvec (batched):  {matvec_ns:>8.1} ns/cycle  ({:.2}×)",
        scalar_ns / matvec_ns
    );

    // Find GOAT
    let best_ns = scalar_ns.min(simd_ns).min(matvec_ns);
    let goat_pass = best_ns < 200.0;
    println!(
        "\nGOAT (<200ns): {} — best = {best_ns:.1} ns",
        if goat_pass { "PASS ✅" } else { "FAIL ❌" }
    );

    // ── Per-Step Breakdown ──
    println!("\n=== Per-Step Breakdown (expand+route+accumulate+evolve) ===");
    let step_scalar_ns = bench_step_scalar(&brain, config);
    let step_matvec_ns = bench_step_matvec(&brain, config);

    println!("Scalar step:       {step_scalar_ns:>8.1} ns");
    println!(
        "Matvec step:       {step_matvec_ns:>8.1} ns  ({:.2}×)",
        step_scalar_ns / step_matvec_ns
    );

    // ── Multi-Entity Batch Expand ──
    println!("\n=== Multi-Entity Batch Expand (per-entity ns) ===");
    println!(
        "{:>4} {:>12} {:>12} {:>12} {:>8}",
        "N", "scalar", "matvec", "batch", "best"
    );
    println!("{}", "-".repeat(52));

    for &n in &[1, 2, 4, 8, 16, 32] {
        let s_ns = bench_scalar_expand_per_entity(&brain, n);
        let m_ns = bench_matvec_expand_per_entity(&brain, n);
        let b_ns = bench_batch_expand(&brain, n);
        let best = s_ns.min(m_ns).min(b_ns);
        let best_label = if best == s_ns {
            "scalar"
        } else if best == m_ns {
            "matvec"
        } else {
            "batch"
        };
        println!("{n:>4} {s_ns:>10.1} ns {m_ns:>10.1} ns {b_ns:>10.1} ns {best_label:>8}");
    }

    // ── Correctness ──
    println!("\n=== Correctness ===");

    // Matvec matches scalar
    let weights = ProjectionWeights::from_brain(&brain);
    let mut state_scalar = ReconstructionState::with_config(brain.hla_state, config);
    let _ = state_scalar.reconstruct(&brain);

    let mut state_matvec = ReconstructionState::with_config(brain.hla_state, config);
    let _ = state_matvec.reconstruct_with_weights(&weights);

    let mut max_diff = 0.0f32;
    for i in 0..8 {
        let diff = (state_scalar.hla()[i] - state_matvec.hla()[i]).abs();
        max_diff = max_diff.max(diff);
    }
    println!("Max HLA diff (scalar vs matvec): {max_diff:.6e}");
    assert!(
        max_diff < 1e-4,
        "Matvec should match scalar, diff={max_diff}"
    );
    println!("Matvec equivalence: PASS ✅");

    // Batch expand matches scalar
    let batch_weights = BatchProjectionWeights::new(&brain, 4);
    let hla_batch = [
        0.3f32, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8, 0.2f32, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7,
        0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8f32, 0.6, 0.4, 0.2, 0.7, 0.5, 0.3, 0.1,
    ];
    let mut activations_out = [0.0f32; 24];
    batch_weights.expand_batch(&hla_batch, &mut activations_out);

    // Verify entity 0 matches scalar expand
    let mut state_0 =
        ReconstructionState::with_config([0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8], config);
    let scalar_acts = state_0.expand(&brain);
    let mut max_batch_diff = 0.0f32;
    for i in 0..6 {
        let diff = (scalar_acts[i] - activations_out[i]).abs();
        max_batch_diff = max_batch_diff.max(diff);
    }
    println!("Max batch diff (entity 0): {max_batch_diff:.6e}");
    assert!(
        max_batch_diff < 1e-4,
        "Batch expand should match scalar, diff={max_batch_diff}"
    );
    println!("Batch equivalence: PASS ✅");
}
