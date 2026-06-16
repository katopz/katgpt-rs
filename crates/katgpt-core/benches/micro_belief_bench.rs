//! MicroRecurrentBeliefState latency benchmark (Plan 276 T1.14).
//!
//! Canonical criterion wiring for the G1.4 latency gate and friends. Replaces
//! the wall-clock `g1_4_attractor_step_32_under_100ns` test as the source of
//! truth for the per-step ns number cited in
//! `katgpt-rs/.issues/024_micro_belief_g1_4_attractor_latency.md` and
//! `katgpt-rs/.benchmarks/276_micro_belief_goat.md`.
//!
//! # GOAT gates covered
//!
//! - **G1.4** — `AttractorKernel::step()` @ dim=32 must be <100 ns/step on
//!   Apple Silicon arm64 release. (Currently FAILs at ~270 ns — attractor is
//!   demoted to opt-in per T5.2; this bench documents the gap, not enforces it.
//!   criterion reports the number; the GOAT decision lives in the plan.)
//! - **LeakyIntegrator baseline** — same dim=32, expected <30 ns/step. This is
//!   the promotable Family C output.
//! - **LatentThoughtKernel** — K=1 must match attractor ±5%; K=3 ~3× attractor.
//! - **project_to_scalars bridge** — K=5 scalars over dim=32, target <50 ns.
//! - **1000-NPC batch** — 1000 leaky kernels at dim=8 (HLA-shaped). Two
//!   variants: serial baseline and rayon `par_iter`. The serial path is the
//!   honest winner because the per-NPC step (~10 ns at dim=8) is ~500× below
//!   rayon's ~5 µs thread-pool breakeven (AGENTS.md: "only parallelize when
//!   per-task work exceeds thread-pool overhead"). The rayon variant is kept
//!   intentionally to document this finding — it is NOT the speedup path at
//!   this work size. Serial target: <15 µs total for 1000 NPCs.
//! - **G3 (BoM)** — `sample_k_states(K=8)` must be ≤ 2× the cost of a single
//!   `step()` call. The batched single-pass design computes the matvec once,
//!   so the K overhead is only the K·dim element-wise noise additions +
//!   sigmoids. Measured for K ∈ {1, 4, 8, 16} (Plan 281 T2.1).
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench micro_belief_bench --features "micro_belief bom_sampling"
//! ```
//!
//! # Feature gate
//!
//! Requires `micro_belief` (Plan 276 Phase 1) + `bom_sampling` (Plan 281).

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::micro_belief::LatentThoughtKernel;
use katgpt_core::{
    AttractorKernel, LeakyIntegrator, MicroRecurrentBeliefState, project_to_scalars,
};
use rayon::prelude::*;
use std::sync::Mutex;

/// Belief-vector dimension used by the G1.4 canonical gate (matches Plan 255 L1
/// budget and the existing `g1_4_attractor_step_32_under_100ns` wall-clock test).
const G1_4_DIM: usize = 32;

/// HLA-shaped dim used by the 1000-NPC batch benchmark (matches
/// `ReconstructionConfig` default of `dim = 8`).
const BATCH_DIM: usize = 8;

/// Number of NPCs in the batch-throughput benchmark (matches the riir-ai
/// NPC scaling budget).
const BATCH_NPCS: usize = 1000;

/// Number of scalar projections used by the bridge benchmark (valence, arousal,
/// desperation, calm, fear — the 5 HLA scalars from AGENTS.md §Latent).
const BRIDGE_K: usize = 5;

// ─── G1.4: per-kernel step latency @ dim=32 ─────────────────────────────────

fn bench_g1_4_attractor_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("micro_belief/g1_4_step");
    group.sample_size(500);

    let kernel = AttractorKernel::from_seed(42, G1_4_DIM);
    let mut state = vec![0.0f32; G1_4_DIM];
    let input = vec![0.5f32; G1_4_DIM];

    group.bench_function("attractor_dim32", |b| {
        b.iter(|| {
            kernel.step(black_box(&mut state), black_box(&input));
        });
    });

    group.finish();
}

fn bench_leaky_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("micro_belief/g1_4_step");
    group.sample_size(500);

    let kernel = LeakyIntegrator::hla_default(G1_4_DIM);
    let mut state = vec![0.0f32; G1_4_DIM];
    let input = vec![0.5f32; G1_4_DIM];

    group.bench_function("leaky_dim32", |b| {
        b.iter(|| {
            kernel.step(black_box(&mut state), black_box(&input));
        });
    });

    group.finish();
}

fn bench_latent_thought_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("micro_belief/g1_4_step");
    group.sample_size(500);

    let mut state = vec![0.0f32; G1_4_DIM];
    let input = vec![0.5f32; G1_4_DIM];

    // K=1 — must match attractor ±5% (G1.6 bit-identical at K=1 means latency
    // should also match; the inner call is the same matvec + sigmoid).
    let k1 = LatentThoughtKernel::from_seed(42, G1_4_DIM, 1);
    group.bench_function("latent_thought_k1_dim32", |b| {
        b.iter(|| {
            k1.step(black_box(&mut state), black_box(&input));
        });
    });

    // K=3 — ~3× the attractor cost (three inner matvec+sigmoid passes).
    let k3 = LatentThoughtKernel::from_seed(42, G1_4_DIM, 3);
    group.bench_function("latent_thought_k3_dim32", |b| {
        b.iter(|| {
            k3.step(black_box(&mut state), black_box(&input));
        });
    });

    group.finish();
}

// ─── project_to_scalars bridge ──────────────────────────────────────────────

fn bench_project_to_scalars(c: &mut Criterion) {
    let mut group = c.benchmark_group("micro_belief/project_to_scalars");
    group.sample_size(500);

    let state = vec![0.5f32; G1_4_DIM];
    // Flattened [K * dim] row-major: 5 directions × 32 dims.
    let directions = vec![0.25f32; BRIDGE_K * G1_4_DIM];
    let mut out = vec![0.0f32; BRIDGE_K];

    group.bench_function("k5_dim32", |b| {
        b.iter(|| {
            project_to_scalars(
                black_box(&state),
                black_box(&directions),
                G1_4_DIM,
                black_box(&mut out),
            );
        });
    });

    group.finish();
}

// ─── 1000-NPC batch throughput (riir-ai NPC scaling story) ──────────────────

fn bench_batch_1000_npcs(c: &mut Criterion) {
    let mut group = c.benchmark_group("micro_belief/batch_1000_npcs");
    group.sample_size(100);

    // HLA-shaped leaky kernels — the promotable Family C output that riir-ai
    // would actually deploy per NPC. Using `Mutex<Vec<...>>` mirrors the
    // temporal_deriv batch bench convention.
    let kernels: Mutex<Vec<LeakyIntegrator>> = Mutex::new(
        (0..BATCH_NPCS)
            .map(|_| LeakyIntegrator::hla_default(BATCH_DIM))
            .collect(),
    );
    // Per-NPC belief vectors (the `par_iter_mut` needs mutable state).
    let states: Mutex<Vec<Vec<f32>>> = Mutex::new(
        (0..BATCH_NPCS)
            .map(|_| vec![0.0f32; BATCH_DIM])
            .collect(),
    );
    let input = vec![0.5f32; BATCH_DIM];

    group.bench_function("leaky_rayon_par_iter_dim8", |b| {
        b.iter(|| {
            let mut kernels = kernels.lock().unwrap();
            let mut states = states.lock().unwrap();
            kernels
                .par_iter_mut()
                .zip(states.par_iter_mut())
                .for_each(|(k, s)| {
                    k.step(black_box(s), black_box(&input));
                });
        });
    });

    // Serial baseline for the speedup reference.
    group.bench_function("leaky_serial_iter_dim8", |b| {
        b.iter(|| {
            let mut kernels = kernels.lock().unwrap();
            let mut states = states.lock().unwrap();
            for (k, s) in kernels.iter_mut().zip(states.iter_mut()) {
                k.step(black_box(s), black_box(&input));
            }
        });
    });

    group.finish();
}

// ─── BoM: sample_k_states latency (Plan 281 T2.1, G3 gate) ───────────────────
//
// Measures sample_k_states(K) for K ∈ {1, 4, 8, 16} on AttractorKernel and
// K=8 on LeakyIntegrator. The G3 gate requires sample_k_states(K=8) ≤ 2× step().
// The single-pass design computes the matvec once, so the K overhead is only
// the K·dim element-wise noise additions + sigmoids.

#[cfg(feature = "bom_sampling")]
fn bench_bom_sample_k_states(c: &mut Criterion) {
    use katgpt_core::{BoMSampler, NoiseQueryConfig};

    let mut group = c.benchmark_group("micro_belief/bom_sample_k_states");
    group.sample_size(500);

    let kernel = AttractorKernel::from_seed(42, G1_4_DIM);
    let s_prev = vec![0.0f32; G1_4_DIM];
    let x = vec![0.5f32; G1_4_DIM];

    for &k in &[1usize, 4, 8, 16] {
        let cfg = NoiseQueryConfig::default().with_k(k).with_sigma(0.3);
        // Fixed pre-allocated queries buffer, filled once with fastrand seed=42.
        let mut rng = fastrand::Rng::with_seed(42);
        let queries: Vec<f32> = (0..k * G1_4_DIM).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let mut out = vec![0.0f32; k * G1_4_DIM];

        let label = format!("attractor_k{k}_dim{G1_4_DIM}");
        group.bench_function(&label, |b| {
            b.iter(|| {
                kernel.sample_k_states(
                    black_box(&s_prev),
                    black_box(&x),
                    black_box(&queries),
                    black_box(&mut out),
                    black_box(&cfg),
                );
            });
        });
    }

    // LeakyIntegrator baseline at K=8.
    {
        let leaky = LeakyIntegrator::hla_default(G1_4_DIM);
        let k = 8usize;
        let cfg = NoiseQueryConfig::default().with_k(k).with_sigma(0.3);
        let mut rng = fastrand::Rng::with_seed(42);
        let queries: Vec<f32> = (0..k * G1_4_DIM).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let mut out = vec![0.0f32; k * G1_4_DIM];

        group.bench_function("leaky_k8_dim32", |b| {
            b.iter(|| {
                leaky.sample_k_states(
                    black_box(&s_prev),
                    black_box(&x),
                    black_box(&queries),
                    black_box(&mut out),
                    black_box(&cfg),
                );
            });
        });
    }

    group.finish();
}

// ─── SIMD sigmoid GOAT gate (Issues 024/025 M1) ───────────────────────────────
//
// Benchmarks the simd_sigmoid feature path for AttractorKernel::step() and
// sample_k_states. Only compiles when `--features simd_sigmoid` is passed.
// Compare against the scalar baseline (run without `simd_sigmoid`) to verify
// the G1.4 < 100ns/step and G3 ≤ 2× targets are met.

#[cfg(feature = "simd_sigmoid")]
fn bench_simd_sigmoid_attractor_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("micro_belief/simd_sigmoid_goat");
    group.sample_size(500);

    let kernel = AttractorKernel::from_seed(42, G1_4_DIM);
    let mut state = vec![0.0f32; G1_4_DIM];
    let input = vec![0.5f32; G1_4_DIM];

    group.bench_function("attractor_step_dim32_simd", |b| {
        b.iter(|| {
            kernel.step(black_box(&mut state), black_box(&input));
        });
    });

    group.finish();
}

#[cfg(all(feature = "simd_sigmoid", feature = "bom_sampling"))]
fn bench_simd_sigmoid_bom_k8(c: &mut Criterion) {
    use katgpt_core::{BoMSampler, NoiseQueryConfig};

    let mut group = c.benchmark_group("micro_belief/simd_sigmoid_goat");
    group.sample_size(500);

    let kernel = AttractorKernel::from_seed(42, G1_4_DIM);
    let s_prev = vec![0.0f32; G1_4_DIM];
    let x = vec![0.5f32; G1_4_DIM];
    let k = 8usize;
    let cfg = NoiseQueryConfig::default().with_k(k).with_sigma(0.3);
    let mut rng = fastrand::Rng::with_seed(42);
    let queries: Vec<f32> = (0..k * G1_4_DIM).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let mut out = vec![0.0f32; k * G1_4_DIM];

    group.bench_function("bom_k8_dim32_simd", |b| {
        b.iter(|| {
            kernel.sample_k_states(
                black_box(&s_prev),
                black_box(&x),
                black_box(&queries),
                black_box(&mut out),
                black_box(&cfg),
            );
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_g1_4_attractor_step,
    bench_leaky_step,
    bench_latent_thought_step,
    bench_project_to_scalars,
    bench_batch_1000_npcs,
);

// `criterion_group!` does not accept `#[cfg]` on its arguments, so feature-
// gated benches live in their own groups and are merged via feature-gated
// `criterion_main!` invocations. The bench's `required-features` already
// includes `bom_sampling`, so both groups are always compiled here — the cfg
// split is defensive for any future change to `required-features`.
#[cfg(feature = "bom_sampling")]
criterion_group!(bom_benches, bench_bom_sample_k_states);

// SIMD sigmoid GOAT gate group — only when `simd_sigmoid` is enabled.
#[cfg(all(feature = "simd_sigmoid", feature = "bom_sampling"))]
criterion_group!(
    simd_sigmoid_benches,
    bench_simd_sigmoid_attractor_step,
    bench_simd_sigmoid_bom_k8
);
#[cfg(all(feature = "simd_sigmoid", not(feature = "bom_sampling")))]
criterion_group!(simd_sigmoid_benches, bench_simd_sigmoid_attractor_step);

#[cfg(all(feature = "bom_sampling", feature = "simd_sigmoid"))]
criterion_main!(benches, bom_benches, simd_sigmoid_benches);

#[cfg(all(feature = "bom_sampling", not(feature = "simd_sigmoid")))]
criterion_main!(benches, bom_benches);

#[cfg(all(not(feature = "bom_sampling"), feature = "simd_sigmoid"))]
criterion_main!(benches, simd_sigmoid_benches);

#[cfg(not(any(feature = "bom_sampling", feature = "simd_sigmoid")))]
criterion_main!(benches);
