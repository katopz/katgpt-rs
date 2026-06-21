//! Criterion latency benchmark for the Personality-Weighted Layer Composition
//! kernel (Plan 297 Phase 4 T4.3 — GOAT gate G4 / G5).
//!
//! # GOAT gates covered
//!
//! - **G4** — `compose_n9_d32` (the production Entity Cognition Stack case)
//!   must be < 1µs per entity. This is the plasma-tier target.
//! - **G5** — zero heap allocation in the compose hot path (verified by
//!   code audit + the absence of any `Vec` / `Box` in `compose_into`; a
//!   future `dhat` run can confirm empirically).
//! - **G1 (supporting)** — `compose_tau_infinity_uniform` smoke test (the
//!   uniform baseline; not a perf gate, included for completeness).
//! - `compose_n9_d32_batch_10k` — crowd-scale case: 10K entities per tick.
//!   Target: serial < 10ms (rayon breakeven is ~5µs/entity, so serial wins
//!   by a wide margin at <1µs/entity).
//! - `drift_n9_d32` — drift update cost (expected comparable to compose).
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench personality_composition_bench --features personality_composition --release
//! ```
//!
//! # Feature gate
//!
//! Requires `personality_composition` (Plan 297 Phase 1).

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::personality_composition::{
    LayerDirectionSource, PersonalityConfig, PersonalityWeightedComposition,
};

/// The production Entity Cognition Stack shape: N=9 layers, D=32 latent dims
/// (matches the HLA belief-vector dimension from Research 242 / Research 146).
const N: usize = 9;
const D: usize = 32;

/// Crowd-scale batch size (matches the riir-ai NPC scaling budget).
const BATCH_10K: usize = 10_000;

// ─── Test layer: zero-allocation static direction ─────────────────────────

/// A minimal `LayerDirectionSource` that returns a precomputed direction.
///
/// In a real game, the host computes these per-tick from KG lookups / faction
/// state. Here we precompute them so the benchmark measures ONLY the
/// composition kernel, not the layer computation.
#[derive(Debug)]
struct BenchLayer {
    direction: [f32; D],
    confidence: f32,
}

impl LayerDirectionSource for BenchLayer {
    #[inline]
    fn direction<'a>(&self, scratch: &'a mut [f32]) -> &'a [f32] {
        scratch[..D].copy_from_slice(&self.direction);
        &scratch[..D]
    }

    fn recent_direction(&self) -> &[f32] {
        &self.direction
    }

    #[inline]
    fn belief_confidence(&self) -> f32 {
        self.confidence
    }
}

fn make_layers() -> [BenchLayer; N] {
    let mut layers = Vec::with_capacity(N);
    for i in 0..N {
        let mut dir = [0.0f32; D];
        // Deterministic non-trivial direction: each layer claims a different
        // 4-element block of the 32-dim vector.
        let block = (i % 8) * 4;
        for j in 0..4 {
            dir[block + j] = 0.25 * ((j + 1) as f32);
        }
        // Spread the rest.
        dir[16 + i.min(15)] = 0.1;
        layers.push(BenchLayer {
            direction: dir,
            confidence: 1.0,
        });
    }
    layers.try_into().expect("N layers")
}

// ─── G4: compose_n9_d32 — per-entity compose latency ─────────────────────

fn bench_g4_compose_n9_d32(c: &mut Criterion) {
    let mut group = c.benchmark_group("personality_composition/g4");
    group.sample_size(1000);

    let kernel = PersonalityWeightedComposition::<N, D>::new(
        PersonalityConfig::default(),
        // Mix of weights so the kernel doesn't take an early-out.
        [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9],
    );
    let layers_storage = make_layers();
    let layers: [&dyn LayerDirectionSource; N] = [
        &layers_storage[0],
        &layers_storage[1],
        &layers_storage[2],
        &layers_storage[3],
        &layers_storage[4],
        &layers_storage[5],
        &layers_storage[6],
        &layers_storage[7],
        &layers_storage[8],
    ];
    let mut scratch = [0.0f32; D];
    let mut out = [0.0f32; D];

    group.bench_function("compose_n9_d32", |b| {
        b.iter(|| {
            kernel.compose_into(
                black_box(&layers),
                black_box(&mut scratch),
                black_box(&mut out),
            );
        });
    });

    group.finish();
}

// ─── Crowd-scale: 10K entities per tick ──────────────────────────────────

fn bench_compose_n9_d32_batch_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group("personality_composition/g4");
    group.sample_size(50); // fewer samples — each iter is 10K composes

    // One kernel per entity (they have independent personalities).
    let kernels: Vec<PersonalityWeightedComposition<N, D>> = (0..BATCH_10K)
        .map(|i| {
            let seed = i as f32 * 0.001;
            PersonalityWeightedComposition::new(
                PersonalityConfig::default(),
                [seed, -seed, 0.5, -0.3, 0.7, -0.1, 0.2, -0.6, 0.4],
            )
        })
        .collect();
    let layers_storage = make_layers();
    let layers: [&dyn LayerDirectionSource; N] = [
        &layers_storage[0],
        &layers_storage[1],
        &layers_storage[2],
        &layers_storage[3],
        &layers_storage[4],
        &layers_storage[5],
        &layers_storage[6],
        &layers_storage[7],
        &layers_storage[8],
    ];

    // Per-entity scratch + out, kept across iterations.
    let mut scratch_buf = vec![0.0f32; D * BATCH_10K];
    let mut out_buf = vec![0.0f32; D * BATCH_10K];

    group.bench_function("compose_n9_d32_batch_10k", |b| {
        b.iter(|| {
            for i in 0..BATCH_10K {
                let s = &mut scratch_buf[i * D..(i + 1) * D];
                let o = &mut out_buf[i * D..(i + 1) * D];
                kernels[i].compose_into(black_box(&layers), s, o);
            }
        });
    });

    group.finish();
}

// ─── Drift cost ──────────────────────────────────────────────────────────

fn bench_drift_n9_d32(c: &mut Criterion) {
    let mut group = c.benchmark_group("personality_composition/g4");
    group.sample_size(1000);

    let mut kernel = PersonalityWeightedComposition::<N, D>::new(
        PersonalityConfig::default(),
        [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9],
    );
    let layers_storage = make_layers();
    let layers: [&dyn LayerDirectionSource; N] = [
        &layers_storage[0],
        &layers_storage[1],
        &layers_storage[2],
        &layers_storage[3],
        &layers_storage[4],
        &layers_storage[5],
        &layers_storage[6],
        &layers_storage[7],
        &layers_storage[8],
    ];

    group.bench_function("drift_n9_d32", |b| {
        b.iter(|| {
            kernel.drift(black_box(&layers), black_box(0.5));
        });
    });

    group.finish();
}

// ─── G1 supporting: τ=∞ uniform baseline ─────────────────────────────────

fn bench_g1_compose_tau_infinity(c: &mut Criterion) {
    let mut group = c.benchmark_group("personality_composition/g1");
    group.sample_size(1000);

    let config = PersonalityConfig {
        tau: f32::INFINITY,
        ..Default::default()
    };
    let kernel = PersonalityWeightedComposition::<N, D>::new(config, [10.0; N]);
    let layers_storage = make_layers();
    let layers: [&dyn LayerDirectionSource; N] = [
        &layers_storage[0],
        &layers_storage[1],
        &layers_storage[2],
        &layers_storage[3],
        &layers_storage[4],
        &layers_storage[5],
        &layers_storage[6],
        &layers_storage[7],
        &layers_storage[8],
    ];
    let mut scratch = [0.0f32; D];
    let mut out = [0.0f32; D];

    group.bench_function("compose_tau_infinity_uniform", |b| {
        b.iter(|| {
            kernel.compose_into(
                black_box(&layers),
                black_box(&mut scratch),
                black_box(&mut out),
            );
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_g4_compose_n9_d32,
    bench_compose_n9_d32_batch_10k,
    bench_drift_n9_d32,
    bench_g1_compose_tau_infinity,
);
criterion_main!(benches);
