//! Viable Manifold Graph — GOAT gate benchmarks (Plan 312 Phase 4, T4.9).
//!
//! Three criterion benches covering the latency-critical primitive operations.
//! All three reuse the criterion convention from `engram_micro.rs` /
//! `karc_forecast_bench.rs` (`criterion_group!` + `criterion_main!`).
//!
//! # Benches
//!
//! - **`pullback_volume/R^4→R^4_identity`** — one `jacobian_svd_at` + reduction.
//!   Target: < 5µs. `f(x) = x` (Jacobian = I, no truncation error contribution).
//! - **`manifold_random_walk/k=4_per_step`** — per-step latency of a 1000-step
//!   random walk on a k=4-nearest-neighbor graph. Reported throughput is
//!   `Elements(1000)` so criterion prints the per-walk time; divide by 1000
//!   for per-step. Target: < 100 ns/step.
//! - **`build_safe_manifold_graph/1000_samples_d4`** — full graph build
//!   (filter + kNN connect + dedup) on 1000 4D samples with a paper-style
//!   viability predicate. Target: < 10 ms (dominated by 1000 SVD calls).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features viable_manifold_graph \
//!   --bench viable_manifold_graph_bench -- --warm-up-time 1 --measurement-time 2 --sample-size 10
//! ```
//!
//! # Feature gate
//!
//! Requires `viable_manifold_graph` (Plan 312).

#![cfg(feature = "viable_manifold_graph")]

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use katgpt_core::viable_manifold_graph::{
    ClosurePredicate, GraphBuildConfig, SafeManifoldGraph, VolumeFieldConfig,
    build_safe_manifold_graph, manifold_random_walk, pullback_volume,
};
use katgpt_core::JacobianSvdScratch;
use fastrand::Rng;

const DIM: usize = 4;

fn f_identity(z: &[f32], out: &mut [f32]) {
    out.copy_from_slice(z);
}

// ─── G-bench 1: pullback_volume on R^4 → R^4 ────────────────────────────────
//
// Target: < 5µs (one SVD call on a 4×4 Jacobian + sum-of-logs reduction).

fn bench_pullback_volume(c: &mut Criterion) {
    let mut group = c.benchmark_group("viable_manifold_graph/pullback_volume");
    group.sample_size(500);

    let mut scratch = JacobianSvdScratch::with_capacity(DIM, DIM);
    let cfg = VolumeFieldConfig::default();
    let z = [0.3_f32, -0.7, 1.2, 0.9];

    group.bench_function("R^4_to_R^4_identity", |b| {
        b.iter(|| {
            let vol = pullback_volume(
                black_box(f_identity),
                black_box(&z),
                black_box(&mut scratch),
                black_box(&cfg),
            );
            black_box(vol);
        });
    });

    group.finish();
}

// ─── G-bench 2: manifold_random_walk per-step (k=4 neighbors) ───────────────
//
// Target: < 100 ns/step. We build the graph once outside the timed region,
// then time one 1000-step walk per iteration. `Throughput::Elements(1000)`
// tells criterion the iteration produces 1000 "units" — the printed median is
// per-walk, so divide by 1000 for the per-step figure.

fn build_k4_graph_for_walk() -> (SafeManifoldGraph, u32) {
    // 4D toy manifold: union of two balls in R^4 connected by a thin slab.
    // Viable iff inside left ball (center [-2,0,0,0], r=1.5) OR right ball
    // (center [+2,0,0,0], r=1.5) OR the bridging slab (|x0|<2 AND |x1|<0.4
    // AND x2,x3 ≈ 0). 50×50 grid in (x0,x1) plane × fixed (0,0) → 2500 samples.
    let viable = |z: &[f32]| {
        let x0 = z[0];
        let x1 = z[1];
        let dl = ((x0 + 2.0).powi(2) + x1 * x1).sqrt();
        let dr = ((x0 - 2.0).powi(2) + x1 * x1).sqrt();
        dl < 1.5 || dr < 1.5 || (x0.abs() < 2.0 && x1.abs() < 0.4)
    };

    let mut samples: Vec<f32> = Vec::with_capacity(50 * 50 * DIM);
    let mut i = -5.0_f32;
    while i <= 5.0 {
        let mut j = -5.0_f32;
        while j <= 5.0 {
            samples.push(i);
            samples.push(j);
            samples.push(0.0);
            samples.push(0.0);
            j += 0.2;
        }
        i += 0.2;
    }

    let mut scratch = JacobianSvdScratch::with_capacity(DIM, DIM);
    let build_cfg = GraphBuildConfig {
        volume_threshold: f32::INFINITY,
        edge_midpoint_check: true,
        k_nearest: 4,
    };
    let g = build_safe_manifold_graph(
        f_identity,
        &samples,
        DIM,
        &ClosurePredicate(viable),
        &VolumeFieldConfig::default(),
        &build_cfg,
        &mut scratch,
    );
    let src = g.nearest_node(&[-2.0, 0.0, 0.0, 0.0]).expect("left ball");
    (g, src)
}

fn bench_random_walk_per_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("viable_manifold_graph/manifold_random_walk");
    group.throughput(Throughput::Elements(1000));
    // Walks are fast; sample_size=500 like the lookup_into bench.
    group.sample_size(500);

    let (g, src) = build_k4_graph_for_walk();
    let m = 1000;

    group.bench_function("k=4_1000_steps", |b| {
        let mut rng = Rng::with_seed(0xBADC0DE);
        b.iter(|| {
            // We deliberately do NOT advance the RNG between iterations —
            // the walk is deterministic across iterations, which is fine for
            // timing (the RNG step cost is the same regardless of the path).
            let walk = manifold_random_walk(black_box(&g), black_box(src), black_box(m), &mut rng);
            black_box(walk.as_ptr());
        });
    });

    group.finish();
}

// ─── G-bench 3: build_safe_manifold_graph on 1000 samples ───────────────────
//
// Target: < 10 ms (1000 SVD calls + O(N²) kNN). Uses a 4D toy manifold with
// roughly 50% viability rate so the kept-set has ~500 nodes — a realistic
// "paper-scale" build.

fn make_1000_samples_4d(seed: u64) -> Vec<f32> {
    let mut rng = Rng::with_seed(seed);
    let mut v = Vec::with_capacity(1000 * DIM);
    for _ in 0..1000 {
        for _ in 0..DIM {
            v.push(rng.f32() * 10.0 - 5.0); // [-5, 5)
        }
    }
    v
}

fn bench_build_safe_manifold_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("viable_manifold_graph/build_safe_manifold_graph");
    group.throughput(Throughput::Elements(1000));
    // Build is ~ms-scale; criterion's default sample_size (100) is fine. We
    // keep the default — each iteration is independent (no shared state to
    // invalidate).

    let samples = make_1000_samples_4d(42);
    let mut scratch = JacobianSvdScratch::with_capacity(DIM, DIM);
    let build_cfg = GraphBuildConfig {
        volume_threshold: f32::INFINITY,
        edge_midpoint_check: true,
        k_nearest: 4,
    };
    // Predicate: roughly half-viable (a hypersphere of radius 3.0 centered
    // at origin in R^4 keeps ~16% of a [-5,5]^4 box by volume — closer to
    // 30% on a discrete 1000-sample uniform draw).
    let viable = |z: &[f32]| {
        let mut d2 = 0.0_f32;
        for zj in z.iter().take(DIM) {
            d2 += zj * zj;
        }
        d2.sqrt() < 3.5
    };

    group.bench_function("1000_samples_d4", |b| {
        b.iter(|| {
            let g = build_safe_manifold_graph(
                black_box(f_identity),
                black_box(&samples),
                black_box(DIM),
                black_box(&ClosurePredicate(viable)),
                black_box(&VolumeFieldConfig::default()),
                black_box(&build_cfg),
                black_box(&mut scratch),
            );
            black_box(g.n_nodes());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pullback_volume,
    bench_random_walk_per_step,
    bench_build_safe_manifold_graph,
);
criterion_main!(benches);
