//! Bench 668 G4 — `effective_degree` zero-alloc steady state (Issue 668 T4).
//!
//! Separate test binary so the global `CountingAllocator` sees only this
//! module's hot path (the `bench_668_effective_degree_goat` timing loops would
//! be skewed by a counting allocator, and parallel lib tests would corrupt the
//! deltas). Single `#[test]` so every measurement runs serially against the
//! one process-wide counter.
//!
//! Contract:
//! - [`effective_degree_along_path`] (scalar) allocates **nothing, ever** — the
//!   whole `(K+1)²` solve lives in fixed-size stack arrays.
//! - [`randomized_cosine_nodes`] allocates nothing (caller-owned `out`).
//! - [`ed_over_pairs`] / [`effective_degree_along_path_multi`] allocate nothing
//!   once the caller's [`EdScratch`] exists. `EdScratch::new` is cold-path
//!   construction and is explicitly outside the gate.

use katgpt_core::effective_degree::{
    EdConfig, EdScratch, ed_over_pairs, effective_degree_along_path,
    effective_degree_along_path_multi, randomized_cosine_nodes,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

const N_WARMUP: usize = 16;
const N_CALLS: usize = 10_000;
const IN_DIM: usize = 6;

fn linear_form(x: &[f32]) -> f32 {
    const W: [f32; IN_DIM] = [0.5, -0.3, 0.8, 0.2, -0.6, 0.4];
    x.iter().zip(W).map(|(a, b)| a * b).sum()
}

fn manifold_points(n: usize, phase: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * IN_DIM);
    for i in 0..n {
        let t = (i as f32) / (n as f32) * core::f32::consts::TAU + phase;
        for d in 0..IN_DIM {
            let f = (d + 1) as f32;
            out.push((0.6 * (f * t).sin() + 0.25 * (0.5 * f * t).cos()) / f.sqrt());
        }
    }
    out.shrink_to_fit();
    out
}

#[test]
fn g4_zero_alloc_steady_state() {
    let cfg = EdConfig::precise();

    // ── Cold-path construction (allowed to allocate) ────────────────────────
    let mut nodes = vec![0.0f32; cfg.resolution];
    randomized_cosine_nodes(cfg.resolution, cfg.seed, &mut nodes).unwrap();
    let scalar_out: Vec<f32> = nodes.iter().map(|&a| a.powi(3) - 0.4 * a + 0.2).collect();
    let multi_out: Vec<f32> = nodes
        .iter()
        .flat_map(|&a| [a * a, 1.0 - a, a.powi(3)])
        .collect();
    let a = manifold_points(cfg.n_pairs, 0.0);
    let b = manifold_points(cfg.n_pairs, 1.7);
    let mut scratch_multi = EdScratch::new(&cfg, 0, 3);
    let mut scratch_pairs = EdScratch::new(&cfg, IN_DIM, 1);

    // ── Warmup — rule out first-call lazy init ──────────────────────────────
    for _ in 0..N_WARMUP {
        black_box(effective_degree_along_path(&scalar_out, &nodes, &cfg).unwrap());
        black_box(
            effective_degree_along_path_multi(&multi_out, 3, &nodes, &cfg, &mut scratch_multi)
                .unwrap(),
        );
        black_box(
            ed_over_pairs(
                |x, y| y[0] = linear_form(x).powi(3),
                &a,
                &b,
                &cfg,
                &mut scratch_pairs,
            )
            .unwrap(),
        );
        randomized_cosine_nodes(cfg.resolution, cfg.seed, &mut nodes).unwrap();
    }

    // ── Measured window ─────────────────────────────────────────────────────
    let before = ALLOC_COUNT.load(Ordering::SeqCst);
    for i in 0..N_CALLS {
        black_box(
            effective_degree_along_path(black_box(&scalar_out), black_box(&nodes), &cfg).unwrap(),
        );
        randomized_cosine_nodes(cfg.resolution, black_box(i as u64), &mut nodes).unwrap();
    }
    let scalar_delta = ALLOC_COUNT.load(Ordering::SeqCst) - before;

    let before = ALLOC_COUNT.load(Ordering::SeqCst);
    for _ in 0..N_CALLS {
        black_box(
            effective_degree_along_path_multi(
                black_box(&multi_out),
                3,
                black_box(&nodes),
                &cfg,
                &mut scratch_multi,
            )
            .unwrap(),
        );
    }
    let multi_delta = ALLOC_COUNT.load(Ordering::SeqCst) - before;

    let before = ALLOC_COUNT.load(Ordering::SeqCst);
    for _ in 0..1_000 {
        black_box(
            ed_over_pairs(
                |x, y| y[0] = linear_form(x).powi(3),
                black_box(&a),
                black_box(&b),
                &cfg,
                &mut scratch_pairs,
            )
            .unwrap(),
        );
    }
    let pairs_delta = ALLOC_COUNT.load(Ordering::SeqCst) - before;

    println!(
        "G4 allocs — scalar+sampler: {scalar_delta} / {N_CALLS}, multi(out_dim=3): {multi_delta} / {N_CALLS}, ed_over_pairs: {pairs_delta} / 1000"
    );
    assert_eq!(scalar_delta, 0, "effective_degree_along_path allocated");
    assert_eq!(
        multi_delta, 0,
        "effective_degree_along_path_multi allocated"
    );
    assert_eq!(
        pairs_delta, 0,
        "ed_over_pairs allocated with reused scratch"
    );
}
