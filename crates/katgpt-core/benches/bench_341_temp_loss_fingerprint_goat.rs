//! TEMP Perturbed-Loss-Vector Diversity Fingerprint — GOAT gate bench
//! (Plan 341 Phase 2).
//!
//! Exercises G3 (perf) and G3-alloc (zero-alloc hot path) against the
//! `temp_loss_fingerprint` primitive.
//!
//! # Gates
//!
//! - **G3 (perf)** — `perturbed_loss_vector` for K=8, N=100, D=8 on a
//!   single-matmul kernel: < 5µs per candidate. Plus `select_diverse_subset`
//!   for n=256, k=32, K=8: < 1ms (re-validated from riir-neuron-db Plan 005
//!   Phase 4 at 156µs).
//! - **G3-alloc** — Zero allocations on the hot path:
//!   - `perturbed_loss_vector`: 0 allocs (writes into caller buffer).
//!   - `select_diverse_subset_into`: the internal hot path is 0-alloc with
//!     pre-allocated workspaces. The `Vec<usize>` return value is 1 alloc
//!     per call (unavoidable output) — noted but not counted against the gate.
//!
//! # Run
//!
//! ```bash
//! cargo run -p katgpt-core --bench bench_341_temp_loss_fingerprint_goat --release -- --nocapture
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use katgpt_core::diversity::temp::{
    LossKernel, extrapolated_snapshot_schedule, perturbed_loss_vector, select_diverse_subset_into,
};
use std::hint::black_box;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── Deterministic xorshift RNG (fixture generation only) ──────────────────

struct FixtureRng(u64);
impl FixtureRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f32 {
        let bits = ((self.next_u64() >> 40) as u32 & 0x007f_ffff) | 0x3f80_0000;
        f32::from_bits(bits) - 1.0
    }
}

// ─── Single-matmul kernel (G3 perf target) ─────────────────────────────────
//
// The plan's perf target is "single matmul, K=8, N=100 = 800 FMA lanes"
// interpreted at D=1. Here we use D=8 (style_weights scale) with N=100
// tokens, giving K*N*D = 8*100*8 = 6400 FMA lanes per candidate — the more
// realistic shard-scale workload. A D=1 variant would be 800 lanes; both are
// well under the 5µs target for pure-FMA arithmetic.

struct MatmulKernel {
    d: usize,
}
impl LossKernel for MatmulKernel {
    fn short_prefix_loss(&self, theta: &[f32], z_prefix: &[f32]) -> f32 {
        let d = self.d.min(theta.len());
        let n_tokens = z_prefix.len() / d;
        let mut sum = 0.0_f32;
        for t in 0..n_tokens {
            let mut dot = 0.0_f32;
            for i in 0..d {
                dot += theta[i] * z_prefix[t * d + i];
            }
            sum += dot;
        }
        sum
    }
}

// ─── G3 perf: perturbed_loss_vector ────────────────────────────────────────

fn bench_perturbed_loss_vector(c: &mut Criterion) {
    const D: usize = 8;
    const K: usize = 8;
    const N_TOKENS: usize = 100;

    let s0 = vec![0.0_f32; D];
    let s1: Vec<f32> = (0..D).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let lambda = [0.0_f32, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0];
    let seeds = [0u64; K];
    let mut theta: Vec<Vec<f32>> = (0..K).map(|_| Vec::with_capacity(D)).collect();
    extrapolated_snapshot_schedule(&s0, &s1, &lambda, &seeds, 0.0, &mut theta);

    let kernel = MatmulKernel { d: D };
    let z_prefix: Vec<f32> = (0..N_TOKENS * D)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let mut out = [0.0_f32; K];

    c.bench_function("perturbed_loss_vector_K8_N100_D8", |b| {
        b.iter(|| {
            perturbed_loss_vector(
                black_box(&kernel),
                black_box(&theta),
                black_box(&z_prefix),
                black_box(&mut out),
            );
        });
    });

    // G3-alloc: zero allocations on the hot path.
    let (_, allocs) = alloc_delta(|| {
        for _ in 0..100 {
            perturbed_loss_vector(&kernel, &theta, &z_prefix, &mut out);
        }
    });
    eprintln!("G3-alloc: perturbed_loss_vector 100 calls: {allocs} allocs (expected 0)");
    assert_eq!(
        allocs, 0,
        "G3-alloc FAIL: perturbed_loss_vector should be zero-alloc on the hot path"
    );
}

// ─── G3 perf: select_diverse_subset ────────────────────────────────────────

fn bench_select_diverse_subset(c: &mut Criterion) {
    let n: usize = 256;
    let k_vec: usize = 8;
    let k_subset: usize = 32;

    let mut rng = FixtureRng::new(42);
    let loss_vecs: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..k_vec).map(|_| rng.uniform()).collect())
        .collect();
    let lvs_refs: Vec<&[f32]> = loss_vecs.iter().map(|v| v.as_slice()).collect();

    let mut scratch = vec![0_usize; k_subset];
    let mut min_dist: Vec<f32> = Vec::with_capacity(n);
    let mut is_selected: Vec<bool> = Vec::with_capacity(n);

    c.bench_function("select_diverse_subset_n256_k32_K8", |b| {
        b.iter(|| {
            select_diverse_subset_into(
                black_box(&lvs_refs),
                black_box(k_subset),
                black_box(&mut scratch),
                black_box(&mut min_dist),
                black_box(&mut is_selected),
            )
        });
    });

    // G3-alloc note: the internal hot path is 0-alloc with pre-allocated
    // workspaces. The `Vec<usize>` return value is 1 alloc per call.
    let (_, allocs) = alloc_delta(|| {
        for _ in 0..100 {
            let _ = select_diverse_subset_into(
                &lvs_refs,
                k_subset,
                &mut scratch,
                &mut min_dist,
                &mut is_selected,
            );
        }
    });
    eprintln!(
        "G3-alloc: select_diverse_subset_into 100 calls: {allocs} allocs (expected ~100 from return Vec<usize>)"
    );
    // The return Vec<usize> allocates once per call. 100 calls ≈ 100 allocs.
    // Any allocs beyond the return-Vec count would indicate workspace leakage.
    assert!(
        allocs <= 100,
        "G3-alloc FAIL: {allocs} allocs for 100 calls — expected ≤100 (return Vec only), \
         excess indicates workspace reallocation"
    );
}

criterion_group!(
    benches,
    bench_perturbed_loss_vector,
    bench_select_diverse_subset
);
criterion_main!(benches);
