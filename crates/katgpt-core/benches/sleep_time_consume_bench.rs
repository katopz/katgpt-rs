//! Sleep-Time consume() hot-path latency benchmark — GOAT gate G6 (Plan 334).
//!
//! Target: `consume()` ≤ 200 ns/call at D=64. For D=8 the target is
//! ≤ 100 ns/call. Matches `EmotionDirections::project` and KARC `forecast`
//! latency profiles (Bench 292 / Bench 308).
//!
//! Note: dimension labels (D8/D64) are intentionally generic. The mapping of
//! these dims to specific runtime artifacts is private (riir-ai).
//!
//! # Run
//!
//! ```bash
//! cargo run -p katgpt-core --features sleep_time_anticipation --bench sleep_time_consume_bench --release -- --nocapture
//! ```

#![cfg(feature = "sleep_time_anticipation")]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::sleep_time::{
    consume, consume_gate, AnticipatedQueryDir, DotPredictabilityScorer, IdentityFunctorOp,
    SleepTimeAnticipator, SleepTimeScratch,
};

/// Build a c' artifact for benchmarking. K directions of the given dim.
fn build_artifact<const D: usize, const K: usize>(
    c: &[f32; D],
) -> katgpt_core::sleep_time::AnticipatedQuerySet<D, K> {
    // Build K orthonormal-ish directions by hand (we don't need true
    // orthonormality for the latency bench — just K distinct vectors).
    let dirs = std::array::from_fn(|i| {
        let mut d = [0.0f32; D];
        // Place a 1.0 in position (i mod D), and a small value elsewhere
        // so no two directions are identical.
        d[i % D] = 1.0;
        if D > 1 {
            d[(i + 1) % D] = 0.1;
        }
        AnticipatedQueryDir::new(d)
    });
    let anticipator = SleepTimeAnticipator::<D, K, IdentityFunctorOp, DotPredictabilityScorer> {
        op: IdentityFunctorOp,
        scorer: DotPredictabilityScorer::default(),
        budgets: [128; K],
        tau: 0.5,
        beta: 4.0,
    };
    let mut scratch = SleepTimeScratch::new();
    anticipator.anticipate(c, &dirs, &mut scratch)
}

fn bench_consume(c: &mut Criterion) {
    // Zero-alloc fresh_think: pure stack arithmetic.
    fn fresh_d8(fq: &[f32; 8]) -> [f32; 8] {
        let mut z = [0.0f32; 8];
        for j in 0..8 {
            z[j] = fq[j] * 0.5 + 1.0;
        }
        z
    }
    fn fresh_d64(fq: &[f32; 64]) -> [f32; 64] {
        let mut z = [0.0f32; 64];
        for j in 0..64 {
            z[j] = fq[j] * 0.5 + 1.0;
        }
        z
    }

    let mut group = c.benchmark_group("sleep_time_consume");
    group.throughput(criterion::Throughput::Elements(1));

    // ── D=8, K=4 (small latent dim, small catalog) ─────────────────────────
    {
        const D: usize = 8;
        const K: usize = 4;
        let c_ctx = [0.3f32; D];
        let artifact = build_artifact::<D, K>(&c_ctx);
        let q = [0.7f32; D];

        group.bench_with_input(BenchmarkId::new("consume", "D8_K4"), &q, |b, &_q| {
            b.iter(|| {
                let out = consume(black_box(&q), black_box(&artifact), 0.5, 4.0, fresh_d8);
                black_box(out);
            });
        });
        group.bench_with_input(BenchmarkId::new("consume_gate", "D8_K4"), &q, |b, &_q| {
            b.iter(|| {
                let r = consume_gate(black_box(&q), black_box(&artifact), 0.5, 4.0);
                black_box(r);
            });
        });
    }

    // ── D=8, K=8 (small latent dim, full catalog) ──────────────────────────
    {
        const D: usize = 8;
        const K: usize = 8;
        let c_ctx = [0.3f32; D];
        let artifact = build_artifact::<D, K>(&c_ctx);
        let q = [0.7f32; D];

        group.bench_with_input(BenchmarkId::new("consume", "D8_K8"), &q, |b, &_q| {
            b.iter(|| {
                let out = consume(black_box(&q), black_box(&artifact), 0.5, 4.0, fresh_d8);
                black_box(out);
            });
        });
    }

    // ── D=64, K=8 (large latent dim, full catalog) ─────────────────────────
    {
        const D: usize = 64;
        const K: usize = 8;
        let c_ctx = [0.3f32; D];
        let artifact = build_artifact::<D, K>(&c_ctx);
        let q = [0.7f32; D];

        group.bench_with_input(BenchmarkId::new("consume", "D64_K8"), &q, |b, &_q| {
            b.iter(|| {
                let out = consume(black_box(&q), black_box(&artifact), 0.5, 4.0, fresh_d64);
                black_box(out);
            });
        });
        group.bench_with_input(BenchmarkId::new("consume_gate", "D64_K8"), &q, |b, &_q| {
            b.iter(|| {
                let r = consume_gate(black_box(&q), black_box(&artifact), 0.5, 4.0);
                black_box(r);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_consume);
criterion_main!(benches);
