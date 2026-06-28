//! best_belief latency benchmark (Plan 336 Phase 2 T2.2 — G2 perf gate).
//!
//! Reports median latency for:
//! - `best_belief_score` across a representative `(S, F)` grid. The LUT hot
//!   path (`S, F ∈ [0, 31]`, standard ε) targets ≤ 100 ns; the cold path
//!   (large `S+F`) is documented but not gated.
//! - `select_best_belief` on 4 and 8 candidates. Two candidate sets are
//!   measured: a realistic all-in-LUT set (target ≤ 500 ns) and an
//!   adversarial mixed set with some out-of-LUT candidates (cold path).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features best_belief --bench best_belief_bench \
//!   -- --warm-up-time 0.5 --measurement-time 1.5 --sample-size 30
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::best_belief::{best_belief_score, select_best_belief};

fn bench_score(c: &mut Criterion) {
    // The first three cases hit the LUT hot path (S, F < 32). The last three
    // fall back to the closed form (LUT miss) — documented cold path.
    let cases: &[(u32, u32)] = &[
        (0, 0),
        (1, 1),
        (10, 1),
        (31, 31), // last in-LUT cell
        (50, 50), // LUT miss
        (1000, 1000), // LUT miss, worst case
    ];
    let mut g = c.benchmark_group("best_belief_score");
    for &(s, f) in cases {
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("S{s}_F{f}")),
            &(s, f),
            |b, &(s, f)| {
                b.iter(|| {
                    black_box(best_belief_score(
                        black_box(s),
                        black_box(f),
                        black_box(0.05),
                    ))
                })
            },
        );
    }
    g.finish();
}

fn bench_select(c: &mut Criterion) {
    // Realistic candidate set: all (S, F) inside the LUT domain. This is the
    // intended hot-path use case (snapshot promotion, shard selection with
    // bounded evidence budgets). Target: ≤ 500 ns.
    let realistic_8: Vec<(u32, u32)> = vec![
        (10, 1),
        (5, 2),
        (3, 0),
        (20, 15),
        (8, 8),
        (1, 10),
        (15, 5),
        (7, 3),
    ];
    let realistic_4 = &realistic_8[..4];

    // Adversarial mixed set: some candidates have large counts (S+F > 62),
    // forcing the closed-form cold path. Documents the fallback latency.
    let adversarial_8: Vec<(u32, u32)> = vec![
        (10, 1),
        (5, 2),
        (100, 90),
        (3, 0),
        (20, 15),
        (8, 8),
        (50, 49),
        (1, 10),
    ];

    c.bench_function("select_4_realistic_in_lut", |b| {
        b.iter(|| {
            black_box(select_best_belief(
                black_box(realistic_4),
                black_box(0.05),
                black_box(Some(0)),
            ))
        })
    });
    c.bench_function("select_8_realistic_in_lut", |b| {
        b.iter(|| {
            black_box(select_best_belief(
                black_box(&realistic_8),
                black_box(0.05),
                black_box(Some(0)),
            ))
        })
    });
    c.bench_function("select_8_adversarial_mixed", |b| {
        b.iter(|| {
            black_box(select_best_belief(
                black_box(&adversarial_8),
                black_box(0.05),
                black_box(Some(0)),
            ))
        })
    });
}

criterion_group!(benches, bench_score, bench_select);
criterion_main!(benches);
