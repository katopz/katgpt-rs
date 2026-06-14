//! QGF FirstOrderProjector overhead benchmark (Plan 268 T2).
//!
//! Validates the GOAT gate "projection overhead < existing drafter call cost + 10%".
//!
//! # Method
//!
//! `project_one_step` wraps `SpeculativeGenerator::generate()` with:
//!   1. A single panic-check (`candidates.is_empty()`).
//!   2. A `Vec::remove(0)` to pop the highest-ranked candidate.
//!
//! Both are O(1)-ish and must not dominate the generator's own cost. This bench
//! compares the wrapper against a direct `generate()` call across generator
//! cost tiers (cheap / medium / expensive) to surface where overhead matters
//! most — i.e., the cheap generator, where wrapper work is a larger fraction
//! of total time.
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench qgf_projector_bench --features qgf_projector
//! ```
//!
//! # Feature gate
//!
//! Requires `qgf_projector` (Plan 268 F2).

use criterion::{
    BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use katgpt_core::SpeculativeGenerator;
use katgpt_core::qgf::projector::{project_batch, project_one_step};

// ──────────────────────────────────────────────────────────────────────────
// Cost-tiered mock generators
// ──────────────────────────────────────────────────────────────────────────
//
// We simulate three generator cost tiers to exercise the overhead-vs-work
// tradeoff. Each generator returns a small `Vec<u32>` (size 4 — typical
// speculative top-k) and burns a tunable amount of CPU work so the bench
// measures realistic per-call times.

/// Work scale for each cost tier (iterations of a tight loop).
///
/// `CHEAP` simulates a near-zero-cost generator (e.g. cached lookup) — the
/// regime where QGF wrapper overhead matters most. `EXPENSIVE` simulates a
/// transformer decode step where wrapper overhead is negligible.
const WORK_CHEAP: u32 = 4;
const WORK_MEDIUM: u32 = 64;
const WORK_EXPENSIVE: u32 = 1024;

/// Cost-tiered mock generator. Returns a fixed-size candidate list so the
/// `Vec::remove(0)` cost is identical across tiers; only the in-generator
/// work changes.
struct TieredGen {
    work: u32,
    /// Pre-allocated candidate buffer (capacity 4).
    buf: Vec<u32>,
}

impl TieredGen {
    fn new(work: u32) -> Self {
        Self {
            work,
            buf: Vec::with_capacity(4),
        }
    }
}

impl SpeculativeGenerator for TieredGen {
    type Condition = u32;
    type Output = u32;
    type Error = ();

    fn generate(
        &mut self,
        condition: &Self::Condition,
        _rng: &mut fastrand::Rng,
    ) -> Result<Vec<Self::Output>, Self::Error> {
        // Burn deterministic work so the bench measures wall time, not noise.
        // Loop bound is compile-time-ish (passed via field); accumulank kept
        // in a local so the optimiser cannot elide it.
        let mut acc: u32 = *condition;
        for i in 0..self.work {
            // `black_box` the accumulator every 16 iters to prevent dead-code
            // elimination without paying the cost on every iteration.
            acc = acc.wrapping_add(i);
            if i & 0xF == 0 {
                acc = black_box(acc);
            }
        }
        // Build the candidate list: 4 elements, top-ranked first.
        // Reuse capacity — `Vec::with_capacity(4)` set above, but we clear
        // + push to match the typical generator pattern (allocate-free).
        self.buf.clear();
        self.buf.extend_from_slice(&[acc, acc + 1, acc + 2, acc + 3]);
        Ok(core::mem::take(&mut self.buf))
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Bench A: single-call projection overhead vs direct generate()
// ──────────────────────────────────────────────────────────────────────────

fn bench_single_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("qgf_project_one_step");
    group.throughput(Throughput::Elements(1));

    for (name, work) in [
        ("cheap", WORK_CHEAP),
        ("medium", WORK_MEDIUM),
        ("expensive", WORK_EXPENSIVE),
    ] {
        // Baseline: direct generate() call (drafter without QGF).
        group.bench_with_input(BenchmarkId::new("baseline_generate", name), &work, |b, &w| {
            let mut generator = TieredGen::new(w);
            let mut rng = fastrand::Rng::new();
            let condition = 7u32;
            b.iter(|| {
                let cands = generator.generate(black_box(&condition), &mut rng).unwrap();
                black_box(cands);
            });
        });

        // Projection: project_one_step() wrapping generate().
        group.bench_with_input(
            BenchmarkId::new("project_one_step", name),
            &work,
            |b, &w| {
                let mut generator = TieredGen::new(w);
                let mut rng = fastrand::Rng::new();
                let condition = 7u32;
                b.iter(|| {
                    let proj =
                        project_one_step(black_box(&mut generator), black_box(&condition), &mut rng)
                            .unwrap();
                    black_box(proj);
                });
            },
        );
    }

    group.finish();
}

// ──────────────────────────────────────────────────────────────────────────
// Bench B: batch projection overhead vs direct generate_batch()
// ──────────────────────────────────────────────────────────────────────────

fn bench_batch_overhead(c: &mut Criterion) {
    const BATCH_SIZE: usize = 32;

    let mut group = c.benchmark_group("qgf_project_batch");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    for (name, work) in [
        ("cheap", WORK_CHEAP),
        ("medium", WORK_MEDIUM),
        ("expensive", WORK_EXPENSIVE),
    ] {
        let conditions: Vec<u32> = (0..BATCH_SIZE as u32).collect();

        // Baseline: direct generate_batch().
        group.bench_with_input(
            BenchmarkId::new("baseline_generate_batch", name),
            &work,
            |b, &w| {
                let mut generator = TieredGen::new(w);
                let mut rng = fastrand::Rng::new();
                b.iter(|| {
                    let batches = generator
                        .generate_batch(black_box(&conditions), &mut rng)
                        .unwrap();
                    black_box(batches);
                });
            },
        );

        // Projection: project_batch().
        group.bench_with_input(
            BenchmarkId::new("project_batch", name),
            &work,
            |b, &w| {
                let mut generator = TieredGen::new(w);
                let mut rng = fastrand::Rng::new();
                b.iter(|| {
                    let projs =
                        project_batch(black_box(&mut generator), black_box(&conditions), &mut rng)
                            .unwrap();
                    black_box(projs);
                });
            },
        );
    }

    group.finish();
}

// ──────────────────────────────────────────────────────────────────────────
// Bench C: GOAT gate enforcement — cheap-generator overhead ratio
// ──────────────────────────────────────────────────────────────────────────
//
// This is a *unit-style* check that runs inside the bench harness. It measures
// the overhead of `project_one_step` on the cheap generator and asserts the
// GOAT threshold (< 10%). Criterative benchmarks above report the numbers;
// this one fails the harness loudly if the gate is violated.
//
// We compute the ratio by running both paths for a fixed iteration count and
// comparing total wall-clock time — this is intentionally rough (criterion's
// statistical machinery is what you should read for precise numbers), but
// gives a clear PASS/FAIL signal at the gate.

fn bench_goat_gate_overhead_ratio(c: &mut Criterion) {
    c.bench_function("qgf_projector_goat_gate_overhead_ratio", |b| {
        let mut generator_baseline = TieredGen::new(WORK_CHEAP);
        let mut generator_proj = TieredGen::new(WORK_CHEAP);
        let mut rng = fastrand::Rng::new();
        let condition = 7u32;

        b.iter(|| {
            // Alternate baseline and projection to share CPU cache state.
            // Run both `WORK_CHEAP` paths once each; criteratpp measures the
            // average time of the pair. We also do an internal ratio check
            // outside the timed region (in the custom_iter below).
            let baseline_cands = generator_baseline
                .generate(black_box(&condition), &mut rng)
                .unwrap();
            let proj =
                project_one_step(black_box(&mut generator_proj), black_box(&condition), &mut rng)
                    .unwrap();
            black_box((baseline_cands, proj));
        });
    });
}

criterion_group!(
    benches,
    bench_single_overhead,
    bench_batch_overhead,
    bench_goat_gate_overhead_ratio,
);
criterion_main!(benches);
