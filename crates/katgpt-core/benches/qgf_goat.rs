//! QGF (Q-Guided Flow) — GOAT gate G4 overhead benchmark (Plan 268 T10).
//!
//! Measures the per-step cost of the QGF tilt hot path (`tilt_logits`) and the
//! end-to-end guided pipeline (`generate_project_tilt_sample`), against the
//! unguided base. This is the **G4 overhead gate**: the tilt must not dominate
//! the generator's own cost.
//!
//! # Gates measured here
//!
//! - **G4a — tilt overhead vs action-space size.** `tilt_logits` is an AXPY
//!   over `n` logits; we expect linear scaling with a tiny constant (one SIMD
//!   FMA per 4/8 lanes). Target: sub-microsecond at `n ≤ 256`.
//! - **G4b — end-to-end pipeline overhead.** `generate_project_tilt_sample`
//!   wraps the generator (one `generate()` call + tilt + sample closure). The
//!   wrapper overhead must be a small fraction of the generator's own cost on
//!   realistic (medium/expensive) generators.
//!
//! # What this bench does NOT measure
//!
//! Downstream task quality (Sudoku solve rate, Bomber win rate) — those are
//! deferred to a riir-ai integration plan (the selling-point layer). The
//! correctness, alloc-free, and stability gates live in `tests/qgf_goat.rs`.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features "qgf,qgf_drafter,qgf_adaptive" --bench qgf_goat
//! ```
//!
//! For a quick (non-statistical) GOAT check, use `--bench` with `--profile=dev`
//! or read the printed ratios from the `qgf_goat_summary` group.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use katgpt_core::qgf::QGuidedDrafter;
use katgpt_core::traits::{QGradientOracle, SpeculativeGenerator};

// ──────────────────────────────────────────────────────────────────────────
// Fixtures
// ──────────────────────────────────────────────────────────────────────────

/// Cost-tiered mock generator (mirrors `qgf_projector_bench.rs::TieredGen`).
/// Burns a tunable amount of CPU work so the bench measures realistic per-call
/// times, then returns a 4-element candidate list (typical spec top-k).
struct TieredGen {
    work: u32,
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
        let mut acc: u32 = *condition;
        for i in 0..self.work {
            acc = acc.wrapping_add(i);
            if i & 0xF == 0 {
                acc = black_box(acc);
            }
        }
        self.buf.clear();
        self.buf
            .extend_from_slice(&[acc, acc + 1, acc + 2, acc + 3]);
        Ok(core::mem::take(&mut self.buf))
    }
}

/// Deterministic oracle that writes a fixed Q-vector into the gradient buffer
/// (no allocation on the hot path).
struct FixedQOracle {
    q: Vec<f32>,
}

impl QGradientOracle for FixedQOracle {
    type State = u32;
    type Action = u32;

    fn q_gradient_at(&self, _s: &u32, _a: &u32) -> Vec<f32> {
        self.q.clone()
    }

    fn q_gradient_into(&self, _s: &u32, _a: &u32, out: &mut [f32]) {
        let n = out.len().min(self.q.len());
        out[..n].copy_from_slice(&self.q[..n]);
        out[n..].fill(0.0);
    }

    fn confidence(&self, _s: &u32) -> f32 {
        1.0
    }
}

// ──────────────────────────────────────────────────────────────────────────
// G4a — tilt_logits overhead vs action-space size
// ──────────────────────────────────────────────────────────────────────────
//
// `tilt_logits` does: (1) a period/weight gate check, (2) one
// `q_gradient_into` call (pure buffer copy), (3) one SIMD AXPY
// (`simd_fused_scale_acc`). We measure it in isolation across action-space
// sizes to confirm linear-in-n scaling with a small constant.

fn bench_tilt_overhead_vs_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("qgf_goat_g4a_tilt_logits");
    group.sample_size(200);

    for &n in &[16usize, 64, 256, 1024] {
        let q = vec![0.5f32; n];
        let oracle = FixedQOracle { q };
        // The generator is never called by `tilt_logits` — it exists only to
        // satisfy the drafter's type bound.
        let drafter = QGuidedDrafter::new(TieredGen::new(0), oracle).with_weight(2.0);
        let mut logits = vec![0.1f32; n];
        let mut grad = vec![0.0f32; n];

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                let applied = drafter.tilt_logits(
                    black_box(&7u32),
                    black_box(&42u32),
                    black_box(&mut logits),
                    black_box(&mut grad),
                    black_box(0),
                );
                black_box(applied);
            });
        });
    }

    group.finish();
}

// ──────────────────────────────────────────────────────────────────────────
// G4b — end-to-end pipeline overhead vs base generate
// ──────────────────────────────────────────────────────────────────────────
//
// Compares the full guided pipeline (`generate_project_tilt_sample`) against
// the unguided `generate()` across generator-cost tiers. The wrapper's
// marginal cost (projection + gradient query + tilt + sample) should be a
// small fraction of the generator's own cost at the medium/expensive tiers —
// the regime where QGF actually runs (real generators are not free).

fn bench_pipeline_overhead_vs_base(c: &mut Criterion) {
    const N_LOGITS: usize = 64;
    let mut group = c.benchmark_group("qgf_goat_g4b_pipeline_vs_base");
    group.sample_size(100);

    for (name, work) in [("cheap", 4u32), ("medium", 64u32), ("expensive", 1024u32)] {
        // Baseline: unguided generate() + greedy argmax sample (the work the
        // caller does WITHOUT QGF).
        group.bench_with_input(BenchmarkId::new("base_generate", name), &work, |b, &w| {
            let mut generator = TieredGen::new(w);
            let mut rng = fastrand::Rng::new();
            let condition = 7u32;
            b.iter(|| {
                let cands = generator.generate(black_box(&condition), &mut rng).unwrap();
                // Simulate the caller's sample step (argmax is the cheapest
                // realistic sampler; representative of the non-QGF path).
                let chosen = cands.into_iter().next().unwrap_or(0);
                black_box(chosen);
            });
        });

        // Guided: full generate_project_tilt_sample pipeline.
        group.bench_with_input(BenchmarkId::new("guided_pipeline", name), &work, |b, &w| {
            let q = vec![0.3f32; N_LOGITS];
            let oracle = FixedQOracle { q };
            let mut drafter = QGuidedDrafter::new(TieredGen::new(w), oracle).with_weight(2.0);
            let mut rng = fastrand::Rng::new();
            let condition = 7u32;
            // Pre-allocated scratch (the documented caller-owned-buffer
            // pattern — reused across iterations, not reallocated).
            let mut logits = vec![0.1f32; N_LOGITS];
            let mut grad = vec![0.0f32; N_LOGITS];
            b.iter(|| {
                let out = drafter
                    .generate_project_tilt_sample(
                        black_box(&condition),
                        &mut rng,
                        black_box(0),
                        &mut logits,
                        &mut grad,
                        |tilted| {
                            // Greedy argmax sampler over the tilted logits.
                            let mut best_idx = 0usize;
                            let mut best_val = f32::NEG_INFINITY;
                            for (i, &v) in tilted.iter().enumerate() {
                                if v > best_val {
                                    best_val = v;
                                    best_idx = i;
                                }
                            }
                            best_idx as u32
                        },
                    )
                    .unwrap();
                black_box(out);
            });
        });
    }

    group.finish();
}

// ──────────────────────────────────────────────────────────────────────────
// G4c — adaptive tilt overhead (F4) vs fixed-weight tilt
// ──────────────────────────────────────────────────────────────────────────
//
// `tilt_logits_adaptive` adds a per-call `confidence()` query + sigmoid per
// query. We confirm the F4 path is not meaningfully more expensive than the
// fixed-weight path (the confidence query + sigmoid are O(1) on top of the
// O(n) AXPY).

fn bench_adaptive_vs_fixed_tilt(c: &mut Criterion) {
    const N: usize = 64;
    let mut group = c.benchmark_group("qgf_goat_g4c_adaptive_vs_fixed");
    group.sample_size(200);

    let q = vec![0.4f32; N];
    let oracle_fixed = FixedQOracle { q: q.clone() };
    let drafter_fixed = QGuidedDrafter::new(TieredGen::new(0), oracle_fixed).with_weight(2.0);
    let oracle_adaptive = FixedQOracle { q };
    let drafter_adaptive = QGuidedDrafter::new(TieredGen::new(0), oracle_adaptive);

    let mut logits_f = vec![0.1f32; N];
    let mut grad_f = vec![0.0f32; N];
    let mut logits_a = vec![0.1f32; N];
    let mut grad_a = vec![0.0f32; N];

    group.bench_function("fixed_weight_tilt", |b| {
        b.iter(|| {
            drafter_fixed.tilt_logits(
                black_box(&7u32),
                black_box(&42u32),
                black_box(&mut logits_f),
                black_box(&mut grad_f),
                black_box(0),
            )
        });
    });

    group.bench_function("adaptive_weight_tilt", |b| {
        b.iter(|| {
            drafter_adaptive.tilt_logits_adaptive(
                black_box(&7u32),
                black_box(&42u32),
                black_box(&mut logits_a),
                black_box(&mut grad_a),
                black_box(0),
                0.5,
                6.0,
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tilt_overhead_vs_size,
    bench_pipeline_overhead_vs_base,
    bench_adaptive_vs_fixed_tilt,
);
criterion_main!(benches);
