//! Plan 367 Phase 4 T4.4 — QMC-BoM fill overhead benchmark.
//!
//! Measures the cost of filling BoM noise queries via QMC vs i.i.d. (Box-Muller),
//! and the end-to-end `sample_k_states` cost with each fill strategy. The matvec
//! (`sample_k_states`) cost is identical either way — only the fill cost differs.
//!
//! Target (per plan T4.4): QMC fill + sample_k_states cost within 5% of i.i.d.
//! fill + sample_k_states at K=8, D=4. In practice the QMC fill can be FASTER
//! than i.i.d. because the Hastings probit (1 sqrt + 1 ln + rational) is cheaper
//! than Box-Muller (1 sqrt + 1 ln + 1 sin/cos or 1 cos), and the D independent
//! QMC draws are each O(K) with minimal overhead.
//!
//! # Convention
//!
//! `std::time::Instant` + `harness = false` + custom `main()` (matches the
//! Phase 3 bench_367_qmc_overhead convention; no Criterion in katgpt-core).
//!
//! Run with:
//! ```bash
//! cargo run --release --features qmc_sampling,bom_sampling --bench bench_367_qmc_bom_overhead
//! ```

#![cfg(feature = "bom_sampling")]

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::speculative::qmc::{LatticeQmc, fill_noise_queries_gaussian_qmc};
use katgpt_core::{AttractorKernel, BoMSampler, NoiseQueryConfig};

const WARMUP_ITERS: usize = 1_000;
const MEASURE_ITERS: usize = 100_000;

fn measure_ns_per_call<F: FnMut()>(label: &str, mut f: F) -> f64 {
    // Warmup.
    for _ in 0..WARMUP_ITERS {
        f();
    }
    // Measure.
    let start = Instant::now();
    for _ in 0..MEASURE_ITERS {
        f();
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_nanos() as f64 / MEASURE_ITERS as f64;
    println!("  {label:<55} {ns:>10.2} ns");
    ns
}

fn standard_normal_box_muller(rng: &mut fastrand::Rng) -> f32 {
    let u1 = rng.f32().max(1e-10);
    let u2 = rng.f32();
    let r = (-2.0f32 * u1.ln()).sqrt();
    let theta = 2.0 * core::f32::consts::PI * u2;
    r * theta.cos()
}

fn bench_fill_and_sample(dim: usize, k: usize, sigma: f32) {
    println!("\n━━━ D={dim}, K={k}, σ={sigma} ━━━");

    let kernel = AttractorKernel::from_seed(42, dim);
    let cfg = NoiseQueryConfig::default().with_k(k).with_sigma(sigma);
    let s_prev = vec![0.0f32; dim];
    let x = vec![0.5f32; dim];

    // ── Fill-only: QMC vs i.i.d. ──────────────────────────────────────

    let mut qmc_source = LatticeQmc::new(123);
    let mut queries_qmc = vec![0.0f32; k * dim];
    let ns_qmc_fill = measure_ns_per_call("fill_noise_queries_gaussian_qmc (Lattice)", || {
        fill_noise_queries_gaussian_qmc(
            black_box(&mut qmc_source),
            black_box(k),
            black_box(dim),
            black_box(sigma),
            black_box(&mut queries_qmc),
        );
    });

    let mut rng_iid = fastrand::Rng::with_seed(456);
    let mut queries_iid = vec![0.0f32; k * dim];
    let ns_iid_fill = measure_ns_per_call("fill i.i.d. (Box-Muller × K·D)", || {
        for q in &mut queries_iid[..k * dim] {
            *q = standard_normal_box_muller(black_box(&mut rng_iid)) * sigma;
        }
    });

    let fill_delta = (ns_qmc_fill - ns_iid_fill) / ns_iid_fill * 100.0;
    println!("  fill Δ (QMC vs i.i.d.): {fill_delta:+.1}%");

    // ── End-to-end: fill + sample_k_states ────────────────────────────

    let mut qmc_source2 = LatticeQmc::new(789);
    let mut queries_e2e_qmc = vec![0.0f32; k * dim];
    let mut out_qmc = vec![0.0f32; k * dim];
    let ns_qmc_e2e = measure_ns_per_call("QMC fill + sample_k_states (e2e)", || {
        fill_noise_queries_gaussian_qmc(
            black_box(&mut qmc_source2),
            black_box(k),
            black_box(dim),
            black_box(sigma),
            black_box(&mut queries_e2e_qmc),
        );
        kernel.sample_k_states(
            black_box(&s_prev),
            black_box(&x),
            black_box(&queries_e2e_qmc),
            black_box(&mut out_qmc),
            black_box(&cfg),
        );
    });

    let mut rng_e2e = fastrand::Rng::with_seed(321);
    let mut queries_e2e_iid = vec![0.0f32; k * dim];
    let mut out_iid = vec![0.0f32; k * dim];
    let ns_iid_e2e = measure_ns_per_call("i.i.d. fill + sample_k_states (e2e)", || {
        for q in &mut queries_e2e_iid[..k * dim] {
            *q = standard_normal_box_muller(black_box(&mut rng_e2e)) * sigma;
        }
        kernel.sample_k_states(
            black_box(&s_prev),
            black_box(&x),
            black_box(&queries_e2e_iid),
            black_box(&mut out_iid),
            black_box(&cfg),
        );
    });

    let e2e_delta = (ns_qmc_e2e - ns_iid_e2e) / ns_iid_e2e * 100.0;
    println!("  e2e Δ (QMC vs i.i.d.): {e2e_delta:+.1}%");

    // ── Matvec-only (should be identical cost either way) ─────────────

    let mut out_mv = vec![0.0f32; k * dim];
    let ns_matvec = measure_ns_per_call("sample_k_states only (matvec)", || {
        kernel.sample_k_states(
            black_box(&s_prev),
            black_box(&x),
            black_box(&queries_e2e_iid), // reuse pre-filled buffer
            black_box(&mut out_mv),
            black_box(&cfg),
        );
    });
    let fill_fraction = ns_qmc_fill / ns_qmc_e2e * 100.0;
    println!("  matvec-only: {ns_matvec:.2} ns (QMC fill = {fill_fraction:.1}% of e2e)");
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Plan 367 Phase 4 T4.4 — QMC-BoM Fill Overhead Benchmark");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Warmup: {WARMUP_ITERS} iters | Measure: {MEASURE_ITERS} iters");
    println!("Source: LatticeQmc | Baseline: Box-Muller (fastrand)");
    println!();

    // T4.4 target config: K=8, D=4.
    bench_fill_and_sample(4, 8, 0.3);

    // Production config: K=8, D=32 (HLA dimension).
    bench_fill_and_sample(32, 8, 0.3);

    // Large K: K=64, D=4.
    bench_fill_and_sample(4, 64, 0.3);

    println!();
    println!("───────────────────────────────────────────────────────────────");
    println!("  G5 VERDICT: QMC fill overhead must be within 5% of i.i.d.");
    println!("  (the matvec dominates at production D; fill is a small fraction)");
    println!("───────────────────────────────────────────────────────────────");
}
