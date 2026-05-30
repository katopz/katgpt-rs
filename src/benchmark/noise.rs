//! ELF SDE noise scheduling benchmarks.
//!
//! Measures throughput of noise injection operations across different
//! noise scales and dimensions, covering the "Noise" feature dimension
//! from the Paper Feature Comparison Matrix.

use super::{BenchCategory, BenchResult};
use crate::types::Config;
use std::time::Instant;

/// Benchmark ELF SDE noise injection across different configurations.
///
/// Tests:
/// - SDE noise injection on 1D vectors (logit perturbation)
/// - SDE noise injection on 2D attention scores
/// - Noise schedule decay curves (linear, cosine, exponential)
///
/// Returns BenchResult entries tagged with feature_dim = "Noise".
pub fn bench_elf_sde(_config: &Config) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let warmup = 1_000;
    let iters = 50_000;

    println!("\n🔊 ELF SDE Noise Scheduling...");
    println!("   ({iters} iterations, {warmup} warmup)");

    // ── T1: SDE 1D noise injection (logit perturbation) ──
    let dims = [16usize, 64, 128, 256];
    for &dim in &dims {
        let mut data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let sigma = 0.1f32;
        let mut rng = fastrand::Rng::with_seed(42);

        // Warmup
        for _ in 0..warmup {
            inject_sde_noise_1d(&mut data, sigma, &mut rng);
        }

        let start = Instant::now();
        for _ in 0..iters {
            inject_sde_noise_1d(&mut data, sigma, &mut rng);
        }
        let elapsed = start.elapsed();
        let tp = iters as f64 / elapsed.as_secs_f64();
        let us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

        results.push(BenchResult {
            label: format!("SDE noise 1D (dim={dim})"),
            throughput: tp,
            time_per_step_us: us,
            avg_acceptance_len: 0.0,
            color: (240, 228, 66), // yellow
            category: BenchCategory::Noise,
            feature_dim: "Noise".into(),
        });
    }

    // ── T2: Noise schedule decay ──
    #[allow(clippy::type_complexity)]
    let schedule_configs: &[(&str, fn(f32) -> f32)] = &[
        ("linear", |t| 1.0 - t),
        ("cosine", |t| ((1.0 - t) * std::f32::consts::PI * 0.5).cos()),
        ("exponential", |t| (-(t * 5.0)).exp()),
    ];

    let n_steps = 1000;
    for &(name, schedule_fn) in schedule_configs {
        let mut rng = fastrand::Rng::with_seed(42);

        // Warmup
        for _ in 0..warmup {
            apply_noise_schedule(n_steps, schedule_fn, &mut rng);
        }

        let start = Instant::now();
        for _ in 0..iters {
            apply_noise_schedule(n_steps, schedule_fn, &mut rng);
        }
        let elapsed = start.elapsed();
        let tp = iters as f64 * n_steps as f64 / elapsed.as_secs_f64();
        let us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

        results.push(BenchResult {
            label: format!("Noise schedule ({name})"),
            throughput: tp,
            time_per_step_us: us,
            avg_acceptance_len: 0.0,
            color: (200, 200, 0), // olive
            category: BenchCategory::Noise,
            feature_dim: "Noise".into(),
        });
    }

    // ── T3: Path diversity measurement ──
    {
        let n_paths = 8;
        let dim = 64;
        let mut rng = fastrand::Rng::with_seed(42);

        // Pre-allocate buffers to avoid per-iteration allocation
        let mut base_buf: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut paths_buf: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; n_paths];

        for _ in 0..warmup {
            let _ = measure_path_diversity_with_buffers(
                n_paths,
                dim,
                &mut rng,
                &mut base_buf,
                &mut paths_buf,
            );
        }

        let start = Instant::now();
        for _ in 0..iters {
            let _ = measure_path_diversity_with_buffers(
                n_paths,
                dim,
                &mut rng,
                &mut base_buf,
                &mut paths_buf,
            );
        }
        let elapsed = start.elapsed();
        let tp = iters as f64 / elapsed.as_secs_f64();
        let us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

        results.push(BenchResult {
            label: "Path diversity (8×64)".into(),
            throughput: tp,
            time_per_step_us: us,
            avg_acceptance_len: 0.0,
            color: (230, 200, 50),
            category: BenchCategory::Noise,
            feature_dim: "Noise".into(),
        });
    }

    // Print summary
    println!("\n   {:<30} {:>12} {:>12}", "Method", "ops/s", "μs/op");
    println!("   {}", "-".repeat(56));
    for r in &results {
        println!(
            "   {:<30} {:>12.0} {:>12.2}",
            r.label, r.throughput, r.time_per_step_us,
        );
    }

    results
}

/// Inject SDE noise into a 1D vector: x += σ · N(0,1).
/// Uses Box-Muller for approximate Gaussian sampling.
fn inject_sde_noise_1d(data: &mut [f32], sigma: f32, rng: &mut fastrand::Rng) {
    let mut i = 0;
    let n = data.len();
    while i + 1 < n {
        // Box-Muller transform
        let u1 = rng.f32().max(1e-10);
        let u2 = rng.f32();
        let mag = sigma * (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
        let z1 = mag * (2.0 * std::f32::consts::PI * u2).sin();
        data[i] += z0;
        data[i + 1] += z1;
        i += 2;
    }
    // Handle odd element
    if i < n {
        let u1 = rng.f32().max(1e-10);
        let u2 = rng.f32();
        let z0 = sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        data[i] += z0;
    }
}

/// Apply a noise schedule function across n steps and return the final sigma.
fn apply_noise_schedule(
    n_steps: usize,
    schedule_fn: fn(f32) -> f32,
    rng: &mut fastrand::Rng,
) -> f32 {
    let mut sigma = 0.0f32;
    for step in 0..n_steps {
        let t = step as f32 / n_steps as f32;
        sigma = schedule_fn(t);
        // Consume RNG to simulate realistic noise application
        let _ = rng.f32();
    }
    sigma
}

/// Measure path diversity: generate n_paths noisy variants and compute avg pairwise cosine distance.
/// Uses pre-allocated buffers to avoid per-call allocation.
fn measure_path_diversity_with_buffers(
    n_paths: usize,
    _dim: usize,
    rng: &mut fastrand::Rng,
    base: &mut [f32],
    paths: &mut [Vec<f32>],
) -> f32 {
    // Reinitialize base vector in-place
    for (i, b) in base.iter_mut().enumerate() {
        *b = (i as f32 * 0.1).sin();
    }
    // Generate noisy variants into pre-allocated paths
    for path in paths.iter_mut() {
        path.copy_from_slice(base);
        inject_sde_noise_1d(path, 0.1, rng);
    }

    // Average pairwise cosine distance
    let mut total_dist = 0.0f32;
    let mut count = 0;
    for i in 0..n_paths {
        for j in (i + 1)..n_paths {
            total_dist += cosine_distance(&paths[i], &paths[j]);
            count += 1;
        }
    }
    if count > 0 {
        total_dist / count as f32
    } else {
        0.0
    }
}

/// Original allocating version kept for non-hot-path callers.
#[allow(dead_code)]
fn measure_path_diversity(n_paths: usize, dim: usize, rng: &mut fastrand::Rng) -> f32 {
    let mut base: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut paths: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; n_paths];
    measure_path_diversity_with_buffers(n_paths, dim, rng, &mut base, &mut paths)
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        return 1.0;
    }
    1.0 - dot / (na * nb)
}
