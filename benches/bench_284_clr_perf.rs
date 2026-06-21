//! Plan 284 Phase 4 — CLR `clr_vote_minimal` performance benchmark (G3).
//!
//! Measures per-call latency of `clr_vote_minimal` at three scales:
//!   - K=8,  K=16,  K=32  (all at M=5, direction_dim=8).
//!
//! Target: ≤200µs/call at K=32 (stretch target ≤50µs).
//!
//! # Deviation from the plan
//!
//! The plan (T4.3) specified `criterion::{black_box, criterion_group, ...}`.
//! However, **criterion is not in the root `katgpt-rs/Cargo.toml`
//! `[dev-dependencies]`** — only `ratatui`, `crossterm`, `tempfile`. All
//! existing root-crate benches (`attn_match_router_bench`,
//! `dual_pool_reachability_bench`, `manifold_power_iter_router_bench`, etc.)
//! use `std::time::Instant` with `harness = false` and a custom `main()`.
//!
//! Adding criterion as a dev-dep would require modifying `[dev-dependencies]`,
//! which violates the task constraint: *"Cargo.toml — EXTEND ONLY to add two
//! new `[[bench]]` entries and (if needed) one `[[test]]` entry. Do NOT modify
//! anything else."*
//!
//! This bench follows the established root-crate convention: `std::time::Instant`
//! + `harness = false` + custom `main()`. The `[[bench]]` entry in `Cargo.toml`
//!   still declares `harness = false` as the plan required.
//!
//! Run with:
//! ```bash
//! cargo run --release --features clr --bench bench_284_clr_perf
//! # or
//! cargo bench --no-default-features --features clr --bench bench_284_clr_perf
//! ```

#![cfg(feature = "clr")]

use std::hint::black_box;
use std::time::Instant;

use fastrand::Rng;
use katgpt_rs::clr::{
    Claim, ClrConfig, ClrScratch, DirectionVectorSource, FnClaimExtractor,
    SigmoidProjectionVerifier, Trajectory, clr_vote_minimal,
};

// ──────────────────────────────────────────────────────────────────────────
// Direction source (same pattern as the test binaries)
// ──────────────────────────────────────────────────────────────────────────

struct FlatDirections {
    dim: usize,
    vectors: Vec<f32>,
}

impl FlatDirections {
    fn from_rows(rows: &[&[f32]]) -> Self {
        let dim = rows[0].len();
        let vectors: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Self { dim, vectors }
    }
}

impl DirectionVectorSource for FlatDirections {
    #[inline]
    fn direction(&self, idx: usize) -> &[f32] {
        &self.vectors[idx * self.dim..(idx + 1) * self.dim]
    }
    #[inline]
    fn blake3(&self) -> [u8; 32] {
        [0u8; 32]
    }
    #[inline]
    fn version(&self) -> u64 {
        1
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Suite builder
// ──────────────────────────────────────────────────────────────────────────

const BENCH_M: usize = 5;
const BENCH_DIM: usize = 8;
const WARMUP_ITERS: usize = 1_000;
const MEASURE_ITERS: usize = 100_000;

/// Build K trajectories with M claims each. Embeddings are parallel to their
/// respective direction vectors (high verdict), mimicking a realistic "all
/// candidates are decent" scenario.
fn build_suite(k: usize, seed: u64) -> (Vec<Trajectory<u8>>, FlatDirections) {
    let mut rng = Rng::with_seed(seed);

    let mut dir_rows: Vec<Vec<f32>> = Vec::with_capacity(BENCH_M);
    for _ in 0..BENCH_M {
        let mut v: Vec<f32> = (0..BENCH_DIM).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in v.iter_mut() {
            *x /= norm;
        }
        dir_rows.push(v);
    }
    let dir_refs: Vec<&[f32]> = dir_rows.iter().map(|v| v.as_slice()).collect();
    let directions = FlatDirections::from_rows(&dir_refs);

    let mut trajectories: Vec<Trajectory<u8>> = Vec::with_capacity(k);
    for i in 0..k {
        let outcome = (i % 4) as u8;
        let claims: Vec<Claim<u8>> = (0..BENCH_M)
            .map(|m| {
                let dir = directions.direction(m);
                let emb: Vec<f32> = (0..BENCH_DIM)
                    .map(|d| dir[d] + (rng.f32() * 2.0 - 1.0) * 0.1)
                    .collect();
                Claim {
                    embedding: emb,
                    payload: outcome,
                }
            })
            .collect();
        trajectories.push(Trajectory {
            outcome,
            tokens_or_steps: 100 + i,
            claims,
            log_probs: None,
        });
    }

    (trajectories, directions)
}

/// Measure per-call latency of `clr_vote_minimal` for a given K.
fn bench_k(k: usize) -> (f64, f64, f64) {
    let (trajectories, directions) = build_suite(k, 42);
    let config = ClrConfig {
        k,
        m: BENCH_M,
        ..ClrConfig::default()
    };
    let extractor = FnClaimExtractor::new(BENCH_M, |t: &Trajectory<u8>| t.claims.clone());
    let verifier = SigmoidProjectionVerifier::new(&directions, BENCH_DIM);
    let outcome_eq = |a: &u8, b: &u8| a == b;
    let mut scratch = ClrScratch::new(config.k, config.m);

    // Warm up: prime caches, JIT (none in Rust but primes the allocator cache).
    let mut sink = 0usize;
    let mut rel_sink = 0.0f32;
    for _ in 0..WARMUP_ITERS {
        let (w, r) = clr_vote_minimal(
            black_box(&trajectories),
            black_box(&extractor),
            black_box(&verifier),
            black_box(&config),
            black_box(&outcome_eq),
            black_box(&mut scratch),
        );
        sink = sink.wrapping_add(w);
        rel_sink += r;
    }
    black_box((sink, rel_sink));

    // Measure.
    let mut samples: Vec<f64> = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let start = Instant::now();
        let (w, r) = clr_vote_minimal(
            black_box(&trajectories),
            black_box(&extractor),
            black_box(&verifier),
            black_box(&config),
            black_box(&outcome_eq),
            black_box(&mut scratch),
        );
        let elapsed = start.elapsed();
        black_box((w, r));
        samples.push(elapsed.as_secs_f64() * 1e6); // → microseconds
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50 = samples[samples.len() / 2];
    let p99 = samples[(samples.len() as f64 * 0.99) as usize];

    (mean, p50, p99)
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Plan 284 Phase 4 — CLR clr_vote_minimal Performance (G3)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Config: M={BENCH_M}, direction_dim={BENCH_DIM}");
    println!("Warmup: {WARMUP_ITERS} iters | Measure: {MEASURE_ITERS} iters");
    println!("Target: ≤200µs/call at K=32 (stretch ≤50µs)");
    println!();
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>10}",
        "K", "mean (µs)", "p50 (µs)", "p99 (µs)", "pass?"
    );
    println!("{}", "─".repeat(6 + 12 + 12 + 12 + 10));

    for &k in &[8usize, 16, 32] {
        let (mean, p50, p99) = bench_k(k);
        let pass = if mean <= 200.0 { "✅" } else { "❌" };
        let stretch = if mean <= 50.0 { " ✨stretch" } else { "" };
        println!(
            "{:>6} {:>12.2} {:>12.2} {:>12.2} {:>10}{}",
            k, mean, p50, p99, pass, stretch
        );
    }

    println!();
    println!("Note: per-call allocations from the extractor (~{} allocs/call for", BENCH_M + 1);
    println!("      K trajectories × clone-based FnClaimExtractor) are included in");
    println!("      this measurement. A pre-extracted hot-path variant would be");
    println!("      faster. The vote arithmetic itself is zero-alloc (see G4).");
}
