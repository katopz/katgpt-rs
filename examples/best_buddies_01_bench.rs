//! Best Buddies Drafting Benchmark — Plan 199.
//!
//! Measures Pearson correlation, mutual agreement, filter_marginals, and
//! end-to-end `build_dd_tree_speculative_best_buddies` overhead.
//!
//! ```sh
//! cargo run --features "speculative_generator,best_buddies" --example best_buddies_01_bench
//! ```

#![cfg(all(feature = "speculative_generator", feature = "best_buddies"))]

use std::time::Instant;

use katgpt_core::traits::{BestBuddyAligner, pearson_correlation};
use katgpt_core::{Config, NoPruner};
use katgpt_rs::speculative::{
    MarginalBestBuddyAligner, MarginalTokenGenerator, TokenConstraintPruner,
    build_dd_tree_speculative, build_dd_tree_speculative_best_buddies,
};

// ── Config ─────────────────────────────────────────────────────

const WARMUP: u64 = 50;
const N_ITERS: u64 = 500;

// Vocabulary sizes to sweep (covers small LLMs to large)
const VOCAB_SIZES: &[usize] = &[128, 1024, 8192, 32768];
// Sequence depths (speculative lookahead lengths)
const DEPTHS: &[usize] = &[5, 10, 20];

// ── Helpers ────────────────────────────────────────────────────

/// Simple stats from a sample of timing measurements.
struct Stats {
    p50: f64,
    p99: f64,
    mean: f64,
    min: f64,
}

fn compute_stats(samples: &[f64]) -> Stats {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    // Nearest rank: `ceil(0.99 * n) - 1`. Was `sorted[(n * 99) / 100]`, which
    // is `n - 1` -- the MAX -- for every n <= 100, and the `.min(sorted[n-1])`
    // clamped a panic, not a wrong statistic. Integer arithmetic on purpose:
    // 0.99 has no exact binary form, so a float `ceil` can round back onto the
    // max at exactly the boundary this fixes. At N_ITERS = 500 the tail
    // support is 6 samples, so this p99 still moves if one preemption lands in
    // it -- read it as a thin tail, not as a stable quantile.
    // (.issues/722; scripts/percentile_index_audit.py.)
    let p99_idx = (n * 99).div_ceil(100).saturating_sub(1).min(n - 1);
    Stats {
        p50: sorted[n / 2],
        p99: sorted[p99_idx],
        mean: sorted.iter().sum::<f64>() / n as f64,
        min: sorted[0],
    }
}

/// Create normalized descending marginals for `n_depths` positions × `vocab_size` tokens.
fn make_marginals(n_depths: usize, vocab_size: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(n_depths);
    for _ in 0..n_depths {
        let mut row = Vec::with_capacity(vocab_size);
        let mut sum = 0.0f32;
        for t in 0..vocab_size {
            let v = 1.0 / ((t + 1) as f32);
            row.push(v);
            sum += v;
        }
        for v in &mut row {
            *v /= sum;
        }
        out.push(row);
    }
    out
}

/// Create anti-correlated target marginals (reverse of draft).
fn make_anti_marginals(draft: &[Vec<f32>]) -> Vec<Vec<f32>> {
    draft
        .iter()
        .map(|row| {
            let mut rev = row.clone();
            rev.reverse();
            rev
        })
        .collect()
}

// ── Bench 1: Pearson correlation per position ──────────────────

fn bench_pearson_per_position() {
    println!("── Bench 1: Pearson correlation per position ──\n");
    println!(
        "{:>12} {:>10} {:>10} {:>10} {:>10}",
        "Vocab", "p50 (μs)", "p99 (μs)", "mean (μs)", "min (μs)"
    );
    println!("{}", "-".repeat(56));

    for &vocab in VOCAB_SIZES {
        let a: Vec<f32> = (0..vocab).map(|i| 1.0 / ((i + 1) as f32)).collect();
        let b: Vec<f32> = (0..vocab).map(|i| 1.0 / ((vocab - i) as f32)).collect();

        // Warmup
        for _ in 0..WARMUP {
            let _ = pearson_correlation(&a, &b);
        }

        let mut samples = Vec::with_capacity(N_ITERS as usize);
        for _ in 0..N_ITERS {
            let start = Instant::now();
            let _ = pearson_correlation(&a, &b);
            samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        }

        let stats = compute_stats(&samples);
        println!(
            "{:>12} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            vocab, stats.p50, stats.p99, stats.mean, stats.min,
        );
    }
    println!();
}

// ── Bench 2: mutual_agreement per position ─────────────────────

fn bench_mutual_agreement() {
    println!("── Bench 2: mutual_agreement per position ──\n");
    println!(
        "{:>12} {:>10} {:>10} {:>10} {:>10}",
        "Vocab", "p50 (μs)", "p99 (μs)", "mean (μs)", "min (μs)"
    );
    println!("{}", "-".repeat(56));

    let aligner = MarginalBestBuddyAligner::default();

    for &vocab in VOCAB_SIZES {
        let a: Vec<f32> = (0..vocab).map(|i| 1.0 / ((i + 1) as f32)).collect();
        let b: Vec<f32> = (0..vocab).map(|i| 1.0 / ((vocab - i) as f32)).collect();

        // Warmup
        for _ in 0..WARMUP {
            let _ = aligner.mutual_agreement(&a, &b);
        }

        let mut samples = Vec::with_capacity(N_ITERS as usize);
        for _ in 0..N_ITERS {
            let start = Instant::now();
            let _ = aligner.mutual_agreement(&a, &b);
            samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        }

        let stats = compute_stats(&samples);
        println!(
            "{:>12} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            vocab, stats.p50, stats.p99, stats.mean, stats.min,
        );
    }
    println!();
}

// ── Bench 3: filter_marginals per decode step ───────────────────

fn bench_filter_marginals() {
    println!("── Bench 3: filter_marginals per decode step ──\n");
    println!(
        "{:>12} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Vocab", "Depth", "p50 (μs)", "p99 (μs)", "mean (μs)", "min (μs)",
    );
    println!("{}", "-".repeat(64));

    for &vocab in VOCAB_SIZES {
        for &depth in DEPTHS {
            let draft = make_marginals(depth, vocab);
            let target = make_anti_marginals(&draft);
            let draft_slices: Vec<&[f32]> = draft.iter().map(|m| m.as_slice()).collect();
            let target_slices: Vec<&[f32]> = target.iter().map(|m| m.as_slice()).collect();

            // Warmup
            {
                let mut aligner = MarginalBestBuddyAligner::default();
                for _ in 0..WARMUP {
                    let _ = aligner.filter_marginals(&draft_slices, &target_slices);
                }
            }

            let mut samples = Vec::with_capacity(N_ITERS as usize);
            for _ in 0..N_ITERS {
                let mut aligner = MarginalBestBuddyAligner::default();
                let start = Instant::now();
                let _ = aligner.filter_marginals(&draft_slices, &target_slices);
                samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
            }

            let stats = compute_stats(&samples);
            println!(
                "{:>12} {:>8} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
                vocab, depth, stats.p50, stats.p99, stats.mean, stats.min,
            );
        }
    }
    println!();
}

// ── Bench 4: End-to-end BB pipeline vs standard speculative ────

fn bench_end_to_end() {
    println!(
        "── Bench 4: End-to-end build_dd_tree_speculative_best_buddies vs build_dd_tree_speculative ──\n"
    );

    let vocab = 128; // Realistic for draft model (top-K narrowed)
    let depth = 10;

    let draft = make_marginals(depth, vocab);
    let target = make_anti_marginals(&draft);
    let draft_slices: Vec<&[f32]> = draft.iter().map(|m| m.as_slice()).collect();
    let target_slices: Vec<&[f32]> = target.iter().map(|m| m.as_slice()).collect();

    let config = Config::draft();

    // Warmup standard
    {
        let mut rng = fastrand::Rng::new();
        let mut sampler = MarginalTokenGenerator { top_k: 10 };
        let pruner = TokenConstraintPruner::new(NoPruner);
        for _ in 0..WARMUP {
            let _ =
                build_dd_tree_speculative(&mut sampler, &pruner, &draft_slices, &config, &mut rng);
        }
    }
    // Warmup BB
    {
        let mut rng = fastrand::Rng::new();
        let mut sampler = MarginalTokenGenerator { top_k: 10 };
        let pruner = TokenConstraintPruner::new(NoPruner);
        let mut aligner = MarginalBestBuddyAligner::default();
        for _ in 0..WARMUP {
            let _ = build_dd_tree_speculative_best_buddies(
                &mut sampler,
                &pruner,
                &draft_slices,
                &target_slices,
                &mut aligner,
                &config,
                &mut rng,
            );
        }
    }

    // Measure standard speculative
    let mut std_samples = Vec::with_capacity(N_ITERS as usize);
    {
        let mut rng = fastrand::Rng::new();
        let mut sampler = MarginalTokenGenerator { top_k: 10 };
        let pruner = TokenConstraintPruner::new(NoPruner);
        for _ in 0..N_ITERS {
            let start = Instant::now();
            let tree =
                build_dd_tree_speculative(&mut sampler, &pruner, &draft_slices, &config, &mut rng);
            std_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
            std::hint::black_box(&tree);
        }
    }

    // Measure BB speculative
    let mut bb_samples = Vec::with_capacity(N_ITERS as usize);
    {
        let mut rng = fastrand::Rng::new();
        let mut sampler = MarginalTokenGenerator { top_k: 10 };
        let pruner = TokenConstraintPruner::new(NoPruner);
        let mut aligner = MarginalBestBuddyAligner::default();
        for _ in 0..N_ITERS {
            let start = Instant::now();
            let tree = build_dd_tree_speculative_best_buddies(
                &mut sampler,
                &pruner,
                &draft_slices,
                &target_slices,
                &mut aligner,
                &config,
                &mut rng,
            );
            bb_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
            std::hint::black_box(&tree);
        }
    }

    let std_stats = compute_stats(&std_samples);
    let bb_stats = compute_stats(&bb_samples);

    println!("  Config: vocab={vocab}, depth={depth}, iters={N_ITERS}\n");
    println!(
        "  {:>24} {:>10} {:>10} {:>10} {:>10}",
        "Path", "p50 (μs)", "p99 (μs)", "mean (μs)", "min (μs)"
    );
    println!("  {}", "-".repeat(66));
    println!(
        "  {:>24} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
        "speculative (standard)", std_stats.p50, std_stats.p99, std_stats.mean, std_stats.min,
    );
    println!(
        "  {:>24} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
        "speculative + BB filter", bb_stats.p50, bb_stats.p99, bb_stats.mean, bb_stats.min,
    );

    let overhead_pct = (bb_stats.mean - std_stats.mean) / std_stats.mean * 100.0;
    let overhead_us = bb_stats.mean - std_stats.mean;

    println!("\n  BB overhead: {overhead_us:.2} μs ({overhead_pct:.1}%)");

    match overhead_pct {
        p if p <= 10.0 => println!("  ✅ Overhead ≤ 10% — BB filter is lightweight"),
        p if p <= 30.0 => {
            println!("  ⚠️  Overhead 10-30% — acceptable if acceptance rate improves");
        }
        _ => println!("  ❌ Overhead > 30% — investigate Pearson optimization"),
    }
    println!();
}

// ── Bench 5: Batch alignment confidence vs per-position ────────

fn bench_batch_vs_per_position() {
    println!("── Bench 5: batch_alignment_confidence vs per-position Pearson ──\n");

    let vocab = 8192;
    let depth = 10;

    let draft_flat: Vec<f32> = (0..depth * vocab)
        .map(|i| 1.0 / ((i % vocab + 1) as f32))
        .collect();
    let target_flat: Vec<f32> = (0..depth * vocab)
        .map(|i| 1.0 / ((vocab - (i % vocab)) as f32))
        .collect();

    let aligner = MarginalBestBuddyAligner::default();

    // Warmup
    {
        let mut results = vec![0.0f32; depth];
        for _ in 0..WARMUP {
            aligner.batch_alignment_confidence(&draft_flat, &target_flat, &mut results);
        }
    }

    // Batch path
    let mut batch_samples = Vec::with_capacity(N_ITERS as usize);
    {
        let mut results = vec![0.0f32; depth];
        for _ in 0..N_ITERS {
            let start = Instant::now();
            aligner.batch_alignment_confidence(&draft_flat, &target_flat, &mut results);
            batch_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        }
    }

    // Per-position path
    let mut per_pos_samples = Vec::with_capacity(N_ITERS as usize);
    {
        for _ in 0..N_ITERS {
            let start = Instant::now();
            for i in 0..depth {
                let offset = i * vocab;
                let _ = pearson_correlation(
                    &draft_flat[offset..offset + vocab],
                    &target_flat[offset..offset + vocab],
                );
            }
            per_pos_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        }
    }

    let batch_stats = compute_stats(&batch_samples);
    let per_pos_stats = compute_stats(&per_pos_samples);

    println!("  Config: vocab={vocab}, depth={depth}, iters={N_ITERS}\n");
    println!(
        "  {:>20} {:>10} {:>10} {:>10}",
        "Path", "p50 (μs)", "mean (μs)", "min (μs)"
    );
    println!("  {}", "-".repeat(56));
    println!(
        "  {:>20} {:>10.2} {:>10.2} {:>10.2}",
        "batch (contiguous)", batch_stats.p50, batch_stats.mean, batch_stats.min,
    );
    println!(
        "  {:>20} {:>10.2} {:>10.2} {:>10.2}",
        "per-position loop", per_pos_stats.p50, per_pos_stats.mean, per_pos_stats.min,
    );

    let speedup = per_pos_stats.mean / batch_stats.mean;
    println!("\n  Batch speedup: {speedup:.2}×");
    println!();
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Best Buddies Drafting Benchmark — Plan 199");
    println!("  Warmup: {WARMUP} iters, Measure: {N_ITERS} iters");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    bench_pearson_per_position();
    bench_mutual_agreement();
    bench_filter_marginals();
    bench_end_to_end();
    bench_batch_vs_per_position();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmark Complete");
    println!("═══════════════════════════════════════════════════════════════");
}

// TL;DR: 5-benchmark suite for Plan 199 Best Buddies — Pearson per position,
// mutual agreement, filter_marginals, end-to-end BB pipeline overhead, batch vs
// per-position. Run with: cargo run --features "speculative_generator,best_buddies"
// --example best_buddies_01_bench
